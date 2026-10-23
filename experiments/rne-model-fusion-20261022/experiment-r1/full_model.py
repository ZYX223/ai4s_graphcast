"""Official GraphCast rollout; timing boundaries match the weekly experiments."""

import argparse
import ctypes as C
import datetime
import importlib.metadata as metadata
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr

from graphcast_bench.assets import (
    ROOT,
    STATS,
    code_snapshot,
    load_case,
    read_json,
    sha256,
    verify_source,
    write_json,
)
from graphcast_bench.metrics import compare, load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("h100", "hygon"), required=True)
    p.add_argument("--mode", choices=("tf32", "fp32", "xf32", "rne_xf32", "all"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--variant",choices=["baseline","fused"],required=True)
    p.add_argument("--logical-catalog", action="store_true")
    args = p.parse_args()
    if (
        args.repeats < 1
        or (args.platform == "h100" and args.mode != "tf32")
        or (args.platform == "hygon" and args.mode == "tf32")
    ):
        p.error("Unsupported platform/mode or repeat count")
    case, prepared = load_case(args.data)
    upstream = verify_source()
    code = code_snapshot()
    config = read_json(ROOT / "configs" / f"{args.platform}.json")
    args.output.mkdir(parents=True, exist_ok=False)
    import haiku as hk
    import jax
    from graphcast import autoregressive, checkpoint, graphcast, normalization, rollout, xarray_jax

    if not Path(graphcast.__file__).resolve().is_relative_to(ROOT / "third_party/graphcast"):
        raise RuntimeError("Imported model is not the pinned official source")
    if metadata.version("jax") != config["jax_version"]:
        raise RuntimeError("JAX version differs from tested platform configuration")
    devices = jax.devices("gpu")
    if len(devices) != 1 or config["expected_device"] not in devices[0].device_kind:
        raise RuntimeError("Expected exactly one configured accelerator")
    if jax.config.jax_enable_x64 or os.environ.get("ROCBLAS_MATH_MODE") is not None:
        raise RuntimeError("Unexpected global precision override")
    device = devices[0]
    bus = C.create_string_buffer(64)
    if args.platform == "hygon":
        from graphcast_bench.router import Router

        if args.variant == "fused":
            from candidate_router import CandidateRouter
            router = CandidateRouter()
        else:
            router = Router()
        driver = C.CDLL("libamdhip64.so")
        assert driver.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
        synchronize = driver.hipDeviceSynchronize
    else:
        router = None
        driver = C.CDLL("libcuda.so.1")
        assert driver.cuInit(0) == 0
        assert driver.cuDeviceGetPCIBusId(bus, len(bus), 0) == 0
        synchronize = driver.cuCtxSynchronize
    if bus.value.decode().lower() != os.environ["GRAPHCAST_EXPECTED_PCI"].lower():
        raise RuntimeError("Container selected the wrong physical accelerator")
    with (args.data / "assets/jax_weights.npz").open("rb") as f:
        weights = checkpoint.load(f, graphcast.CheckPoint)
    assert all(np.asarray(v).dtype == np.float32 for v in jax.tree_util.tree_leaves(weights.params))
    inputs, forcings, target = [
        xr.load_dataset(args.data / "assets" / name, decode_timedelta=True)
        for name in ("inputs.nc", "rollout_forcings.nc", "target_template.nc")
    ]
    stats = {
        name.removesuffix(".nc"): xr.load_dataset(args.data / "assets/stats" / name).astype(
            np.float32
        )
        for name in STATS
    }
    targets = rollout.extend_targets_template(target.drop_vars("datetime", errors="ignore"), 40)
    targets = targets.assign_coords(datetime=forcings.datetime)
    reference = load_reference(args.data / "reference.npz", case)

    from model_hook import RneHook
    policy=read_json(Path("/experiment/policy.json"))
    hook=RneHook({row["module"]:row for row in policy["layers"]})
    experiment_hashes={p.name:sha256(p) for p in Path("/experiment").iterdir() if p.is_file()}

    @hk.transform_with_state
    def predict(inputs, targets_template, forcings):
        model = graphcast.GraphCast(weights.model_config, weights.task_config)
        model = normalization.InputsAndResiduals(model, **stats)
        model = autoregressive.Predictor(model, gradient_checkpointing=False)
        if args.variant == "fused":
            with hk.intercept_methods(hook):
                return model(inputs, targets_template=targets_template, forcings=forcings)
        return model(inputs, targets_template=targets_template, forcings=forcings)

    rows, warmups, audit = [], [], []
    report = {
        "schema": "graphcast.rne_model_fusion.v1",
        "variant":args.variant,
        "experiment_sha256":experiment_hashes,
        "status": "running",
        "platform": args.platform,
        "mode": args.mode,
        "source": upstream,
        "code_sha256": code,
        "data": prepared,
        "original_case_sha256": case["source_case_sha256"],
        "versions": {
            k: metadata.version(k) for k in ("jax", "jaxlib", "dm-haiku", "numpy", "xarray")
        },
        "device": device.device_kind,
        "pci": bus.value.decode(),
        "profile_only": args.profile,
        "warmup_steps": 5,
        "logical_catalog_only": args.logical_catalog,
        "measured_steps": 1 if args.profile or args.logical_catalog else 40,
        "repeats": 1 if args.profile or args.logical_catalog else args.repeats,
        "rows": rows,
        "warmups": warmups,
        "router_build": router.build if router else None,
        "storage": "float32",
        "jax_matmul_precision": "tensorfloat32",
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "timing": "Synchronized predictor calls; preloaded-data rollout wall includes transfer and autoregressive update. Excludes load, compile, warmup, metrics and save. Hygon includes existing routing/audit and RNE kernels. Graphs disabled.",
    }
    write_json(args.output / "report.json", report)

    def telemetry():
        if args.platform == "h100":
            cmd = [
                "nvidia-smi",
                "--query-gpu=uuid,name,pci.bus_id,memory.used,utilization.gpu,clocks.sm",
                "--format=csv,noheader",
            ]
        else:
            cmd = [
                "/opt/hyhal/bin/hy-smi",
                "-d",
                os.environ["GRAPHCAST_PHYSICAL_DEVICE"],
                "--showuse",
                "--showmeminfo",
                "vram",
                "--showclocks",
                "--json",
            ]
        return subprocess.check_output(cmd, text=True)

    with jax.default_device(device), jax.default_matmul_precision("tensorfloat32"):
        params = jax.device_put(weights.params)
        apply = jax.jit(
            lambda rng, inputs, targets_template, forcings: predict.apply(
                params, {}, rng, inputs, targets_template, forcings
            )[0]
        )

        def execute(mode, label, steps):
            if router:
                router.select(mode, label)
            times, first = [], None

            def timed_apply(**kwargs):
                nonlocal first
                if first is None:
                    first = kwargs
                before = router.snapshot() if router else None
                start = time.perf_counter()
                result = apply(**kwargs)
                jax.block_until_ready(result)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                counts = router.verify(before, mode) if router else {}
                audit.append({"label": label, "step": len(times), "seconds": elapsed, **counts})
                return result

            assert synchronize() == 0
            start = time.perf_counter()
            last = None
            for chunk in rollout.chunked_prediction_generator(
                predictor_fn=timed_apply,
                rng=jax.random.PRNGKey(0),
                inputs=inputs,
                targets_template=targets.isel(time=slice(0, steps)),
                forcings=forcings.isel(time=slice(0, steps)),
                num_steps_per_chunk=1,
            ):
                last = chunk
            jax.block_until_ready(last)
            wall = time.perf_counter() - start
            assert last is not None and len(times) == steps
            return times, wall, last, first

        all_orders = [
            ["fp32", "xf32", "rne_xf32"],
            ["xf32", "rne_xf32", "fp32"],
            ["rne_xf32", "fp32", "xf32"],
        ]
        for i in range(report["repeats"]):
            order = all_orders[i % 3] if args.mode == "all" else [args.mode]
            for mode in order:
                name = f"round{i + 1}-{mode}"
                print("Warmup:", name, flush=True)
                wt, ww, _, kwargs = execute(mode, name + "-warmup", 5)
                warmups.append({"run": name, "seconds": wt, "wall_seconds": ww})
                if len(warmups) == 1:
                    (args.output / "optimized-hlo.txt").write_text(
                        apply.lower(**kwargs).compile().as_text()
                    )
                    if args.logical_catalog:
                        from .logical_inventory import export_jaxpr
                        export_jaxpr(jax.make_jaxpr(apply)(**kwargs), args.output / "logical")
                if args.variant == "fused":
                    hook.verify_traced()
                cache = apply._cache_size()
                before = telemetry()
                if args.profile:
                    runtime = C.CDLL("libcudart.so" if args.platform == "h100" else "libamdhip64.so")
                    start_profile = getattr(runtime, "cudaProfilerStart" if args.platform == "h100" else "hipProfilerStart")
                    stop_profile = getattr(runtime, "cudaProfilerStop" if args.platform == "h100" else "hipProfilerStop")
                    assert start_profile() == 0
                try:
                    times, wall, last, _ = execute(mode, name, report["measured_steps"])
                finally:
                    if args.profile:
                        assert stop_profile() == 0
                after = telemetry()
                if apply._cache_size() != cache:
                    raise RuntimeError("Compilation occurred during measurement")
                row = {
                    "run": name,
                    "mode": mode,
                    "round": i + 1,
                    "apply_seconds": times,
                    "apply_median_seconds": statistics.median(times),
                    "apply_sum_seconds": sum(times),
                    "rollout_wall_seconds": wall,
                    "jit_cache_before": cache,
                    "jit_cache_after": apply._cache_size(),
                    "device_before": before,
                    "device_after": after,
                }
                folder = args.output / name
                folder.mkdir()
                if not args.logical_catalog:
                    host = xr.Dataset(
                        {
                            k: (
                                v.dims,
                                np.asarray(jax.device_get(xarray_jax.unwrap_data(v))),
                                target[k].attrs,
                            )
                            for k, v in last.data_vars.items()
                        },
                        coords=last.coords,
                    )
                    accuracy = None if args.profile else compare(host, reference, case=case)
                    host.to_netcdf(folder / "prediction.nc")
                    row.update(accuracy=accuracy, output_sha256=sha256(folder / "prediction.nc"))
                    del host
                write_json(folder / "report.json", row)
                rows.append({k: v for k, v in row.items() if k != "accuracy"})
                write_json(args.output / "report.json", report)
                print(
                    name,
                    "wall seconds:",
                    wall,
                    "accuracy:",
                    (row.get("accuracy") or {}).get("overall_mean_percent"),
                    flush=True,
                )
                del last
    load_case(args.data)
    if code_snapshot() != code:
        raise RuntimeError("Code changed during execution")
    if {p.name:sha256(p) for p in Path("/experiment").iterdir() if p.is_file()}!=experiment_hashes:
        raise RuntimeError("Experiment changed during execution")
    write_json(args.output / "hook-audit.json",dict(seen=sorted(hook.seen),expected=sorted(hook.targets),variant=args.variant))
    write_json(args.output / "step-audit.json", audit)
    report.update(
        status="completed",
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        optimized_hlo_sha256=sha256(args.output / "optimized-hlo.txt"),
    )
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
