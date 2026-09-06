#!/usr/bin/env python3
"""Run one strict inference reference. Use separate Torch/JAX environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.inference import inference_reference as reference  # noqa: E402
from scripts.inference.diagnose_inference import sha256  # noqa: E402


def reference_environment(environment: dict[str, str]) -> dict[str, str]:
    result = {key: value for key, value in environment.items()
              if not key.startswith(("GRAPHCAST_", "FLAGGEMS_"))}
    result.update(GRAPHCAST_FLAGOS_MODE="off", GRAPHCAST_BACKEND="nvidia",
                  NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    return result


def torch_arguments(case_path: Path, case: dict, output: Path, device: str) -> list[str]:
    trajectory = case["lead_hours"] != case["steps"] * case["step_hours"]
    options = {f"--{name.replace('_', '-')}": str(case_path / reference.ASSETS[name])
               for name in ("config", "weights", "inputs", "rollout_forcings", "target_template")}
    options.update({"--stats-dir": str(case_path / "assets/stats"),
        "--output": str(output / ("final_prediction.nc" if trajectory else "prediction.nc")),
        "--report": str(output / "report.json"), "--platform": "h100", "--device": device,
        "--model-execution-profile": "baseline", "--precision": "fp32",
        "--steps": str(case["steps"]), "--step-hours": str(case["step_hours"]), "--warmup-calls": "1"})
    if trajectory:
        options["--trajectory-output"] = str(output / "prediction.nc")
    return [value for pair in options.items() for value in pair]


def project_source() -> dict:
    paths = sorted(path for folder in (ROOT / "src", ROOT / "scripts") for path in folder.rglob("*.py"))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    return {"tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "files": hashes}


def verify_official_source(root: Path) -> dict:
    """A nominal version is insufficient: check the bytes imported from the pin."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    try:
        if git("rev-parse", "HEAD").decode().strip() != reference.OFFICIAL_COMMIT:
            raise ValueError("official GraphCast checkout must match the pinned commit")
        files = git("ls-tree", "-r", "--name-only", reference.OFFICIAL_COMMIT, "graphcast", "setup.py").decode().splitlines()
        hashes = {}
        for name in files:
            if not name.endswith(".py"):
                continue
            expected = hashlib.sha256(git("show", f"{reference.OFFICIAL_COMMIT}:{name}")).hexdigest()
            if sha256(root / name) != expected:
                raise ValueError(f"official source differs from pinned bytes: {name}")
            hashes[name] = expected
        actual = {str(path.relative_to(root)) for path in (root / "graphcast").rglob("*.py")}
        if actual != {name for name in hashes if name.startswith("graphcast/")}:
            raise ValueError("untracked Python files in official source tree")
        if "graphcast/graphcast.py" not in hashes:
            raise ValueError("official GraphCast source missing")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify official GraphCast checkout") from error
    return {"commit": reference.OFFICIAL_COMMIT, "verified_official": True,
            "tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def validate_torch_report(report: dict) -> None:
    precision, workload = report.get("precision", {}), report.get("workload", {})
    if (report.get("status") != "passed" or report.get("runtime", {}).get("flagos_mode") != "off"
            or precision.get("effective") != "strict_fp32" or precision.get("float32_math_mode") != "ieee_fp32"
            or precision.get("autocast", {}).get("enabled") is not False
            or any(precision.get(key) != "torch.float32" for key in (
                "parameter_dtype", "state_dtype", "forcing_dtype", "model_input_dtype",
                "model_output_dtype", "feedback_dtype", "final_output_dtype"))):
        raise ValueError("Torch run did not demonstrate strict Native FP32 throughout")
    if (workload.get("model_execution_profile", {}).get("name") != "baseline"
            or workload.get("edge_input_projection") != "concatenated"
            or workload.get("mesh2grid_full_grid_fast_path") is not False
            or workload.get("index_add_execution") != "inline"):
        raise ValueError("Torch run did not use the baseline model profile")


def run_torch(case_path: Path, case: dict, output: Path, device: str) -> dict:
    import torch

    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("Torch reference requires an explicitly selected H100 CUDA device")
    device_name = torch.cuda.get_device_name(torch.device(device))
    if "H100" not in device_name:
        raise ValueError(f"expected H100, found {device_name}")
    # No parallel copy of the model/rollout; execute the same production entry.
    with (output / "console.log").open("w") as stream:
        subprocess.run([sys.executable, str(ROOT / "scripts/inference/run_operational_40step_inference.py"),
                        *torch_arguments(case_path, case, output, device)],
                       env=reference_environment(dict(os.environ)), stdout=stream, stderr=subprocess.STDOUT, check=True)
    raw = json.loads((output / "report.json").read_text())
    validate_torch_report(raw)
    for key, value in reference.case_source(case).items():
        if key == "jax_weights_sha256":  # Production Torch does not consume JAX weights.
            continue
        actual = raw["source"].get(key)
        if key == "normalization_stats":
            actual = {name: {"sha256": actual[name]["sha256"]} for name in value}
        if actual != value:
            raise ValueError(f"Torch source identity mismatch: {key}")
    return {"device_name": device_name, "device_type": "cuda", "device": device,
            "flagos_mode": "off", "versions": raw["runtime"]["versions"]}


def run_jax(case_path: Path, case: dict, output: Path, device: str, source: Path) -> dict:
    # Configure before imports/tracing; dtype alone does not exclude TF32.
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ["JAX_ENABLE_X64"] = "false"
    sys.path.insert(0, str(source.resolve()))
    import jax
    import haiku as hk
    from graphcast import autoregressive, checkpoint, graphcast, normalization, rollout, xarray_jax

    if not Path(graphcast.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("JAX imported a different GraphCast installation")
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)
    platform, ordinal = device.split(":")
    selected = jax.devices(platform)[int(ordinal)]
    with (case_path / reference.ASSETS["jax_weights"]).open("rb") as stream:
        weights = checkpoint.load(stream, graphcast.CheckPoint)
    if any(np.asarray(value).dtype != np.float32 for value in jax.tree_util.tree_leaves(weights.params)):
        raise ValueError("official checkpoint parameters are not FP32")
    inputs, forcings, target = [xr.load_dataset(case_path / reference.ASSETS[key], decode_timedelta=True)
                               for key in ("inputs", "rollout_forcings", "target_template")]
    stats = {name.removesuffix(".nc"): xr.load_dataset(case_path / "assets/stats" / name).astype(np.float32)
             for name in reference.diagnostic.STATS_FILES}
    # The official helper expands only the template lazily, not the truth or
    # forcing. Drop datetime here because the helper expects a 1-D coordinate;
    # reattach the frozen bundle's batch/time coordinate without recomputation.
    targets = rollout.extend_targets_template(target.drop_vars("datetime", errors="ignore"), case["steps"])
    targets = targets.assign_coords(datetime=forcings.datetime)

    @hk.transform_with_state
    def predict(inputs, targets_template, forcings):
        model = graphcast.GraphCast(weights.model_config, weights.task_config)
        model = normalization.InputsAndResiduals(model, **stats)
        model = autoregressive.Predictor(model, gradient_checkpointing=False)
        return model(inputs, targets_template=targets_template, forcings=forcings)

    with jax.default_device(selected), jax.default_matmul_precision("highest"):
        params = jax.device_put(weights.params)
        apply = jax.jit(lambda rng, inputs, targets_template, forcings:
                        predict.apply(params, {}, rng, inputs, targets_template, forcings)[0])
        saved = []
        for chunk in rollout.chunked_prediction_generator(
            predictor_fn=apply, rng=jax.random.PRNGKey(0), inputs=inputs,
            targets_template=targets, forcings=forcings, num_steps_per_chunk=1,
        ):
            # Diagnostic run only. Check each autoregressive result, retain the
            # selected lead plus final horizon, and avoid a full 40-lead RAM copy.
            host = xr.Dataset({name: (value.dims, np.asarray(jax.device_get(
                xarray_jax.unwrap_data(value))), target[name].attrs) for name, value in chunk.data_vars.items()},
                              coords=chunk.coords)
            if any(value.dtype != np.float32 or not np.isfinite(value.values).all() for value in host.data_vars.values()):
                raise ValueError("non-finite or non-FP32 official JAX prediction")
            lead = int(host.time.values[0] / np.timedelta64(1, "h"))
            if lead in (case["lead_hours"], case["steps"] * case["step_hours"]):
                saved.append(host)
    prediction = xr.concat(saved, dim="time") if len(saved) > 1 else saved[0]
    temporary = output / ".prediction.nc.tmp"
    prediction.to_netcdf(temporary)
    temporary.replace(output / "prediction.nc")
    return {"device_name": selected.device_kind, "device_type": selected.platform,
            "device": str(selected), "flagos_mode": "off", "jax_jit": True,
            "versions": {"jax": jax.__version__, "haiku": hk.__version__, "numpy": np.__version__}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--engine", choices=("jax", "torch"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Torch: cuda:0; JAX: gpu:0 (or explicitly cpu:0)")
    parser.add_argument("--jax-source-root", type=Path)
    parser.add_argument("--check-only", action="store_true", help="verify assets/source only; no imports of Torch/JAX or device work")
    args = parser.parse_args(argv)
    try:
        case_path, output = args.case.resolve(), args.output_dir.resolve()
        case = reference.load_case(case_path)
        case_digest = sha256(case_path / "manifest.json")
        code = project_source()
        if args.engine == "jax" and args.jax_source_root is None:
            raise ValueError("JAX requires --jax-source-root with the pinned official checkout")
        engine_source = verify_official_source(args.jax_source_root) if args.engine == "jax" else code
        if output.exists():
            raise FileExistsError(output)
        if args.check_only:
            print("assets/source verified; no device run, no qualified baseline")
            return 0
        # Each process must start independently, not reuse warmed hidden model state.
        if "torch" in sys.modules or "jax" in sys.modules or "flag_gems" in sys.modules:
            raise ValueError("launch the reference runner in a fresh Python process")
        environment = reference_environment(dict(os.environ))
        os.environ.clear()
        os.environ.update(environment)
        output.mkdir(parents=True, exist_ok=False)
        runtime = (run_torch(case_path, case, output, args.device) if args.engine == "torch" else
                   run_jax(case_path, case, output, args.device, args.jax_source_root))
        reference.load_case(case_path)
        if sha256(case_path / "manifest.json") != case_digest or project_source() != code:
            raise ValueError("case or runner source changed during execution")
        if args.engine == "jax" and verify_official_source(args.jax_source_root) != engine_source:
            raise ValueError("official source changed during execution")
        reference.write_json(output / "reference_report.json", {
            "schema": "graphcast.inference_reference_run.v1", "status": "passed",
            "role": "pytorch_gpu" if args.engine == "torch" else "jax_fp32_no_bf16",
            "execution_id": str(uuid.uuid4()), "case_sha256": case_digest,
            "source": reference.case_source(case), "runner_sha256": sha256(Path(__file__)),
            "engine_source": engine_source, "runtime": runtime,
            "precision": {"effective": "strict_fp32", "parameter_dtype": "float32", "data_dtype": "float32",
                          "matmul_precision": "highest", "autocast_enabled": False},
            "workload": {"steps": case["steps"], "step_hours": case["step_hours"],
                         "model_profile": "baseline", "compile": False},
            "artifacts": {"prediction": {"sha256": sha256(output / "prediction.nc")}},
            "performance_eligible": False,
        })
        print("reference run recorded; dual-reference qualification still required")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        print(f"reference run failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
