"""Full-model short training, one precision and rollout horizon per process.

Each process resumes its own trajectory. Recompilation and cache lifetime cannot
make later horizons retain earlier executables. Native SGEMM precision is selected
at runtime; JAX differentiates the original floating-point model.
"""
import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata as md
import json
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from graphcast import autoregressive, checkpoint, data_utils, normalization, xarray_jax
from remat_policy import rematerialize, CheckpointedGraphCast
from runtime import Precision, Runtime, tree_digest, verify_reload
from data_transfer import to_device
from model import make_loss
import recipe, remat_policy
from curriculum import select_updates
from streamed_microbatches import two_microbatch_grads

FROZEN_TRAIN_SHA256 = "77df16253040abf114c6ca24cd024d907e5489de29a8a7800671a0c5b09e1969"
assert (
    hashlib.sha256(Path("/original/experiment/train.py").read_bytes()).hexdigest()
    == FROZEN_TRAIN_SHA256
)


def save_report(output, name, obj):
    p = output / name
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(obj, indent=2) + "\n")
    q.replace(p)


def tree_numpy(tree):
    return jax.tree.map(lambda x: np.asarray(x), jax.device_get(tree))


def flatten(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])


def comparison(actual, reference):
    sq = 0.0
    ref_sq = 0.0
    l1 = 0.0
    ref_l1 = 0.0
    mx = 0.0
    for i in range(0, actual.size, 1000000):
        a = actual[i : i + 1000000].astype(np.float64)
        b = reference[i : i + 1000000].astype(np.float64)
        d = a - b
        sq += float(d @ d)
        ref_sq += float(b @ b)
        l1 += float(np.abs(d).sum())
        ref_l1 += float(np.abs(b).sum())
        mx = max(mx, float(np.max(np.abs(d))))
    return dict(
        relative_l2=float(np.sqrt(sq / ref_sq)) if ref_sq else None,
        relative_l1=l1 / ref_l1 if ref_l1 else None,
        rmse=float(np.sqrt(sq / actual.size)),
        max_abs=mx,
    )


def run_warmups(params, state, example, grad, optimizer, control, mode, iterations):
    """Warm gradient/averaging/AdamW on fresh copies, preserving formal state."""
    if iterations < 1:
        raise ValueError("warmup-iters must be positive")
    (before_params, before_state) = (tree_digest(params), tree_digest(state))
    avg = jax.jit(
        lambda x, y: jax.tree.map(lambda a, b: (a + b) * 0.5, x, y), donate_argnums=(0,)
    )

    def apply(params, state, grads, lr):
        (updates, state) = optimizer.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + lr * u, params, updates)
        return (params, state)

    def copy_to_device(tree):
        arrays = jax.tree.map(
            lambda value: np.array(value, copy=True), tree_numpy(tree)
        )
        return jax.device_put(arrays)

    update = None
    counts = []
    for index in range(iterations):
        control.select(mode, "warmup-%d" % (index + 1))
        (loss, grads) = grad(params, example)
        jax.block_until_ready((loss, grads))
        if not np.isfinite(float(loss)):
            raise ValueError("Nonfinite warmup loss")
        if update is None:
            update = (
                jax.jit(apply, donate_argnums=(0, 1))
                .lower(params, state, grads, jnp.float32(0))
                .compile()
            )
            avg.lower(grads, grads).compile()
        warm_grads = avg(copy_to_device(grads), grads)
        (warm_params, warm_state) = copy_to_device((params, state))
        warm_result = update(warm_params, warm_state, warm_grads, jnp.float32(0))
        jax.block_until_ready(warm_result)
        counts.append(
            [
                int(np.asarray(leaf))
                for leaf in jax.tree.leaves(warm_result[1])
                if leaf.shape == () and np.dtype(leaf.dtype).kind in "iu"
            ]
        )
        del warm_grads, warm_result, warm_params, warm_state, grads, loss
    (after_params, after_state) = (tree_digest(params), tree_digest(state))
    if (before_params, before_state) != (after_params, after_state):
        raise ValueError("Warmup mutated formal training state")
    audit = dict(
        iterations=iterations,
        params_before=before_params,
        params_after=after_params,
        optimizer_before=before_state,
        optimizer_after=after_state,
        optimizer_step_counts=counts,
    )
    return (avg, update, audit)


def delivery_training_helper(name, package="training"):
    import importlib.util

    source = Path(__file__).resolve().parents[3] / (package + "/" + name + ".py")
    spec = importlib.util.spec_from_file_location("h100_delivery_" + name, str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_predecessor(path):
    return delivery_training_helper("resume").validate_resume(path)


def snapshot_checkpoint_state(tree):
    host = tree_numpy(tree)
    if not all((np.isfinite(leaf).all() for leaf in jax.tree.leaves(host))):
        raise ValueError("Nonfinite parameter or optimizer state in checkpoint")
    return host


def record_update_diagnostics(row, params, grads, previous_flat, a, context):
    """Inspect completed gradients/state without advancing the optimizer."""
    u, lr = row["update"], row["lr"]
    OUT = context.output
    grad_host = tree_numpy(grads)
    flat = flatten(grad_host)
    assert np.isfinite(flat).all() and np.isfinite(row["loss"])
    row["gradient_norm"] = float(
        np.sqrt(
            sum(
                (
                    float(np.square(x.astype(np.float64)).sum())
                    for x in jax.tree.leaves(grad_host)
                )
            )
        )
    )
    assert row["gradient_norm"] > 0
    row["gradient_groups"] = {
        k: float(
            np.sqrt(
                sum(
                    (
                        float(np.square(v.astype(np.float64)).sum())
                        for v in group.values()
                    )
                )
            )
        )
        for (k, group) in grad_host.items()
    }
    current_flat = flatten(tree_numpy(params))
    assert np.isfinite(current_flat).all()
    row["parameter_update"] = comparison(current_flat, previous_flat)
    if lr == 0:
        assert row["parameter_update"]["max_abs"] == 0
    else:
        assert row["parameter_update"]["max_abs"] > 0
    row["gradient_sha256"] = hashlib.sha256(flat.tobytes()).hexdigest()
    if a.mode == "fp32":
        np.save(OUT / ("gradient-%02d.npy" % u), flat, allow_pickle=False)
    else:
        ref = (
            Path("/experiment-root")
            / ("fp32-r%02d" % a.rollout)
            / ("gradient-%02d.npy" % u)
        )
        if ref.exists():
            row["gradient_vs_fp32"] = comparison(flat, np.load(ref, mmap_mode="r"))
    return current_flat


def run_update(
    row0,
    params,
    state,
    previous_flat,
    *,
    a,
    context,
    get_batch,
    control,
    grad,
    avg,
    update,
    dev,
    report,
    save,
    monitor
):
    """Stage two microbatches, update once, then record diagnostics outside capture."""
    row = dict(row0)
    u = row["update"]
    lr = row["lr"]
    row["device_before_sample"] = monitor.sample()["device_sample"]
    stage_t = time.perf_counter()
    emit(
        "Training",
        "update={} | horizon={} | prepare two microbatches".format(u, a.rollout),
    )
    host_batches = [get_batch(s) for s in row["starts"]]
    row["batch_sha256"] = [tree_digest(b) for b in host_batches]
    batches = [to_device(host_batches[0])]
    jax.block_until_ready(batches)
    row["input_preparation_and_h2d_seconds"] = time.perf_counter() - stage_t
    emit(
        "Training",
        "update={} | executing | input_staging_s={:.3f}".format(
            u, row["input_preparation_and_h2d_seconds"]
        ),
    )
    before = control.counts()
    control.select(a.mode, "update-%02d" % u)
    with update_interval(u, jax.profiler) as interval:
        t = time.perf_counter()
        (l0, g0, l1, g1, stream_audit) = two_microbatch_grads(
            params,
            batches,
            host_batches[1],
            grad,
            to_device,
            jax.block_until_ready,
            lambda phase: control.select(a.mode, phase),
            dev.memory_stats,
        )
        del host_batches
        t2 = time.perf_counter()
        control.select(a.mode, "accumulation-optimizer")
        grads = avg(g0, g1)
        (params, state) = update(params, state, grads, jnp.float32(lr))
        jax.block_until_ready((params, state, grads, l0, l1))
        elapsed = time.perf_counter() - t
    row.update(
        forward_backward_microbatch_seconds=stream_audit[
            "forward_backward_microbatch_seconds"
        ],
        accumulation_optimizer_seconds=time.perf_counter() - t2,
        loss=(float(l0) + float(l1)) * 0.5,
        micro_losses=[float(l0), float(l1)],
        synchronized_update_seconds=elapsed,
        trace_begin_ns=interval["trace_begin_ns"],
        trace_end_ns=interval["trace_end_ns"],
        gemm=control.delta(before),
        jax_memory=dev.memory_stats(),
    )
    row["streamed_input_audit"] = stream_audit
    row["second_microbatch_h2d_seconds"] = stream_audit["second_microbatch_h2d_seconds"]
    row["compute_phase_sync_seconds"] = (
        sum(row["forward_backward_microbatch_seconds"])
        + row["accumulation_optimizer_seconds"]
    )
    if a.mode == "rne_xf32":
        assert (
            row["gemm"]["calls"] > 0 and row["gemm"]["rounded"] == row["gemm"]["calls"]
        ), row["gemm"]
    previous_flat = record_update_diagnostics(
        row, params, grads, previous_flat, a, context
    )
    row["device_after_sample"] = monitor.sample()["device_sample"]
    report["rows"].append(row)
    save("report.json", report)
    emit(
        "Training",
        "update={} | horizon={} | loss={:.8g} | gradient_norm={:.6g} | synchronized_s={:.3f}".format(
            u, a.rollout, row["loss"], row["gradient_norm"], elapsed
        ),
    )
    del batches, l0, l1, g0, g1, grads
    gc.collect()
    return params, state, previous_flat


def parse_arguments(argv):
    """Validate stage controls before importing data or allocating model state."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--rollout", type=int, required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--limit", type=int, default=38)
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--expected-updates", default="")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args(argv)
    if a.warmup_iters < 1:
        p.error("--warmup-iters must be positive")
    if a.save_every is not None and a.save_every < 1:
        p.error("--save-every must be positive")
    return a


def prepare_training_data(w, a, context):
    """Verify the fixed sample plan and expose batches without staging them on GPU."""
    data_root = context.path("training_data")
    source = json.loads((data_root / "source.json").read_text())
    data_path = data_root / "hres-20220101-025-13-12.nc"
    h = hashlib.sha256()
    with data_path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == source["sha256"]
    raw = xr.load_dataset(
        data_root
        / (
            "hres-20220101-025-13-1.nc" if a.preflight else "hres-20220101-025-13-12.nc"
        ),
        decode_timedelta=True,
    )
    if a.tiny:
        raw = raw.isel(lat=slice(None, None, 120), lon=slice(None, None, 120))
        cfg = dataclasses.replace(
            w.model_config, resolution=30, mesh_size=1, latent_size=8, gnn_msg_steps=2
        )
    else:
        cfg = w.model_config
    assert a.tiny or (
        raw.sizes["lat"] == 721
        and raw.sizes["lon"] == 1440
        and (raw.sizes["level"] == 13)
    )
    assert (raw.sizes["time"] >= 14 or a.preflight) and raw.sizes["batch"] == 1
    if a.preflight:
        assert a.rollout == 1
        plan = [dict(r, starts=[0, 0]) for r in recipe.schedule()]
    else:
        plan = recipe.sample_plan(raw.sizes["time"], 27)
    stats = {
        n: xr.load_dataset(
            context.path("assets") / "assets/stats" / (n + ".nc")
        ).astype(np.float32)
        for n in ["stddev_by_level", "mean_by_level", "diffs_stddev_by_level"]
    }

    def get_batch(start):
        selected = raw.isel(time=slice(start, start + a.rollout + 2))
        b = data_utils.extract_inputs_targets_forcings(
            selected,
            target_lead_times=slice("6h", str(6 * a.rollout) + "h"),
            **dataclasses.asdict(w.task_config)
        )
        b = tuple((x.astype(np.float32) for x in b))
        assert b[0].sizes["time"] == 2 and b[1].sizes["time"] == a.rollout
        assert all(
            (np.isfinite(v.values).all() for ds in b for v in ds.data_vars.values())
        )
        assert all((np.any(v.values != 0) for v in b[1].data_vars.values()))
        return b

    return cfg, source, plan, stats, get_batch


def main(argv=None, context=None):
    if context is None:
        (context, argv) = delivery_training_helper(
            "context", "platforms"
        ).context_arguments(argv)
    context.require_kind("training")
    global capture_stage, update_interval, finish_stage, emit, TrainingMonitor
    emit = delivery_training_helper("progress", ".").emit
    TrainingMonitor = delivery_training_helper("telemetry").TrainingMonitor
    profiling = delivery_training_helper("profiling")
    capture_stage, update_interval = profiling.capture_stage, profiling.update_interval
    finish_stage = profiling.finish_stage
    OUT = context.output
    OUT.mkdir(parents=True, exist_ok=True)

    def save(name, obj):
        save_report(OUT, name, obj)

    a = parse_arguments(argv)
    dev = jax.devices()[0]
    assert len(jax.devices()) == 1
    hip = Runtime()
    bus = hip.bus
    started = time.perf_counter()
    assert bus.value.decode().lower() == context.device["pci"]
    jax.config.update("jax_compilation_cache_dir", str(OUT / "cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    with open(context.path("assets") / "assets/jax_weights.npz", "rb") as f:
        w = checkpoint.load(
            f, __import__("graphcast.graphcast", fromlist=["CheckPoint"]).CheckPoint
        )
    cfg, source, plan, stats, get_batch = prepare_training_data(w, a, context)
    save("sample-plan.json", plan)
    loss_fn = make_loss(cfg, w.task_config, stats)

    def loss_apply(params, batch):
        (loss, state) = loss_fn.apply(params, {}, jax.random.PRNGKey(27), *batch)
        assert not state
        return loss

    control = Precision()
    control.select(a.mode, "initialization")
    # Resume state is validated before selecting or compiling formal updates.
    first = next((r for r in plan if r["rollout"] == a.rollout))
    example = get_batch(first["starts"][0])
    params = w.params
    if a.tiny:
        (params, _) = loss_fn.init(jax.random.PRNGKey(27), *example)
    optimizer = recipe.optimizer(params)
    state = optimizer.init(params)
    completed = 0
    if a.resume:
        predecessor = validate_predecessor(a.resume)
        with open(a.resume, "rb") as f:
            saved = pickle.load(f)
        assert saved["updates"] == predecessor["rows"][-1]["update"]
        assert (
            tree_digest(saved["params"])
            == predecessor["checkpoint_reload"]["params_sha256"]
        )
        assert (
            tree_digest(saved["opt_state"])
            == predecessor["checkpoint_reload"]["optimizer_sha256"]
        )
        assert saved["mode"] == a.mode and saved["data_sha256"] == source["sha256"]
        assert saved["plan"] == plan and saved["tiny"] == a.tiny
        (params, state, completed) = (
            saved["params"],
            saved["opt_state"],
            saved["updates"],
        )
    selected = select_updates(plan, a.rollout, completed, a.limit)
    checkpoints = delivery_training_helper("checkpointing").CheckpointWriter(
        OUT,
        a.mode,
        source["sha256"],
        plan,
        a.tiny,
        a.rollout,
        a.save_every,
        snapshot_checkpoint_state,
        verify_reload,
        preflight=a.preflight,
    )
    expected_updates = (
        [int(u) for u in a.expected_updates.split(",")]
        if a.expected_updates
        else [r["update"] for r in selected]
    )
    if [r["update"] for r in selected] != expected_updates:
        raise ValueError("Selected updates differ from expected global IDs")
    initial_params_digest = tree_digest(params)
    initial_opt_digest = tree_digest(state)
    initial_batch_digest = tree_digest(example)
    (params, state) = jax.device_put((params, state))
    example = to_device(example)
    report = dict(
        schema="graphcast.training38.v1",
        status="compiling",
        mode=a.mode,
        rollout=a.rollout,
        tiny=a.tiny,
        preflight=a.preflight,
        warmup_iters=a.warmup_iters,
        completed_before=completed,
        expected_updates=expected_updates,
        device=dict(
            index=context.device["index"],
            pci=bus.value.decode(),
            driver_name=dev.device_kind,
            user_name="H100" if a.mode == "tf32" else "BW1100",
        ),
        model=dataclasses.asdict(cfg),
        task=dataclasses.asdict(w.task_config),
        versions={n: md.version(n) for n in ["jax", "jaxlib", "dm-haiku", "optax"]},
        data_sha256=source["sha256"],
        scope="Short mechanics/numerics test on a small real HRES example; not convergence or weather-skill validation",
        precision="FP32 arrays/states; H100 TF32 without explicit RNE"
        if a.mode == "tf32"
        else "FP32 arrays/states; rocBLAS SGEMM/GEMM_EX both operands RNE + XF32 for forward/backward/recomputation; no producer fusion",
        initial_params_sha256=initial_params_digest,
        initial_optimizer_sha256=initial_opt_digest,
        initial_batch_sha256=initial_batch_digest,
        seed=27,
        baseline_modified=False,
        source_revision="08cf73625c9d12bd9aaa038868bcb2fe488f2a22",
        remat_policy_file=remat_policy.__file__,
        remat_policy_sha256=hashlib.sha256(
            Path(remat_policy.__file__).read_bytes()
        ).hexdigest(),
        optimizer=dict(
            name="AdamW",
            betas=[0.9, 0.95],
            eps=1e-08,
            matrix_decay=0.1,
            clip_global_norm=32,
            accumulation=2,
        ),
        rows=[],
    )
    report["input_staging"] = dict(
        mode="sequential_microbatches",
        frozen_trainer_sha256=FROZEN_TRAIN_SHA256,
        candidate_trainer_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        synchronized_update_scope="Includes second microbatch H2D and input release/diagnostics; compare compute_phase_sync_seconds separately",
        input_preparation_scope="Host preparation for both fixed batches plus first H2D only",
        mfu_scope="All formal-update GPU compute kernels interval union; H2D copies excluded; useful FLOPs unchanged",
    )
    save("report.json", report)
    monitor = TrainingMonitor(context, dev.memory_stats)
    monitor.start()
    primary_error = None
    try:
        t = time.perf_counter()
        emit(
            "Compile",
            "Training horizon={} | compiling gradient executable".format(a.rollout),
        )
        grad = jax.jit(jax.value_and_grad(loss_apply)).lower(params, example).compile()
        report["compile_seconds"] = time.perf_counter() - t
        report["compiled_memory_analysis"] = str(grad.memory_analysis())
        (OUT / "optimized-hlo.txt").write_text(grad.as_text())
        emit(
            "Compile",
            "Gradient executable ready | seconds={:.3f}".format(
                report["compile_seconds"]
            ),
        )
        (OUT / "compiled-hlo-proto.pb").write_bytes(
            grad.memory_analysis().serialized_hlo_proto
        )
        if context.debug["compile_only"]:
            report["status"] = "compiled_only"
            save("report.json", report)
            return
        emit(
            "Warmup",
            "horizon={} | iterations={} | formal state is preserved".format(
                a.rollout, a.warmup_iters
            ),
        )
        (avg, update, report["warmup_audit"]) = run_warmups(
            params, state, example, grad, optimizer, control, a.mode, a.warmup_iters
        )
        del example
        gc.collect()
        report["status"] = "running"
        save("report.json", report)
        previous_flat = flatten(tree_numpy(params))
        # Per-update intervals exclude diagnostics and periodic checkpoint saves.
        with capture_stage(OUT, hip, context.debug["trace"], jax.profiler):
            for row0 in selected:
                u = row0["update"]
                params, state, previous_flat = run_update(
                    row0,
                    params,
                    state,
                    previous_flat,
                    a=a,
                    context=context,
                    get_batch=get_batch,
                    control=control,
                    grad=grad,
                    avg=avg,
                    update=update,
                    dev=dev,
                    report=report,
                    save=save,
                    monitor=monitor,
                )
                if u != selected[-1]["update"]:
                    if checkpoints.save(params, state, u) is not None:
                        report["checkpointing"] = checkpoints.summary()
                        save("report.json", report)
        committed = checkpoints.save(params, state, selected[-1]["update"], final=True)
        report["checkpoint_reload"] = committed["checkpoint_reload"]
        report["checkpoint_sha256"] = committed["checkpoint_sha256"]
        params_host = tree_numpy(params)
        report["final_params_sha256"] = hashlib.sha256(
            flatten(params_host).tobytes()
        ).hexdigest()
        if a.mode == "rne_xf32":
            ref = (
                Path("/experiment-root") / ("fp32-r%02d" % a.rollout) / "checkpoint.pkl"
            )
            if ref.exists():
                with ref.open("rb") as f:
                    ref = pickle.load(f)
                report["parameters_vs_fp32"] = comparison(
                    flatten(params_host), flatten(ref["params"])
                )
        report["status"] = "completed"
    except BaseException as e:
        primary_error = e
        emit(
            "Training",
            "horizon={} failed: {}: {}".format(a.rollout, type(e).__name__, e),
            "ERROR",
        )
        report.update(status="failed", error=repr(e))
        raise
    finally:
        finish_stage(report, monitor, checkpoints, save, started, primary_error)


if __name__ == "__main__":
    main()
