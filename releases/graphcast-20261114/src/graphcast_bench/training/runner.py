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
from graphcast import checkpoint

from graphcast_bench.data.transfer import to_device
from graphcast_bench.model import remat as remat_policy
from graphcast_bench.model.loss import make_loss
from graphcast_bench.platforms.runtime import Precision, Runtime
from graphcast_bench.training import recipe
from graphcast_bench.training.checkpoint import (
    snapshot_state,
    tree_digest,
    verify_reload,
)
from graphcast_bench.training.checkpointing import CheckpointWriter
from graphcast_bench.training.resume import validate_resume
from graphcast_bench.training.curriculum import select_updates
from graphcast_bench.training.diagnostics import comparison, flatten, tree_numpy
from graphcast_bench.training.microbatches import two_microbatch_grads
from graphcast_bench.training.warmup import run_warmup
from graphcast_bench.training.profiling import (
    capture_stage,
    update_interval,
    finish_stage,
)
from graphcast_bench.progress import emit
from graphcast_bench.training.telemetry import TrainingMonitor
from graphcast_bench.delivery.options import TRAINING_DEFAULTS, positive_integer
from graphcast_bench.platforms.context import context_arguments

FROZEN_TRAIN_SHA256 = "77df16253040abf114c6ca24cd024d907e5489de29a8a7800671a0c5b09e1969"


def save_report(output, name, obj):
    p = output / name
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(obj, indent=2) + "\n")
    q.replace(p)


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
                float(np.square(x.astype(np.float64)).sum())
                for x in jax.tree.leaves(grad_host)
            )
        )
    )
    assert row["gradient_norm"] > 0
    row["gradient_groups"] = {
        k: float(
            np.sqrt(
                sum(
                    float(np.square(v.astype(np.float64)).sum()) for v in group.values()
                )
            )
        )
        for k, group in grad_host.items()
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
            Path(context.debug["fp32_reference"] or (OUT / "no-fp32-reference"))
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
        l0, g0, l1, g1, stream_audit = two_microbatch_grads(
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
        params, state = update(params, state, grads, jnp.float32(lr))
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
    p.add_argument(
        "--warmup-iters", type=int, default=TRAINING_DEFAULTS["warmup_iters"]
    )
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args(argv)
    positive_integer("warmup_iters", a.warmup_iters)
    if a.save_every is not None:
        positive_integer("save_every", a.save_every)
    return a


def main(argv=None, context=None):
    if context is None:
        context, argv = context_arguments(argv)
    context.require_kind("training")
    OUT = context.output

    def save(name, obj):
        save_report(OUT, name, obj)

    a = parse_arguments(argv)
    dev = jax.devices()[0]
    assert len(jax.devices()) == 1
    hip = Runtime(context)
    bus = hip.bus
    started = time.perf_counter()
    assert bus.value.decode().lower() == context.device["pci"]
    jax.config.update("jax_compilation_cache_dir", str(OUT / "cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    with (context.path("assets") / "assets/jax_weights.npz").open("rb") as f:
        w = checkpoint.load(
            f, __import__("graphcast.graphcast", fromlist=["CheckPoint"]).CheckPoint
        )
    from graphcast_bench.data.training import prepare_training_data

    cfg, source, plan, stats, get_batch = prepare_training_data(
        w,
        a.rollout,
        context.path("training_data"),
        context.path("assets"),
        a.tiny,
        a.preflight,
    )
    save("sample-plan.json", plan)
    loss_fn = make_loss(cfg, w.task_config, stats)

    def loss_apply(params, batch):
        loss, state = loss_fn.apply(params, {}, jax.random.PRNGKey(27), *batch)
        assert not state
        return loss

    control = Precision(context)
    control.select(a.mode, "initialization")
    # Resume state is validated before selecting or compiling formal updates.
    first = next(r for r in plan if r["rollout"] == a.rollout)
    example = get_batch(first["starts"][0])
    params = w.params
    if a.tiny:
        params, _ = loss_fn.init(jax.random.PRNGKey(27), *example)
    optimizer = recipe.optimizer(params)
    state = optimizer.init(params)
    completed = 0
    if a.resume:
        predecessor = validate_resume(a.resume)
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
        params, state, completed = saved["params"], saved["opt_state"], saved["updates"]
    selected = select_updates(plan, a.rollout, completed, a.limit)
    checkpoints = CheckpointWriter(
        OUT,
        a.mode,
        source["sha256"],
        plan,
        a.tiny,
        a.rollout,
        a.save_every,
        snapshot_state,
        verify_reload,
        preflight=a.preflight,
    )
    initial_params_digest = tree_digest(params)
    initial_opt_digest = tree_digest(state)
    initial_batch_digest = tree_digest(example)
    params, state = jax.device_put((params, state))
    example = to_device(example)
    report = dict(
        schema="graphcast.training38.v1",
        status="compiling",
        mode=a.mode,
        rollout=a.rollout,
        warmup_iters=a.warmup_iters,
        tiny=a.tiny,
        preflight=a.preflight,
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
        precision=(
            "FP32 arrays/states; H100 TF32 without explicit RNE"
            if a.mode == "tf32"
            else "FP32 arrays/states; rocBLAS SGEMM/GEMM_EX both operands RNE + XF32 for forward/backward/recomputation; no producer fusion"
        ),
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
            eps=1e-8,
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
    report["training_capture_format"] = (
        "xspace" if context.debug["trace"] else "disabled"
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
        control.select(a.mode, "warmup")
        avg = jax.jit(
            lambda x, y: jax.tree.map(lambda a, b: (a + b) * 0.5, x, y),
            donate_argnums=(0,),
        )

        def apply(params, state, grads, lr):
            updates, state = optimizer.update(grads, state, params)
            params = jax.tree.map(lambda p, u: p + lr * u, params, updates)
            return params, state

        update = jax.jit(apply, donate_argnums=(0, 1))
        emit(
            "Warmup",
            "horizon={} | iterations={} | formal state is preserved".format(
                a.rollout, a.warmup_iters
            ),
        )
        run_warmup(
            grad,
            avg,
            lambda p, s, g: update(p, s, g, jnp.float32(0)),
            params,
            state,
            example,
            a.warmup_iters,
            lambda tree: jax.device_put(tree_numpy(tree)),
            jax.block_until_ready,
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
                Path(context.debug["fp32_reference"] or (OUT / "no-fp32-reference"))
                / ("fp32-r%02d" % a.rollout)
                / "checkpoint.pkl"
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
