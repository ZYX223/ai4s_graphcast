"""Orchestrate 38 training updates using the frozen full38 numerical engine.

Stage h1 has 16 updates; h2-h12 have two each. Separate stage processes
release device memory and resume audited params, AdamW state and update counts.
"""

import json
import time
from pathlib import Path

from graphcast_bench.validation.loss_metrics import aggregate_comparisons

from . import runtime, shapes
from .catalog import read

CASES = {
    "small-1deg-13l-b1": "small-b1",
    "small-1deg-13l-b2": "small-b2",
    "graphcast-025deg-37l-b1": "large37-b1",
}


def stages(context):
    if context["case"] not in CASES or context["workload"] != "train-full38":
        raise ValueError("Unknown full38 case/workload")
    count = 2 if CASES[context["case"]] == "large37-b1" else 1
    devices = context["devices"]
    if len(devices) != count or len({d["index"] for d in devices}) != count:
        raise ValueError("Full38 requires %d distinct devices" % count)
    if context["horizons"] != list(range(1, 13)):
        raise ValueError("Full38 requires all twelve horizons")
    return [
        dict(
            name="h%02d" % h,
            action="train",
            horizon=h,
            reference="h%02d" % h,
            count=16 if h == 1 else 2,
        )
        for h in range(1, 13)
    ]


def environment(context, stage, router=None):
    large = CASES[context["case"]] == "large37-b1"
    proxy = dict(context, engine="graphcast37_two" if large else "small_b1")
    env = shapes.runtime_environment(proxy, stage, router)
    env.update(
        FULL38_CASE=CASES[context["case"]],
        DEVICE_COUNT=str(len(context["devices"])),
        EXPECTED_PCIS=json.dumps([d["pci"] for d in context["devices"]]),
        BENCHMARK_GPU_UUIDS=json.dumps(
            [d["uuid"] for d in context["devices"]] if context["platform"] == "h100" else []
        ),
        TRAIN_LAYERNORM_REMAT="1" if large else "0",
    )
    # H100 h9+ uses the successfully tested preallocation policy; quota stays .94.
    # BW retains .80 and growable allocation. Neither changes arithmetic/batch.
    if large and context["platform"] == "h100" and stage["horizon"] >= 9:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    return env


def verify_predecessor(report, previous):
    pairs = [
        ("initial_params_sha256", previous["final_params_sha256"]),
        ("initial_optimizer_sha256", previous["final_optimizer_sha256"]),
        ("resume_checkpoint_sha256", previous["checkpoint_reload"]["sha256"]),
    ]
    for key, value in pairs:
        if report.get(key) != value:
            raise ValueError("Checkpoint chain mismatch: " + key)


def prepare(context, directory, previous):
    release = Path(context["release"])
    runtime.alias("/repo", release)
    runtime.alias("/spike", release / "src/graphcast_bench/engines/full38")
    runtime.alias("/case", context["data"])
    runtime.alias("/output", directory)
    if previous is None:
        empty = directory / "empty-resume"
        empty.mkdir()
        runtime.alias("/resume", empty)
    else:
        runtime.alias("/resume", previous)
    if CASES[context["case"]] == "large37-b1":
        runtime.alias("/reference-graph", Path(context["model"]) / "graph")
    if context["platform"] == "bw1100":
        # The tested full38 router records device IDs on both single and two GPUs.
        return runtime.build_router(release, context["build"], kind="training_two")
    return None


def await_admission(context, step, directory):
    """Host guard rechecks physical devices between stages, before GPU launch."""
    shapes.publish_state(context, step, directory, "waiting")
    deadline = time.monotonic() + context.get("admission_timeout", 45)
    path = directory / "stage-admission.json"
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Host stage admission did not arrive")
        time.sleep(0.1)
    value = read(path)
    if value.get("status") != "passed":
        raise RuntimeError("Host stage admission rejected: " + str(value))


def run(context):
    output = Path(context["output"])
    output.mkdir(parents=True, exist_ok=True)
    completed, current, previous = [], None, None
    code = 1
    schedule = stages(context)
    if any(
        (output / name).exists()
        for name in ["result.json", "active-stage.json"] + [s["name"] for s in schedule]
    ):
        raise FileExistsError("Full38 output already used")
    try:
        for step in schedule:
            directory = output / step["name"]
            directory.mkdir()
            (directory / "tmp").mkdir()
            step["started_ns"] = int(time.time() * 1e9)
            current = (step, directory)
            router = prepare(context, directory, previous)
            env = environment(context, step, router)
            command = shapes.compute_command(context)
            (directory / "launch.json").write_text(
                json.dumps(
                    dict(
                        command=command,
                        case=context["case"],
                        horizon=step["horizon"],
                        devices=context["devices"],
                        previous=None if previous is None else str(previous),
                        environment={
                            k: v
                            for k, v in env.items()
                            if k.startswith(
                                (
                                    "JAX_",
                                    "XLA_",
                                    "FULL38_",
                                    "DEVICE_",
                                    "EXPECTED_",
                                    "BENCHMARK_",
                                    "TRAIN_",
                                    "TRAINING_",
                                    "SMALL_",
                                    "GRAPHCAST_",
                                    "ROCR_",
                                    "CUDA_",
                                    "HIP_",
                                )
                            )
                        },
                    ),
                    indent=2,
                )
                + "\n"
            )
            await_admission(context, step, directory)
            shapes.publish_state(context, step, directory, "running")
            code = shapes.execute(
                command,
                env,
                directory,
                on_start=lambda pid: shapes.publish_state(context, step, directory, "running", pid),
            )
            if code:
                raise RuntimeError("Training process exited with code %d" % code)
            code = 1
            report = read(directory / "report.json")
            # The analyzer validates the frozen schedule, real exit and counters.
            if report.get("status") != "completed" or len(report.get("rows", [])) != step["count"]:
                raise ValueError("Incomplete full38 stage")
            if previous is not None:
                verify_predecessor(report, read(previous / "report.json"))
            shapes.publish_state(context, step, directory, "analyzing")
            shapes.wait_for_monitor(
                directory,
                report,
                context["platform"] == "bw1100",
                context.get("monitor_timeout", 15),
            )
            engine = Path(context["release"]) / "src/graphcast_bench/engines/full38"
            reference = Path(context["reference"]) / step["reference"]
            for name, command in [
                (
                    "analysis",
                    [context["python"], str(engine / "analyze_full38.py"), str(directory)],
                ),
                (
                    "comparison",
                    [
                        context["python"],
                        str(engine / "compare_full38.py"),
                        str(directory),
                        str(reference),
                        "--diagnostic-only",
                    ],
                ),
            ]:
                code = shapes.execute(command, shapes.cpu_environment(env), directory, name)
                if code:
                    raise RuntimeError("%s exited with code %d" % (name, code))
            code = 1
            comparison = read(directory / "comparison.json")
            analysis = read(directory / "analysis-full38.json")
            if analysis["performance_qualification"]["qualified"] is not True:
                raise ValueError("Performance samples lack clean ownership/coverage")
            completed.append(
                dict(
                    stage=step["name"],
                    updates=[r["update"] for r in report["rows"]],
                    comparison=comparison,
                    analysis=analysis,
                )
            )
            shapes.publish_state(context, step, directory, "completed")
            previous = directory
        if [u for s in completed for u in s["updates"]] != list(range(1, 39)):
            raise ValueError("Full38 trajectory incomplete")
        loss = aggregate_comparisons(
            [s["comparison"] for s in completed], expected_updates=range(1, 39)
        )
        total_f = sum(s["analysis"]["useful_gemm_flops"] for s in completed)
        active = sum(s["analysis"]["gpu_active_total_seconds"] for s in completed)
        wall = sum(s["analysis"]["sync_wall_total_seconds"] for s in completed)
        peak = (
            len(context["devices"]) * (494.7 if context["platform"] == "h100" else 170.3936) * 1e12
        )
        result = dict(
            status="completed",
            updates=38,
            actual_nonzero_lr_updates=36,
            useful_gemm_flops=total_f,
            gpu_active_seconds=active,
            sync_wall_seconds=wall,
            mfu_percent=100 * total_f / (active * peak),
            stages=completed,
            loss_validation=loss,
            accuracy_status="passed" if loss["metric_passed"] else "failed",
            max_loss_error_percent=loss["max_relative_error_percent"],
            timing="GPU compute+collective union; excludes copies/idle/compile/warmup; synchronized wall separate",
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0 if loss["metric_passed"] else 1
    except BaseException as exc:
        if current:
            shapes.publish_state(context, current[0], current[1], "failed")
        (output / "result.json").write_text(
            json.dumps(
                dict(
                    status="failed",
                    error=repr(exc),
                    exit_code=code,
                    completed=[s["stage"] for s in completed],
                ),
                indent=2,
            )
            + "\n"
        )
        raise
