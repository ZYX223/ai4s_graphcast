"""Portable orchestration of the frozen shape engines; no model arithmetic.

Every short-training horizon starts from the same pretrained parameters and
performs two updates. It is never a substitute for the 38-update curriculum.
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path

from . import runtime
from .catalog import read


def stages(context):
    engine, workload = context["engine"], context["workload"]
    if engine not in ("small_b1", "small_b2", "graphcast37", "graphcast37_two"):
        raise ValueError("Unknown shape engine")
    if context["platform"] not in ("h100", "bw1100"):
        raise ValueError("Unknown platform")
    devices = context["devices"]
    expected = 2 if engine == "graphcast37_two" else 1
    if len(devices) != expected or len({d["index"] for d in devices}) != expected:
        raise ValueError("Shape engine requires %d distinct devices" % expected)
    if workload == "train-short":
        if engine == "graphcast37":
            raise ValueError("37-level short training requires its two-device engine")
        horizons = context.get("horizons", [1, 6, 12])
        if list(horizons) != [1, 6, 12]:
            raise ValueError("Published short training consists of independent h1, h6 and h12")
        return [
            dict(name="train-h%d" % h, action="train", horizon=h, reference="training/h%d" % h)
            for h in horizons
        ]
    if engine == "graphcast37_two" or workload not in ("single", "full40"):
        raise ValueError("Unsupported shape workload; full38 is not published")
    names = ["single", "full40"] if workload == "full40" else ["single"]
    return [dict(name=n, action=n, horizon=1, reference=n) for n in names]


def engine_path(context):
    return Path(context["release"]) / "src/graphcast_bench/engines" / context["engine"]


def runtime_environment(context, stage, router=None):
    env = dict(os.environ)
    # A reused container must not inherit a previous platform's numerical path.
    for key in (
        "LD_PRELOAD",
        "NVIDIA_TF32_OVERRIDE",
        "ROCBLAS_MATH_MODE",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "XLA_MODE",
        "TRAINING_ROUTER",
        "TRAINING_GEMM_LOG",
        "TRAIN_LAYERNORM_REMAT",
        "RNE_PRE_ROUNDED",
        "RNE_LOWERING_AUDIT",
        "GRAPHCAST_REFERENCE_GRAPH",
        "GRAPHCAST_REFERENCE_GRAPH_SHA256",
        "GRAPHCAST_REFERENCE_GRAPH_AUDIT",
    ):
        env.pop(key, None)
    two = context["engine"] == "graphcast37_two"
    bw = context["platform"] == "bw1100"
    devices = context["devices"]
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        JAX_ENABLE_X64="false",
        JAX_DEFAULT_MATMUL_PRECISION="tensorfloat32",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".94" if two and not bw else ".80",
        XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0",
        SMALL_PLATFORM="bw" if bw else "h100",
        SMALL_ACTION=stage["action"],
        SMALL_HORIZON=str(stage["horizon"]),
        SMALL_PYTHON=context["python"],
        PYTHONPATH="/spike:/repo/src:/repo/third_party/graphcast:/minimal/deps:/minimal/official",
        TMPDIR="/output/tmp",
    )
    if two:
        env.update(
            EXPECTED_PCIS=json.dumps([d["pci"] for d in devices]),
            BENCHMARK_GPU_UUIDS=json.dumps([d["uuid"] for d in devices] if not bw else []),
            TRAIN_LAYERNORM_REMAT="1",
        )
    else:
        env.update(
            EXPECTED_PCI=devices[0]["pci"], BENCHMARK_GPU_UUID=str(devices[0].get("uuid") or "")
        )
    if bw:
        if not router:
            raise ValueError("BW engine requires the verified RNE router")
        local = ",".join(str(i) for i in range(len(devices)))
        env.update(
            JAX_PLATFORMS="rocm",
            ROCR_VISIBLE_DEVICES=",".join(str(d["index"]) for d in devices),
            HIP_VISIBLE_DEVICES=local,
            CUDA_VISIBLE_DEVICES=local,
            XLA_MODE="fl_only",
            XLA_FLAGTREE_LOG="1",
            XLA_FLAGTREE_COMPILE_WORKERS="8",
            XLA_FLAGTREE_PERSISTENT_WORKERS="8",
            TRAINING_ROUTER=str(router),
            LD_PRELOAD=str(router),
            TRAINING_GEMM_LOG="/output/gemm.jsonl",
            LD_LIBRARY_PATH="/opt/hyhal/lib:" + env.get("LD_LIBRARY_PATH", ""),
        )
    else:
        if any(not d.get("uuid") for d in devices):
            raise ValueError("H100 selection requires physical GPU UUIDs")
        selected = ",".join(d["uuid"] for d in devices)
        env.update(
            JAX_PLATFORMS="cuda",
            CUDA_VISIBLE_DEVICES=selected,
            NVIDIA_VISIBLE_DEVICES=selected,
            NVIDIA_DRIVER_CAPABILITIES="compute,utility",
        )
    if context["engine"].startswith("graphcast37"):
        meta = read(Path(context["model"]) / "graph/reference-graph.json")
        env.update(
            GRAPHCAST_REFERENCE_GRAPH="/reference-graph",
            GRAPHCAST_REFERENCE_GRAPH_SHA256=meta["file_sha256"],
            GRAPHCAST_REFERENCE_GRAPH_AUDIT="/output/reference-graph-audit.jsonl",
        )
    return env


def cpu_environment(env):
    result = dict(env)
    for key in (
        "LD_PRELOAD",
        "TRAINING_ROUTER",
        "TRAINING_GEMM_LOG",
        "XLA_MODE",
        "RNE_PRE_ROUNDED",
        "RNE_LOWERING_AUDIT",
    ):
        result.pop(key, None)
    result.update(
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        NVIDIA_VISIBLE_DEVICES="void",
        XLA_FLAGS="",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    return result


def prepare_runtime(context, stage, directory):
    release = Path(context["release"])
    runtime.alias("/repo", release)
    runtime.alias("/spike", engine_path(context))
    runtime.alias("/case", context["data"])
    runtime.alias("/output", directory)
    if context["engine"].startswith("graphcast37"):
        runtime.alias("/reference-graph", Path(context["model"]) / "graph")
    if context["platform"] == "bw1100":
        socket = Path("/tmp/flagtree-r10-0")
        if (
            not socket.is_dir()
            or socket.is_symlink()
            or socket.stat().st_uid != os.getuid()
            or socket.stat().st_mode & 0o777 != 0o700
        ):
            raise ValueError("Expected this locked container's owned 0700 FlagTree socket mount")
        kind = "training_two" if context["engine"] == "graphcast37_two" else "training"
        return runtime.build_router(release, context["build"], kind=kind)
    return None


def compute_command(context):
    action = ["bash", "/spike/run_cross.sh"]
    if context["platform"] == "h100":
        helper = (
            Path(context["release"])
            / "src/graphcast_bench/engines/operational_h100/clone3_compat_exec.py"
        )
        return [context["python"], str(helper), "--"] + action
    return action


def analysis_commands(context, stage, directory):
    engine, python = engine_path(context), context["python"]
    reference = Path(context["reference"]) / stage["reference"]
    if context["engine"] != "graphcast37_two":
        script = str(engine / "analyze_cross.py")
        return [
            ("analysis", [python, script, str(directory)]),
            ("comparison", [python, script, str(directory), "--reference", str(reference)]),
        ]
    analyzer = "analyze_bw_two.py" if context["platform"] == "bw1100" else "analyze_two_gpu.py"
    # Keep trace/path checks separate from the short-training loss verdict.
    # The comparison adapter uses the shared full-interval <=5% statistic.
    compare = (
        "import json,sys; from pathlib import Path; from analyze_bw_two import compare; "
        "p,r,proof=map(Path,sys.argv[1:]); "
        'c=json.loads((p/"report.json").read_text()); '
        'v=json.loads((r/"report.json").read_text()); '
        "result=compare(c,v,p,r,proof); "
        '(p/"comparison.json").write_text(json.dumps(result,indent=2)+"\\n")'
    )
    proof = Path(context["reference"]) / "qualification/h12-decoded-comparison.json"
    return [
        ("analysis", [python, str(engine / analyzer), str(directory)]),
        ("comparison", [python, "-c", compare, str(directory), str(reference), str(proof)]),
    ]


def enforce_threshold(comparison, action):
    if action == "train":
        from graphcast_bench.validation.loss_metrics import enforce_loss_threshold

        enforce_loss_threshold(comparison)
        return
    if comparison.get("stage_within_10_percent") is not True:
        raise ValueError("Reference comparison did not pass the <=10% stage threshold")
    names = ["max_overall_percent", "max_z500_percent"]
    if "max_common13_overall_percent" in comparison:
        names.append("max_common13_overall_percent")
    for name in names:
        value = comparison.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 10
        ):
            raise ValueError("Invalid or excessive reference error: " + name)


def validate_report(report, context, stage):
    expected_platform = "bw" if context["platform"] == "bw1100" else "h100"
    if (
        report.get("status") != "completed"
        or report.get("action") != stage["action"]
        or report.get("platform") != expected_platform
    ):
        raise ValueError("Engine report is incomplete or belongs to another workload/platform")
    rows = report.get("rows", [])
    expected = 2 if stage["action"] == "train" else 3 if stage["action"] == "full40" else 1
    if len(rows) != expected:
        raise ValueError("Engine report has an unexpected number of formal rows")
    if stage["action"] == "train":
        if report.get("horizon") != stage["horizon"] or [r.get("update") for r in rows] != [1, 2]:
            raise ValueError("Expected two independent short-training updates")
    elif any(r.get("steps") != (40 if stage["action"] == "full40" else 1) for r in rows):
        raise ValueError("Engine report has an incorrect inference horizon")
    if context["engine"] == "graphcast37_two" and report.get("device_count") != 2:
        raise ValueError("Training report did not use both devices")


def _last_sample(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            sample = json.loads(line)
        except ValueError:
            continue  # The host may currently be appending the final line.
        if isinstance(sample.get("time_ns"), int):
            return sample["time_ns"]
    return None


def wait_for_monitor(directory, report, process_required, timeout=15):
    final_ns = max(r["trace_end_ns"] for r in report["rows"])
    files = [directory / "host-memory.jsonl"]
    if process_required:
        files.append(directory / "process-monitor.jsonl")
    deadline = time.monotonic() + timeout
    while True:
        samples = [_last_sample(p) for p in files]
        if all(value is not None and value >= final_ns for value in samples):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Host monitoring did not bracket the completed formal ranges")
        time.sleep(0.1)


def publish_state(context, stage, directory, status, pid=None):
    path = Path(context["output"]) / "active-stage.json"
    data = dict(
        schema="graphcast.delivery.stage.v1",
        stage=stage["name"],
        path=str(directory),
        status=status,
        pid=pid,
        started_ns=stage["started_ns"],
        updated_ns=int(time.time() * 1e9),
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(str(temporary), str(path))


def execute(command, env, directory, name="engine", on_start=None):
    log = "console.log" if name == "engine" else name + ".log"
    status = "exit-code.txt" if name == "engine" else name + "-exit-code.txt"
    with (directory / log).open("x") as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            if on_start:
                on_start(child.pid)
            code = child.wait()
        except BaseException:
            child.terminate()
            code = child.wait()
            (directory / status).write_text(str(code) + "\n")
            raise
    (directory / status).write_text(str(code) + "\n")
    return code


def run(context):
    output = Path(context["output"])
    output.mkdir(parents=True, exist_ok=True)
    current = None
    code = 1
    completed = []
    try:
        plan = stages(context)
        if (output / "result.json").exists() or (output / "active-stage.json").exists():
            raise FileExistsError("Output has a previous delivery result")
        for step in plan:
            if (output / step["name"]).exists():
                raise FileExistsError("Output stage already exists: " + step["name"])
        for step in plan:
            directory = output / step["name"]
            directory.mkdir(exist_ok=False)
            (directory / "tmp").mkdir()
            step["started_ns"] = int(time.time() * 1e9)
            current = (step, directory)
            router = prepare_runtime(context, step, directory)
            env = runtime_environment(context, step, router)
            env["DELIVERY_STAGE_PATH"] = str(directory)
            command = compute_command(context)
            public_keys = (
                "JAX_",
                "XLA_",
                "SMALL_",
                "EXPECTED_",
                "BENCHMARK_",
                "GRAPHCAST_REFERENCE_",
                "ROCR_",
                "HIP_",
                "CUDA_",
                "NVIDIA_",
                "TRAINING_",
                "TRAIN_",
            )
            launch = dict(
                command=command,
                engine=context["engine"],
                action=step["action"],
                horizon=step["horizon"],
                devices=context["devices"],
                environment={k: v for k, v in env.items() if k.startswith(public_keys)},
                reference=str(Path(context["reference"]) / step["reference"]),
            )
            (directory / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            publish_state(context, step, directory, "running")
            code = execute(
                command,
                env,
                directory,
                on_start=lambda pid: publish_state(context, step, directory, "running", pid),
            )
            if code:
                raise RuntimeError("Shape engine exited with code %s" % code)
            code = 1
            report = read(directory / "report.json")
            validate_report(report, context, step)
            publish_state(context, step, directory, "analyzing")
            wait_for_monitor(
                directory,
                report,
                context["engine"] == "graphcast37_two" and context["platform"] == "bw1100",
                context.get("monitor_timeout", 15),
            )
            for name, command in analysis_commands(context, step, directory):
                code = execute(command, cpu_environment(env), directory, name=name)
                if code:
                    raise RuntimeError("%s process exited with code %s" % (name, code))
            code = 1
            comparison = read(directory / "comparison.json")
            enforce_threshold(comparison, step["action"])
            completed.append(
                dict(
                    stage=step["name"],
                    reference=launch["reference"],
                    comparison=str(directory / "comparison.json"),
                )
            )
            publish_state(context, step, directory, "completed")
        result = dict(
            status="completed",
            stages=completed,
            platform=context["platform"],
            workload=context["workload"],
            threshold_percent=5 if context["workload"] == "train-short" else 10,
            scope="Independent pretrained two-update h1/h6/h12 probes; not full38 or convergence"
            if context["workload"] == "train-short"
            else "Same-case frozen H100 reference comparison",
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0
    except Exception as error:
        if current:
            publish_state(context, current[0], current[1], "failed")
        result = dict(
            status="failed",
            completed_stages=completed,
            error=str(error),
            error_type=type(error).__name__,
            exit_code=code,
        )
        # Never overwrite a prior run's final record, including on reuse errors.
        result_path = output / "result.json"
        if not result_path.exists():
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        return code or 1
