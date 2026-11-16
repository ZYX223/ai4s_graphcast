"""Run the frozen Operational H100 engines from a relocatable delivery.

The frozen model math and training recipe are preserved. Delivery wrappers
provide explicit inference and training controls while retaining fixed-preset
validators. Execution uses the pinned, exclusively locked delivery container.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Keep maintained-package imports inside orchestration functions. This file also
# runs as --validate-full40 with only the frozen graphcast_bench on PYTHONPATH.

CANDIDATE = "mesh-group8-m2g-save-stats-vjp-finalmesh-g2m"
ENGINE_RELATIVE = "src/graphcast_bench/engines/operational_h100"
WORKLOADS = ("single", "full40", "train-full38")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location("delivery_h100_" + name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources(engine):
    engine = Path(engine).resolve()
    records = read_json(engine / "ORIGINS.json")["files"]
    for name, record in records.items():
        path = (engine / name).resolve()
        if engine not in path.parents or sha256(path) != record.get(
            "delivery_sha256", record["sha256"]
        ):
            raise ValueError("Curated H100 source differs: " + name)
    return records


def training_stages(python, engine, training=None):
    """The complete 38-update curriculum; no representative-update shortcut."""
    if training is not None:
        from graphcast_bench.delivery.options import training_plan
        completed = training.get("completed_updates", 0)
        result, previous = [], training.get("resume")
        for stage in training_plan(training, completed):
            horizon = stage["horizon"]
            command = [
                python, str(engine / "clone3_compat_exec.py"), "--", "bash",
                "/candidate/run-h100.sh", "--runtime-context", "/output/runtime-context.json",
                "--mode", "tf32", "--rollout", str(horizon),
                "--limit", str(stage["updates"][-1]),
                "--warmup-iters", str(training["warmup_iters"]),
                "--expected-updates", ",".join(map(str, stage["updates"])),
            ]
            if training.get("save_every") is not None:
                command += ["--save-every", str(training["save_every"])]
            if previous:
                command += ["--resume", previous]
            result.append(dict(
                stage, command=command,
                memory_fraction=".94" if horizon < 11 else ".96",
            ))
            previous = "/run-chain/r%02d/checkpoint.pkl" % horizon
        return result
    result = []
    for horizon in range(1, 13):
        command = [
            python,
            str(engine / "clone3_compat_exec.py"),
            "--",
            "bash",
            "/candidate/run-h100.sh",
            "--runtime-context", "/output/runtime-context.json",
            "--mode",
            "tf32",
            "--rollout",
            str(horizon),
        ]
        if horizon > 1:
            command += ["--resume", "/run-chain/r%02d/checkpoint.pkl" % (horizon - 1)]
        result.append(
            {
                "horizon": horizon,
                "command": command,
                "memory_fraction": ".94" if horizon < 11 else ".96",
            }
        )
    return result


def runtime_environment(context):
    from graphcast_bench.platforms.environment import h100_environment

    return h100_environment(context, CANDIDATE)


def training_context(context, stage):
    from graphcast_bench.platforms.context import RuntimeContext

    runtime = RuntimeContext('training', 'h100', str(stage), context['devices'][0],
                             dict(assets='/weekly', training_data='/training-data'),
                             dict(training=dict(candidate=CANDIDATE)))
    runtime.write(Path(stage) / 'runtime-context.json')
    return runtime


def inference_context(context, stage, recipe, values=None):
    """Write explicit worker options; legacy environment is only the shell ABI."""
    from graphcast_bench.platforms.context import RuntimeContext

    if values is None:
        values = dict(steps=1 if recipe == 'single' else 40, warmup_steps=5,
                      repeats=2 if recipe == 'single' else 3, warmup_scope='per-repeat')
    runtime = RuntimeContext('inference', 'h100', str(stage), context['devices'][0],
        dict(assets=str(Path(context['data']) / 'prepared-inference'),
             inference_engine=str(Path(context['release']) / ENGINE_RELATIVE / 'inference')),
        dict(recipe=recipe, inference=dict(values), variant='shifted_mesh'))
    runtime.write(Path(stage) / 'inference-context.json')
    return runtime


def validate_inference_worker(context, stage, values):
    from graphcast_bench.inference.h100.provenance import validate_manifest

    return validate_manifest(dict(context, output=str(stage)), values)


def _inside(path, roots):
    path = Path(path).resolve()
    return any(path == root.resolve() or root.resolve() in path.parents for root in roots)


def install_alias(alias, target, allowed_roots=None):
    """Only replace our private symlinks, never an image directory or mount."""
    alias, target = Path(alias), Path(target).resolve()
    roots = [Path(p) for p in (allowed_roots or ("/bundle", "/work"))]
    if not target.exists() or not _inside(target, roots):
        raise ValueError("Alias target is outside this delivery: " + str(target))
    if alias.is_symlink():
        if not _inside(alias.resolve(), roots):
            raise ValueError("Refusing to replace an unrelated alias: " + str(alias))
        alias.unlink()
    elif alias.exists():
        raise ValueError("Refusing to replace a directory or mount: " + str(alias))
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)


def require_fresh_output(output):
    output = Path(output)
    if not output.is_dir():
        raise ValueError("The launcher must create a fresh output directory")
    occupied = [
        "adapter-status.json",
        "adapter-context.json",
        ".runtime",
        "run",
        "preflight-single",
    ]
    occupied += ["r%02d" % horizon for horizon in range(1, 13)]
    if any((output / name).exists() for name in occupied):
        raise ValueError("H100 output already contains a previous run: " + str(output))


def prepare_aliases(context, engine):
    if not Path("/.dockerenv").is_file() or os.environ.get("GRAPHCAST_DELIVERY_LOCKED") != "1":
        raise ValueError("H100 adapter requires the exclusively locked delivery container")
    output, data = Path(context["output"]), Path(context["data"])
    private = output / ".runtime"
    original = private / "original"
    original.mkdir(parents=True)
    (original / "experiment").symlink_to(engine / "original_training")
    aliases = {
        "/repo": engine / "frozen",
        "/experiment": engine / "inference",
        "/candidate": engine / "training",
        "/original": original,
        "/minimal/official": engine / "minimal-official",
        "/data": data / "prepared-inference",
        "/weekly": data / "prepared-inference",  # Frozen trainer's internal ABI.
        "/training-data": data / "training",
        "/frozen-reference": Path(context["original_reference"]),
        "/run-chain": output,
        "/output": output,
    }
    # Check the whole plan before changing any alias.
    for alias, target in aliases.items():
        path = Path(alias)
        if (path.exists() and not path.is_symlink()) or (
            path.is_symlink() and not _inside(path.resolve(), [Path("/bundle"), Path("/work")])
        ):
            raise ValueError("Container path is not a delivery alias: " + alias)
        if not target.exists() or not _inside(target, [Path("/bundle"), Path("/work")]):
            raise ValueError("Missing or external delivery input: " + str(target))
    for alias, target in aliases.items():
        install_alias(alias, target)


def publish_stage(root, output, name, status, pid=None, started_ns=None):
    path = Path(root) / "active-stage.json"
    previous = read_json(path) if path.exists() else {}
    same = previous.get("path") == str(output) and previous.get("stage") == name
    write_json(
        path,
        {
            "schema": "graphcast.delivery.stage.v1",
            "stage": name,
            "path": str(output),
            "status": status,
            "started_ns": started_ns
            if started_ns is not None
            else (previous["started_ns"] if same else int(time.time() * 1e9)),
            "pid": pid if pid is not None else (previous.get("pid") if same else None),
        },
    )


def execute(command, environment, output, monitor_root=None, stage_name=None, forward_output=False):
    """Retain the real process status and its complete console log."""
    output = Path(output)
    write_json(output / "engine-command.json", {"command": command})
    if monitor_root is not None:
        publish_stage(
            monitor_root, output, stage_name, "running", started_ns=int(time.time() * 1e9)
        )
    with (output / "engine-console.log").open("a") as console:
        process = subprocess.Popen(
            command, env=environment,
            stdout=subprocess.PIPE if forward_output else console, stderr=subprocess.STDOUT,
            universal_newlines=True, encoding='utf-8', errors='replace',
        )
        if monitor_root is not None:
            publish_stage(monitor_root, output, stage_name, "running", pid=process.pid)
        try:
            if forward_output:
                # Preserve raw engine output here; the host relay owns presentation.
                for line in process.stdout:
                    console.write(line)
                    console.flush()
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except (OSError, ValueError):
                        pass  # A closed terminal must not discard the engine log.
            result = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            (output / "engine-exit-code.txt").write_text(str(process.returncode) + "\n")
    if monitor_root is not None:
        publish_stage(monitor_root, output, stage_name, "failed" if result else "analyzing")
    return result


def stage_environment(environment, stage):
    """Profiler and compiler scratch files belong to the external work volume."""
    from graphcast_bench.platforms.environment import scratch_environment

    return scratch_environment(environment, stage)


def validate_training_stage(
    engine, stage, reference, original_reference, horizon, environment=None, expected=None, previous=None, warmup_iters=1
):
    compare = load_module(engine / "training/compare.py", "loss").compare_reports
    report = read_json(stage / "report.json")
    generic = expected is not None
    if not generic:
        expected = list(range(1, 17)) if horizon == 1 else [13 + 2 * horizon, 14 + 2 * horizon]
    if (
        report.get("status") != "completed"
        or report.get("rollout") != horizon
        or report.get("mode") != "tf32"
        or report.get("tiny") is not False
        or [r["update"] for r in report["rows"]] != expected
    ):
        raise ValueError("Training stage is incomplete or has a different curriculum")
    if sha256(stage / "checkpoint.pkl") != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint file hash differs from completed report")
    comparisons = {}
    references = () if generic else (
        ("optimized_h100", reference), ("original_h100", original_reference)
    )
    for label, root in references:
        path = root / "training" / ("r%02d" % horizon) / "report.json"
        result = compare(report, read_json(path), same_initial_state=horizon == 1)
        result["reference_sha256"] = sha256(path)
        comparisons[label] = result
    manifest = read_json(stage / "candidate-manifest.json")
    if manifest["candidate"] != CANDIDATE or manifest["candidate_xla_flags"] != "":
        raise ValueError("Runtime training candidate differs from the frozen engine")
    expected_sources = {}
    for prefix, folder, patterns in (
        ("/candidate/", engine / "training", ("*.py", "*.sh")),
        ("/original/experiment/", engine / "original_training", ("*.py",)),
    ):
        for pattern in patterns:
            expected_sources.update(
                {prefix + source.name: sha256(source) for source in folder.glob(pattern)}
            )
    if manifest["source_sha256"] != expected_sources:
        raise ValueError("Runtime source set differs from the curated training engine")
    if environment is not None:
        for key in (
            "NVIDIA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "EXPECTED_PCI",
            "PHYSICAL_DEVICE",
            "STREAM_MICROBATCHES",
            "TRAINING_CANDIDATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "JAX_ENABLE_X64",
            "JAX_DEFAULT_MATMUL_PRECISION",
            "JAX_PLATFORMS",
        ):
            if manifest["environment"].get(key) != environment[key]:
                raise ValueError("Runtime training environment differs: " + key)
        if report["device"]["pci"].lower() != environment["EXPECTED_PCI"]:
            raise ValueError("Training report names a different physical GPU")
    for name, digest in manifest["source_sha256"].items():
        if name.startswith("/candidate/"):
            source = engine / "training" / name[len("/candidate/") :]
        elif name.startswith("/original/experiment/"):
            source = engine / "original_training" / name[len("/original/experiment/") :]
        else:
            raise ValueError("Unrecognized runtime training source: " + name)
        if sha256(source) != digest:
            raise ValueError("Runtime source identity differs: " + name)
    if generic:
        from graphcast_bench.delivery.h100_validation import (
            validate_training_report, validate_training_checkpoint,
            validate_optimization_audits,
        )
        validate_training_report(report, horizon, expected, warmup_iters, previous)
        validate_training_checkpoint(stage, report, expected)
        validate_optimization_audits(stage, training=True)
    else:
        write_json(stage / "loss-validation.json", comparisons)
    return report, comparisons


def validate_training_timing(engine, stage, report):
    trace = stage / "train-trace.sqlite"
    integrity = load_module(engine / "training/trace_integrity.py", "trace").inspect_trace(trace)
    if not integrity["passed"]:
        raise ValueError("Nsight kernel capture is incomplete")
    timing = load_module(engine / "training/timing.py", "timing")
    kernels = timing.nsight_update_kernels(
        trace, report["rows"], report.get("checkpointing", {}).get("saves", ()))
    rows = []
    for row in report["rows"]:
        selected = kernels[row["update"]]
        if any("round_rne" in name.lower() for _, _, name in selected):
            raise ValueError("Explicit RNE kernel unexpectedly present on H100")
        rows.append(
            {
                "update": row["update"],
                "gpu_kernel_union_seconds": timing.interval_union([(a, b) for a, b, _ in selected])
                / 1e9,
                "synchronized_update_seconds": row["synchronized_update_seconds"],
                "compute_phase_sync_seconds": row["compute_phase_sync_seconds"],
                "kernel_count": len(selected),
                "tf32_named_kernel_count": sum("tf32" in name.lower() for _, _, name in selected),
            }
        )
    result = {
        "status": "passed",
        "rows": rows,
        "trace_integrity": integrity,
        "scope": "Formal-update GPU kernel interval union and synchronized call time are separate; no end-to-end or convergence claim.",
    }
    write_json(stage / "delivery-timing.json", result)
    return result


def validate_full40(context):
    """Second evidence layer: newly run optimized H100 versus its frozen output."""
    engine = Path(context["release"]) / ENGINE_RELATIVE
    helper = load_module(engine / "inference/compare.py", "inference_compare")
    output = Path(context["output"])
    candidate = output / "run"
    reference = Path(context["reference"]) / "runs/inference-full40-r1/run"
    reports = {
        name: read_json(root / "report.json")
        for name, root in (("candidate", candidate), ("reference", reference))
    }
    for report in reports.values():
        if any(
            report.get(key) != value
            for key, value in {
                "status": "completed",
                "measured_steps": 40,
                "warmup_steps": 5,
                "mode": "tf32",
                "storage": "float32",
                "repeats": 3,
                "router_build": None,
            }.items()
        ):
            raise ValueError("Full40 comparison requires three completed independent TF32 rollouts")
        if sorted(row["round"] for row in report["rows"]) != [1, 2, 3]:
            raise ValueError("Missing or duplicated full40 rounds")
    if reports["candidate"]["data"] != reports["reference"]["data"]:
        raise ValueError("Full40 reference uses different prepared inputs")
    if reports["candidate"]["source"]["commit"] != reports["reference"]["source"]["commit"]:
        raise ValueError("Full40 reference uses different official model source")
    case = read_json(engine / "frozen/configs/case.json")

    def prediction(root, row):
        folder = root / row["run"]
        path = folder / "prediction.nc"
        if sha256(path) != read_json(folder / "report.json")["output_sha256"]:
            raise ValueError("Full40 prediction hash differs from its report")
        return helper.load(path)

    refs = {row["round"]: prediction(reference, row) for row in reports["reference"]["rows"]}
    comparisons = []
    for row in reports["candidate"]["rows"]:
        actual = prediction(candidate, row)
        for reference_round, expected in sorted(refs.items()):
            result = helper.compare(actual, expected, case=case)
            result.update(
                candidate_round=row["round"],
                reference_round=reference_round,
                primary_pair=row["round"] == reference_round,
            )
            comparisons.append(result)
    result = {
        "status": "passed" if all(r["metrics_within_limits"] for r in comparisons) else "failed",
        "reference": "Frozen optimized H100 TF32; no explicit RNE",
        "reference_report_sha256": sha256(reference / "report.json"),
        "scope": "Independent 40-step rollouts; +240h final frame only; 6 upper-air variables x 13 levels equally averaged, Z500 separately; H100 denominator.",
        "internal_limit_percent": 5,
        "rows": comparisons,
    }
    write_json(output / "accuracy-optimized-reference.json", result)
    return 0 if result["status"] == "passed" else 1


def write_execution_summary(output, options, details):
    from graphcast_bench.validation.execution import write_summary

    write_summary(output, options, details)


def validate_generic_inference(context, options):
    from graphcast_bench.delivery.h100_validation import validate_inference

    return validate_inference(context, options)


def validate_forcing_frames(context, options):
    import xarray as xr

    path = Path(context["data"]) / "prepared-inference/assets/rollout_forcings.nc"
    with xr.open_dataset(path, decode_cf=False) as forcings:
        if forcings.sizes.get("time", 0) < max(options["steps"], options["warmup_steps"]):
            raise ValueError("Insufficient prepared forcing frames")


def run_generic_inference(context, engine, output, environment, options):
    from graphcast_bench.platforms.environment import h100_inference_environment

    values = options["inference"]
    validate_forcing_frames(context, values)
    inference_context(context, output, 'custom', values)
    env = h100_inference_environment(environment, output, values=values)
    command = [
        context["python"], str(engine / "clone3_compat_exec.py"),
        "--", "bash", "/experiment/run.sh",
    ]
    result = execute(command, env, output, output, "infer", forward_output=True)
    if result:
        return result
    details = validate_generic_inference(context, values)
    write_execution_summary(output, options, details)
    publish_stage(output, output, "infer", "completed")
    return 0


def run_training(context, engine, output, environment, options):
    from graphcast_bench.platforms.environment import h100_training_environment
    from graphcast_bench.training.resume import validate_resume
    from graphcast_bench.delivery.options import resume_progress

    values = dict(options["training"])
    previous = None
    if values.get("resume"):
        previous = validate_resume(values["resume"])
        completed = resume_progress(values["resume"])
        if values.get("completed_updates", completed) != completed:
            raise ValueError("Resume progress differs from verified predecessor")
        values["completed_updates"] = completed
    elif values.get("completed_updates", 0):
        raise ValueError("Completed updates require a verified resume checkpoint")
    from graphcast_bench.progress import emit

    reports, comparisons, timings = [], [], []
    preset = options["preset"]
    for spec in training_stages(context["python"], engine, values):
        horizon = spec["horizon"]
        stage = output / ("r%02d" % horizon)
        stage.mkdir()
        install_alias("/output", stage)
        training_context(context, stage)
        env = h100_training_environment(environment, stage, spec["memory_fraction"])
        write_json(
            stage / "launch.json",
            dict(
                spec, candidate=CANDIDATE, full_chain=preset,
                warmup_iters=values["warmup_iters"], device=context["devices"][0],
                stream_microbatches=True,
            ),
        )
        name = "train-h%02d" % horizon
        emit("Training", "Stage {}/{} | updates={}..{} | warmup_iters={}".format(
            horizon, values["end_stage"], spec["updates"][0], spec["updates"][-1], values["warmup_iters"]))
        result = execute(spec["command"], env, stage, output, name, forward_output=True)
        if result:
            return result
        report, comparison = validate_training_stage(
            engine, stage, Path(context["reference"]) if preset else None,
            Path(context["original_reference"]) if preset else None, horizon, env,
            expected=None if preset else spec["updates"], previous=previous,
            warmup_iters=values["warmup_iters"],
        )
        timings.append(validate_training_timing(engine, stage, report))
        reports.append(report)
        comparisons.append(comparison)
        previous = report
        publish_stage(output, stage, name, "completed")
    if preset:
        return finish_training_chain(context, engine, output, reports, comparisons, timings)
    summary = dict(
        status="passed",
        updates=[row["update"] for report in reports for row in report["rows"]],
        horizons=[report["rollout"] for report in reports],
        checkpoint_chain_verified=True,
        gpu_kernel_union_seconds=sum(
            row["gpu_kernel_union_seconds"]
            for timing in timings for row in timing["rows"]
        ),
    )
    write_json(output / "training-validation.json", summary)
    write_execution_summary(output, options, summary)
    return 0


def finish_training_chain(context, engine, output, reports, comparisons, timings):
    """Retain the fixed full38 trajectory and both frozen-reference loss gates."""
    audit = load_module(engine / "training/verify_chain.py", "chain").audit_reports
    summary = audit(reports, read_json(Path(context["reference"]) / "training/r01/report.json"))
    from graphcast_bench.validation.loss_metrics import aggregate_comparisons

    summary["loss_comparisons"] = comparisons
    summary["loss_validation"] = {
        label: aggregate_comparisons(
            [c[label] for c in comparisons], expected_updates=range(1, 39)
        )
        for label in ("optimized_h100", "original_h100")
    }
    summary["accuracy_status"] = (
        "passed"
        if all(c["metric_passed"] for c in summary["loss_validation"].values())
        else "failed"
    )
    summary["training_scope"] = (
        "38 corresponding updates with checkpoint continuity; not convergence or inference comparison of separately trained checkpoints."
    )
    summary["gpu_kernel_union_seconds"] = sum(
        row["gpu_kernel_union_seconds"] for stage in timings for row in stage["rows"]
    )
    write_json(output / "training-validation.json", summary)
    if summary["accuracy_status"] != "passed":
        return 1
    return 0


def run_generic_training(context, engine, output, environment, options):
    """Compatibility entry for callers of the former custom-only executor."""
    return run_training(context, engine, output, environment, options)


def run_inference(context, engine, output, environment, options):
    """Run the single preflight and fixed rollout, or one explicit custom stage."""
    from graphcast_bench.platforms.environment import h100_inference_environment
    from graphcast_bench.progress import emit

    if not options['preset']:
        return run_generic_inference(context, engine, output, environment, options)
    workload = options['recipe']
    stages = (
        [("single", output)]
        if workload == "single"
        else [("preflight-single", output / "preflight-single"), ("full40", output)]
    )
    for name, stage in stages:
        if stage != output:
            stage.mkdir()
        install_alias("/output", stage)
        single = name != "full40"
        runtime = inference_context(context, stage, 'single' if single else 'full40')
        env = h100_inference_environment(environment, stage, single=single)
        command = [
            context["python"],
            str(engine / "clone3_compat_exec.py"),
            "--",
            "bash",
            "/experiment/run.sh",
        ]
        emit('Initialize', 'Start H100 inference stage: {}; output {}'.format(name, stage))
        result = execute(command, env, stage, output, name, forward_output=True)
        if result:
            return result
        validate_inference_worker(context, stage, runtime.config['inference'])
        if single:
            if read_json(stage / "single-step-validation.json").get("status") != "passed":
                raise ValueError("Single-step numerical validation did not pass")
            if (
                read_json(stage / "single-step-timing.json").get("status")
                != "measured_and_verified"
            ):
                raise ValueError("Single-step timing validation did not pass")
        else:
            env = dict(
                env,
                PYTHONPATH=str(engine / "frozen/src")
                + ":"
                + str(engine / "frozen/third_party/graphcast"),
            )
            # Separate interpreter prevents the launcher's package imports from
            # replacing the frozen benchmark's identically named modules.
            command = [
                context["python"],
                str(Path(__file__).resolve()),
                "--validate-full40",
                str(output / "adapter-context.json"),
            ]
            result = execute(command, env, output / ".runtime", forward_output=True)
            if result:
                return result
        publish_stage(output, stage, name, "completed")
    return 0


def run(context):
    """Run a requested workload inside the shared launcher's locked container."""
    from graphcast_bench.platforms.environment import h100_training_environment, h100_inference_environment
    from graphcast_bench.delivery.options import options_for_context
    options = options_for_context(dict(context, platform="h100"))
    if options['kind'] == 'inference':
        context = dict(context, inference_worker_schema='graphcast.h100_inference_worker.v1')
    workload = options["recipe"] if options["preset"] else options["workload"]
    if options["preset"] and workload not in WORKLOADS:
        raise ValueError("Unsupported Operational H100 workload: " + workload)
    engine = Path(context["release"]).resolve() / ENGINE_RELATIVE
    output = Path(context["output"]).resolve()
    reference = Path(context["reference"]).resolve()
    original_reference = Path(context["original_reference"]).resolve()
    require_fresh_output(output)
    verify_sources(engine)
    environment = runtime_environment(context)
    for root in ((reference, original_reference) if options["preset"] else ()):
        required = (
            [root / "training" / ("r%02d" % h) / "report.json" for h in range(1, 13)]
            if workload == "train-full38"
            else [root / "report.json"]
            if root == original_reference
            else [root / "runs/inference-full40-r1/run/report.json"]
        )
        if any(not path.is_file() for path in required):
            raise ValueError("Missing frozen H100 reference material: " + str(root))
    prepare_aliases(context, engine)
    write_json(output / "adapter-context.json", context)
    write_json(output / "adapter-status.json", {"status": "running", "workload": workload})
    result = 1
    try:
        if options['kind'] == 'inference':
            result = run_inference(context, engine, output, environment, options)
            return result
        result = run_training(context, engine, output, environment, options)
        return result
    except Exception as error:
        result = 1
        write_json(
            output / "adapter-error.json", {"error": str(error), "type": type(error).__name__}
        )
        raise
    finally:
        install_alias("/output", output)
        if result and (output / "active-stage.json").exists():
            active = read_json(output / "active-stage.json")
            publish_stage(output, active["path"], active["stage"], "failed")
        write_json(
            output / "adapter-status.json",
            {
                "status": "passed" if result == 0 else "failed",
                "exit_code": result,
                "workload": workload,
                "scope": "Result of this delivery execution; source provenance alone is not a new GPU validation.",
            },
        )


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--validate-full40":
        raise SystemExit("This adapter is invoked by the delivery launcher")
    raise SystemExit(validate_full40(read_json(sys.argv[2])))
