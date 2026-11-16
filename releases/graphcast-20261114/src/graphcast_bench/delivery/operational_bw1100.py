"""Bind the previously delivered BW numerical engine to versioned materials."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from graphcast_bench.validation.loss_metrics import aggregate_comparisons
from graphcast_bench.validation.execution import write_summary
from .options import options_for_context, training_plan, resume_progress

from .runtime import build_router
from graphcast_bench.progress import emit, phase


def run(context):
    release, out = Path(context["release"]), Path(context["output"])
    model, data = Path(context["model"]), Path(context["data"])
    paths = dict(
        assets=str(data / "prepared-inference"),
        training_data=str(data / "training"),
        inference_graph=str(model / "inference-graph"),
        training_graph=str(model / "training-graph"),
        library_dirs=["/opt/hyhal/lib"],
    )
    kind = context["kind"]
    if kind == 'inference':
        with phase('Prepare', 'Build or reuse the native inference library'):
            paths[kind + "_router"] = build_router(release, context["build"], kind)
    else:
        paths[kind + "_router"] = build_router(release, context["build"], kind)
    config = out / "paths.json"
    config.write_text(json.dumps(paths, indent=2) + "\n")
    device = context["devices"][0]
    base = [
        sys.executable,
        str(release / "scripts" / ("train.py" if kind == "training" else "inference.py")),
        "--paths",
        str(config),
        "--device",
        str(device["index"]),
        "--pci",
        device["pci"],
    ]
    validate = [sys.executable, str(release / "scripts/validate.py")]

    def publish(target, status):
        # Atomic replacement prevents the host sampler from seeing partial JSON.
        state = out / "active-stage.json"
        temporary = state.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(
                schema="graphcast.delivery.stage.v1",
                stage=target.name, path=str(target), status=status,
            )) + "\n"
        )
        temporary.replace(state)

    @contextmanager
    def stage(name, args, completion):
        target = out / name
        publish(target, "running")
        try:
            if kind == 'inference':
                emit('Initialize', 'Start inference worker: ' + name)
            subprocess.check_call(base + ["--output", str(target)] + args)
            publish(target, "analyzing")
            if kind == 'inference':
                with phase('Validation', 'Validate execution, saved outputs and audits: ' + name):
                    subprocess.check_call(validate + ["completion", str(target)] + completion)
            else:
                subprocess.check_call(validate + ["completion", str(target)] + completion)
            # Include accuracy checks performed by the caller in this stage.
            yield target
        except BaseException:
            publish(target, "failed")
            if kind == 'inference':
                emit('Failure', 'Inference stage failed: ' + name + '; see the full diagnostic log', 'ERROR')
            raise
        else:
            publish(target, "completed")

    options = options_for_context(context)
    if kind == "inference":
        return run_inference(context, options, stage, validate)
    return run_training(context, options, stage, validate, publish)


def inference_flags(values):
    return ["--steps", str(values["steps"]), "--warmup-steps", str(values["warmup_steps"]),
            "--repeats", str(values["repeats"]), "--warmup-scope", values["warmup_scope"]]


def run_inference(context, options, stage, validate):
    out = Path(context["output"])
    if not options["preset"]:
        flags = inference_flags(options["inference"])
        with stage("inference", ["--profile", "custom"] + flags, ["inference"] + flags):
            pass
        write_summary(out, options, dict(stages=["inference"]))
        values = options['inference']
        warmups = 1 if values['warmup_scope'] == 'once' else values['repeats']
        emit('Validation', 'Execution passed: {} warmup steps, {} measured steps, {} rounds'.format(
            warmups * values['warmup_steps'], values['steps'] * values['repeats'], values['repeats']))
        emit('Accuracy', 'not evaluated: custom inference has no frozen-reference acceptance')
        return 0

    # Full reproduction retains the single-step accuracy gate before full40.
    profiles = ["single", "full40"] if options["recipe"] == "full40" else ["single"]
    for profile in profiles:
        with stage(profile, ["--profile", profile], [profile]) as target:
            comparison = out / ("single-comparison" if profile == "single" else "comparison")
            with phase('Accuracy', 'Compare against frozen H100 reference: ' + profile):
                subprocess.check_call(validate + [
                    "inference", "--accept-stage-threshold", "--candidate", str(target / "run"),
                    "--reference", context["reference"], "--steps", "1" if profile == "single" else "40",
                    "--output", str(comparison),
                ])
                metrics = json.loads((comparison / 'summary.json').read_text())
                if metrics['all_pairs_at_most_10_percent'] is not True:
                    raise RuntimeError('Frozen-reference comparison did not pass')
                emit('Accuracy', 'Stage threshold passed | overall relative MAE={:.6g}% | Z500={:.6g}% | internal <5%={}'.format(
                    metrics['max_overall_percent'], metrics['max_z500_percent'],
                    'passed' if metrics['all_pairs_under_5_percent'] else 'not passed'))
    return 0


def run_training(context, options, stage, validate, publish):
    out = Path(context["output"])
    values = options["training"]
    previous = values["resume"]
    completed = resume_progress(previous) if previous else 0
    stages = training_plan(values, completed)
    comparisons = []
    for selected in stages:
        horizon, updates = selected["horizon"], selected["updates"]
        emit('Training', 'Stage {}/{} | updates={}..{} | warmup_iters={}'.format(
            horizon, values['end_stage'], updates[0], updates[-1], values['warmup_iters']))
        args = ["--rollout", str(horizon), "--limit", str(updates[-1]),
                "--warmup-iters", str(values["warmup_iters"])]
        if values.get("save_every") is not None:
            args += ["--save-every", str(values["save_every"])]
        check = ["training-stage", "--updates", ",".join(map(str, updates)),
                 "--rollout", str(horizon), "--warmup-iters", str(values["warmup_iters"])]
        check.append("--check-trace")
        if previous:
            args += ["--resume", str(previous)]
            check += ["--predecessor", str(previous)]
        with stage("r%02d" % horizon, args, check) as target:
            if options["preset"]:
                subprocess.check_call(validate + [
                    "training-loss", "--diagnostic-only", str(target),
                    "--reference-root", context["reference"],
                    "--output", str(target / "loss-comparison.json"),
                ])
                comparisons.append(json.loads((target / "loss-comparison.json").read_text()))
            previous = target / "checkpoint.pkl"
    if options["preset"]:
        loss = aggregate_comparisons(comparisons, expected_updates=range(1, 39))
        (out / "training-loss-summary.json").write_text(
            json.dumps(loss, indent=2, allow_nan=False) + "\n")
        if not loss["metric_passed"]:
            publish(target, "failed")
            return 1
    else:
        write_summary(out, options, dict(completed_before=completed, stages=stages,
                                         completed_updates=stages[-1]["updates"][-1]))
    return 0
