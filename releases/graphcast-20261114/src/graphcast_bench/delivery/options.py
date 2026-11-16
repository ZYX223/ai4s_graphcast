"""Public run controls, independent of containers and accelerator libraries.

Presets select frozen acceptance recipes. Generic runs select their own execution
bounds; successfully executing those bounds is not a numerical acceptance result.
"""

from copy import deepcopy
from pathlib import Path

ALIASES = {"single": "infer-single", "full40": "infer-full40"}
PRESETS = {"infer-single": "single", "infer-full40": "full40", "train-full38": "train-full38"}
WORKLOADS = ("infer", "train", "infer-single", "infer-full40", "train-full38", "single", "full40")
INFERENCE_DEFAULTS = dict(steps=40, warmup_steps=5, repeats=3, warmup_scope="per-repeat")
TRAINING_DEFAULTS = dict(end_stage=12, max_updates=38, warmup_iters=1, resume=None, save_every=None)


def positive_integer(name, value, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        bound = "1..{}".format(maximum) if maximum else "a positive integer"
        raise ValueError("--{} must be {}".format(name.replace("_", "-"), bound))


def resolve_options(workload, platform, overrides=None):
    if platform not in ("bw1100", "h100"):
        raise ValueError("Unknown platform: " + str(platform))
    workload = ALIASES.get(workload, workload)
    if workload not in WORKLOADS:
        raise ValueError("Unknown workload: " + str(workload))
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    training = workload in ("train", "train-full38")
    kind = "training" if training else "inference"
    preset = workload in PRESETS
    defaults = TRAINING_DEFAULTS if training else INFERENCE_DEFAULTS
    if preset and overrides:
        raise ValueError("Fixed preset {} rejects custom controls; use --workload {}".format(
            workload, "train" if training else "infer"))
    invalid = set(overrides) - set(defaults)
    if invalid:
        raise ValueError("Unsupported {} controls: {}".format(
            kind, ", ".join("--" + key.replace("_", "-") for key in sorted(invalid))))
    values = dict(defaults, **overrides)
    if training:
        positive_integer("end_stage", values["end_stage"], 12)
        positive_integer("max_updates", values["max_updates"], 38)
        positive_integer("warmup_iters", values["warmup_iters"])
        if values["save_every"] is not None:
            positive_integer("save_every", values["save_every"])
        if values["resume"] is not None:
            values["resume"] = str(values["resume"])
            if not values["resume"].strip():
                raise ValueError("--resume must not be empty; specify a checkpoint file or one run directory")
    else:
        if workload == "infer-single":
            values.update(steps=1, repeats=2 if platform == "h100" else 1)
        positive_integer("steps", values["steps"], 40)
        positive_integer("warmup_steps", values["warmup_steps"], 40)
        positive_integer("repeats", values["repeats"])
        if values["warmup_scope"] not in ("once", "per-repeat"):
            raise ValueError("--warmup-scope must be once or per-repeat")
    return dict(workload=workload, recipe=PRESETS.get(workload, "train-full38" if training else "full40"),
                kind=kind, preset=preset, platform=platform, **{kind: values})


def options_for_context(context):
    """Support historical context files while new launchers resolve options once."""
    if "run_options" in context:
        return context["run_options"]
    platform = context.get("platform", "h100" if context.get("engine", "").endswith("h100") else "bw1100")
    return resolve_options(context["workload"], platform)


def training_plan(options, completed=0):
    """Select contiguous global update IDs from the unchanged 38-update recipe."""
    positive_integer("end_stage", options["end_stage"], 12)
    positive_integer("max_updates", options["max_updates"], 38)
    if type(completed) is not int or not 0 <= completed <= 38:
        raise ValueError("Invalid completed update count")
    stages = []
    for horizon in range(1, options["end_stage"] + 1):
        ids = range(1, 17) if horizon == 1 else (2 * horizon + 13, 2 * horizon + 14)
        updates = [i for i in ids if completed < i <= options["max_updates"]]
        if updates:
            stages.append(dict(horizon=horizon, updates=updates))
    if not stages:
        raise ValueError("No training updates remain within --end-stage / --max-updates")
    return stages


def resume_progress(path):
    """Read verified checkpoint evidence, never unpickle on the host."""
    from graphcast_bench.training.resume import completed_updates, validate_resume
    return completed_updates(validate_resume(path))


def prepare_resume(options, work):
    """Resolve host evidence and map its path into the existing /work mount."""
    result = deepcopy(options)
    if result["kind"] != "training":
        return result
    values = result["training"]
    completed = 0
    if values["resume"] is not None:
        from graphcast_bench.training.resume import resolve_checkpoint, validate_resume
        path = Path(values["resume"]).expanduser().resolve()
        try:
            relative = path.relative_to(Path(work).expanduser().resolve())
        except ValueError:
            raise ValueError("--resume checkpoint must be inside --work")
        path = resolve_checkpoint(path)
        relative = path.resolve().relative_to(Path(work).expanduser().resolve())
        report = validate_resume(path)
        expected_mode = {"bw1100": "rne_xf32", "h100": "tf32"}.get(result.get("platform"))
        if expected_mode and report.get("mode", expected_mode) != expected_mode:
            raise ValueError("Checkpoint mode does not match requested platform")
        if report.get("tiny", False) or report.get("preflight", False):
            raise ValueError("Cannot resume tiny/preflight checkpoints in delivery training")
        completed = resume_progress(path)
        values["resume"] = "/work/" + str(relative)
    values["completed_updates"] = completed
    values["stages"] = training_plan(values, completed)
    return result
