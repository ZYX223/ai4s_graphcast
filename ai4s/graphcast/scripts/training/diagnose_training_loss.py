#!/usr/bin/env python3
"""Compare aligned training losses; this is not a weather-skill evaluation.

Adapted from the project team's diagnostic_package/diagnose_training_loss.py.
Source SHA256 and adaptations are recorded in docs/training-source-provenance.md.
The ratio-of-sums metric and inclusive threshold retain the colleague algorithm.
Only the Python standard library is required; no model or accelerator is loaded.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

# Bypass package __init__, which intentionally imports model dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/graphcast_compat"))
from training_diagnostics import (
    SEGMENT_SCHEMA, SOURCE_UPDATE_ALIASES, source_context,
)


STEP_COLUMNS = ("optimizer_step", "step", "global_step", "iteration", "iter")
LOSS_COLUMNS = ("loss", "pytorch_loss", "train_loss", "training_loss", "input_loss",
                "candidate_loss", "jax_loss")
UPDATE_FIELDS = ("epoch", "batch_index", "rollout_steps", "phase", "learning_rate",
                 "effective_samples", "accumulation_steps")
CONTRACT_FIELDS = ("data_sha256", "stats_sha256", "config_sha256", "schedule_digest",
                   "window_indices_sha256", "epoch_policy", "seed", "shuffle",
                   "global_batch_size", "micro_batch_size_per_rank", "dp", "mp")
POLICY_FIELDS = ("storage_dtype", "tensor_core", "per_variable_weights", "optimizer", "normalization")


class DiagnosticError(RuntimeError):
    pass


def as_float(value, label):
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a number")
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DiagnosticError(f"cannot parse {label}: {value!r}") from exc
    if not math.isfinite(value):
        raise DiagnosticError(f"{label} is not finite: {value!r}")
    return value


def as_int(value, label):
    number = as_float(value, label)
    if not number.is_integer() or number < 0:
        raise DiagnosticError(f"{label} must be a non-negative integer: {value!r}")
    return int(number)


def pick_column(fieldnames, candidates, explicit, kind):
    if explicit:
        if explicit not in fieldnames:
            raise DiagnosticError(f"{kind} column {explicit!r} missing; available: {fieldnames}")
        return explicit
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise DiagnosticError(f"cannot identify {kind} column; available: {fieldnames}")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise DiagnosticError(f"cannot read JSON {path}: {exc}") from exc


def _load_rows(path, objects, step_column, loss_column, metadata):
    rows, run = {}, metadata
    for line_number, obj in enumerate(objects, 1):
        if not isinstance(obj, dict):
            raise DiagnosticError(f"{path}:{line_number} must contain an object")
        if obj.get("event") == "start":
            run = obj
            continue
        if obj.get("event") not in (None, "update"):
            raise DiagnosticError(f"{path}:{line_number} unsupported event: {obj.get('event')!r}")
        if "training_sampling" in obj:
            obj = {**obj, **{target: obj[source] for target, source in SOURCE_UPDATE_ALIASES.items()
                            if source in obj}}
        step_name = pick_column(list(obj), STEP_COLUMNS, step_column, "step")
        loss_name = pick_column(list(obj), LOSS_COLUMNS, loss_column, "loss")
        step = as_int(obj.get(step_name), "step")
        loss = as_float(obj.get(loss_name), "loss")
        if step in rows:
            raise DiagnosticError(f"{path}: duplicate step {step}; split overlapping/restarted logs first")
        if "finite" in obj and str(obj["finite"]).lower() != "true":
            raise DiagnosticError(f"{path}: step {step} finite check failed")
        if "gradient_norm" in obj:
            as_float(obj["gradient_norm"], "gradient_norm finite check")
        rows[step] = {"loss": loss, "fields": obj, "run": run}
    if not rows:
        raise DiagnosticError(f"{path}: no loss records")
    return rows


def load_loss_file(path, step_column=None, loss_column=None, metadata=None):
    path = Path(path)
    if path.is_dir():
        if metadata is not None:
            raise DiagnosticError("directory comparisons require per-segment sidecars")
        directory = path / "metrics" if (path / "metrics").is_dir() else path
        rows = {}
        for segment in sorted(directory.glob("segment_*.jsonl")):
            chunk = load_loss_file(segment, step_column, loss_column)
            if rows.keys() & chunk.keys():
                raise DiagnosticError("duplicate step across metric segments")
            rows.update(chunk)
        if not rows:
            raise DiagnosticError(f"{path}: no loss records")
        return rows
    if metadata is None and path.name.startswith("segment_") and path.with_suffix(".start.json").exists():
        metadata = path.with_suffix(".start.json")
    supplied = _read_json(metadata) if metadata is not None else None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            if path.suffix.lower() == ".csv":
                return _load_rows(path, csv.DictReader(handle), step_column, loss_column, supplied)
            # JSONL also supports extensionless logs; do not reinterpret malformed
            # JSONL as CSV and accidentally hide the original error.
            objects = (json.loads(line) for line in handle if line.strip())
            return _load_rows(path, objects, step_column, loss_column, supplied)
    except (ValueError, OSError, csv.Error) as exc:
        raise DiagnosticError(f"cannot load {path}: {exc}") from exc


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _comparison_context(row):
    if "training_sampling" in row["fields"]:
        return source_context(row["run"])
    comparison = _get(row["run"], "loss_comparison")
    context = {name: _get(comparison, name) for name in (
        "start_optimizer_step", "starting_parameters_sha256", "starting_optimizer_sha256")}
    context.update({name: _get(comparison, "resume_contract", name) for name in CONTRACT_FIELDS})
    context.update({f"policy.{name}": _get(comparison, "resume_contract", "policy", name)
                    for name in POLICY_FIELDS})
    return context


def _check_runtime(row, label, missing, mismatches):
    run = row["run"]
    if "training_sampling" in row["fields"]:
        _check_source_runtime(row, label, missing, mismatches)
        return
    requested = _get(run, "requested_backend")
    policy_backend = _get(run, "loss_comparison", "resume_contract", "policy", "backend")
    mode = _get(run, "resolved_backend", "flagos_mode")
    status = _get(run, "resolved_backend", "flagos_status")
    for name, value in (("requested_backend", requested), ("policy.backend", policy_backend),
                        ("flagos_mode", mode), ("flagos_status", status)):
        if value in (None, "", "unverified"):
            missing.add(f"{label}: missing runtime evidence: {name}")
    if requested not in (None, "", "unverified"):
        if requested not in ("native", "flaggems") or label == "reference" and requested != "native":
            mismatches.add(f"{label}: expected Native reference or Native/FlagGems candidate, got {requested}")
        if policy_backend not in (None, "", "unverified", requested):
            mismatches.add(f"{label}: policy.backend disagrees with requested_backend")
        expected = ("off", "disabled") if requested == "native" else ("flaggems", "enabled_only_enable")
        if mode not in (None, "", "unverified") and status not in (None, "", "unverified"):
            if (mode, status) != expected:
                mismatches.add(f"{label}: resolved backend disagrees with requested_backend")
    precision = _get(run, "resolved_precision")
    tensor_core = _get(run, "loss_comparison", "resume_contract", "policy", "tensor_core")
    if precision in (None, "", "unverified"):
        missing.add(f"{label}: missing resolved_precision")
    elif tensor_core in ("enabled", "disabled"):
        allowed = ("tf32", "hf32") if tensor_core == "enabled" else ("ieee_fp32",)
        if precision not in allowed:
            mismatches.add(f"{label}: resolved_precision disagrees with tensor_core request")


def _source_container(value, kind, label, *, missing_allowed=True):
    """Check artifact structure before traversal; missing evidence stays missing."""
    if value is None and missing_allowed:
        return kind()
    if not isinstance(value, kind):
        expected = "object" if kind is dict else "array"
        raise DiagnosticError(f"{label} must be a JSON {expected}")
    return value


def _validate_source_structure(row, label):
    run = _source_container(row["run"], dict, f"{label} segment metadata")
    for name in ("resume_contract", "stats_sha256", "optimizer", "per_variable_weights", "execution_policy"):
        _source_container(run.get(name), dict, f"{label} {name}")
    _source_container(run.get("data_sha256"), list, f"{label} data_sha256")
    for name in ("runtime_by_rank", "states_by_rank"):
        records = _source_container(run.get(name), list, f"{label} {name}")
        for index, record in enumerate(records):
            record_label = f"{label} {name}[{index}]"
            _source_container(record, dict, record_label, missing_allowed=False)
            if name == "runtime_by_rank":
                _source_container(record.get("runtime"), dict, f"{record_label}.runtime")
                for field in ("operators", "skipped"):
                    _source_container(record.get(field), list, f"{record_label}.{field}")
    origins = _source_container(row["fields"].get("origins_by_rank"), list,
                                f"{label} origins_by_rank")
    for index, origin in enumerate(origins):
        origin_label = f"{label} origins_by_rank[{index}]"
        _source_container(origin, dict, origin_label, missing_allowed=False)
        for name in ("global_window_indices", "local_window_indices", "sample_weights",
                     "global_sample_weights", "reference_datetimes", "target_datetimes_by_sample"):
            values = _source_container(origin.get(name), list, f"{origin_label}.{name}")
            if name == "target_datetimes_by_sample":
                for sample_index, times in enumerate(values):
                    _source_container(times, list, f"{origin_label}.{name}[{sample_index}]",
                                      missing_allowed=False)


def _check_source_runtime(row, label, missing, mismatches):
    _validate_source_structure(row, label)
    run = row["run"] or {}
    if run.get("schema") != SEGMENT_SCHEMA:
        missing.add(f"{label}: missing source segment comparison schema")
    contract = run.get("resume_contract") or {}
    runtimes = run.get("runtime_by_rank") or []
    states = run.get("states_by_rank") or []
    expected_ranks = list(range(as_int(contract.get("data_parallel_replicas", 0), "replicas")))
    for name, records in (("runtime", runtimes), ("state", states)):
        if not records or [r.get("rank") for r in records] != expected_ranks:
            missing.add(f"{label}: missing complete rank {name} evidence")
    for replacement in runtimes:
        runtime = replacement.get("runtime") or {}
        requested = replacement.get("mode")
        if requested == "torch_native":
            requested = "native"
        if requested not in ("native", "flaggems"):
            missing.add(f"{label}: missing requested runtime mode")
            continue
        if label == "reference" and requested != "native":
            mismatches.add("reference: expected Native runtime")
        expected = ("off", "disabled", False) if requested == "native" else ("flaggems", "enabled_only_enable", True)
        observed = (runtime.get("flagos_mode"), runtime.get("flagos_status"), replacement.get("enabled"))
        if any(v is None for v in observed):
            missing.add(f"{label}: missing resolved backend evidence")
        elif observed != expected:
            mismatches.add(f"{label}: resolved backend disagrees with requested runtime")
        if requested == "native" and replacement.get("operators"):
            mismatches.add(f"{label}: Native runtime has replacements")
        if requested == "flaggems" and not replacement.get("operators"):
            missing.add(f"{label}: missing replacement operator evidence")
    precision = run.get("resolved_precision")
    enabled = contract.get("tensor_core_enabled")
    if precision is None or enabled is None:
        missing.add(f"{label}: missing resolved precision")
    elif precision not in (("tf32", "hf32") if enabled else ("ieee_fp32",)):
        mismatches.add(f"{label}: resolved precision disagrees with tensor_core_enabled")
    if contract.get("precision") not in (None, "fp32"):
        mismatches.add(f"{label}: expected FP32 storage")
    origins = row["fields"].get("origins_by_rank") or []
    if not origins or [r.get("rank") for r in origins] != expected_ranks:
        missing.add(f"{label}: missing complete sample origins_by_rank")
    for origin in origins:
        for name in ("global_window_indices", "local_window_indices", "sample_weights",
                     "global_sample_weights", "reference_datetimes", "target_datetimes_by_sample"):
            if not origin.get(name):
                missing.add(f"{label}: missing sample provenance {name}")


def _comparability(candidate, reference, steps):
    missing, mismatches = set(), set()
    positive_lr = False
    for step in steps:
        cand, ref = candidate[step], reference[step]
        _check_runtime(cand, "candidate", missing, mismatches)
        _check_runtime(ref, "reference", missing, mismatches)
        contexts = (_comparison_context(cand), _comparison_context(ref))
        for name in contexts[0].keys() | contexts[1].keys():
            a, b = contexts[0].get(name), contexts[1].get(name)
            if any(value is None or value == "" or isinstance(value, (list, dict)) and not value
                   for value in (a, b)):
                missing.add(f"missing run metadata: {name}")
            elif a != b:
                mismatches.add(f"run metadata differs: {name}")
        for label, context in zip(("candidate", "reference"), contexts):
            start = context.get("start_optimizer_step")
            if start is not None and as_int(start, "start_optimizer_step") >= step:
                mismatches.add(f"{label} step {step} is not after its run start")
        for name in UPDATE_FIELDS:
            a, b = cand["fields"].get(name), ref["fields"].get(name)
            if a is None or b is None or a == "" or b == "":
                missing.add(f"missing update metadata: {name}")
                continue
            if name != "phase":
                parse = as_float if name == "learning_rate" else as_int
                a, b = parse(a, name), parse(b, name)
                if name == "learning_rate" and (a < 0 or b < 0):
                    raise DiagnosticError("learning_rate must be non-negative")
            if a != b:
                mismatches.add(f"step {step} differs: {name}")
            if name == "learning_rate":
                positive_lr = positive_lr or (a > 0 and b > 0)
        for row in (cand, ref):
            if "finite" not in row["fields"] or "gradient_norm" not in row["fields"]:
                missing.add("missing finite/gradient evidence")
        if "training_sampling" in cand["fields"] or "training_sampling" in ref["fields"]:
            for name in ("training_sampling", "origins_by_rank"):
                a, b = cand["fields"].get(name), ref["fields"].get(name)
                if a is not None and b is not None and a != b:
                    mismatches.add(f"step {step} differs: {name}")
    if not positive_lr:
        missing.add("no paired positive-learning-rate update in the requested interval")
    status = "incomparable_training" if mismatches else "unverified_comparison" if missing else "verified"
    return status, sorted(mismatches | missing)


def compare_losses(candidate, baseline, *, start_step, end_step, threshold=5.,
                   step_column=None, loss_column=None, baseline_step_column=None,
                   baseline_loss_column=None, candidate_metadata=None, baseline_metadata=None):
    threshold = as_float(threshold, "threshold")
    if threshold < 0:
        raise DiagnosticError("threshold must be non-negative")
    start_step, end_step = as_int(start_step, "start step"), as_int(end_step, "end step")
    if end_step < start_step:
        raise DiagnosticError("end step must not precede start step")
    candidate_rows = load_loss_file(candidate, step_column, loss_column, candidate_metadata)
    baseline_rows = load_loss_file(baseline, baseline_step_column, baseline_loss_column, baseline_metadata)
    steps = list(range(start_step, end_step + 1))
    for label, records in (("candidate", candidate_rows), ("reference", baseline_rows)):
        missing = sorted(set(steps) - set(records))
        if missing:
            raise DiagnosticError(f"{label}: incomplete requested interval; missing steps {missing[:20]}")

    # Colleague's ratio-of-sums algorithm. Per-step percentages are explanatory,
    # not the aggregate threshold. Preserve <= threshold for the loss metric.
    rows = []
    sum_abs_diff = sum_signed_diff = sum_abs_ref = sum_candidate = sum_ref = 0.
    max_step_relative, max_step, sum_step_relative = -1., None, 0.
    for step in steps:
        ref = baseline_rows[step]["loss"]
        candidate_loss = candidate_rows[step]["loss"]
        signed_diff = candidate_loss - ref
        abs_diff, abs_ref = abs(signed_diff), abs(ref)
        step_relative = (0. if abs_diff == 0. else math.inf) if abs_ref == 0. else abs_diff / abs_ref * 100.
        sum_abs_diff += abs_diff
        sum_signed_diff += signed_diff
        sum_abs_ref += abs_ref
        sum_candidate += candidate_loss
        sum_ref += ref
        sum_step_relative += step_relative
        if step_relative > max_step_relative:
            max_step_relative, max_step = step_relative, step
        rows.append({"step": step, "reference_loss": ref, "candidate_loss": candidate_loss,
                     "abs_diff": abs_diff, "signed_diff": signed_diff,
                     "step_relative_percent": step_relative,
                     "metric_passed": step_relative <= threshold})
    mean_relative = ((0. if sum_abs_diff == 0. else math.inf) if sum_abs_ref == 0.
                     else sum_abs_diff / sum_abs_ref * 100.)
    comparison_status, issues = _comparability(candidate_rows, baseline_rows, steps)
    # Only roundoff at the agreed inclusive 5% boundary is tolerated (8 ULPs).
    boundary_roundoff = threshold == 5. and abs(mean_relative - 5.) <= 8 * math.ulp(5.)
    metric_passed = mean_relative <= threshold or boundary_roundoff
    status = "passed" if metric_passed else "failed_loss"
    if comparison_status != "verified":
        status = comparison_status
    if sum_abs_ref == 0. or not all(math.isfinite(v) for v in (
            sum_abs_diff, sum_signed_diff, sum_abs_ref, sum_candidate, sum_ref, mean_relative)):
        status = "invalid_reference"
        issues.append("zero reference magnitude or nonfinite aggregate; not a valid relative-loss gate")
    count = float(len(steps))
    return {"schema": "graphcast.training_loss_diagnostic.v1", "status": status,
            "candidate": str(Path(candidate).resolve()), "baseline": str(Path(baseline).resolve()),
            "metric": "sum(abs(candidate-reference))/sum(abs(reference))*100",
            "threshold_percent": threshold, "threshold_comparison": "less_than_or_equal",
            "evaluated_steps": steps, "candidate_steps": len(candidate_rows), "reference_steps": len(baseline_rows),
            "reference_mean_loss": sum_ref / count, "candidate_mean_loss": sum_candidate / count,
            "mean_abs_loss_diff": sum_abs_diff / count, "mean_signed_loss_diff": sum_signed_diff / count,
            "mean_relative_loss_percent": mean_relative, "mean_step_relative_percent": sum_step_relative / count,
            "max_step_relative_percent": max_step_relative, "max_step": max_step,
            "metric_passed": metric_passed, "comparison_status": comparison_status,
            "comparison_issues": issues, "weather_skill": "not_evaluated", "records": rows}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="candidate run/metrics directory, segment JSONL, legacy updates.jsonl or CSV")
    parser.add_argument("--baseline", type=Path, required=True, help="same-recipe Native reference log; no implicit JAX baseline")
    parser.add_argument("--start-step", type=int, required=True, help="first required optimizer step (inclusive)")
    parser.add_argument("--end-step", type=int, required=True, help="last required optimizer step (inclusive)")
    parser.add_argument("--threshold", type=float, default=5.)
    parser.add_argument("--step-column")
    parser.add_argument("--loss-column")
    parser.add_argument("--baseline-step-column")
    parser.add_argument("--baseline-loss-column")
    parser.add_argument("--candidate-metadata", type=Path, help="start-event JSON for legacy/CSV logs")
    parser.add_argument("--baseline-metadata", type=Path, help="start-event JSON for legacy/CSV logs")
    parser.add_argument("--output-json", type=Path, required=True, help="new output file; never overwrites an existing file")
    args = parser.parse_args(argv)
    try:
        inputs = [args.candidate, args.baseline, args.candidate_metadata, args.baseline_metadata]
        if args.output_json.resolve() in [path.resolve() for path in inputs if path is not None]:
            raise DiagnosticError("output must not overwrite an input")
        if args.output_json.exists():
            raise DiagnosticError(f"output already exists: {args.output_json}")
        result = compare_losses(args.candidate, args.baseline,
            start_step=args.start_step, end_step=args.end_step, threshold=args.threshold,
            step_column=args.step_column, loss_column=args.loss_column,
            baseline_step_column=args.baseline_step_column, baseline_loss_column=args.baseline_loss_column,
            candidate_metadata=args.candidate_metadata, baseline_metadata=args.baseline_metadata)
        payload = json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print(json.dumps(_json_safe({key: value for key, value in result.items() if key != "records"}),
                         ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "passed" else 1 if result["status"] == "failed_loss" else 2
    except (DiagnosticError, OSError) as exc:
        print(f"diagnostic failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
