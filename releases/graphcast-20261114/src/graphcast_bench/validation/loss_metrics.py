"""Shared loss statistic, independent of JAX and GPU profiling.

Adapters verify the case, frozen reference, completion and backend evidence.
This module then checks the specified update interval and aligned row metadata.
Continuous training aggregates raw pairs once; stage percentages are diagnostic.
"""

import hashlib
import json
import math

METRIC_VERSION = "sugon.training_loss.sum_abs.v1"
THRESHOLD_PERCENT = 5.0


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Non-finite or invalid " + name)
    return float(value)


def within_threshold(error):
    # Permit only binary64 round-off at the inclusive boundary, not a practical
    # tolerance band. For example, (1.05 - 1) * 100 is 5.000000000000004.
    return error <= THRESHOLD_PERCENT or math.isclose(
        error, THRESHOLD_PERCENT, rel_tol=0, abs_tol=8 * math.ulp(THRESHOLD_PERCENT)
    )


def _updates(rows, expected):
    actual = [row.get("update") for row in rows]
    if any(type(i) is not int for i in actual) or actual != expected:
        raise ValueError("Missing, duplicate, reordered or unexpected updates")


def compare_loss_rows(candidate, reference, *, expected_updates, invariant_keys=()):
    """Compare a complete, explicit interval; never silently intersect logs."""
    expected = list(expected_updates)
    if not expected or any(type(i) is not int or i < 1 for i in expected):
        raise ValueError("Expected updates must be a non-empty positive interval")
    if expected != list(range(expected[0], expected[-1] + 1)):
        raise ValueError("Expected updates must be a contiguous interval")
    _updates(candidate, expected)
    _updates(reference, expected)
    pairs = []
    for c, r in zip(candidate, reference):
        for key in ("lr", *invariant_keys):
            if key not in c or key not in r or c[key] != r[key]:
                raise ValueError("Changed or missing update invariant: " + key)
        lr = _number(c["lr"], "learning rate")
        _number(r["lr"], "reference learning rate")
        if lr < 0:
            raise ValueError("Negative learning rate")
        a, b = _number(c["loss"], "candidate loss"), _number(r["loss"], "reference loss")
        difference = abs(a - b)
        error = 100 * difference / abs(b) if b != 0 else None
        if not math.isfinite(difference) or (error is not None and not math.isfinite(error)):
            raise ValueError("Non-finite loss difference")
        pairs.append(
            dict(
                update=c["update"],
                lr=lr,
                candidate_loss=a,
                reference_loss=b,
                absolute_difference=difference,
                relative_error_percent=error,
            )
        )
    if not any(row["lr"] > 0 for row in pairs):
        raise ValueError("At least one paired positive learning-rate update is required")
    numerator = math.fsum(row["absolute_difference"] for row in pairs)
    denominator = math.fsum(abs(row["reference_loss"]) for row in pairs)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Total reference loss magnitude must be finite and positive")
    error = 100 * (numerator / denominator)
    if not math.isfinite(error):
        raise ValueError("Non-finite aggregate loss error")
    individual = [row["relative_error_percent"] for row in pairs]
    undefined = sum(value is None for value in individual)
    return dict(
        metric_version=METRIC_VERSION,
        formula="100 * sum(abs(candidate_loss-reference_loss)) / sum(abs(reference_loss))",
        threshold_percent=THRESHOLD_PERCENT,
        loss_error_percent=error,
        sum_abs_diff=numerator,
        sum_abs_ref=denominator,
        metric_passed=within_threshold(error),
        comparison_status="paired_updates_verified",
        updates=len(pairs),
        expected_updates=expected,
        positive_lr_updates=sum(row["lr"] > 0 for row in pairs),
        rows=pairs,
        undefined_relative_error_updates=undefined,
        max_relative_error_percent=None if undefined else max(individual),
        mean_relative_error_percent=None if undefined else math.fsum(individual) / len(individual),
        last_relative_error_percent=individual[-1],
    )


def aggregate_comparisons(comparisons, *, expected_updates):
    """Recompute from raw pairs, retaining all updates across horizon boundaries."""
    comparisons = list(comparisons)
    candidate, reference = [], []
    segments = []
    has_trajectory = any("trajectory_segments" in item for item in comparisons)
    identities = {result.get("comparison_identity") for result in comparisons}
    if len(identities) > 1:
        raise ValueError("Stage comparison identity differs")
    for result in comparisons:
        if (
            result.get("metric_version") != METRIC_VERSION
            or result.get("comparison_status") != "paired_updates_verified"
        ):
            raise ValueError("Comparison lacks the current metric and pairing checks")
        if has_trajectory:
            current = result.get("trajectory_segments")
            if not current:
                raise ValueError("Stage comparison lacks trajectory evidence")
            offset = 0
            for segment in current:
                length = len(segment["updates"])
                paired = result["rows"][offset : offset + length]
                if [row["update"] for row in paired] != segment["updates"] or _digest(
                    paired
                ) != segment["pairs_sha256"]:
                    raise ValueError("Trajectory evidence does not match loss pairs")
                for platform in ("candidate", "reference"):
                    state = segment[platform]
                    for key in (
                        "initial_params_sha256",
                        "initial_optimizer_sha256",
                        "final_params_sha256",
                        "final_optimizer_sha256",
                        "canonical_report_sha256",
                    ):
                        if not isinstance(state.get(key), str) or not state[key]:
                            raise ValueError("Missing trajectory state: " + key)
                    if segments:
                        previous = segments[-1][platform]
                        for kind in ("params", "optimizer"):
                            if (
                                state["initial_" + kind + "_sha256"]
                                != previous["final_" + kind + "_sha256"]
                            ):
                                raise ValueError(
                                    platform + " trajectory is not continuous: " + kind
                                )
                segments.append(segment)
                offset += length
            if offset != len(result["rows"]):
                raise ValueError("Trajectory evidence omits loss pairs")
        for row in result["rows"]:
            candidate.append(dict(update=row["update"], lr=row["lr"], loss=row["candidate_loss"]))
            reference.append(dict(update=row["update"], lr=row["lr"], loss=row["reference_loss"]))
    result = compare_loss_rows(candidate, reference, expected_updates=expected_updates)
    if identities and None not in identities:
        result["comparison_identity"] = identities.pop()
    if has_trajectory:
        result.update(trajectory_segments=segments, trajectory_verified=True)
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def attach_trajectory(result, candidate, reference, *, layout="operational"):
    """Bind raw loss pairs to each platform's independent parameter/optimizer chain.

    Operational initial hashes use tree_digest: compare them with checkpoint
    reload hashes, never with its flattened final_params hash. Full38 uses the
    same tree convention for its initial and final hashes.
    """
    if layout not in ("operational", "full38"):
        raise ValueError("Unknown training report layout")
    segment = dict(updates=result["expected_updates"], pairs_sha256=_digest(result["rows"]))
    for label, report in (("candidate", candidate), ("reference", reference)):
        reload = report["checkpoint_reload"]
        if reload["status"] != "passed" or reload["optimizer_counters"] != [
            result["expected_updates"][-1]
        ]:
            raise ValueError("Checkpoint reload or optimizer counter is invalid")
        segment[label] = dict(
            initial_params_sha256=report["initial_params_sha256"],
            initial_optimizer_sha256=report["initial_optimizer_sha256"],
            final_params_sha256=reload["params_sha256"]
            if layout == "operational"
            else report["final_params_sha256"],
            final_optimizer_sha256=reload["optimizer_sha256"]
            if layout == "operational"
            else report["final_optimizer_sha256"],
            canonical_report_sha256=_digest(report),
        )
    result["trajectory_segments"] = [segment]
    return result


def comparison_identity(candidate, reference, keys):
    """Identify stable case/config metadata, excluding independently updated state."""
    metadata = {
        label: {key: report[key] for key in keys}
        for label, report in (("candidate", candidate), ("reference", reference))
    }
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def enforce_loss_threshold(result):
    """Accept only a new-format, finite, complete-interval numerical verdict."""
    if (
        result.get("metric_version") != METRIC_VERSION
        or result.get("comparison_status") != "paired_updates_verified"
    ):
        raise ValueError("Training comparison requires the current loss metric")
    checked = aggregate_comparisons([result], expected_updates=result["expected_updates"])
    if not checked["metric_passed"]:
        raise ValueError("Full-interval loss difference exceeds 5%")


def operational_updates(horizon):
    """The existing 38-update Operational schedule, not a GenCast default."""
    if type(horizon) is not int or not 1 <= horizon <= 12:
        raise ValueError("Invalid Operational horizon")
    return list(range(1, 17)) if horizon == 1 else [13 + 2 * horizon, 14 + 2 * horizon]


def main(argv=None):
    """Aggregate previously validated stage comparisons without overwriting them."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", type=Path, nargs="+")
    parser.add_argument("--first-update", type=int, required=True)
    parser.add_argument("--last-update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Summary output must be new")
    comparisons = [json.loads(path.read_text()) for path in args.comparisons]
    if not comparisons or any(not c.get("comparison_identity") for c in comparisons):
        raise ValueError("Stage comparison lacks verified case/config identity")
    if any(not c.get("trajectory_segments") for c in comparisons):
        raise ValueError("Stage comparison lacks checkpoint trajectory evidence")
    references = {value.get("reference") for value in comparisons}
    if len(references) != 1:
        raise ValueError("Stage reference identities differ")
    result = aggregate_comparisons(
        comparisons, expected_updates=range(args.first_update, args.last_update + 1)
    )
    result["reference"] = comparisons[0].get("reference")
    result["source_comparisons"] = [
        dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in args.comparisons
    ]
    # Exclusive creation protects a historical summary even if another writer
    # appears between the existence check and the write.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["metric_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
