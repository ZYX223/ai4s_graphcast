"""Corresponding-update loss guard; no inference/checkpoint-output comparison."""

import argparse
import json
from pathlib import Path

from graphcast_bench.validation.loss_metrics import (
    attach_trajectory,
    compare_loss_rows,
    comparison_identity,
    operational_updates,
)


def compare_reports(candidate, reference, same_initial_state=False):
    for record in (candidate, reference):
        if record.get("status") != "completed" or not record.get("rows"):
            raise ValueError("Both reports must have completed updates")
    fixed = ["mode", "rollout", "tiny", "seed", "data_sha256", "optimizer", "model", "task"]
    if same_initial_state:
        fixed += ["initial_params_sha256", "initial_optimizer_sha256"]
    for key in fixed:
        if candidate[key] != reference[key]:
            raise ValueError("Changed invariant: " + key)
    result = compare_loss_rows(
        candidate["rows"],
        reference["rows"],
        expected_updates=operational_updates(reference["rollout"]),
        invariant_keys=("rollout", "starts", "batch_sha256"),
    )
    result["comparison_identity"] = comparison_identity(
        candidate, reference, ("mode", "tiny", "seed", "data_sha256", "optimizer", "model", "task")
    )
    result["same_initial_state"] = same_initial_state
    attach_trajectory(result, candidate, reference)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--same-initial-state", action="store_true")
    args = parser.parse_args()
    result = compare_reports(
        json.loads(args.candidate.read_text()),
        json.loads(args.reference.read_text()),
        args.same_initial_state,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["metric_passed"] else 1)


if __name__ == "__main__":
    main()
