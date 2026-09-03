#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_PLATFORMS = {"h100", "910c", "bw3000", "s5000", "c550", "810e"}
PATH_MATCH = {
    "strict_fp32": "strict_fp32",
    "tf32": "fast_fp32",
    "hf32": "fast_fp32",
}
CORE7_OPS = {
    "addmm",
    "cat",
    "native_layer_norm",
    "index_add_",
    "silu",
    "index",
    "add",
}


def positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def normalize_operator_takeover(record: dict[str, Any]) -> dict[str, Any]:
    platform = record["platform"]
    takeover = record.get("operator_takeover")
    if not isinstance(takeover, dict):
        raise ValueError(f"{platform}: operator_takeover must be an object")
    requested = takeover.get("requested")
    observed = takeover.get("observed_ops")
    missing = takeover.get("missing_ops")
    if not isinstance(requested, bool):
        raise ValueError(f"{platform}: operator_takeover.requested must be boolean")
    if not isinstance(observed, list) or not all(isinstance(op, str) for op in observed):
        raise ValueError(f"{platform}: observed_ops must be a string list")
    if not isinstance(missing, list) or not all(isinstance(op, str) for op in missing):
        raise ValueError(f"{platform}: missing_ops must be a string list")
    if len(set(observed)) != len(observed) or len(set(missing)) != len(missing):
        raise ValueError(f"{platform}: takeover operator lists must not contain duplicates")
    if requested:
        if set(observed).isdisjoint(missing) and set(observed) | set(missing) == CORE7_OPS:
            display = f"{len(observed)}/7"
        else:
            raise ValueError(
                f"{platform}: observed_ops and missing_ops must partition the seven core operators"
            )
    else:
        if observed or missing or takeover.get("status") != "not_applicable_native":
            raise ValueError(f"{platform}: Native baseline must not claim FlagGems takeover")
        display = "不适用（Native baseline）"
    return {
        **takeover,
        "observed_count": len(observed),
        "total_count": 7,
        "display": display,
    }


def calculate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "graphcast.path_matched_mfu_input.v1":
        raise ValueError("unexpected input schema")
    useful_flops = positive_number(payload.get("useful_flops_40"), "useful_flops_40")
    records = payload.get("platforms")
    if not isinstance(records, list):
        raise ValueError("platforms must be a list")
    names = [record.get("platform") for record in records if isinstance(record, dict)]
    if len(records) != 6 or set(names) != EXPECTED_PLATFORMS or len(set(names)) != 6:
        raise ValueError("platforms must contain exactly six unique platform records")

    output = []
    for record in records:
        operator_takeover = normalize_operator_takeover(record)
        precision = record.get("execution_precision")
        denominator_mode = record.get("denominator_mode")
        if PATH_MATCH.get(precision) != denominator_mode:
            raise ValueError(
                f"{record['platform']}: execution precision {precision!r} does not "
                f"match denominator mode {denominator_mode!r}"
            )
        samples = record.get("e2e_seconds")
        if not isinstance(samples, list) or len(samples) != 3:
            raise ValueError(f"{record['platform']}: exactly three E2E samples are required")
        e2e_values = [positive_number(value, "e2e_seconds") for value in samples]
        roof = positive_number(record.get("roof_median_tflops"), "roof_median_tflops")
        roof_ci = record.get("roof_ci95")
        if not isinstance(roof_ci, list) or len(roof_ci) != 2:
            raise ValueError(f"{record['platform']}: roof_ci95 must contain two values")
        roof_low, roof_high = [positive_number(value, "roof_ci95") for value in roof_ci]
        if not roof_low <= roof <= roof_high:
            raise ValueError(f"{record['platform']}: roof median is outside its CI")
        mean_e2e = statistics.fmean(e2e_values)
        effective_tflops = useful_flops / mean_e2e / 1.0e12
        output.append(
            {
                **record,
                "operator_takeover": operator_takeover,
                "mean_e2e_seconds": mean_e2e,
                "effective_tflops": effective_tflops,
                "mfu": effective_tflops / roof,
                "mfu_from_roof_ci95": [
                    effective_tflops / roof_high,
                    effective_tflops / roof_low,
                ],
            }
        )
    return {
        "schema": "graphcast.path_matched_mfu.v1",
        "metric": "empirical_path_matched_mfu",
        "is_vendor_theoretical_mfu": False,
        "useful_flops_40": useful_flops,
        "platforms": output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = calculate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
