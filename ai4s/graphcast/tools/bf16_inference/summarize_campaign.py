#!/usr/bin/env python3
"""Validate BF16 campaign evidence and calculate path-matched MFU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path_value: object, field: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(path_value, str):
        raise ValueError(f"{field} must be a path string")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return path, payload


def _validate_diagnostic(path_value: object, field: str) -> dict[str, Any]:
    path, payload = _read_json(path_value, field)
    if payload.get("status") != "measured":
        raise ValueError(f"{field} is not a completed measurement")
    parity = payload.get("implementation_parity")
    worst = payload.get("worst")
    if not isinstance(parity, dict) or parity.get("finite") is not True:
        raise ValueError(f"{field} did not remain finite")
    if not isinstance(worst, dict):
        raise ValueError(f"{field} has no worst-case metrics")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "lead_count": int(_positive(payload.get("lead_count"), f"{field}.lead_count")),
        "relative_l2": parity.get("relative_l2"),
        "normalized_max_abs": worst.get("normalized_max_abs"),
        "normalized_rmse": worst.get("normalized_rmse"),
    }


def _summarize_mode(
    mode: str,
    paths: object,
    *,
    platform: str,
    expected_profile: str,
    roof_tflops: float,
    expected_coverage: str | None,
    core_coverage: str | None,
) -> tuple[dict[str, Any] | None, float | None]:
    if not isinstance(paths, list):
        raise ValueError(f"{mode}_reports must be a list")
    if not paths:
        return None, None
    if len(paths) != 3:
        raise ValueError(f"{mode} requires exactly three fresh performance reports")
    samples: list[float] = []
    useful_flops: float | None = None
    evidence: list[dict[str, str]] = []
    for index, value in enumerate(paths, start=1):
        path, report = _read_json(value, f"{mode}_reports[{index}]")
        measurement = report.get("measurement")
        precision = report.get("precision")
        if report.get("status") != "passed" or not isinstance(measurement, dict):
            raise ValueError(f"{path}: performance report did not pass")
        if measurement.get("performance_eligible") is not True:
            raise ValueError(f"{path}: run is not performance eligible")
        if measurement.get("finite") is not True:
            raise ValueError(f"{path}: run output is not finite")
        if not isinstance(precision, dict) or precision.get("requested") != "bf16":
            raise ValueError(f"{path}: report is not BF16")
        if precision.get("effective") != "mixed_precision":
            raise ValueError(f"{path}: BF16 mixed precision was not verified")
        workload = report.get("workload")
        profile = (
            workload.get("model_execution_profile")
            if isinstance(workload, dict)
            else None
        )
        if (
            not isinstance(profile, dict)
            or profile.get("name") != expected_profile
            or profile.get("platform") != platform
        ):
            raise ValueError(
                f"{path}: model execution profile does not match "
                f"{platform}/{expected_profile}"
            )
        sample_flops = _positive(measurement.get("useful_flops"), "useful_flops")
        if useful_flops is None:
            useful_flops = sample_flops
        elif sample_flops != useful_flops:
            raise ValueError(f"{path}: useful FLOPs differ across repetitions")
        samples.append(_positive(measurement.get("e2e_seconds"), "e2e_seconds"))
        evidence.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        if mode == "flaggems":
            runtime = report.get("runtime")
            dispatch = runtime.get("flaggems_dispatch") if isinstance(runtime, dict) else None
            coverage = dispatch.get("active_coverage") if isinstance(dispatch, dict) else None
            if (
                not isinstance(dispatch, dict)
                or dispatch.get("status") != "passed"
                or dispatch.get("missing_ops") != []
                or coverage != expected_coverage
            ):
                raise ValueError(
                    f"{path}: FlagGems coverage {coverage!r} does not match "
                    f"{expected_coverage!r}"
                )
    assert useful_flops is not None
    median_seconds = statistics.median(samples)
    effective_tflops = useful_flops / median_seconds / 1.0e12
    return (
        {
            "mode": mode,
            "samples_e2e_seconds": samples,
            "median_e2e_seconds": median_seconds,
            "effective_tflops": effective_tflops,
            "bf16_roof_tflops": roof_tflops,
            "mfu": effective_tflops / roof_tflops,
            "operator_coverage": core_coverage if mode == "flaggems" else None,
            "dispatch_coverage": expected_coverage if mode == "flaggems" else None,
            "evidence": evidence,
        },
        useful_flops,
    )


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "graphcast.bf16_inference_campaign_input.v1":
        raise ValueError("unexpected input schema")
    platform_records = payload.get("platforms")
    if not isinstance(platform_records, list) or not platform_records:
        raise ValueError("platforms must be a non-empty list")
    names: set[str] = set()
    output: list[dict[str, Any]] = []
    campaign_flops: float | None = None
    for record in platform_records:
        if not isinstance(record, dict) or not isinstance(record.get("platform"), str):
            raise ValueError("each platform must be an object with a name")
        platform = record["platform"]
        if platform in names:
            raise ValueError(f"duplicate platform: {platform}")
        names.add(platform)
        roof = _positive(record.get("bf16_roof_tflops"), "bf16_roof_tflops")
        expected_profile = record.get("expected_model_execution_profile")
        if not isinstance(expected_profile, str) or not expected_profile:
            raise ValueError("expected_model_execution_profile must be a non-empty string")
        expected_coverage = record.get("expected_flaggems_coverage")
        if expected_coverage is not None and not isinstance(expected_coverage, str):
            raise ValueError("expected_flaggems_coverage must be a string or null")
        core_coverage = record.get("core_flaggems_coverage", expected_coverage)
        if core_coverage is not None and not isinstance(core_coverage, str):
            raise ValueError("core_flaggems_coverage must be a string or null")
        diagnostics = record.get("diagnostic_reports")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"{platform}: diagnostic_reports must be an object")
        diagnostic_output = {
            mode: _validate_diagnostic(path, f"{platform}.{mode}_diagnostic")
            for mode, path in diagnostics.items()
        }
        modes: list[dict[str, Any]] = []
        for mode in ("native", "flaggems"):
            result, useful_flops = _summarize_mode(
                mode,
                record.get(f"{mode}_reports", []),
                platform=platform,
                expected_profile=expected_profile,
                roof_tflops=roof,
                expected_coverage=expected_coverage,
                core_coverage=core_coverage,
            )
            if result is not None:
                modes.append(result)
                if campaign_flops is None:
                    campaign_flops = useful_flops
                elif useful_flops != campaign_flops:
                    raise ValueError("useful FLOPs differ across platforms")
        output.append(
            {
                "platform": platform,
                "model_execution_profile": expected_profile,
                "bf16_roof_tflops": roof,
                "diagnostics": diagnostic_output,
                "modes": modes,
            }
        )
    if campaign_flops is None:
        raise ValueError("campaign contains no performance reports")
    return {
        "schema": "graphcast.bf16_inference_campaign_summary.v1",
        "metric": "empirical_path_matched_bf16_mfu",
        "is_vendor_theoretical_mfu": False,
        "useful_flops": campaign_flops,
        "platforms": output,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
