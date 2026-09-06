#!/usr/bin/env python3
"""Offline project 5% prediction acceptance against the handoff NPZ baselines.

The metric and two-reference policy come from checkpoint_prediction_validation/
diagnostic_package (2026-06-10). This is prediction consistency, not weather skill.
No Torch/FlagGems import, device execution, or performance timing is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[2]
if str(GRAPHCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_ROOT))

from scripts.inference.evaluate_forecast_skill import (  # noqa: E402
    _canonical_forecast,
    _initialization_times,
    _validate_forecast_layout,
    _validate_time_semantics,
)

THRESHOLD_PERCENT = 5.0
MAX_REPEATABILITY_PERCENT = 0.1
REFERENCE_KEYS = ("jax_fp32_no_bf16", "pytorch_gpu")
VARIABLES = {
    "u_component_of_wind": "U", "v_component_of_wind": "V", "geopotential": "P",
    "temperature": "T", "specific_humidity": "Q", "vertical_velocity": "W",
}
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CORE_OPS = ("addmm", "cat", "native_layer_norm", "index_add_", "silu", "index", "add")
SOURCE_KEYS = ("weights_sha256", "inputs_sha256", "rollout_forcings_sha256", "target_template_sha256")
STATS_FILES = ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_digest(actual: str, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
        raise ValueError(f"{label} SHA-256 does not match its bound evidence")


def _validate_provenance(metadata: dict, report: dict) -> bool:
    if metadata.get("schema") not in ("graphcast.fixed_baseline.v1", "graphcast.fixed_baseline.v2"):
        raise ValueError("unsupported baseline metadata schema")
    status = metadata.get("provenance_status")
    if status not in ("verified", "unverified"):
        raise ValueError("baseline provenance_status must be verified or unverified")
    if status == "unverified":
        return False
    if not metadata.get("provenance_evidence") or not metadata.get("initialization_time"):
        raise ValueError("verified baselines require provenance evidence and initialization time")
    expected, actual = metadata.get("source", {}), report.get("source", {})
    for name in SOURCE_KEYS:
        _check_digest(actual.get(name), expected.get(name), name)
    if metadata["schema"] == "graphcast.fixed_baseline.v2":
        _check_digest(actual.get("config_sha256"), expected.get("config_sha256"), "config")
    for name in STATS_FILES:
        _check_digest(
            actual.get("normalization_stats", {}).get(name, {}).get("sha256"),
            expected.get("normalization_stats", {}).get(name, {}).get("sha256"),
            name,
        )
    return True


def _reference_qualified(metadata: dict) -> bool:
    # Legacy provenance behavior is preserved; newly rebuilt references also
    # require an independent numerical qualification, not a renamed prediction.
    if metadata["schema"] == "graphcast.fixed_baseline.v1":
        return True
    qualification = metadata.get("qualification", {})
    if qualification.get("status") != "qualified":
        return False
    _check_digest(qualification.get("baseline_sha256"), metadata.get("baseline_sha256"), "qualification baseline")
    _check_digest(qualification.get("case_sha256"), metadata.get("case_sha256"), "qualification case")
    cross = qualification.get("cross_reference", {})
    repeat_limit = qualification.get("repeatability_percent", 0)
    repeats = qualification.get("repeatability", [])
    def within(row, limit):
        return all(isinstance(row.get(key), (int, float)) and np.isfinite(row[key])
                   and 0 <= row[key] < limit for key in ("overall_mean_percent", "z500_percent"))
    if (not qualification.get("cross_reference_passed") or not qualification.get("repeatability_passed")
            or not within(cross, THRESHOLD_PERCENT) or not 0 < repeat_limit <= MAX_REPEATABILITY_PERCENT
            or len(repeats) < 4 or not all(within(row, repeat_limit) for row in repeats)):
        raise ValueError("qualified reference is missing passing numerical evidence")
    return True


def _select_prediction(
    dataset: xr.Dataset, baseline: Any, metadata: dict, verified: bool, horizon: int
) -> xr.Dataset:
    candidate = _canonical_forecast(dataset)
    _validate_forecast_layout(candidate, label="candidate")
    _validate_time_semantics(candidate, label="candidate")
    if candidate.sizes["init_time"] != 1:
        raise ValueError("one fixed baseline case requires exactly one initialization")
    lead_hours = float(np.asarray(baseline["lead_hour"]).item())
    if not np.isfinite(lead_hours) or lead_hours <= 0 or metadata.get("lead_hours") != lead_hours:
        raise ValueError("invalid or inconsistent baseline lead_hours")
    leads = np.asarray(candidate["lead_time"].values) / np.timedelta64(1, "h")
    if not np.isfinite(leads).all() or np.any(leads <= 0) or np.max(leads) != horizon:
        raise ValueError("prediction lead times do not match the recorded rollout horizon")
    matches = np.flatnonzero(leads == lead_hours)
    if matches.size != 1 or np.unique(leads).size != leads.size:
        raise ValueError(f"candidate must contain exactly one {lead_hours:g}h lead")
    if verified:
        expected_init = np.datetime64(metadata["initialization_time"], "ns")
        if np.isnat(expected_init) or np.asarray(_initialization_times(candidate))[0] != expected_init:
            raise ValueError("candidate initialization differs from the fixed baseline case")
    for original, canonical in (("level", "level"), ("lat", "latitude"), ("lon", "longitude")):
        if canonical not in candidate.coords:
            raise ValueError(f"candidate is missing {canonical}")
        values, expected = np.asarray(candidate[canonical]), np.asarray(baseline[original])
        if (values.shape != expected.shape or not np.isfinite(expected).all()
                or np.unique(expected).size != expected.size
                or not np.allclose(values, expected, atol=1e-6, rtol=0)):
            raise ValueError(f"candidate {canonical} differs from baseline")
    for name in VARIABLES:
        if name not in candidate or "level" not in candidate[name].dims:
            raise ValueError(f"candidate is missing required pressure variable {name}")
        expected_units = metadata.get("units", {}).get(name)
        if verified and not expected_units:
            raise ValueError(f"verified baseline is missing units for {name}")
        if expected_units and candidate[name].attrs.get("units") != expected_units:
            raise ValueError(f"candidate units differ for {name}")
    return candidate.isel(init_time=0, lead_time=int(matches[0]), drop=True)


def _compare(candidate: xr.Dataset, baseline: Any) -> list[dict]:
    comparisons = []
    titles = dict(zip(baseline["baseline_keys"].tolist(), baseline["baseline_titles"].tolist(), strict=True))
    for key in REFERENCE_KEYS:
        rows = []
        for name, label in VARIABLES.items():
            # Load one variable, not both complete 550 MB reference volumes or a full rollout.
            reference = np.asarray(baseline[f"{key}__{name}"], dtype=np.float64)
            values = np.asarray(candidate[name].transpose("level", "latitude", "longitude"), dtype=np.float64)
            if reference.shape != values.shape:
                raise ValueError(f"baseline shape differs for {key}/{name}")
            if not np.isfinite(reference).all():
                raise ValueError(f"non-finite reference: {key}/{name}")
            for index, level in enumerate(LEVELS):
                ref, diff = reference[index], values[index] - reference[index]
                denominator = float(np.mean(np.abs(ref)))
                if denominator == 0:
                    raise ValueError(f"zero reference denominator: {key}/{name}/{level}")
                mae = float(np.mean(np.abs(diff)))
                percent = mae / denominator * 100
                row = {
                    "variable": name, "label": label, "level_hpa": level,
                    "relative_error_percent": percent, "mae": mae,
                    "rmse": float(np.sqrt(np.mean(diff * diff))),
                    "max_abs": float(np.max(np.abs(diff))), "bias": float(np.mean(diff)),
                    "reference_mean_abs": denominator,
                }
                if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError(f"non-finite diagnostic arithmetic: {key}/{name}/{level}")
                rows.append(row)
        overall = float(np.mean([row["relative_error_percent"] for row in rows]))
        z500 = next(row for row in rows if row["variable"] == "geopotential" and row["level_hpa"] == 500)
        comparisons.append({
            "reference": key, "title": titles[key], "overall_mean_percent": overall,
            "z500_percent": z500["relative_error_percent"],
            "z500_mae_m": z500["mae"] / 9.80665, "z500_rmse_m": z500["rmse"] / 9.80665,
            "metrics_within_limits": overall < THRESHOLD_PERCENT and z500["relative_error_percent"] < THRESHOLD_PERCENT,
            "worst_layer": max(rows, key=lambda row: row["relative_error_percent"]), "layers": rows,
        })
    return comparisons


def diagnose_run(run_dir: Path, baseline_path: Path, metadata: dict) -> dict:
    """Evaluate one immutable runner result; unknown provenance never passes."""
    if not isinstance(metadata, dict):
        raise ValueError("baseline metadata must be a JSON object")
    prediction_path, report_path = run_dir / "prediction.nc", run_dir / "report.json"
    baseline_digest = sha256(baseline_path)
    _check_digest(baseline_digest, metadata.get("baseline_sha256"), "baseline")
    report_digest = sha256(report_path)
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or report.get("schema") != "graphcast.operational_rollout.v2":
        raise ValueError("expected an operational rollout v2 report")
    if report.get("status") not in ("passed", "failed_nonfinite"):
        raise ValueError("runner did not complete prediction publication")
    prediction_digest = sha256(prediction_path)
    _check_digest(prediction_digest, report.get("artifacts", {}).get("prediction", {}).get("sha256"), "prediction")
    verified = _validate_provenance(metadata, report)
    qualified = _reference_qualified(metadata)
    runtime = report.get("runtime", {})
    counts = runtime.get("flaggems_dispatch", {}).get("counts")
    coverage = None
    if runtime.get("flagos_mode") == "flaggems":
        coverage = (f"{sum(isinstance(counts.get(op), int) and counts[op] > 0 for op in CORE_OPS)}/7"
                    if isinstance(counts, dict) else "unverified")
    result = {
        "run_dir": str(run_dir.resolve()), "platform": runtime.get("chip"),
        "mode": runtime.get("flagos_mode"), "precision": report.get("precision"),
        "core_operator_coverage": coverage, "runtime": runtime, "workload": report.get("workload"),
        "source": report.get("source"), "provenance_verified": verified,
        "reference_qualified": qualified if metadata["schema"] == "graphcast.fixed_baseline.v2" else None,
        "inputs": {"prediction_sha256": prediction_digest, "run_report_sha256": report_digest,
                   "baseline_sha256": baseline_digest},
        "acceptance_passed": False, "comparisons": [],
    }
    with np.load(baseline_path, allow_pickle=False) as baseline, xr.open_dataset(
        prediction_path, decode_timedelta=True
    ) as dataset:
        if (sorted(baseline["baseline_keys"].tolist()) != sorted(REFERENCE_KEYS)
                or sorted(baseline["variable_names"].tolist()) != sorted(VARIABLES)
                or baseline["level"].tolist() != LEVELS):
            raise ValueError("baseline must contain the two project references and all 6 variables / 13 levels")
        workload = report.get("workload", {})
        if not all(type(workload.get(key)) is int and workload[key] > 0 for key in ("steps", "step_hours")):
            raise ValueError("run steps and step_hours must be positive integers")
        horizon = workload["steps"] * workload["step_hours"]
        if horizon < metadata.get("lead_hours", 0):
            raise ValueError("run horizon is shorter than the reference lead")
        candidate = _select_prediction(dataset, baseline, metadata, verified, horizon)
        if (report["status"] == "failed_nonfinite"
                or report.get("measurement", {}).get("finite") is False
                or not all(np.isfinite(candidate[name]).all().item() for name in VARIABLES)):
            result["status"] = "failed_nonfinite"
        else:
            result["comparisons"] = _compare(candidate, baseline)
            within_limits = all(row["metrics_within_limits"] for row in result["comparisons"])
            result["status"] = ("unverified_reference" if not verified else
                                "unqualified_reference" if not qualified else
                                "passed" if within_limits else "failed")
            result["acceptance_passed"] = verified and qualified and within_limits
    # Never publish metrics from files that changed during evaluation.
    for path, digest in ((prediction_path, prediction_digest), (report_path, report_digest),
                         (baseline_path, baseline_digest)):
        _check_digest(sha256(path), digest, str(path))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        metadata_digest = sha256(args.baseline_metadata)
        metadata = json.loads(args.baseline_metadata.read_text())
        baseline_digest = sha256(args.baseline)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print(f"diagnostic input/output error: {error}", file=sys.stderr)
        return 2
    runs = []
    for run_dir in args.run_dir:
        try:
            result = diagnose_run(run_dir, args.baseline, metadata)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result = {"run_dir": str(run_dir.resolve()), "status": "invalid_inputs",
                      "acceptance_passed": False, "error": str(error), "comparisons": []}
        runs.append(result)
        print(f"{run_dir.name}: {result['status']}; core coverage={result.get('core_operator_coverage')}")
    details = args.output_dir / "details.csv"
    columns = ["run_dir", "reference", "variable", "label", "level_hpa", "relative_error_percent",
               "mae", "rmse", "max_abs", "bias", "reference_mean_abs"]
    with details.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in runs:
            for comparison in result["comparisons"]:
                for row in comparison["layers"]:
                    writer.writerow({"run_dir": result["run_dir"], "reference": comparison["reference"], **row})
    if sha256(args.baseline_metadata) != metadata_digest or sha256(args.baseline) != baseline_digest:
        print("baseline evidence changed during campaign; no terminal summary published", file=sys.stderr)
        return 2
    summary = {
        "schema": "graphcast.fixed_baseline_diagnostics.v1",
        "methodology": {"threshold_percent": THRESHOLD_PERCENT, "comparison": "strict_less_than",
                        "metric": "mean(abs(candidate-reference))/mean(abs(reference))*100",
                        "aggregation": "equal_mean_of_78_variable_level_percentages",
                        "gates": "overall_and_z500_against_both_references", "spatial_weight": "uniform",
                        "weather_skill": "not_evaluated", "affects_e2e_timing": False},
        "baseline": {"path": str(args.baseline.resolve()), "sha256": baseline_digest,
                     "metadata_sha256": metadata_digest, "metadata": metadata},
        "implementation_sha256": sha256(Path(__file__)),
        "details_sha256": sha256(details), "runs": runs,
    }
    temporary = args.output_dir / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "summary.json")
    if any(row["status"] in ("invalid_inputs", "unverified_reference", "unqualified_reference") for row in runs):
        return 2
    return 0 if all(row["acceptance_passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
