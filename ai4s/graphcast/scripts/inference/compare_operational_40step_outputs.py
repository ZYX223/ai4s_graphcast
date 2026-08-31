#!/usr/bin/env python3
"""Compare final 40-step GraphCast outputs in the model normalization space."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


NORMALIZED_MAX_ABS_LIMIT = 1.0e-2
NORMALIZED_RMSE_LIMIT = 1.0e-3


def compare_outputs(
    native: xr.Dataset,
    candidate: xr.Dataset,
    *,
    mean: xr.Dataset,
    stddev: xr.Dataset,
) -> dict[str, Any]:
    """Return a unit-independent final-state correctness envelope."""

    if set(native.data_vars) != set(candidate.data_vars):
        raise ValueError("Native and candidate outputs have different variables")

    passed = True
    rows = []
    for name in sorted(native.data_vars):
        if name not in mean or name not in stddev:
            raise ValueError(f"normalization stats are missing {name!r}")
        reference = native[name]
        actual = candidate[name]
        if reference.dims != actual.dims or reference.shape != actual.shape:
            raise ValueError(f"Native and candidate layouts differ for {name!r}")

        reference_values = np.asarray(reference.values, dtype=np.float64)
        actual_values = np.asarray(actual.values, dtype=np.float64)
        physical_difference = np.abs(actual_values - reference_values)

        normalized_reference = np.asarray(
            ((reference - mean[name]) / stddev[name]).values,
            dtype=np.float64,
        )
        normalized_actual = np.asarray(
            ((actual - mean[name]) / stddev[name]).values,
            dtype=np.float64,
        )
        normalized_difference = np.abs(normalized_actual - normalized_reference)
        finite = bool(
            np.isfinite(normalized_reference).all()
            and np.isfinite(normalized_actual).all()
        )
        normalized_max_abs = float(np.max(normalized_difference))
        normalized_rmse = float(
            np.sqrt(np.mean(np.square(normalized_difference)))
        )
        within_envelope = bool(
            finite
            and normalized_max_abs <= NORMALIZED_MAX_ABS_LIMIT
            and normalized_rmse <= NORMALIZED_RMSE_LIMIT
        )
        passed = passed and within_envelope
        rows.append(
            {
                "variable": name,
                "finite": finite,
                "within_envelope": within_envelope,
                "physical_max_abs": float(np.max(physical_difference)),
                "normalized_max_abs": normalized_max_abs,
                "normalized_rmse": normalized_rmse,
                "normalized_p99_abs": float(
                    np.quantile(normalized_difference, 0.99)
                ),
                "normalized_p999_abs": float(
                    np.quantile(normalized_difference, 0.999)
                ),
            }
        )

    return {
        "schema": "graphcast.round1_correctness.v2",
        "status": "passed" if passed else "failed",
        "comparison_space": "(physical_value - mean_by_level) / stddev_by_level",
        "limits": {
            "normalized_max_abs": NORMALIZED_MAX_ABS_LIMIT,
            "normalized_rmse": NORMALIZED_RMSE_LIMIT,
        },
        "variables": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    native = xr.open_dataset(args.native).load()
    candidate = xr.open_dataset(args.candidate).load()
    mean = xr.open_dataset(args.stats_dir / "mean_by_level.nc").load()
    stddev = xr.open_dataset(args.stats_dir / "stddev_by_level.nc").load()
    result = compare_outputs(native, candidate, mean=mean, stddev=stddev)
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
