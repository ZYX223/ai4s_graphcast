#!/usr/bin/env python3
"""Compare two GraphCast trajectories without pretending they are weather truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import xarray as xr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_terminal(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stddev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-atol", type=float, default=1.0e-5)
    parser.add_argument("--raw-rtol", type=float, default=1.0e-5)
    args = parser.parse_args(argv)
    paths = (args.reference, args.candidate, args.stddev)
    if len({path.resolve() for path in (*paths, args.output)}) != 4:
        parser.error("all input and output paths must be distinct")
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.project_root, args.project_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from scripts.inference.evaluate_forecast_skill import evaluate_paired_forecasts

    identities = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in (
            ("reference", args.reference),
            ("candidate", args.candidate),
            ("stddev", args.stddev),
        )
    }
    with ExitStack() as datasets:
        reference = datasets.enter_context(xr.open_dataset(args.reference))
        candidate = datasets.enter_context(xr.open_dataset(args.candidate))
        stddev = datasets.enter_context(xr.open_dataset(args.stddev))
        full = evaluate_paired_forecasts(
            reference,
            reference,
            candidate,
            stddev=stddev,
            raw_atol=args.raw_atol,
            raw_rtol=args.raw_rtol,
        )
    fields = (
        "variable",
        "level",
        "lead_hours",
        "finite",
        "normalized_max_abs",
        "normalized_rmse",
        "parity_max_abs",
        "parity_mean_abs",
        "parity_rmse",
        "parity_relative_l2",
        "parity_allclose",
    )
    records = [{field: record[field] for field in fields} for record in full["records"]]
    normalized_rmse = [
        record["normalized_rmse"]
        for record in records
        if record["normalized_rmse"] is not None
    ]
    payload: dict[str, object] = {
        "schema": "graphcast.implementation_trajectory_diagnostics.v1",
        "status": full["status"],
        "methodology": {
            "comparison": "candidate_minus_reference_at_matching_autoregressive_lead",
            "normalization": "graphcast_stddev_by_level",
            "weather_skill": "not_measured_without_time_matched_truth_and_climatology",
        },
        "inputs": identities,
        "lead_count": full["lead_count"],
        "records": records,
        "implementation_parity": full["implementation_parity"],
        "worst": {
            "normalized_max_abs": full["worst"]["normalized_max_abs"],
            "normalized_rmse": max(normalized_rmse, default=None),
        },
    }
    _write_terminal(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True), flush=True)
    return 0 if payload["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
