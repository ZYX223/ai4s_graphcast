#!/usr/bin/env python3
"""Evaluate GraphCast forecasts against real weather targets."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def _weighted_rmse(target: xr.DataArray, forecast: xr.DataArray) -> float | None:
    if "lat" not in target.dims:
        raise ValueError(f"forecast variable {target.name!r} has no latitude dimension")
    target_values = np.asarray(target.values, dtype=np.float64)
    forecast_values = np.asarray(forecast.values, dtype=np.float64)
    if not np.isfinite(target_values).all() or not np.isfinite(forecast_values).all():
        return None
    latitudes = np.asarray(target.coords["lat"].values, dtype=np.float64)
    weights = np.cos(np.deg2rad(latitudes))
    shape = [1] * target_values.ndim
    shape[target.get_axis_num("lat")] = weights.size
    broadcast_weights = weights.reshape(shape)
    squared_error = np.square(forecast_values - target_values)
    denominator = float(np.sum(np.broadcast_to(broadcast_weights, squared_error.shape)))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("latitude weights must have a finite positive sum")
    return float(np.sqrt(np.sum(squared_error * broadcast_weights) / denominator))


def _coordinate_value(value: Any) -> Any:
    scalar = np.asarray(value).item()
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def evaluate_forecast(target: xr.Dataset, forecast: xr.Dataset) -> dict[str, Any]:
    """Return latitude-weighted RMSE against real targets."""

    if set(target.data_vars) != set(forecast.data_vars):
        raise ValueError("target and forecast variables differ")
    rows: list[dict[str, Any]] = []
    all_finite = True
    for name in sorted(target.data_vars):
        expected = target[name]
        actual = forecast[name]
        if expected.dims != actual.dims or expected.shape != actual.shape:
            raise ValueError(f"target and forecast layouts differ for {name!r}")
        for dimension in expected.dims:
            if dimension in expected.coords and not expected.coords[dimension].equals(
                actual.coords[dimension]
            ):
                raise ValueError(f"target and forecast coordinates differ for {name!r}")
        rmse = _weighted_rmse(expected, actual)
        finite = rmse is not None
        all_finite = all_finite and finite
        levels: list[dict[str, Any]] | None = None
        if "level" in expected.dims:
            levels = []
            for level in expected.coords["level"].values:
                level_target = expected.sel(level=level)
                level_forecast = actual.sel(level=level)
                levels.append(
                    {
                        "level": _coordinate_value(level),
                        "latitude_weighted_rmse": _weighted_rmse(
                            level_target, level_forecast
                        ),
                    }
                )
        rows.append(
            {
                "variable": name,
                "finite": finite,
                "latitude_weighted_rmse": rmse,
                "levels": levels,
            }
        )
    return {
        "schema": "graphcast.forecast_skill.v1",
        "status": "measured" if all_finite else "failed_nonfinite",
        "metric": "latitude_weighted_rmse",
        "latitude_weight": "cos(latitude)",
        "variables": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = xr.open_dataset(args.target).load()
    forecast = xr.open_dataset(args.forecast).load()
    result = evaluate_forecast(target, forecast)
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
