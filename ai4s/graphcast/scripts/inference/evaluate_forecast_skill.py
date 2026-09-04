#!/usr/bin/env python3
"""Evaluate GraphCast forecasts against real weather targets."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def _canonical_spatial_names(dataset: xr.Dataset) -> xr.Dataset:
    renames: dict[str, str] = {}
    if "lat" in dataset.dims or "lat" in dataset.coords:
        renames["lat"] = "latitude"
    if "lon" in dataset.dims or "lon" in dataset.coords:
        renames["lon"] = "longitude"
    return dataset.rename(renames) if renames else dataset


def _canonical_forecast(dataset: xr.Dataset) -> xr.Dataset:
    dataset = _canonical_spatial_names(dataset)
    renames: dict[str, str] = {}
    if "time" in dataset.dims and "lead_time" not in dataset.dims:
        renames["time"] = "lead_time"
    if "batch" in dataset.dims and "init_time" not in dataset.dims:
        renames["batch"] = "init_time"
    dataset = dataset.rename(renames) if renames else dataset
    if "datetime" in dataset.coords:
        if "valid_time" in dataset.coords and not dataset["datetime"].equals(
            dataset["valid_time"]
        ):
            raise ValueError("datetime and valid_time coordinates differ")
        if "valid_time" in dataset.coords:
            dataset = dataset.drop_vars("datetime")
        else:
            dataset = dataset.rename({"datetime": "valid_time"})
    if "valid_time" in dataset.coords and "lead_time" in dataset.coords:
        leads = dataset["lead_time"]
        valid_time = dataset["valid_time"]
        if np.issubdtype(leads.dtype, np.timedelta64) and valid_time.dims == (
            "init_time",
            "lead_time",
        ):
            references = valid_time - leads
            reference = references.isel(lead_time=0, drop=True)
            expected, _ = xr.broadcast(reference, leads)
            if not np.array_equal(
                np.asarray(references.values), np.asarray(expected.values)
            ):
                raise ValueError(
                    "valid_time must contain regular leads from one initialization"
                )
            canonical_reference = xr.DataArray(
                np.asarray(reference.values),
                dims=("init_time",),
                coords={"init_time": dataset["init_time"]},
            )
            if "forecast_reference_time" in dataset.coords and not np.array_equal(
                np.asarray(dataset["forecast_reference_time"].values),
                np.asarray(canonical_reference.values),
            ):
                raise ValueError(
                    "forecast_reference_time conflicts with valid_time and lead_time"
                )
            dataset = dataset.assign_coords(
                forecast_reference_time=canonical_reference
            )
    return dataset


def _validate_forecast_layout(dataset: xr.Dataset, *, label: str) -> None:
    if not dataset.data_vars:
        raise ValueError(f"{label} requires a non-empty variable set")
    for dimension in ("init_time", "lead_time", "latitude", "longitude"):
        if dimension not in dataset.dims or int(dataset.sizes[dimension]) <= 0:
            raise ValueError(f"{label} requires a non-empty {dimension} dimension")
        if dimension not in dataset.coords:
            raise ValueError(f"{label} requires a {dimension} coordinate")
    allowed = {"init_time", "lead_time", "level", "latitude", "longitude"}
    required = {"init_time", "lead_time", "latitude", "longitude"}
    for name, variable in dataset.data_vars.items():
        dimensions = set(variable.dims)
        if not required.issubset(dimensions):
            missing = ", ".join(sorted(required - dimensions))
            raise ValueError(f"{label} variable {name!r} is missing dimensions: {missing}")
        if not dimensions.issubset(allowed):
            extra = ", ".join(sorted(dimensions - allowed))
            raise ValueError(f"{label} variable {name!r} has unexpected dimensions: {extra}")
        if "level" in dimensions and int(variable.sizes["level"]) <= 0:
            raise ValueError(f"{label} variable {name!r} requires a non-empty level dimension")


def _initialization_times(dataset: xr.Dataset) -> xr.DataArray:
    if np.issubdtype(dataset["init_time"].dtype, np.datetime64):
        if "forecast_reference_time" in dataset.coords:
            reference = dataset["forecast_reference_time"]
            if reference.dims != ("init_time",) or not np.array_equal(
                np.asarray(reference.values), np.asarray(dataset["init_time"].values)
            ):
                raise ValueError(
                    "datetime init_time conflicts with valid_time minus lead_time"
                )
        return dataset["init_time"]
    if "forecast_reference_time" in dataset.coords:
        reference = dataset["forecast_reference_time"]
        if reference.dims != ("init_time",) or not np.issubdtype(
            reference.dtype, np.datetime64
        ):
            raise ValueError(
                "forecast_reference_time must be datetime64 over init_time"
            )
        return reference
    raise ValueError("forecast diagnostics require auditable initialization times")


def _validate_time_semantics(dataset: xr.Dataset, *, label: str) -> None:
    leads = dataset["lead_time"]
    if leads.dims != ("lead_time",) or not np.issubdtype(
        leads.dtype, np.timedelta64
    ):
        raise ValueError(f"{label} lead_time must be a one-dimensional duration")
    initialization = _initialization_times(dataset)
    expected, _ = xr.broadcast(initialization, leads)
    expected = expected + leads
    if "valid_time" in dataset.coords:
        valid_time = dataset["valid_time"]
        if valid_time.dims != expected.dims or not np.array_equal(
            np.asarray(valid_time.values), np.asarray(expected.values)
        ):
            raise ValueError(
                f"{label} valid_time must equal init_time plus lead_time"
            )


def _latitude_cell_weights(latitude: xr.DataArray) -> xr.DataArray:
    """Return WeatherBench-style spherical cell-area weights."""

    points = np.deg2rad(np.asarray(latitude.values, dtype=np.float64))
    if points.ndim != 1 or points.size == 0:
        raise ValueError("latitude must be a non-empty one-dimensional coordinate")
    increasing = points.size == 1 or bool(np.all(np.diff(points) > 0))
    decreasing = points.size > 1 and bool(np.all(np.diff(points) < 0))
    if not increasing and not decreasing:
        raise ValueError("latitude coordinate must be strictly monotonic")
    ordered = points if increasing else points[::-1]
    bounds = np.concatenate(
        (
            np.asarray([-np.pi / 2], dtype=np.float64),
            (ordered[:-1] + ordered[1:]) / 2,
            np.asarray([np.pi / 2], dtype=np.float64),
        )
    )
    weights = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    if decreasing:
        weights = weights[::-1]
    return xr.DataArray(weights, dims=("latitude",), coords={"latitude": latitude})


def _spatial_mean(array: xr.DataArray) -> xr.DataArray:
    for dimension in ("latitude", "longitude"):
        if dimension not in array.dims:
            raise ValueError(f"forecast variable {array.name!r} has no {dimension} dimension")
    return array.weighted(_latitude_cell_weights(array["latitude"])).mean(
        ("latitude", "longitude"),
        skipna=False,
    )


def _mean_scalar(array: xr.DataArray) -> float:
    value = float(array.mean(skipna=False).item())
    return value


def _forecast_metrics(
    truth: xr.DataArray, forecast: xr.DataArray
) -> dict[str, float | None]:
    if not np.isfinite(truth.values).all() or not np.isfinite(forecast.values).all():
        return {name: None for name in ("mse", "rmse", "mae", "bias")}
    difference = forecast.astype(np.float64) - truth.astype(np.float64)
    mse = _mean_scalar(_spatial_mean(difference**2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": _mean_scalar(_spatial_mean(abs(difference))),
        "bias": _mean_scalar(_spatial_mean(difference)),
    }


def _coordinate_scalar(value: Any) -> Any:
    scalar = np.asarray(value).item()
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _lead_hours(value: Any) -> float:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.timedelta64):
        return float(array / np.timedelta64(1, "h"))
    return float(array.item())


def _validate_matching_forecasts(
    truth: xr.Dataset,
    native: xr.Dataset,
    candidate: xr.Dataset,
) -> None:
    if set(truth.data_vars) != set(native.data_vars) or set(truth.data_vars) != set(
        candidate.data_vars
    ):
        raise ValueError("truth, Native, and candidate variables differ")
    for name in sorted(truth.data_vars):
        expected = truth[name]
        for label, actual in (("Native", native[name]), ("candidate", candidate[name])):
            if expected.dims != actual.dims or expected.shape != actual.shape:
                raise ValueError(f"truth and {label} layouts differ for {name!r}")
            if expected.attrs.get("units") != actual.attrs.get("units"):
                raise ValueError(f"truth and {label} units differ for {name!r}")
            expected_coordinates = set(expected.coords)
            actual_coordinates = set(actual.coords)
            if expected_coordinates != actual_coordinates:
                raise ValueError(
                    f"truth and {label} coordinate sets differ for {name!r}"
                )
            for coordinate in sorted(expected_coordinates):
                if not expected[coordinate].equals(actual[coordinate]):
                    raise ValueError(
                        f"truth and {label} coordinates differ for {name!r}: {coordinate}"
                    )


def _valid_time(truth: xr.DataArray, lead_value: Any) -> xr.DataArray:
    if "valid_time" in truth.coords:
        return truth["valid_time"]
    if "init_time" in truth.coords and np.issubdtype(
        truth["init_time"].dtype, np.datetime64
    ):
        lead = np.asarray(lead_value)
        if np.issubdtype(lead.dtype, np.timedelta64):
            return truth["init_time"] + lead
    lead = np.asarray(lead_value)
    if np.issubdtype(lead.dtype, np.datetime64):
        return xr.DataArray(lead)
    raise ValueError("ACC requires valid_time or datetime init_time plus timedelta lead_time")


def _climatology_for_record(
    climatology: xr.Dataset,
    name: str,
    truth: xr.DataArray,
    *,
    lead_value: Any,
    level_value: Any | None,
) -> xr.DataArray:
    climatology = _canonical_spatial_names(climatology)
    climatology_name = name if name in climatology else f"{name}_mean"
    if climatology_name not in climatology:
        raise ValueError(f"climatology is missing {name!r}")
    result = climatology[climatology_name]
    if result.attrs.get("units") != truth.attrs.get("units"):
        raise ValueError(f"climatology units differ for {name!r}")
    if level_value is not None and "level" not in result.dims:
        raise ValueError(f"climatology for {name!r} requires a level dimension")
    if level_value is None and "level" in result.dims:
        raise ValueError(f"climatology for {name!r} has an unexpected level dimension")
    if level_value is not None:
        result = result.sel(level=level_value)
    if "dayofyear" in result.dims:
        valid_time = _valid_time(truth, lead_value)
        selection: dict[str, xr.DataArray] = {"dayofyear": valid_time.dt.dayofyear}
        if "hour" in result.dims:
            selection["hour"] = valid_time.dt.hour
        result = result.sel(selection)
    unexpected = set(result.dims) - set(truth.dims)
    if unexpected:
        raise ValueError(
            f"climatology for {name!r} has unexpected dimensions: "
            + ", ".join(sorted(unexpected))
        )
    if not np.isfinite(result.values).all():
        raise ValueError(f"climatology for {name!r} must be finite")
    for dimension in ("latitude", "longitude"):
        if dimension not in result.dims or not np.array_equal(
            np.asarray(result[dimension].values), np.asarray(truth[dimension].values)
        ):
            raise ValueError(f"climatology coordinates differ for {name!r}: {dimension}")
    try:
        result, _ = xr.broadcast(result, truth)
    except ValueError as error:
        raise ValueError(f"climatology coordinates differ for {name!r}") from error
    return result


def _acc(
    truth: xr.DataArray,
    forecast: xr.DataArray,
    climatology: xr.DataArray,
) -> float | None:
    forecast_anomaly = forecast.astype(np.float64) - climatology.astype(np.float64)
    truth_anomaly = truth.astype(np.float64) - climatology.astype(np.float64)
    numerator = _spatial_mean(forecast_anomaly * truth_anomaly)
    denominator = np.sqrt(
        _spatial_mean(forecast_anomaly**2) * _spatial_mean(truth_anomaly**2)
    )
    per_initialization = numerator / denominator
    value = _mean_scalar(per_initialization)
    return value if math.isfinite(value) else None


def _normalized_drift(
    native: xr.DataArray,
    candidate: xr.DataArray,
    stddev: xr.DataArray,
) -> tuple[float, float]:
    scale = np.asarray(stddev.values, dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("normalization stddev must be positive finite")
    normalized = np.asarray(((candidate - native) / stddev).values, dtype=np.float64)
    absolute = np.abs(normalized)
    return float(np.max(absolute)), float(np.sqrt(np.mean(normalized**2)))


class _ParityAccumulator:
    """Accumulate Native/candidate numerical parity one diagnostic record at a time."""

    def __init__(self, *, atol: float, rtol: float) -> None:
        if not math.isfinite(atol) or atol < 0.0:
            raise ValueError("raw parity atol must be finite and nonnegative")
        if not math.isfinite(rtol) or rtol < 0.0:
            raise ValueError("raw parity rtol must be finite and nonnegative")
        self.atol = atol
        self.rtol = rtol
        self.values = 0
        self.finite = True
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.sum_sq_diff = 0.0
        self.sum_sq_native = 0.0
        self.allclose = True

    @staticmethod
    def _relative_l2(
        sum_sq_diff: float, sum_sq_native: float
    ) -> float | None:
        numerator = math.sqrt(sum_sq_diff)
        denominator = math.sqrt(sum_sq_native)
        if denominator <= 1.0e-12:
            return 0.0 if numerator <= 1.0e-12 else None
        return numerator / denominator

    def update(self, native: xr.DataArray, candidate: xr.DataArray) -> dict[str, Any]:
        native_values = np.asarray(native.values, dtype=np.float64)
        candidate_values = np.asarray(candidate.values, dtype=np.float64)
        values = int(native_values.size)
        self.values += values
        finite = bool(
            np.isfinite(native_values).all() and np.isfinite(candidate_values).all()
        )
        self.finite = self.finite and finite
        if not finite:
            self.allclose = False
            return {
                "parity_max_abs": None,
                "parity_mean_abs": None,
                "parity_rmse": None,
                "parity_relative_l2": None,
                "parity_allclose": False,
            }

        difference = candidate_values - native_values
        absolute = np.abs(difference)
        max_abs = float(np.max(absolute))
        sum_abs = float(np.sum(absolute, dtype=np.float64))
        sum_sq_diff = float(np.sum(difference * difference, dtype=np.float64))
        sum_sq_native = float(np.sum(native_values * native_values, dtype=np.float64))
        allclose = bool(
            np.allclose(
                candidate_values,
                native_values,
                atol=self.atol,
                rtol=self.rtol,
                equal_nan=False,
            )
        )
        self.max_abs = max(self.max_abs, max_abs)
        self.sum_abs += sum_abs
        self.sum_sq_diff += sum_sq_diff
        self.sum_sq_native += sum_sq_native
        self.allclose = self.allclose and allclose
        return {
            "parity_max_abs": max_abs,
            "parity_mean_abs": sum_abs / values,
            "parity_rmse": math.sqrt(sum_sq_diff / values),
            "parity_relative_l2": self._relative_l2(sum_sq_diff, sum_sq_native),
            "parity_allclose": allclose,
        }

    def result(self) -> dict[str, Any]:
        complete = self.finite and self.values > 0
        return {
            "finite": self.finite,
            "values": self.values,
            "max_abs": self.max_abs if complete else None,
            "mean_abs": self.sum_abs / self.values if complete else None,
            "rmse": math.sqrt(self.sum_sq_diff / self.values) if complete else None,
            "relative_l2": (
                self._relative_l2(self.sum_sq_diff, self.sum_sq_native)
                if complete
                else None
            ),
            "allclose": self.allclose if complete else False,
            "atol": self.atol,
            "rtol": self.rtol,
        }


def _climatology_method(climatology: xr.Dataset | None) -> str:
    if climatology is None:
        return "not_provided"
    dimensions = set(climatology.dims)
    if "dayofyear" in dimensions and "hour" in dimensions:
        return "dayofyear_and_hour"
    if "dayofyear" in dimensions:
        return "dayofyear"
    return "static_field"


def evaluate_paired_forecasts(
    truth: xr.Dataset,
    native: xr.Dataset,
    candidate: xr.Dataset,
    *,
    climatology: xr.Dataset | None = None,
    stddev: xr.Dataset | None = None,
    raw_atol: float = 1.0e-5,
    raw_rtol: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare Native and candidate forecasts using WeatherBench-style metrics."""

    truth = _canonical_forecast(truth)
    native = _canonical_forecast(native)
    candidate = _canonical_forecast(candidate)
    for label, forecast in (("truth", truth), ("Native", native), ("candidate", candidate)):
        _validate_forecast_layout(forecast, label=label)
        _validate_time_semantics(forecast, label=label)
    lead_count = int(truth.sizes["lead_time"])
    _validate_matching_forecasts(truth, native, candidate)
    canonical_stddev = _canonical_spatial_names(stddev) if stddev is not None else None
    parity = _ParityAccumulator(atol=raw_atol, rtol=raw_rtol)

    records: list[dict[str, Any]] = []
    all_finite = True
    for name in sorted(truth.data_vars):
        levels: list[Any | None] = (
            list(truth[name]["level"].values) if "level" in truth[name].dims else [None]
        )
        for lead_value in truth[name]["lead_time"].values:
            for level_value in levels:
                selection: dict[str, Any] = {"lead_time": lead_value}
                if level_value is not None:
                    selection["level"] = level_value
                truth_record = truth[name].sel(selection).load()
                native_record = native[name].sel(selection).load()
                candidate_record = candidate[name].sel(selection).load()
                finite = bool(
                    np.isfinite(truth_record.values).all()
                    and np.isfinite(native_record.values).all()
                    and np.isfinite(candidate_record.values).all()
                )
                all_finite = all_finite and finite
                native_metrics = _forecast_metrics(truth_record, native_record)
                candidate_metrics = _forecast_metrics(truth_record, candidate_record)
                native_acc: float | None = None
                candidate_acc: float | None = None
                if climatology is not None:
                    climatology_record = _climatology_for_record(
                        climatology,
                        name,
                        truth_record,
                        lead_value=lead_value,
                        level_value=level_value,
                    )
                    native_acc = _acc(truth_record, native_record, climatology_record)
                    candidate_acc = _acc(truth_record, candidate_record, climatology_record)

                normalized_max_abs: float | None = None
                normalized_rmse: float | None = None
                if canonical_stddev is not None and finite:
                    if name not in canonical_stddev:
                        raise ValueError(f"normalization stddev is missing {name!r}")
                    scale = canonical_stddev[name]
                    if level_value is not None and "level" in scale.dims:
                        scale = scale.sel(level=level_value)
                    unexpected_scale_dimensions = set(scale.dims) - set(
                        truth_record.dims
                    )
                    if unexpected_scale_dimensions:
                        raise ValueError(
                            f"normalization stddev for {name!r} has unexpected dimensions: "
                            + ", ".join(sorted(unexpected_scale_dimensions))
                        )
                    for dimension in scale.dims:
                        if dimension in scale.coords and not np.array_equal(
                            np.asarray(scale[dimension].values),
                            np.asarray(truth_record[dimension].values),
                        ):
                            raise ValueError(
                                f"normalization stddev coordinates differ for {name!r}: "
                                f"{dimension}"
                            )
                    normalized_max_abs, normalized_rmse = _normalized_drift(
                        native_record,
                        candidate_record,
                        scale,
                    )

                native_rmse = native_metrics["rmse"]
                candidate_rmse = candidate_metrics["rmse"]
                relative_change = (
                    (candidate_rmse - native_rmse) / native_rmse
                    if native_rmse not in (None, 0.0) and candidate_rmse is not None
                    else None
                )
                records.append(
                    {
                        "variable": name,
                        "level": (
                            _coordinate_scalar(level_value)
                            if level_value is not None
                            else None
                        ),
                        "lead_hours": _lead_hours(lead_value),
                        "finite": finite,
                        **{f"native_{key}": value for key, value in native_metrics.items()},
                        **{
                            f"candidate_{key}": value
                            for key, value in candidate_metrics.items()
                        },
                        "native_acc": native_acc,
                        "candidate_acc": candidate_acc,
                        "rmse_delta": (
                            candidate_rmse - native_rmse
                            if candidate_rmse is not None and native_rmse is not None
                            else None
                        ),
                        "rmse_relative_change": relative_change,
                        "acc_delta": (
                            candidate_acc - native_acc
                            if candidate_acc is not None and native_acc is not None
                            else None
                        ),
                        "normalized_max_abs": normalized_max_abs,
                        "normalized_rmse": normalized_rmse,
                        **parity.update(native_record, candidate_record),
                    }
                )

    relative_changes = [
        row["rmse_relative_change"]
        for row in records
        if row["rmse_relative_change"] is not None
    ]
    acc_deltas = [row["acc_delta"] for row in records if row["acc_delta"] is not None]
    normalized_maxima = [
        row["normalized_max_abs"]
        for row in records
        if row["normalized_max_abs"] is not None
    ]
    return {
        "schema": "graphcast.forecast_diagnostics.v2",
        "status": "measured" if all_finite else "failed_nonfinite",
        "methodology": {
            "reference": "WeatherBench deterministic metrics",
            "area_weight": "spherical_latitude_cell_area",
            "rmse_aggregation": "sqrt(mean_mse_over_init_time)",
            "acc_climatology": _climatology_method(climatology),
            "normalized_drift": (
                "candidate_minus_native_divided_by_stddev"
                if stddev is not None
                else "not_requested"
            ),
            "streaming_granularity": "one_variable_lead_level_record",
            "implementation_parity": "candidate_minus_native_raw_values",
        },
        "init_count": int(truth.sizes.get("init_time", 1)),
        "lead_count": lead_count,
        "records": records,
        "implementation_parity": parity.result(),
        "worst": {
            "rmse_relative_increase": max(relative_changes, default=None),
            "acc_decrease": min(acc_deltas, default=None),
            "normalized_max_abs": max(normalized_maxima, default=None),
        },
    }


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


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    os.replace(_stage_json(path, payload), path)


def diagnostics_to_dataset(diagnostics: dict[str, Any]) -> xr.Dataset:
    """Encode long-form diagnostic records as a portable NetCDF dataset."""

    records = diagnostics["records"]
    numeric_fields = (
        "native_mse",
        "native_rmse",
        "native_mae",
        "native_bias",
        "native_acc",
        "candidate_mse",
        "candidate_rmse",
        "candidate_mae",
        "candidate_bias",
        "candidate_acc",
        "rmse_delta",
        "rmse_relative_change",
        "acc_delta",
        "normalized_max_abs",
        "normalized_rmse",
        "parity_max_abs",
        "parity_mean_abs",
        "parity_rmse",
        "parity_relative_l2",
    )
    dataset = xr.Dataset(
        data_vars={
            field: (
                ("record",),
                np.asarray(
                    [
                        np.nan if record[field] is None else float(record[field])
                        for record in records
                    ],
                    dtype=np.float64,
                ),
            )
            for field in numeric_fields
        },
        coords={
            "record": np.arange(len(records), dtype=np.int64),
            "variable": (
                ("record",),
                np.asarray([record["variable"] for record in records], dtype=str),
            ),
            "level": (
                ("record",),
                np.asarray(
                    [
                        np.nan if record["level"] is None else float(record["level"])
                        for record in records
                    ],
                    dtype=np.float64,
                ),
            ),
            "lead_hours": (
                ("record",),
                np.asarray([record["lead_hours"] for record in records], dtype=np.float64),
            ),
        },
        attrs={
            "schema": diagnostics["schema"],
            "status": diagnostics["status"],
            "methodology_json": json.dumps(
                diagnostics["methodology"], sort_keys=True, separators=(",", ":")
            ),
            "init_count": int(diagnostics["init_count"]),
            "lead_count": int(diagnostics["lead_count"]),
            "inputs_json": json.dumps(
                diagnostics.get("inputs", {}), sort_keys=True, separators=(",", ":")
            ),
            "worst_json": json.dumps(
                diagnostics["worst"], sort_keys=True, separators=(",", ":")
            ),
            "implementation_parity_json": json.dumps(
                diagnostics["implementation_parity"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    )
    dataset["finite"] = (
        ("record",),
        np.asarray([record["finite"] for record in records], dtype=np.int8),
    )
    dataset["finite"].attrs["encoding"] = "0=false,1=true"
    dataset["parity_allclose"] = (
        ("record",),
        np.asarray([record["parity_allclose"] for record in records], dtype=np.int8),
    )
    dataset["parity_allclose"].attrs["encoding"] = "0=false,1=true"
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_netcdf(path: Path, dataset: xr.Dataset) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        dataset.to_netcdf(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _require_distinct_paths(paths: dict[str, Path]) -> None:
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        canonical = path.resolve()
        if canonical in resolved:
            raise ValueError(
                f"diagnostic input and output paths must be distinct: "
                f"{resolved[canonical]} and {label}"
            )
        resolved[canonical] = label


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--climatology", type=Path)
    parser.add_argument(
        "--stddev",
        type=Path,
        help="Optional GraphCast stddev_by_level.nc for Native/candidate drift metrics",
    )
    parser.add_argument("--raw-atol", type=float, default=1.0e-5)
    parser.add_argument("--raw-rtol", type=float, default=1.0e-5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-netcdf", type=Path)
    parser.add_argument("--target", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--forecast", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    legacy_names = ("target", "forecast", "output")
    paired_names = (
        "truth",
        "native",
        "candidate",
        "climatology",
        "output_json",
        "output_netcdf",
    )
    has_legacy = any(getattr(args, name) is not None for name in legacy_names)
    has_paired = args.stddev is not None or any(
        getattr(args, name) is not None for name in paired_names
    )
    if has_legacy and has_paired:
        parser.error("legacy single-forecast and paired diagnostic arguments cannot be mixed")
    required = legacy_names if has_legacy else paired_names
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{name}" for name in missing))
    args.legacy_single_forecast = has_legacy
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.legacy_single_forecast:
        _require_distinct_paths(
            {"target": args.target, "forecast": args.forecast, "output": args.output}
        )
        for path in (args.target, args.forecast):
            if not path.is_file():
                raise FileNotFoundError(path)
        if args.output.exists():
            raise FileExistsError(args.output)
        target = xr.open_dataset(args.target).load()
        forecast = xr.open_dataset(args.forecast).load()
        result = evaluate_forecast(target, forecast)
        _write_json(args.output, result)
        print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
        return 0 if result["status"] == "measured" else 1

    input_paths: dict[str, Path] = {
        "truth": args.truth,
        "native": args.native,
        "candidate": args.candidate,
        "climatology": args.climatology,
    }
    if args.stddev is not None:
        input_paths["stddev"] = args.stddev
    output_paths = {
        "output_json": args.output_json,
        "output_netcdf": args.output_netcdf,
    }
    _require_distinct_paths({**input_paths, **output_paths})
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(path)
    input_identities = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in input_paths.items()
    }
    with ExitStack() as datasets:
        truth = datasets.enter_context(xr.open_dataset(args.truth))
        native = datasets.enter_context(xr.open_dataset(args.native))
        candidate = datasets.enter_context(xr.open_dataset(args.candidate))
        climatology = datasets.enter_context(xr.open_dataset(args.climatology))
        stddev = (
            datasets.enter_context(xr.open_dataset(args.stddev))
            if args.stddev is not None
            else None
        )
        result = evaluate_paired_forecasts(
            truth,
            native,
            candidate,
            climatology=climatology,
            stddev=stddev,
            raw_atol=args.raw_atol,
            raw_rtol=args.raw_rtol,
        )
        for name, path in input_paths.items():
            if _sha256(path) != input_identities[name]["sha256"]:
                raise RuntimeError(
                    f"diagnostic input changed while it was being read: {name}"
                )
    result["inputs"] = input_identities
    staged_netcdf: Path | None = None
    staged_json: Path | None = None
    published_netcdf = False
    try:
        staged_netcdf = _stage_netcdf(
            args.output_netcdf, diagnostics_to_dataset(result)
        )
        result["publication"] = {
            "schema": "graphcast.forecast_diagnostics_publication.v1",
            "terminal": "output_json",
            "metrics_netcdf": {
                "path": str(args.output_netcdf.resolve()),
                "sha256": _sha256(staged_netcdf),
            },
        }
        staged_json = _stage_json(args.output_json, result)
        os.replace(staged_netcdf, args.output_netcdf)
        staged_netcdf = None
        published_netcdf = True
        os.replace(staged_json, args.output_json)
        staged_json = None
    except BaseException:
        if staged_netcdf is not None:
            staged_netcdf.unlink(missing_ok=True)
        if staged_json is not None:
            staged_json.unlink(missing_ok=True)
        if published_netcdf and not args.output_json.exists():
            args.output_netcdf.unlink(missing_ok=True)
        raise
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0 if result["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
