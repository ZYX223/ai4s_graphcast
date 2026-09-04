from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
if str(GRAPHCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_ROOT))

from scripts.inference import evaluate_forecast_skill as forecast_skill  # noqa: E402
from scripts.inference.run_operational_40step_inference import (  # noqa: E402
    TrajectoryNetCDFWriter,
)

evaluate_forecast = forecast_skill.evaluate_forecast


def _paired_evaluator():
    evaluator = getattr(forecast_skill, "evaluate_paired_forecasts", None)
    assert callable(evaluator), "paired official-style forecast evaluator is missing"
    return evaluator


def test_latitude_weighted_rmse_uses_grid_cell_area() -> None:
    target = xr.Dataset(
        {
            "temperature": (
                ("batch", "time", "lat", "lon"),
                np.zeros((1, 1, 2, 2), dtype=np.float32),
            )
        },
        coords={"batch": [0], "time": [6], "lat": [0.0, 60.0], "lon": [0.0, 180.0]},
    )
    forecast = target.copy(deep=True)
    forecast["temperature"].values[:, :, 0, :] = 2.0
    forecast["temperature"].values[:, :, 1, :] = 4.0

    result = evaluate_forecast(target, forecast)

    # sqrt((2^2 * cos(0) + 4^2 * cos(60)) / (cos(0) + cos(60)))
    assert np.isclose(result["variables"][0]["latitude_weighted_rmse"], np.sqrt(8.0))
    assert result["variables"][0]["finite"] is True


def test_skill_evaluation_reports_each_pressure_level_separately() -> None:
    target = xr.Dataset(
        {
            "wind": (
                ("batch", "time", "level", "lat", "lon"),
                np.zeros((1, 1, 2, 1, 1), dtype=np.float32),
            )
        },
        coords={
            "batch": [0],
            "time": [6],
            "level": [500, 850],
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    forecast = target.copy(deep=True)
    forecast["wind"].values[:, :, 0, :, :] = 1.0
    forecast["wind"].values[:, :, 1, :, :] = 3.0

    result = evaluate_forecast(target, forecast)

    assert result["variables"][0]["levels"] == [
        {"level": 500, "latitude_weighted_rmse": 1.0},
        {"level": 850, "latitude_weighted_rmse": 3.0},
    ]


def _trajectory_dataset(values: np.ndarray, *, name: str = "temperature") -> xr.Dataset:
    lead_count = int(np.asarray(values).shape[1])
    return xr.Dataset(
        {
            name: (
                ("init_time", "lead_time", "latitude", "longitude"),
                np.asarray(values, dtype=np.float64),
            )
        },
        coords={
            "init_time": np.asarray(["2022-01-01T00", "2022-01-02T00"], dtype="datetime64[h]"),
            "lead_time": np.arange(1, lead_count + 1) * np.timedelta64(6, "h"),
            "latitude": [-60.0, 0.0, 60.0],
            "longitude": [0.0, 180.0],
        },
    )


def test_paired_diagnostics_average_mse_before_taking_rmse() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    native["temperature"].values[1, ...] = 2.0
    candidate["temperature"].values[1, ...] = 4.0

    diagnostics = _paired_evaluator()(truth, native, candidate)

    assert diagnostics["schema"] == "graphcast.forecast_diagnostics.v2"
    assert diagnostics["lead_count"] == 1
    assert diagnostics["methodology"]["rmse_aggregation"] == "sqrt(mean_mse_over_init_time)"
    assert diagnostics["methodology"]["area_weight"] == "spherical_latitude_cell_area"
    assert len(diagnostics["records"]) == 1
    record = diagnostics["records"][0]
    assert record["native_mse"] == pytest.approx(2.0)
    assert record["native_rmse"] == pytest.approx(np.sqrt(2.0))
    assert record["candidate_mse"] == pytest.approx(8.0)
    assert record["candidate_rmse"] == pytest.approx(np.sqrt(8.0))
    assert record["rmse_delta"] == pytest.approx(np.sqrt(8.0) - np.sqrt(2.0))
    assert record["rmse_relative_change"] == pytest.approx(1.0)


def test_paired_diagnostics_compute_official_style_acc_from_climatology() -> None:
    truth = xr.Dataset(
        {
            "geopotential": (
                ("init_time", "lead_time", "latitude", "longitude"),
                np.asarray([[[[-1.0, 1.0]]]], dtype=np.float64),
            )
        },
        coords={
            "init_time": np.asarray(["2022-01-01T00"], dtype="datetime64[h]"),
            "lead_time": np.asarray([6], dtype="timedelta64[h]"),
            "latitude": [0.0],
            "longitude": [0.0, 180.0],
        },
    )
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    candidate["geopotential"] *= -1.0
    climatology = xr.Dataset(
        {
            "geopotential": (
                ("dayofyear", "hour", "latitude", "longitude"),
                np.zeros((1, 1, 1, 2), dtype=np.float64),
            )
        },
        coords={
            "dayofyear": [1],
            "hour": [6],
            "latitude": [0.0],
            "longitude": [0.0, 180.0],
        },
    )

    diagnostics = _paired_evaluator()(
        truth,
        native,
        candidate,
        climatology=climatology,
    )

    record = diagnostics["records"][0]
    assert record["native_acc"] == pytest.approx(1.0)
    assert record["candidate_acc"] == pytest.approx(-1.0)
    assert record["acc_delta"] == pytest.approx(-2.0)
    assert diagnostics["methodology"]["acc_climatology"] == "dayofyear_and_hour"


def test_paired_diagnostics_report_each_lead_and_level_with_normalized_drift() -> None:
    coords = {
        "init_time": np.asarray(["2022-01-01T00"], dtype="datetime64[h]"),
        "lead_time": np.asarray([6, 12], dtype="timedelta64[h]"),
        "level": [500, 850],
        "latitude": [0.0],
        "longitude": [0.0, 180.0],
    }
    truth = xr.Dataset(
        {
            "temperature": (
                ("init_time", "lead_time", "level", "latitude", "longitude"),
                np.zeros((1, 2, 2, 1, 2), dtype=np.float64),
            )
        },
        coords=coords,
    )
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    candidate["temperature"].loc[{"lead_time": np.timedelta64(12, "h"), "level": 850}] = 4.0
    stddev = xr.Dataset(
        {"temperature": (("level",), np.asarray([2.0, 4.0]))},
        coords={"level": [500, 850]},
    )

    diagnostics = _paired_evaluator()(
        truth,
        native,
        candidate,
        stddev=stddev,
    )

    assert diagnostics["lead_count"] == 2
    assert [(row["lead_hours"], row["level"]) for row in diagnostics["records"]] == [
        (6.0, 500),
        (6.0, 850),
        (12.0, 500),
        (12.0, 850),
    ]
    worst = diagnostics["records"][-1]
    assert worst["normalized_max_abs"] == pytest.approx(1.0)
    assert worst["normalized_rmse"] == pytest.approx(1.0)


def test_paired_diagnostics_include_raw_native_candidate_parity() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    native["temperature"].values[...] = 1.0
    candidate["temperature"].values[...] = 3.0

    diagnostics = _paired_evaluator()(
        truth,
        native,
        candidate,
        raw_atol=2.1,
        raw_rtol=0.0,
    )

    record = diagnostics["records"][0]
    assert record["parity_max_abs"] == pytest.approx(2.0)
    assert record["parity_mean_abs"] == pytest.approx(2.0)
    assert record["parity_rmse"] == pytest.approx(2.0)
    assert record["parity_relative_l2"] == pytest.approx(2.0)
    assert record["parity_allclose"] is True
    assert diagnostics["implementation_parity"] == {
        "finite": True,
        "values": 12,
        "max_abs": pytest.approx(2.0),
        "mean_abs": pytest.approx(2.0),
        "rmse": pytest.approx(2.0),
        "relative_l2": pytest.approx(2.0),
        "allclose": True,
        "atol": 2.1,
        "rtol": 0.0,
    }


def test_raw_parity_zero_native_norm_remains_strict_json() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    candidate["temperature"].values[...] = 1.0

    diagnostics = _paired_evaluator()(truth, native, candidate)

    assert diagnostics["records"][0]["parity_relative_l2"] is None
    assert diagnostics["implementation_parity"]["relative_l2"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_paired_diagnostics_reject_an_empty_trajectory() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2))).isel(lead_time=slice(0, 0))

    with pytest.raises(ValueError, match="non-empty lead_time"):
        _paired_evaluator()(truth, truth.copy(deep=True), truth.copy(deep=True))


def test_paired_diagnostics_reject_coordinate_mismatch() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    candidate = truth.copy(deep=True)
    candidate = candidate.assign_coords(longitude=[1.0, 181.0])

    with pytest.raises(ValueError, match="coordinates differ"):
        _paired_evaluator()(truth, truth.copy(deep=True), candidate)


def test_paired_diagnostics_reject_valid_time_mismatch() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2))).assign_coords(
        valid_time=(
            ("init_time", "lead_time"),
            np.asarray(
                [["2022-01-01T06"], ["2022-01-02T06"]], dtype="datetime64[h]"
            ),
        )
    )
    candidate = truth.copy(deep=True).assign_coords(
        valid_time=truth["valid_time"] + np.timedelta64(1, "h")
    )

    with pytest.raises(ValueError, match="valid_time"):
        _paired_evaluator()(truth, truth.copy(deep=True), candidate)


def test_paired_diagnostics_reject_shared_but_incorrect_valid_times() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2))).assign_coords(
        valid_time=(
            ("init_time", "lead_time"),
            np.asarray(
                [["2030-01-01T06"], ["2030-01-02T06"]], dtype="datetime64[h]"
            ),
        )
    )

    with pytest.raises(ValueError, match="init_time.*lead_time"):
        _paired_evaluator()(truth, truth.copy(deep=True), truth.copy(deep=True))


def test_runner_trajectory_matches_conventional_graphcast_truth_schema(
    tmp_path: Path,
) -> None:
    valid_times = xr.DataArray(
        np.asarray([["2022-01-01T06", "2022-01-01T12"]], dtype="datetime64[h]"),
        dims=("batch", "time"),
        coords={"batch": [0], "time": np.asarray([6, 12], dtype="timedelta64[h]")},
    )
    truth = xr.Dataset(
        {
            "temperature": (
                ("batch", "time", "lat", "lon"),
                np.asarray([[[[1.0]], [[2.0]]]]),
            )
        },
        coords={
            "batch": [0],
            "time": np.asarray([6, 12], dtype="timedelta64[h]"),
            "lat": [0.0],
            "lon": [0.0],
            "datetime": (("batch", "time"), valid_times.values),
        },
    )
    trajectory_path = tmp_path / "trajectory.nc"
    writer = TrajectoryNetCDFWriter(
        trajectory_path,
        steps=2,
        step_hours=6,
        valid_times=valid_times,
    )
    writer.append(0, truth.isel(time=slice(0, 1)).drop_vars("datetime"))
    writer.append(1, truth.isel(time=slice(1, 2)).drop_vars("datetime"))
    writer.finalize()
    with xr.open_dataset(trajectory_path) as dataset:
        trajectory = dataset.load()

    diagnostics = _paired_evaluator()(truth, trajectory, trajectory.copy(deep=True))

    assert diagnostics["lead_count"] == 2
    assert [record["lead_hours"] for record in diagnostics["records"]] == [6.0, 12.0]


def test_paired_diagnostics_reject_unit_mismatch() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    truth["temperature"].attrs["units"] = "K"
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    candidate["temperature"].attrs["units"] = "degC"

    with pytest.raises(ValueError, match="units differ"):
        _paired_evaluator()(truth, native, candidate)


def test_paired_diagnostics_fail_nonfinite_without_emitting_json_nan() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    candidate = truth.copy(deep=True)
    candidate["temperature"].values[0, 0, 0, 0] = np.nan

    diagnostics = _paired_evaluator()(truth, truth.copy(deep=True), candidate)

    assert diagnostics["status"] == "failed_nonfinite"
    assert diagnostics["records"][0]["candidate_rmse"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_normalized_drift_rejects_a_non_positive_stddev() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    stddev = xr.Dataset({"temperature": xr.DataArray(0.0)})

    with pytest.raises(ValueError, match="positive finite"):
        _paired_evaluator()(
            truth,
            truth.copy(deep=True),
            truth.copy(deep=True),
            stddev=stddev,
        )


@pytest.mark.parametrize("kind", ("variables", "init_time", "longitude"))
def test_paired_diagnostics_reject_empty_required_axes(kind: str) -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    if kind == "variables":
        truth = truth.drop_vars(list(truth.data_vars))
    else:
        truth = truth.isel({kind: slice(0, 0)})

    with pytest.raises(ValueError, match="non-empty"):
        _paired_evaluator()(truth, truth.copy(deep=True), truth.copy(deep=True))


def test_acc_rejects_level_less_climatology_for_a_pressure_variable() -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)), name="temperature").expand_dims(
        level=[500, 850], axis=2
    )
    climatology = xr.Dataset(
        {
            "temperature": (
                ("dayofyear", "latitude", "longitude"),
                np.zeros((2, 3, 2)),
            )
        },
        coords={
            "dayofyear": [1, 2],
            "latitude": [-60.0, 0.0, 60.0],
            "longitude": [0.0, 180.0],
        },
    )

    with pytest.raises(ValueError, match="level dimension"):
        _paired_evaluator()(
            truth,
            truth.copy(deep=True),
            truth.copy(deep=True),
            climatology=climatology,
        )


def test_diagnostic_cli_writes_json_and_netcdf_records(tmp_path: Path) -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    native["temperature"].values[...] = 1.0
    candidate["temperature"].values[...] = 2.0
    climatology = xr.Dataset(
        {
            "temperature": (
                ("dayofyear", "hour", "latitude", "longitude"),
                np.zeros((2, 1, 3, 2), dtype=np.float64),
            )
        },
        coords={
            "dayofyear": [1, 2],
            "hour": [6],
            "latitude": [-60.0, 0.0, 60.0],
            "longitude": [0.0, 180.0],
        },
    )
    stddev = xr.Dataset(
        {"temperature": xr.DataArray(2.0)},
    )
    paths = {
        "truth": tmp_path / "truth.nc",
        "native": tmp_path / "native.nc",
        "candidate": tmp_path / "candidate.nc",
        "climatology": tmp_path / "climatology.nc",
        "stddev": tmp_path / "stddev.nc",
        "json": tmp_path / "diagnostics.json",
        "netcdf": tmp_path / "diagnostics.nc",
    }
    truth.to_netcdf(paths["truth"])
    native.to_netcdf(paths["native"])
    candidate.to_netcdf(paths["candidate"])
    climatology.to_netcdf(paths["climatology"])
    stddev.to_netcdf(paths["stddev"])

    exit_code = forecast_skill.main(
        [
            "--truth",
            str(paths["truth"]),
            "--native",
            str(paths["native"]),
            "--candidate",
            str(paths["candidate"]),
            "--climatology",
            str(paths["climatology"]),
            "--stddev",
            str(paths["stddev"]),
            "--output-json",
            str(paths["json"]),
            "--output-netcdf",
            str(paths["netcdf"]),
        ]
    )

    assert exit_code == 0
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["schema"] == "graphcast.forecast_diagnostics.v2"
    assert payload["inputs"]["truth"]["sha256"]
    assert payload["publication"]["terminal"] == "output_json"
    assert payload["publication"]["metrics_netcdf"]["sha256"] == (
        forecast_skill._sha256(paths["netcdf"])
    )
    with xr.open_dataset(paths["netcdf"]) as metrics:
        assert metrics.attrs["schema"] == "graphcast.forecast_diagnostics.v2"
        assert metrics.sizes["record"] == 1
        assert metrics["native_rmse"].item() == pytest.approx(1.0)
        assert metrics["candidate_rmse"].item() == pytest.approx(2.0)
        assert metrics["normalized_max_abs"].item() == pytest.approx(0.5)
        assert json.loads(metrics.attrs["inputs_json"])["truth"]["sha256"]
        assert json.loads(metrics.attrs["worst_json"]) == payload["worst"]


def test_diagnostic_cli_does_not_eagerly_load_whole_trajectories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = _trajectory_dataset(np.zeros((2, 2, 3, 2)))
    native = truth.copy(deep=True)
    candidate = truth.copy(deep=True)
    climatology = xr.Dataset(
        {
            "temperature": (
                ("dayofyear", "hour", "latitude", "longitude"),
                np.zeros((2, 2, 3, 2), dtype=np.float64),
            )
        },
        coords={
            "dayofyear": [1, 2],
            "hour": [6, 12],
            "latitude": [-60.0, 0.0, 60.0],
            "longitude": [0.0, 180.0],
        },
    )
    paths = {
        name: tmp_path / f"{name}.nc"
        for name in ("truth", "native", "candidate", "climatology")
    }
    truth.to_netcdf(paths["truth"])
    native.to_netcdf(paths["native"])
    candidate.to_netcdf(paths["candidate"])
    climatology.to_netcdf(paths["climatology"])

    original_open_dataset = xr.open_dataset

    class LazyDatasetGuard:
        def __init__(self, dataset: xr.Dataset) -> None:
            self.dataset = dataset

        def __enter__(self) -> "LazyDatasetGuard":
            return self

        def __exit__(self, *_args: object) -> None:
            self.dataset.close()

        def __getattr__(self, name: str):
            return getattr(self.dataset, name)

        def __contains__(self, name: str) -> bool:
            return name in self.dataset

        def __getitem__(self, name: str):
            return self.dataset[name]

        def load(self) -> None:
            raise AssertionError("whole-dataset load defeats streaming diagnostics")

    def guarded_open_dataset(path: Path | str, *args: object, **kwargs: object):
        return LazyDatasetGuard(original_open_dataset(path, *args, **kwargs))

    monkeypatch.setattr(forecast_skill.xr, "open_dataset", guarded_open_dataset)

    exit_code = forecast_skill.main(
        [
            "--truth", str(paths["truth"]),
            "--native", str(paths["native"]),
            "--candidate", str(paths["candidate"]),
            "--climatology", str(paths["climatology"]),
            "--output-json", str(tmp_path / "diagnostics.json"),
            "--output-netcdf", str(tmp_path / "diagnostics.nc"),
        ]
    )

    assert exit_code == 0


def test_diagnostic_cli_rejects_colliding_input_and_output_paths(tmp_path: Path) -> None:
    truth = _trajectory_dataset(np.zeros((2, 1, 3, 2)))
    inputs = {
        name: tmp_path / f"{name}.nc"
        for name in ("truth", "native", "candidate", "climatology")
    }
    truth.to_netcdf(inputs["truth"])
    truth.to_netcdf(inputs["native"])
    truth.to_netcdf(inputs["candidate"])
    xr.Dataset(
        {"temperature": (("latitude", "longitude"), np.zeros((3, 2)))},
        coords={"latitude": [-60.0, 0.0, 60.0], "longitude": [0.0, 180.0]},
    ).to_netcdf(inputs["climatology"])

    with pytest.raises(ValueError, match="distinct"):
        forecast_skill.main(
            [
                "--truth", str(inputs["truth"]),
                "--native", str(inputs["native"]),
                "--candidate", str(inputs["candidate"]),
                "--climatology", str(inputs["climatology"]),
                "--output-json", str(inputs["truth"]),
                "--output-netcdf", str(tmp_path / "diagnostics.nc"),
            ]
        )


def test_legacy_single_forecast_cli_remains_available(tmp_path: Path) -> None:
    target = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.zeros((1, 1, 1)))},
        coords={"time": [6], "lat": [0.0], "lon": [0.0]},
    )
    forecast = target.copy(deep=True)
    forecast["temperature"] += 2.0
    target_path = tmp_path / "target.nc"
    forecast_path = tmp_path / "forecast.nc"
    output_path = tmp_path / "skill.json"
    target.to_netcdf(target_path)
    forecast.to_netcdf(forecast_path)

    exit_code = forecast_skill.main(
        [
            "--target",
            str(target_path),
            "--forecast",
            str(forecast_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "graphcast.forecast_skill.v1"
