from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
if str(GRAPHCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_ROOT))

from scripts.inference.evaluate_forecast_skill import evaluate_forecast  # noqa: E402


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
