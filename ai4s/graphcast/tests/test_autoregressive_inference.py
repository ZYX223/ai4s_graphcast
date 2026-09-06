from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat.autoregressive_inference import (
    FeedbackUpdate,
    ForcingUpdate,
    advance_device_rollout,
    normalized_output_to_prediction,
    prepare_device_rollout,
)


def _datasets():
    history_coords = {
        "batch": [0],
        "time": [-6, 0],
        "lat": [10.0, -10.0],
        "lon": [0.0, 90.0],
    }
    inputs = xr.Dataset(
        {
            "state": xr.DataArray(
                np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
                dims=("batch", "time", "lat", "lon"),
                coords=history_coords,
            ),
            "forcing": xr.DataArray(
                np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2) + 100,
                dims=("batch", "time", "lat", "lon"),
                coords=history_coords,
            ),
        }
    )
    target_coords = {
        "batch": [0],
        "time": [6],
        "lat": [10.0, -10.0],
        "lon": [0.0, 90.0],
    }
    target = xr.Dataset(
        {
            "state": xr.DataArray(
                np.zeros((1, 1, 2, 2), dtype=np.float32),
                dims=("batch", "time", "lat", "lon"),
                coords=target_coords,
            )
        }
    )
    forcing_coords = {**target_coords, "time": [6, 12]}
    forcings = xr.Dataset(
        {
            "forcing": xr.DataArray(
                np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2) + 200,
                dims=("batch", "time", "lat", "lon"),
                coords=forcing_coords,
            )
        }
    )
    stats = SimpleNamespace(
        mean_by_level=xr.Dataset(
            {"state": xr.DataArray(np.float32(10)), "forcing": xr.DataArray(np.float32(100))}
        ),
        stddev_by_level=xr.Dataset(
            {"state": xr.DataArray(np.float32(2)), "forcing": xr.DataArray(np.float32(10))}
        ),
        diffs_stddev_by_level=xr.Dataset({"state": xr.DataArray(np.float32(1))}),
    )
    config = SimpleNamespace(
        input_variables=("state", "forcing"),
        target_variables=("state",),
        forcing_variables=("forcing",),
        node_input_dim=8,
        output_dim=1,
    )
    return inputs, target, forcings, stats, config


def test_prepare_and_advance_keep_history_and_forcings_on_device() -> None:
    inputs, target, forcings, stats, config = _datasets()
    prepared = prepare_device_rollout(
        inputs,
        target,
        forcings,
        config=config,
        stats=stats,
        device=torch.device("cpu"),
    )
    assert prepared.steps == 2
    assert prepared.initial_state.device.type == "cpu"
    assert prepared.forcing_features.shape == (2, 1, 4, 1)

    state = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
    feedback = FeedbackUpdate("state", slice(0, 1), slice(1, 2), slice(0, 1), torch.tensor([0.5]))
    forcing = ForcingUpdate("forcing", slice(2, 3), slice(3, 4), slice(0, 1))
    compact = SimpleNamespace(feedback_updates=(feedback,), forcing_updates=(forcing,))
    result = advance_device_rollout(
        state,
        torch.tensor([[[50.0]]]),
        torch.tensor([[[2.0]]]),
        compact,
        advance_state=True,
        return_absolute_output=True,
    )
    torch.testing.assert_close(result, torch.tensor([[[21.0]]]))
    torch.testing.assert_close(state, torch.tensor([[[20.0, 21.0, 40.0, 50.0]]]))


def test_final_normalized_output_preserves_target_units_without_mutating_template() -> None:
    _, target, _, stats, _ = _datasets()
    target["state"].attrs["units"] = "K"
    original_target = target.copy(deep=True)

    prediction = normalized_output_to_prediction(
        torch.tensor([[[-1.0], [0.0], [1.5], [2.0]]]),
        target,
        stats,
    )

    np.testing.assert_array_equal(
        prediction["state"].values,
        np.asarray([8.0, 10.0, 13.0, 14.0], dtype=np.float32).reshape(1, 1, 2, 2),
    )
    assert prediction["state"].attrs["units"] == "K"
    xr.testing.assert_identical(target, original_target)
