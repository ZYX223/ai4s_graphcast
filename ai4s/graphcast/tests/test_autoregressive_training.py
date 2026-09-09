from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat import autoregressive_training as training


class ScalarResidualModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.residual = torch.nn.Parameter(torch.tensor(value))
        self.seen_current: list[float] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.seen_current.append(float(features[0, 0, 1].detach()))
        return self.residual.expand(features.shape[0], features.shape[1], 1)


def _scalar_batch(*, sample_weights: torch.Tensor | None = None):
    return training.AutoregressiveBatch(
        initial_input_features=torch.tensor([[[0.0, 1.0]]]),
        forcing_features=torch.empty((2, 1, 1, 0)),
        target_absolute_features=torch.tensor([[[[2.0]]], [[[4.0]]]]),
        feedback_updates=(
            training.FeedbackUpdate(
                name="state",
                input_previous=slice(0, 1),
                input_current=slice(1, 2),
                output=slice(0, 1),
                residual_to_absolute=torch.tensor([1.0]),
            ),
        ),
        forcing_updates=(),
        latitude_weights=torch.tensor([1.0]),
        channel_weights=torch.tensor([1.0]),
        target_channel_slices={"state": slice(0, 1)},
        lat_count=1,
        lon_count=1,
        sample_weights=sample_weights,
    )


def test_two_lead_feedback_keeps_full_bptt_graph():
    model = ScalarResidualModel(0.5)

    result = training.autoregressive_loss(model, _scalar_batch())
    result.loss.backward()

    assert model.seen_current == [1.0, 1.5]
    assert float(result.loss.detach()) == pytest.approx(2.125)
    assert float(model.residual.grad) == pytest.approx(-4.5)
    assert result.lead_losses.shape == (2,)
    assert float(result.final_input_features[0, 0, 1].detach()) == pytest.approx(2.0)


@pytest.mark.parametrize("implementation", ["non_reentrant", "reentrant"])
@pytest.mark.parametrize("segments", [0, 2])
def test_optional_variable_diagnostics_preserve_loss_and_gradient(
    implementation: str,
    segments: int,
):
    reference_model = ScalarResidualModel(0.25)
    candidate_model = ScalarResidualModel(0.25)
    kwargs = {
        "checkpoint_segments": segments,
        "checkpoint_implementation": implementation,
    }

    expected = training.autoregressive_loss(reference_model, _scalar_batch(), **kwargs)
    actual = training.autoregressive_loss(
        candidate_model,
        _scalar_batch(),
        collect_variable_diagnostics=False,
        **kwargs,
    )
    expected.loss.backward()
    actual.loss.backward()

    assert actual.per_variable_lead_losses == {}
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_model.residual.grad,
        reference_model.residual.grad,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("implementation", ["non_reentrant", "reentrant"])
def test_segment_checkpoint_matches_uncheckpointed_loss_gradient_and_state(implementation):
    ordinary_model = ScalarResidualModel(0.5)
    checkpointed_model = ScalarResidualModel(0.5)

    ordinary = training.autoregressive_loss(ordinary_model, _scalar_batch())
    checkpointed = training.autoregressive_loss(
        checkpointed_model,
        _scalar_batch(),
        checkpoint_segments=2,
        checkpoint_implementation=implementation,
    )
    ordinary.loss.backward()
    checkpointed.loss.backward()

    torch.testing.assert_close(checkpointed.loss, ordinary.loss)
    torch.testing.assert_close(checkpointed_model.residual.grad, ordinary_model.residual.grad)
    torch.testing.assert_close(checkpointed.final_input_features, ordinary.final_input_features)


def test_zero_weight_padding_does_not_change_loss_or_gradient():
    single = _scalar_batch()
    padded = training.AutoregressiveBatch(
        initial_input_features=torch.cat(
            (single.initial_input_features, torch.tensor([[[0.0, 100.0]]])), dim=0
        ),
        forcing_features=torch.cat(
            (single.forcing_features, torch.empty((2, 1, 1, 0))), dim=1
        ),
        target_absolute_features=torch.cat(
            (
                single.target_absolute_features,
                torch.tensor([[[[1000.0]]], [[[2000.0]]]]),
            ),
            dim=1,
        ),
        feedback_updates=single.feedback_updates,
        forcing_updates=single.forcing_updates,
        latitude_weights=single.latitude_weights,
        channel_weights=single.channel_weights,
        target_channel_slices=single.target_channel_slices,
        lat_count=1,
        lon_count=1,
        sample_weights=torch.tensor([1.0, 0.0]),
    )
    expected_model = ScalarResidualModel(0.5)
    padded_model = ScalarResidualModel(0.5)

    expected = training.autoregressive_loss(expected_model, single)
    actual = training.autoregressive_loss(padded_model, padded)
    expected.loss.backward()
    actual.loss.backward()

    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(padded_model.residual.grad, expected_model.residual.grad)


def test_all_padding_produces_differentiable_zero_loss():
    model = ScalarResidualModel(0.5)

    result = training.autoregressive_loss(
        model,
        _scalar_batch(sample_weights=torch.tensor([0.0])),
    )
    result.loss.backward()

    assert float(result.loss.detach()) == 0.0
    assert float(model.residual.grad) == 0.0


def test_latitude_and_pressure_weights_match_pinned_upstream_functions():
    latitudes = np.asarray([90.0, 0.0, -90.0], dtype=np.float32)
    coordinate = xr.DataArray(
        np.zeros(3, dtype=np.float32),
        dims=("lat",),
        coords={"lat": latitudes},
    )
    expected_latitude = np.asarray(
        training.deepmind_losses.normalized_latitude_weights(coordinate).values,
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        training._normalized_latitude_weights(latitudes),
        expected_latitude,
    )
    assert expected_latitude[1] > expected_latitude[0]

    levels = np.asarray([500, 1000], dtype=np.int32)
    targets = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.zeros((1, 1, 3, 1, 2), dtype=np.float32),
                dims=("batch", "time", "lat", "lon", "level"),
                coords={
                    "batch": [0],
                    "time": [np.timedelta64(6, "h")],
                    "lat": latitudes,
                    "lon": [0.0],
                    "level": levels,
                },
            )
        }
    )
    total, diagnostic, _ = training._target_channel_weights(
        targets,
        per_variable_weights={"temperature": 0.25},
    )
    expected_pressure = np.asarray(
        training.deepmind_losses.normalized_level_weights(targets["temperature"]).values,
        dtype=np.float32,
    ) / len(levels)

    np.testing.assert_array_equal(diagnostic, expected_pressure)
    np.testing.assert_array_equal(total, expected_pressure * 0.25)
    assert diagnostic[0] != diagnostic[1]


def test_chunked_grid_loss_preserves_values_and_gradients():
    torch.manual_seed(23)
    batch = training.AutoregressiveBatch(
        initial_input_features=torch.empty((2, 15, 0)),
        forcing_features=torch.empty((1, 2, 15, 0)),
        target_absolute_features=torch.empty((1, 2, 15, 4)),
        feedback_updates=(),
        forcing_updates=(),
        latitude_weights=torch.tensor([0.4, 1.0, 0.4]),
        channel_weights=torch.tensor([1.0, 0.25, 0.5, 2.0]),
        diagnostic_channel_weights=torch.ones(4),
        target_channel_slices={"surface": slice(0, 1), "atmosphere": slice(1, 4)},
        lat_count=3,
        lon_count=5,
        sample_weights=torch.tensor([1.0, 0.5]),
    )
    target = torch.randn(2, 15, 4)
    full_prediction = torch.randn(2, 15, 4, requires_grad=True)
    chunked_prediction = full_prediction.detach().clone().requires_grad_(True)

    full_total, full_variables = training._weighted_losses(
        full_prediction,
        target,
        batch=batch,
    )
    chunked_total, chunked_variables = training._weighted_losses(
        chunked_prediction,
        target,
        batch=batch,
        grid_chunk_size=4,
    )
    (full_total + full_variables.sum()).backward()
    (chunked_total + chunked_variables.sum()).backward()

    torch.testing.assert_close(chunked_total, full_total, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(chunked_variables, full_variables, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(
        chunked_prediction.grad,
        full_prediction.grad,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_preparation_uses_requested_levels_and_sample_weights():
    levels = np.asarray([50, 100], dtype=np.int32)
    input_coords = {
        "batch": [0],
        "time": [-6, 0],
        "lat": np.asarray([0.5, -0.5], dtype=np.float32),
        "lon": np.asarray([0.0], dtype=np.float32),
        "level": levels,
    }
    target_coords = {**input_coords, "time": [6, 12]}
    inputs = xr.Dataset(
        {
            "state": xr.DataArray(
                np.full((1, 2, 2, 1, 2), 4.0, dtype=np.float32),
                dims=("batch", "time", "lat", "lon", "level"),
                coords=input_coords,
            ),
            "forcing": xr.DataArray(
                np.zeros((1, 2, 2, 1), dtype=np.float32),
                dims=("batch", "time", "lat", "lon"),
                coords={key: input_coords[key] for key in ("batch", "time", "lat", "lon")},
            ),
        }
    )
    targets = xr.Dataset(
        {
            "state": xr.DataArray(
                np.full((1, 2, 2, 1, 2), 8.0, dtype=np.float32),
                dims=("batch", "time", "lat", "lon", "level"),
                coords=target_coords,
            )
        }
    )
    forcings = xr.Dataset(
        {
            "forcing": xr.DataArray(
                np.zeros((1, 2, 2, 1), dtype=np.float32),
                dims=("batch", "time", "lat", "lon"),
                coords={key: target_coords[key] for key in ("batch", "time", "lat", "lon")},
            )
        }
    )
    stats_levels = np.asarray([1, 50, 100], dtype=np.int32)
    stats = SimpleNamespace(
        mean_by_level=xr.Dataset(
            {
                "state": xr.DataArray(
                    np.zeros(3, dtype=np.float32),
                    dims=("level",),
                    coords={"level": stats_levels},
                )
            }
        ),
        stddev_by_level=xr.Dataset(
            {
                "state": xr.DataArray(
                    np.asarray([9.0, 2.0, 4.0], dtype=np.float32),
                    dims=("level",),
                    coords={"level": stats_levels},
                )
            }
        ),
        diffs_stddev_by_level=xr.Dataset(
            {
                "state": xr.DataArray(
                    np.asarray([9.0, 1.0, 1.0], dtype=np.float32),
                    dims=("level",),
                    coords={"level": stats_levels},
                )
            }
        ),
    )
    config = SimpleNamespace(
        input_variables=("state", "forcing"),
        target_variables=("state",),
        forcing_variables=("forcing",),
    )

    prepared = training.prepare_autoregressive_batch(
        inputs,
        targets,
        forcings,
        config=config,
        stats=stats,
        input_device=torch.device("cpu"),
        storage_device=torch.device("cpu"),
        per_variable_weights={"state": 2.0},
        sample_weights=[0.5],
    )

    assert prepared.rollout_steps == 2
    assert prepared.feedback_updates[0].residual_to_absolute.tolist() == [0.5, 0.25]
    assert prepared.sample_weights.tolist() == [0.5]
    # Pressure weights [2/3, 4/3] are averaged across two levels, then the
    # configured variable multiplier of two is applied.
    assert prepared.channel_weights.tolist() == pytest.approx([2 / 3, 4 / 3])
