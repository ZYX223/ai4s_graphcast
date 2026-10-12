"""Only immutable raw data may be staged outside checkpoint replay."""
from dataclasses import replace

import pytest
import torch

from test_autoregressive_training import _scalar_batch, ScalarResidualModel
from graphcast_compat import autoregressive_training as training


@pytest.mark.parametrize("implementation", ["reentrant", "non_reentrant"])
def test_staged_raw_targets_preserve_bptt_and_are_not_overwritten(implementation):
    prepare = getattr(training, "stage_autoregressive_batch", None)
    assert callable(prepare), "missing once-per-microbatch device staging"
    batch = _scalar_batch()
    batch = replace(batch, initial_input_features=batch.initial_input_features.double())
    original = batch.target_absolute_features.clone()
    staged = prepare(batch, input_device=torch.device("cpu"),
                     output_device=torch.device("cpu"), dtype=torch.float64)
    assert staged.target_absolute_features.dtype == torch.float64
    assert staged.forcing_features.dtype == torch.float64
    assert staged.feedback_updates[0].residual_to_absolute.dtype == torch.float64
    assert batch.target_absolute_features.dtype == torch.float32
    model = ScalarResidualModel(0.5).double()
    result = training.autoregressive_loss(model, staged, checkpoint_segments=2,
                                         checkpoint_implementation=implementation)
    result.loss.backward()
    assert result.loss.item() == pytest.approx(2.125)
    assert model.residual.grad.item() == pytest.approx(-4.5)
    torch.testing.assert_close(staged.target_absolute_features, original.double(), rtol=0, atol=0)
    torch.testing.assert_close(batch.target_absolute_features, original, rtol=0, atol=0)


def test_staging_does_not_cache_across_samples_or_replace_initial_state():
    prepare = getattr(training, "stage_autoregressive_batch", None)
    assert callable(prepare), "missing per-call data staging"
    first = _scalar_batch()
    second = replace(first, target_absolute_features=first.target_absolute_features + 10)
    kwargs = dict(input_device=torch.device("cpu"), output_device=torch.device("cpu"), dtype=torch.float64)
    one, two = prepare(first, **kwargs), prepare(second, **kwargs)
    assert one.initial_input_features is first.initial_input_features
    torch.testing.assert_close(two.target_absolute_features - one.target_absolute_features,
                               torch.full_like(one.target_absolute_features, 10))


def test_static_graph_staging_moves_nonpersistent_buffers_not_parameters():
    from graphcast_compat.training_model import GoogleSmallCompatibleModel, GoogleSmallConfig
    model = GoogleSmallCompatibleModel(GoogleSmallConfig(latent_size=4, gnn_msg_steps=2))
    model.register_buffer("grid2mesh_senders", torch.tensor([1, 0, 1]), persistent=False)
    prepare = getattr(model, "stage_static_graph", None)
    assert callable(prepare), "static topology still uploads on each replay"
    prepare(torch.device("meta"))
    assert model.grid2mesh_senders.device.type == "meta"
    assert model.grid2mesh_senders.dtype == torch.int64
    assert next(model.parameters()).device.type == "cpu"
    assert "grid2mesh_senders" not in model.state_dict()
