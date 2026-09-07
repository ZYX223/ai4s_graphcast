from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat import training_recipe


class ToyParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.ones(3, 2))
        self.vector = torch.nn.Parameter(torch.ones(3))
        self.scalar = torch.nn.Parameter(torch.ones(()))


def test_adamw_preserves_zero_lr_and_matrix_only_decay():
    model = ToyParameters()

    optimizer = training_recipe.build_graphcast_adamw(model, lr=0.0)

    assert optimizer.defaults["lr"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    decay_by_parameter = {
        id(parameter): float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert decay_by_parameter[id(model.matrix)] == 0.1
    assert decay_by_parameter[id(model.vector)] == 0.0
    assert decay_by_parameter[id(model.scalar)] == 0.0


@pytest.mark.parametrize("learning_rate", [-1.0, float("nan"), float("inf")])
def test_adamw_rejects_invalid_learning_rates(learning_rate):
    with pytest.raises(ValueError, match="finite and non-negative"):
        training_recipe.build_graphcast_adamw(ToyParameters(), lr=learning_rate)


def test_gradient_clip_reports_preclip_norm_and_caps_global_norm():
    model = ToyParameters()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 20.0)

    preclip_norm = training_recipe.clip_graphcast_gradients(model)
    postclip_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
    )

    assert preclip_norm > training_recipe.GRAD_CLIP_MAX_NORM
    assert float(postclip_norm) <= training_recipe.GRAD_CLIP_MAX_NORM + 1.0e-4
