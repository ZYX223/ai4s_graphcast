from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from graphcast_compat.google_small import (
    GoogleSmallCompatibleModel as InferenceModel,
)
from graphcast_compat.google_small import GoogleSmallConfig
from graphcast_compat.precision import PrecisionLinear
from graphcast_compat.training_model import (
    GoogleSmallCompatibleModel as TrainingModel,
)
from graphcast_compat.training_model import GoogleSmallConfig as TrainingConfig


def _tiny_config() -> GoogleSmallConfig:
    return GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=2,
        node_input_dim=6,
        edge_input_dim=4,
        output_dim=3,
    )


def test_training_model_preserves_seeded_inference_parameter_contract() -> None:
    config = _tiny_config()
    training_config = TrainingConfig(**asdict(config))

    torch.manual_seed(27)
    inference = InferenceModel(config)
    torch.manual_seed(27)
    training = TrainingModel(training_config)

    assert list(inference.state_dict()) == list(training.state_dict())
    for name, value in inference.state_dict().items():
        torch.testing.assert_close(value, training.state_dict()[name], rtol=0, atol=0)


def test_training_model_selects_precision_linear_without_changing_inference() -> None:
    config = _tiny_config()
    inference = InferenceModel(config)
    training = TrainingModel(TrainingConfig(**asdict(config)))

    inference_linears = [module for module in inference.modules() if type(module) is nn.Linear]
    training_linears = [
        module for module in training.modules() if isinstance(module, PrecisionLinear)
    ]

    assert inference_linears
    assert training_linears
    assert not any(isinstance(module, PrecisionLinear) for module in inference.modules())
