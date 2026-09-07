"""Published GraphCast optimizer settings shared by compatible runtimes."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


ADAMW_BETAS = (0.9, 0.95)
MATRIX_WEIGHT_DECAY = 0.1
GRAD_CLIP_MAX_NORM = 32.0
FULL_OPERATIONAL_MODEL_CONTRACT = {
    "resolution": 0.25,
    "mesh_size": 6,
    "latent_size": 512,
    "hidden_layers": 1,
    "gnn_msg_steps": 16,
    "radius_query_fraction_edge_length": 0.5999912857713345,
    "mesh2grid_edge_normalization_factor": 0.6180338738074472,
    "node_input_dim": 184,
    "edge_input_dim": 4,
    "output_dim": 83,
    "extra_mesh2grid_final_output_heads": {"mesh_nodes": 83},
}


def validate_full_operational_model_config(config: Any) -> None:
    """Reject reduced GraphCast architectures in official training runs."""

    mismatches = []
    for name, expected in FULL_OPERATIONAL_MODEL_CONTRACT.items():
        actual = getattr(config, name, None)
        matches = (
            math.isclose(float(actual), expected)
            if isinstance(expected, float) and actual is not None
            else actual == expected
        )
        if not matches:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError(
            "official GraphCast training requires the complete operational "
            "0.25-degree model; reduced network models are forbidden: "
            + ", ".join(mismatches)
        )


def build_graphcast_adamw(model: nn.Module, *, lr: float) -> torch.optim.AdamW:
    """Build the paper's AdamW recipe with decay only on weight matrices."""

    learning_rate = float(lr)
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("GraphCast AdamW learning rate must be finite and non-negative")
    matrix_parameters = []
    other_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            matrix_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    if not matrix_parameters and not other_parameters:
        raise ValueError("GraphCast optimizer received no trainable parameters")
    groups = []
    if matrix_parameters:
        groups.append({"params": matrix_parameters, "weight_decay": MATRIX_WEIGHT_DECAY})
    if other_parameters:
        groups.append({"params": other_parameters, "weight_decay": 0.0})
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=ADAMW_BETAS,
        weight_decay=0.0,
    )


def clip_graphcast_gradients(model: nn.Module) -> float:
    """Apply the paper's global max-norm clip and return the pre-clip norm."""

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=GRAD_CLIP_MAX_NORM,
        error_if_nonfinite=True,
    )
    return float(total_norm.detach().item())


def graphcast_optimizer_metadata(*, lr: float) -> dict[str, Any]:
    return {
        "name": "AdamW",
        "lr": float(lr),
        "betas": list(ADAMW_BETAS),
        "matrix_weight_decay": MATRIX_WEIGHT_DECAY,
        "weight_decay_scope": "trainable_parameters_with_ndim_greater_than_or_equal_to_2",
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "lr_schedule": "external_current_lr",
    }
