"""FP32-only precision-aware Linear layers for GraphCast training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .recompute_flops import record_linear_recompute


class PrecisionLinear(nn.Linear):
    """Linear with recomputation accounting and compatible state keys."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        record_linear_recompute(inputs, self.weight)
        return F.linear(inputs, self.weight, self.bias)


def configure_model_precision(
    model: nn.Module,
    precision: str,
    *,
    fp8_backend: str = "native",
) -> int:
    """Validate the approved FP32 policy and count managed linears.

    The fp8_backend keyword remains accepted for source runner call
    compatibility; it has no effect because this integration intentionally
    exposes no mixed- or low-precision training modes.
    """

    del fp8_backend
    if precision != "fp32":
        raise ValueError(
            f"unsupported training precision: {precision!r}; only 'fp32' is approved"
        )
    return sum(
        isinstance(module, PrecisionLinear)
        for module in model.modules()
    )
