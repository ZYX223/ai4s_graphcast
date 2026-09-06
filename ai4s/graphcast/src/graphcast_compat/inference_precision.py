"""Precision policy for device-resident GraphCast inference."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class InferencePrecisionPolicy:
    """Immutable precision boundary for one inference process."""

    requested: Literal["fp32", "tf32", "bf16"]
    effective: Literal["strict_fp32", "fast_fp32", "mixed_precision"]
    autocast_dtype: torch.dtype | None
    fast_float32_math: bool

    def context(self, device_type: str):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device_type, dtype=self.autocast_dtype)


@dataclass(frozen=True)
class PrecisionForwardResult:
    """FP32 feedback tensor plus the dtypes observed around model forward."""

    prediction: torch.Tensor
    model_input_dtype: str
    model_output_dtype: str


def resolve_inference_precision(name: str) -> InferencePrecisionPolicy:
    if name == "fp32":
        return InferencePrecisionPolicy("fp32", "strict_fp32", None, False)
    if name == "tf32":
        return InferencePrecisionPolicy("tf32", "fast_fp32", None, True)
    if name == "bf16":
        return InferencePrecisionPolicy(
            "bf16", "mixed_precision", torch.bfloat16, False
        )
    raise ValueError(f"unsupported inference precision: {name!r}")


def run_precision_forward(
    model: torch.nn.Module,
    state: torch.Tensor,
    forcing: torch.Tensor,
    *,
    policy: InferencePrecisionPolicy,
    device_type: str,
) -> PrecisionForwardResult:
    """Run one model forward and return an FP32 tensor for state feedback."""

    if state.dtype != torch.float32 or forcing.dtype != torch.float32:
        raise TypeError("persistent state and forcing must remain FP32")
    with policy.context(device_type):
        if policy.autocast_dtype is None:
            model_input = torch.cat((state, forcing), dim=-1)
        else:
            model_input = torch.cat(
                (
                    state.to(dtype=policy.autocast_dtype),
                    forcing.to(dtype=policy.autocast_dtype),
                ),
                dim=-1,
            )
        model_output = model(model_input)
    if not model_output.is_floating_point():
        raise TypeError("model prediction must be floating point")
    if policy.autocast_dtype is not None and model_output.dtype != policy.autocast_dtype:
        raise RuntimeError(
            f"BF16 autocast produced {model_output.dtype}; expected {policy.autocast_dtype}"
        )
    return PrecisionForwardResult(
        prediction=model_output.to(dtype=torch.float32),
        model_input_dtype=str(model_input.dtype),
        model_output_dtype=str(model_output.dtype),
    )
