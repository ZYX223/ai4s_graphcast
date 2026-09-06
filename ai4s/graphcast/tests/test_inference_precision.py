from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat import inference_precision  # noqa: E402


def test_fp32_policy_preserves_fp32_model_output() -> None:
    policy = inference_precision.resolve_inference_precision("fp32")
    state = torch.ones((1, 2, 1), dtype=torch.float32)
    forcing = torch.full((1, 2, 1), 2.0, dtype=torch.float32)

    result = inference_precision.run_precision_forward(
        torch.nn.Identity(),
        state,
        forcing,
        policy=policy,
        device_type="cpu",
    )

    assert policy.fast_float32_math is False
    assert result.prediction.dtype == torch.float32
    assert result.model_input_dtype == "torch.float32"
    assert result.model_output_dtype == "torch.float32"
    assert result.prediction.tolist() == [[[1.0, 2.0], [1.0, 2.0]]]


def test_tf32_policy_keeps_fp32_tensors_and_requests_fast_float32_math() -> None:
    policy = inference_precision.resolve_inference_precision("tf32")
    state = torch.ones((1, 2, 1), dtype=torch.float32)
    forcing = torch.full((1, 2, 1), 2.0, dtype=torch.float32)

    result = inference_precision.run_precision_forward(
        torch.nn.Identity(),
        state,
        forcing,
        policy=policy,
        device_type="cpu",
    )

    assert policy.requested == "tf32"
    assert policy.effective == "fast_fp32"
    assert policy.autocast_dtype is None
    assert policy.fast_float32_math is True
    assert result.model_input_dtype == "torch.float32"
    assert result.model_output_dtype == "torch.float32"
    assert result.prediction.dtype == torch.float32


def test_bf16_policy_autocasts_forward_and_returns_fp32_feedback_tensor() -> None:
    policy = inference_precision.resolve_inference_precision("bf16")
    model = torch.nn.Linear(2, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    state = torch.ones((1, 2, 1), dtype=torch.float32)
    forcing = torch.full((1, 2, 1), 2.0, dtype=torch.float32)

    result = inference_precision.run_precision_forward(
        model,
        state,
        forcing,
        policy=policy,
        device_type="cpu",
    )

    assert policy.fast_float32_math is False
    assert next(model.parameters()).dtype == torch.float32
    assert state.dtype == torch.float32
    assert forcing.dtype == torch.float32
    assert result.model_input_dtype == "torch.bfloat16"
    assert result.model_output_dtype == "torch.bfloat16"
    assert result.prediction.dtype == torch.float32
    assert result.prediction.tolist() == [[[5.0], [5.0]]]


def test_bf16_policy_casts_the_temporary_model_input_like_official_graphcast() -> None:
    class InputDtypeProbe(torch.nn.Linear):
        observed_input_dtype: torch.dtype | None = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.observed_input_dtype = value.dtype
            return super().forward(value)

    model = InputDtypeProbe(2, 1, bias=False).eval()
    result = inference_precision.run_precision_forward(
        model,
        torch.ones((1, 1, 1), dtype=torch.float32),
        torch.ones((1, 1, 1), dtype=torch.float32),
        policy=inference_precision.resolve_inference_precision("bf16"),
        device_type="cpu",
    )

    assert model.observed_input_dtype == torch.bfloat16
    assert result.model_input_dtype == "torch.bfloat16"
    assert result.model_output_dtype == "torch.bfloat16"
    assert result.prediction.dtype == torch.float32


@pytest.mark.parametrize("wrong_operand", ["state", "forcing"])
def test_forward_rejects_non_fp32_persistent_inputs(wrong_operand: str) -> None:
    state = torch.ones((1, 1, 1), dtype=torch.float32)
    forcing = torch.ones((1, 1, 1), dtype=torch.float32)
    if wrong_operand == "state":
        state = state.to(torch.bfloat16)
    else:
        forcing = forcing.to(torch.bfloat16)

    with pytest.raises(TypeError, match="state and forcing must remain FP32"):
        inference_precision.run_precision_forward(
            torch.nn.Identity(),
            state,
            forcing,
            policy=inference_precision.resolve_inference_precision("bf16"),
            device_type="cpu",
        )


def test_bf16_policy_rejects_an_fp32_only_forward() -> None:
    class Fp32OnlyForward(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.float()

    with pytest.raises(RuntimeError, match="BF16 autocast produced torch.float32"):
        inference_precision.run_precision_forward(
            Fp32OnlyForward(),
            torch.ones((1, 1, 1), dtype=torch.float32),
            torch.ones((1, 1, 1), dtype=torch.float32),
            policy=inference_precision.resolve_inference_precision("bf16"),
            device_type="cpu",
        )


def test_resolver_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="unsupported inference precision"):
        inference_precision.resolve_inference_precision("float16")
