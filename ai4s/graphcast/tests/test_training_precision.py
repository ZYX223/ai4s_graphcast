from __future__ import annotations

import pytest
import torch

from graphcast_compat.precision import PrecisionLinear, configure_model_precision


def test_precision_linear_keeps_fp32_storage_and_forward_contract() -> None:
    layer = PrecisionLinear(3, 2)
    inputs = torch.randn(4, 3, dtype=torch.float32)

    expected = torch.nn.functional.linear(inputs, layer.weight, layer.bias)
    actual = layer(inputs)

    assert layer.weight.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert configure_model_precision(layer, "fp32", fp8_backend="native") == 1


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp8"])
def test_training_precision_rejects_non_fp32_expansion(precision: str) -> None:
    with pytest.raises(ValueError, match="unsupported training precision"):
        configure_model_precision(PrecisionLinear(1, 1), precision)
