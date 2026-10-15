"""Check the inference-only kernel, separately from the immutable source tests."""
import sys

import pytest
import torch
import flag_gems


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
@pytest.mark.parametrize("shape", [(19, 512), (3, 5, 512)])
@pytest.mark.parametrize("affine", [False, True])
@torch.no_grad()
def test_post_layer_norm_inference_uses_kernel_without_fallback(monkeypatch, shape, affine):
    module = sys.modules[flag_gems.post_layer_norm_residual.__module__]
    torch.manual_seed(20261015)
    x = torch.randn(shape, device=flag_gems.device)
    residual = torch.randn_like(x)
    weight = torch.randn(512, device=flag_gems.device) if affine else None
    bias = torch.randn_like(weight) if affine else None
    expected = torch.nn.functional.layer_norm(x, (512,), weight, bias, 1e-5) + residual

    def forbid_fallback(*args, **kwargs):
        pytest.fail("inference-only test reached the unfused fallback")

    monkeypatch.setattr(module, "_common_post_layer_norm_residual", forbid_fallback)
    assert module._can_use_flat_fp32_kernel(x, residual, (512,), weight, bias)
    actual = flag_gems.post_layer_norm_residual(x, residual, (512,), weight, bias, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
