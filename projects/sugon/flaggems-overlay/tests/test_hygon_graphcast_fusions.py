# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import pytest
import torch

import flag_gems


ROOT = Path(__file__).parents[1]
HYGON_FUSION = (
    ROOT / "src/flag_gems/runtime/backend/_hygon/fused/add_add_silu.py"
)
HYGON_POST_LAYERNORM = (
    ROOT
    / "src/flag_gems/runtime/backend/_hygon/fused/post_layernorm_residual.py"
)
HYGON_ADDMM = ROOT / "src/flag_gems/runtime/backend/_hygon/ops/addmm.py"
HYGON_ADDMM_SILU = (
    ROOT / "src/flag_gems/runtime/backend/_hygon/fused/addmm_silu.py"
)
HYGON_GATHER_ADD_SILU = (
    ROOT
    / "src/flag_gems/runtime/backend/_hygon/fused/gather_gather_add_silu.py"
)
HYGON_FUSED_INIT = ROOT / "src/flag_gems/runtime/backend/_hygon/fused/__init__.py"
TRITON_ONLY_AUDIT = ROOT / "tests/test_triton_only_target_ops.py"


def test_hygon_add_add_silu_is_one_owned_triton_kernel():
    source = HYGON_FUSION.read_text()

    assert "@triton.jit" in source
    assert "def _add_add_silu_kernel(" in source
    assert source.count("tl.store(") == 1
    assert source.count("tl.load(") == 3
    assert "from .add_add_silu import add_add_silu" in HYGON_FUSED_INIT.read_text()
    assert str(HYGON_FUSION.relative_to(ROOT)) in TRITON_ONLY_AUDIT.read_text()
    for forbidden in (
        "torch.add",
        "torch.addmm",
        "torch.ops.aten",
        "torch.library.get_kernel",
        "redispatch",
        "rocblas",
    ):
        assert forbidden not in source


def test_hygon_post_layernorm_residual_is_one_owned_triton_kernel():
    source = HYGON_POST_LAYERNORM.read_text()

    assert "@triton.jit" in source
    assert "def _post_layer_norm_residual_kernel(" in source
    assert source.count("tl.store(") == 1
    assert "normalized + residual" in source
    assert (
        "from .post_layernorm_residual import post_layer_norm_residual"
        in HYGON_FUSED_INIT.read_text()
    )
    assert str(HYGON_POST_LAYERNORM.relative_to(ROOT)) in TRITON_ONLY_AUDIT.read_text()
    for forbidden in (
        "torch.add",
        "torch.ops.aten",
        "torch.library.get_kernel",
        "redispatch",
        "rocblas",
    ):
        assert forbidden not in source


def test_hygon_addmm_silu_is_an_owned_triton_epilogue():
    source = HYGON_ADDMM.read_text()
    fused_source = HYGON_ADDMM_SILU.read_text()

    assert "FUSE_SILU: tl.constexpr" in source
    assert "if FUSE_SILU:" in source
    assert "def addmm_silu(" in source
    assert "from .addmm_silu import addmm_silu" in HYGON_FUSED_INIT.read_text()
    assert str(HYGON_ADDMM_SILU.relative_to(ROOT)) in TRITON_ONLY_AUDIT.read_text()
    for forbidden in (
        "torch.addmm",
        "torch.ops.aten",
        "torch.library.get_kernel",
        "redispatch",
        "rocblas",
    ):
        assert forbidden not in fused_source


def test_hygon_gather_gather_add_silu_is_one_owned_triton_kernel():
    source = HYGON_GATHER_ADD_SILU.read_text()

    assert "@triton.jit" in source
    assert "def _gather_gather_add_silu_kernel(" in source
    assert source.count("tl.store(") == 1
    assert (
        "from .gather_gather_add_silu import gather_gather_add_silu"
        in HYGON_FUSED_INIT.read_text()
    )
    assert str(HYGON_GATHER_ADD_SILU.relative_to(ROOT)) in TRITON_ONLY_AUDIT.read_text()
    for forbidden in (
        "torch.index_select",
        "torch.ops.aten",
        "torch.library.get_kernel",
        "redispatch",
        "rocblas",
    ):
        assert forbidden not in source


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
@pytest.mark.parametrize("shape", [(17,), (19, 37), (3, 5, 7)])
def test_hygon_add_add_silu_matches_ordered_fp32_composition(shape):
    torch.manual_seed(20260910)
    addend = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    residual_a = torch.randn_like(addend)
    residual_b = torch.randn_like(addend)

    expected = flag_gems.silu(
        flag_gems.add(flag_gems.add(addend, residual_a), residual_b)
    )
    actual = flag_gems.add_add_silu(addend, residual_a, residual_b)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
@pytest.mark.parametrize("shape", [(19, 512), (3, 5, 512)])
@pytest.mark.parametrize("affine", [False, True])
def test_hygon_post_layernorm_residual_matches_fp32_composition(shape, affine):
    torch.manual_seed(20260910)
    x = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = (
        torch.randn((shape[-1],), device=flag_gems.device, dtype=torch.float32)
        if affine
        else None
    )
    bias = torch.randn_like(weight) if affine else None

    normalized, _, _ = flag_gems.native_layer_norm(
        x, (shape[-1],), weight, bias, 1e-5
    )
    expected = flag_gems.add(normalized, residual)
    actual = flag_gems.post_layer_norm_residual(
        x, residual, (shape[-1],), weight, bias, 1e-5
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
@pytest.mark.parametrize("shape", [(19, 37, 23), (257, 512, 512)])
def test_hygon_addmm_silu_matches_materialized_fp32_boundary(shape):
    m, n, k = shape
    torch.manual_seed(20260910)
    mat1 = torch.randn((m, k), device=flag_gems.device)
    mat2 = torch.randn((k, n), device=flag_gems.device)
    bias = torch.randn((n,), device=flag_gems.device)

    expected = flag_gems.silu(flag_gems.addmm(bias, mat1, mat2))
    actual = flag_gems.addmm_silu(bias, mat1, mat2)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
def test_hygon_gather_gather_add_silu_matches_ordered_composition():
    torch.manual_seed(20260910)
    hidden = torch.randn((19, 37), device=flag_gems.device)
    sender_projection = torch.randn((11, 37), device=flag_gems.device)
    receiver_projection = torch.randn((13, 37), device=flag_gems.device)
    senders = torch.randint(0, 11, (19,), device=flag_gems.device)
    receivers = torch.randint(0, 13, (19,), device=flag_gems.device)

    expected = torch.nn.functional.silu(
        hidden + sender_projection[senders] + receiver_projection[receivers]
    )
    actual = flag_gems.gather_gather_add_silu(
        hidden,
        sender_projection,
        senders,
        receiver_projection,
        receivers,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
