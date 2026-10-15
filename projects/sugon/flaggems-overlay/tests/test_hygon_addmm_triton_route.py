# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from pathlib import Path

import pytest
import torch

import flag_gems


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "src/flag_gems/runtime/backend/_hygon/ops/addmm.py"
)
INIT_PATH = SOURCE_PATH.parent / "__init__.py"


def _load_pure_function(name):
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "K,stride_bk,stride_bn,expected",
    [
        (4, 1, 1536, "skinny_k"),
        (184, 1, 1536, "k184"),
        (512, 512, 1, "k512_rhs_row"),
        (512, 1, 512, "k512_rhs_k_contiguous"),
        (512, 1, 1536, "k512_rhs_k_contiguous"),
        (1024, 1, 1536, "k1024_plus"),
        (384, 1, 384, "general"),
    ],
)
def test_hygon_addmm_classification_is_shape_class_based(
    K, stride_bk, stride_bn, expected
):
    classify = _load_pure_function("classify_hygon_addmm")

    assert classify(K, stride_bk, stride_bn) == expected


@pytest.mark.parametrize(
    "kernel_class,fp32_operands,fast_enabled,expected",
    [
        ("skinny_k", True, True, "ieee"),
        ("k184", True, True, "ieee"),
        ("k512_rhs_row", True, True, "fp16"),
        ("k512_rhs_k_contiguous", True, True, "fp16"),
        ("k1024_plus", True, True, "ieee"),
        ("general", True, True, "ieee"),
        ("k512_rhs_row", False, True, "ieee"),
        ("k512_rhs_row", True, False, "ieee"),
    ],
)
def test_hygon_fast_fp32_allowlist_is_limited_to_k512(
    kernel_class, fp32_operands, fast_enabled, expected
):
    select = _load_pure_function("select_hygon_input_precision")

    assert select(kernel_class, fp32_operands, fast_enabled) == expected


def test_hygon_k512_fast_path_downcasts_inside_owned_triton_kernel():
    source = SOURCE_PATH.read_text()
    general_kernel = source.split("def addmm_kernel(", 1)[1].split(
        "\ndef addmm_skinny_k_kernel", 1
    )[0]

    assert 'if INPUT_PRECISION == "fp16":' in general_kernel
    assert "a = a.to(tl.float16)" in general_kernel
    assert "b = b.to(tl.float16)" in general_kernel


def test_hygon_backend_exports_all_addmm_overloads():
    source = INIT_PATH.read_text()

    assert "from .addmm import addmm, addmm_dtype, addmm_dtype_out, addmm_out" in source
    for name in ("addmm", "addmm_dtype", "addmm_dtype_out", "addmm_out"):
        assert f'"{name}"' in source


def test_hygon_addmm_has_no_native_or_exact_call_routing():
    source = SOURCE_PATH.read_text()

    for forbidden in (
        "redispatch",
        "get_kernel",
        "call_boxed",
        "torch.addmm",
        "torch.ops.aten.addmm",
        "call_index",
        "131072",
        "327660",
    ):
        assert forbidden not in source


def test_hygon_skinny_k_uses_padded_dot_instead_of_scalar_outer_products():
    source = SOURCE_PATH.read_text()
    skinny_source = source.split("def addmm_skinny_k_kernel(", 1)[1].split(
        "\ndef _addmm_impl", 1
    )[0]

    assert "_accumulate_dot(" in skinny_source
    assert "tl.dot(" in source
    assert "accumulator += a[:, None] * b[None, :]" not in skinny_source


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon-only test")
def test_hygon_public_addmm_resolves_to_backend_triton_source():
    assert flag_gems.addmm.__module__ == "_hygon.ops.addmm"
    M, N, K = 257, 512, 184
    mat1 = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device)
    mat2 = torch.randn((N, K), dtype=torch.float32, device=flag_gems.device).t()
    bias = torch.randn((N,), dtype=torch.float32, device=flag_gems.device)
    reference = torch.addmm(bias, mat1, mat2)
    with flag_gems.use_gems():
        result = torch.addmm(bias, mat1, mat2)

    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon-only test")
def test_hygon_k512_fast_fp32_path_keeps_fp32_output_and_bounded_error():
    torch.manual_seed(20260910)
    mat1 = torch.randn((257, 512), dtype=torch.float32, device=flag_gems.device)
    mat2 = torch.randn((512, 512), dtype=torch.float32, device=flag_gems.device)
    bias = torch.randn((512,), dtype=torch.float32, device=flag_gems.device)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        strict = flag_gems.addmm(bias, mat1, mat2)
        torch.backends.cuda.matmul.allow_tf32 = True
        fast = flag_gems.addmm(bias, mat1, mat2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    assert fast.dtype == torch.float32
    relative_l2 = torch.linalg.vector_norm(fast - strict) / torch.linalg.vector_norm(
        strict
    ).clamp_min(torch.finfo(torch.float32).tiny)
    assert relative_l2.item() < 5e-4
