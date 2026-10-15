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

import importlib.util
from pathlib import Path
import sys


AUDIT_PATH = Path(__file__).parents[1] / "tools/check_triton_only_target_ops.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "_flaggems_test_triton_only_audit", AUDIT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
audit_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = audit_module
MODULE_SPEC.loader.exec_module(audit_module)

audit_paths = audit_module.audit_paths

REPOSITORY_ROOT = Path(__file__).parents[1]
TARGET_OPERATOR_PATHS = [
    REPOSITORY_ROOT / "src/flag_gems/ops/addmm.py",
    REPOSITORY_ROOT / "src/flag_gems/ops/add.py",
    REPOSITORY_ROOT / "src/flag_gems/fused/addmm_silu.py",
    REPOSITORY_ROOT / "src/flag_gems/fused/add_add_silu.py",
    REPOSITORY_ROOT / "src/flag_gems/fused/silu_addmm.py",
    REPOSITORY_ROOT / "src/flag_gems/fused/add_add_silu_addmm.py",
    REPOSITORY_ROOT / "src/flag_gems/fused/post_layernorm_residual.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_ascend/ops/addmm.py",
    REPOSITORY_ROOT / "src/flag_gems/runtime/backend/_ascend/ops/add.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_ascend/fused/addmm_silu.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_ascend/fused/add_add_silu.py",
    REPOSITORY_ROOT / "src/flag_gems/runtime/backend/_hygon/ops/addmm.py",
    REPOSITORY_ROOT / "src/flag_gems/runtime/backend/_hygon/ops/index_add.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_hygon/fused/add_add_silu.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_hygon/fused/addmm_silu.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_hygon/fused/gather_gather_add_silu.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_hygon/fused/post_layernorm_residual.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_mthreads/ops/addmm.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_mthreads/fused/add_add_silu.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_mthreads/fused/packed_mlp.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_mthreads/fused/post_layernorm_residual.py",
    REPOSITORY_ROOT
    / "src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py",
]


def test_audit_rejects_captured_native_compute_routes(tmp_path):
    source = tmp_path / "addmm.py"
    source.write_text(
        "import torch\n"
        "def addmm(a, b, c):\n"
        "    kernel = torch.library.get_kernel('aten::addmm', 'PrivateUse1')\n"
        "    return kernel.call_boxed(a, b, c)\n"
    )

    violations = audit_paths([source])

    assert {item.symbol for item in violations} == {
        "kernel.call_boxed",
        "torch.library.get_kernel",
    }


def test_audit_rejects_redispatch_and_torch_compute_aliases(tmp_path):
    source = tmp_path / "routes.py"
    source.write_text(
        "import torch as pt\n"
        "from torch import addmm as eager_addmm\n"
        "from torch.library import get_kernel as capture\n"
        "def routes(a, b, c, keyset):\n"
        "    a.redispatch(keyset, b)\n"
        "    eager_addmm(a, b, c)\n"
        "    pt.ops.npu.npu_linear.default(a, b, c)\n"
        "    return capture('aten::addmm', 'PrivateUse1')\n"
    )

    violations = audit_paths([source])

    assert {item.symbol for item in violations} == {
        "a.redispatch",
        "torch.addmm",
        "torch.library.get_kernel",
        "torch.ops.npu.npu_linear.default",
    }


def test_audit_rejects_torch_and_vendor_target_compute(tmp_path):
    source = tmp_path / "vendor.py"
    source.write_text(
        "import torch\n"
        "import torch_npu\n"
        "import rocblas\n"
        "def routes(a, b, c):\n"
        "    torch.mm(a, b)\n"
        "    torch.matmul(a, b)\n"
        "    torch.add(a, b)\n"
        "    torch.ops.aten.addmm.default(c, a, b)\n"
        "    rocblas.gemm(a, b, c)\n"
        "    return torch_npu.npu_linear(a, b, c)\n"
    )

    violations = audit_paths([source])

    assert {item.symbol for item in violations} == {
        "torch.add",
        "torch.matmul",
        "torch.mm",
        "torch.ops.aten.addmm.default",
        "rocblas.gemm",
        "torch_npu.npu_linear",
    }


def test_audit_accepts_triton_and_metadata_only_torch(tmp_path):
    source = tmp_path / "addmm.py"
    source.write_text(
        "import torch\n"
        "import triton\n"
        "def addmm(a, m, n):\n"
        "    out = torch.empty((m, n), device=a.device, dtype=a.dtype)\n"
        "    view = a.contiguous().broadcast_to((m, n))\n"
        "    kernel[(triton.cdiv(m, 32),)](view, out)\n"
        "    return out\n"
    )

    assert audit_paths([source]) == []


def test_audit_recurses_over_python_only_and_sorts_results(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.py").write_text("import torch\ntorch.add(1, 2)\n")
    (nested / "a.py").write_text("x.redispatch(keys, 1)\n")
    (nested / "ignored.txt").write_text("torch.mm(a, b)\n")

    violations = audit_paths([tmp_path])

    assert [(item.path.name, item.line, item.symbol) for item in violations] == [
        ("a.py", 1, "x.redispatch"),
        ("b.py", 2, "torch.add"),
    ]


def test_all_production_addmm_and_add_targets_are_triton_only():
    assert all(path.is_file() for path in TARGET_OPERATOR_PATHS)
    assert audit_paths(TARGET_OPERATOR_PATHS) == []
