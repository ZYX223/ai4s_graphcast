# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib
from pathlib import Path

import pytest
import torch

import flag_gems


ROOT = Path(__file__).parents[1]
HYGON_INDEX_ADD = (
    ROOT / "src/flag_gems/runtime/backend/_hygon/ops/index_add.py"
)
TRITON_ONLY_AUDIT = ROOT / "tests/test_triton_only_target_ops.py"


def test_hygon_index_add_uses_owned_triton_bounds_check():
    source = HYGON_INDEX_ADD.read_text()

    assert "def _index_bounds_kernel(" in source
    assert "def use_trusted_index_add_inference():" in source
    assert "_index_is_in_bounds_on_device" in source
    assert "def _index_add_sorted_run_kernel(" in source
    assert "def _index_uniform_run_length_three_kernel(" in source
    assert str(HYGON_INDEX_ADD.relative_to(ROOT)) in TRITON_ONLY_AUDIT.read_text()
    for forbidden in (
        "torch.ops.aten",
        "redispatch",
        "_FALLBACK_KEYSET",
        "rocblas",
    ):
        assert forbidden not in source


def _hygon_index_add_module():
    return importlib.import_module(
        "flag_gems.runtime.backend._hygon.ops.index_add"
    )


def _make_case():
    index_root = torch.arange(
        32, dtype=torch.int64, device=flag_gems.device
    ) // 2
    index_views = (index_root[:16], index_root[16:])
    inp = torch.zeros((1, 32, 8), device=flag_gems.device)
    src = torch.ones((1, 16, 8), device=flag_gems.device)
    return inp, index_root, index_views, src


def _record_bounds_checks(monkeypatch, module):
    original = module._index_is_in_bounds_on_device
    reads = []

    def record(index, upper_bound):
        reads.append((index.numel(), upper_bound))
        return original(index, upper_bound)

    monkeypatch.setattr(module, "_index_is_in_bounds_on_device", record)
    return reads


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
@pytest.mark.parametrize("trusted", [False, True])
def test_hygon_trusted_index_scope_reuses_complete_root_validation(
    monkeypatch, trusted
):
    module = _hygon_index_add_module()
    inp, _, index_views, src = _make_case()
    reads = _record_bounds_checks(monkeypatch, module)
    scope = (
        module.use_trusted_index_add_inference()
        if trusted
        else __import__("contextlib").nullcontext()
    )

    with torch.inference_mode(), scope:
        for view in index_views:
            module.index_add_(inp.clone(), 1, view, src)

    expected = [(32, 32)] if trusted else [(16, 32), (16, 32)]
    assert reads == expected


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
def test_hygon_trusted_index_scope_checks_whole_root_before_writing(monkeypatch):
    module = _hygon_index_add_module()
    inp, index_root, index_views, src = _make_case()
    index_root[-1] = inp.size(1)
    before = inp.clone()
    reads = _record_bounds_checks(monkeypatch, module)

    with torch.inference_mode(), module.use_trusted_index_add_inference():
        with pytest.raises(
            AssertionError, match=r"0 <= index < self\.size\(dim\)"
        ):
            module.index_add_(inp, 1, index_views[0], src)

    assert reads == [(32, 32)]
    torch.testing.assert_close(inp, before, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
def test_hygon_trusted_index_scope_falls_back_after_mutation(monkeypatch):
    module = _hygon_index_add_module()
    inp, index_root, index_views, src = _make_case()
    reads = _record_bounds_checks(monkeypatch, module)

    with torch.no_grad(), module.use_trusted_index_add_inference():
        module.index_add_(inp.clone(), 1, index_views[0], src)
        index_root[0] = 1
        module.index_add_(inp.clone(), 1, index_views[0], src)
        module.index_add_(inp.clone(), 1, index_views[1], src)

    assert reads == [(32, 32), (16, 32), (16, 32)]


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon only")
def test_hygon_exact_three_receiver_runs_use_non_atomic_kernel(monkeypatch):
    module = _hygon_index_add_module()
    monkeypatch.setattr(module, "_SORTED_RUN_INDEX_MIN_ELEMENTS", 1)
    index_root = torch.arange(
        12, dtype=torch.int64, device=flag_gems.device
    ) // 3
    index = index_root[1:11]
    inp = torch.randn((1, 4, 8), device=flag_gems.device)
    src = torch.randn((1, index.numel(), 8), device=flag_gems.device)
    expected = torch.index_add(inp, 1, index, src)

    def atomic_path_must_not_run(*args, **kwargs):
        raise AssertionError("atomic suffix path unexpectedly selected")

    monkeypatch.setattr(
        module, "_run_contiguous_suffix_path", atomic_path_must_not_run
    )
    with torch.inference_mode(), module.use_trusted_index_add_inference():
        actual = module.index_add(inp, 1, index, src)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
