from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat import backend


def test_hygon_fp32_uses_legacy_switches_without_mixing_precision_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matmul = SimpleNamespace(allow_tf32=True, fp32_precision="tf32")
    cudnn = SimpleNamespace(
        allow_tf32=True,
        conv=SimpleNamespace(fp32_precision="tf32"),
    )
    monkeypatch.setattr(
        backend.torch,
        "backends",
        SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
    )
    monkeypatch.setattr(
        backend.torch,
        "version",
        SimpleNamespace(hip="6.3.0", cuda=None),
    )

    mode = backend.configure_float32_math(False, "cuda")

    assert mode == "ieee_fp32"
    assert matmul.allow_tf32 is False
    assert cudnn.allow_tf32 is False
    assert matmul.fp32_precision == "tf32"
    assert cudnn.conv.fp32_precision == "tf32"


def test_ascend_fp32_disables_and_verifies_hf32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npu = SimpleNamespace(
        matmul=SimpleNamespace(allow_hf32=True),
        conv=SimpleNamespace(allow_hf32=True),
    )
    monkeypatch.setattr(backend, "_import_torch_npu", lambda: None)
    monkeypatch.setattr(backend.torch, "npu", npu, raising=False)

    mode = backend.configure_float32_math(False, "npu")

    assert mode == "ieee_fp32"
    assert npu.matmul.allow_hf32 is False
    assert npu.conv.allow_hf32 is False
