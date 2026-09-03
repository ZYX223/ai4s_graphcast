from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
if str(GRAPHCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_ROOT))

from tools.diagnose_bf16_stages import tensor_error_metrics  # noqa: E402


def test_tensor_error_metrics_reports_relative_error_and_dtypes() -> None:
    reference = torch.tensor([3.0, 4.0], dtype=torch.float32)
    candidate = torch.zeros(2, dtype=torch.bfloat16)

    result = tensor_error_metrics(reference, candidate)

    assert result == {
        "candidate_dtype": "torch.bfloat16",
        "finite": True,
        "max_abs": 4.0,
        "reference_dtype": "torch.float32",
        "reference_rms": pytest.approx((12.5) ** 0.5),
        "relative_rmse": 1.0,
        "rmse": pytest.approx((12.5) ** 0.5),
        "shape": [2],
    }


def test_tensor_error_metrics_rejects_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        tensor_error_metrics(torch.zeros(2), torch.zeros(1, 2))
