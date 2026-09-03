from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
if str(GRAPHCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_ROOT))

from scripts.inference.compare_operational_40step_outputs import compare_outputs  # noqa: E402


def _stats(scale: float) -> tuple[xr.Dataset, xr.Dataset]:
    level = [1000]
    mean = xr.Dataset({"wind": (("level",), np.asarray([100.0], dtype=np.float32))})
    stddev = xr.Dataset(
        {"wind": (("level",), np.asarray([scale], dtype=np.float32))},
        coords={"level": level},
    )
    mean = mean.assign_coords(level=level)
    return mean, stddev


def test_comparison_uses_normalized_error_instead_of_one_physical_unit_tolerance() -> None:
    native = xr.Dataset(
        {
            "wind": (
                ("batch", "time", "level", "lat", "lon"),
                np.linspace(100.0, 110.0, 100, dtype=np.float32).reshape(
                    1, 1, 1, 1, 100
                ),
            )
        }
    )
    candidate = native.copy(deep=True)
    candidate["wind"].values[..., -1] += 0.05
    mean, stddev = _stats(scale=10.0)

    result = compare_outputs(native, candidate, mean=mean, stddev=stddev)

    assert result["status"] == "passed"
    row = result["variables"][0]
    assert row["physical_max_abs"] > 1.0e-4
    assert row["normalized_max_abs"] < 1.0e-2


def test_comparison_rejects_material_normalized_drift() -> None:
    native = xr.Dataset(
        {
            "wind": (
                ("batch", "time", "level", "lat", "lon"),
                np.asarray([[[[[100.0, 110.0]]]]], dtype=np.float32),
            )
        }
    )
    candidate = native.copy(deep=True)
    candidate["wind"].values[..., 1] += 1.0
    mean, stddev = _stats(scale=10.0)

    result = compare_outputs(native, candidate, mean=mean, stddev=stddev)

    assert result["status"] == "failed"
    assert result["variables"][0]["normalized_max_abs"] > 1.0e-2


def test_cross_precision_comparison_reports_drift_without_fp32_acceptance_gate() -> None:
    native = xr.Dataset(
        {
            "wind": (
                ("batch", "time", "level", "lat", "lon"),
                np.asarray([[[[[100.0, 110.0]]]]], dtype=np.float32),
            )
        }
    )
    candidate = native.copy(deep=True)
    candidate["wind"].values[..., 1] += 1.0
    mean, stddev = _stats(scale=10.0)

    result = compare_outputs(
        native,
        candidate,
        mean=mean,
        stddev=stddev,
        comparison_kind="cross_precision_drift",
    )

    assert result["status"] == "measured"
    assert result["comparison_kind"] == "cross_precision_drift"
    assert result["acceptance_gate_applied"] is False
    assert result["limits"] is None
    assert result["variables"][0]["within_envelope"] is None
