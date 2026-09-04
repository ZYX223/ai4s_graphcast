from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tools" / "bf16_inference" / "summarize_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_bf16_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report(
    path: Path,
    seconds: float,
    *,
    coverage: str | None = None,
    profile: str = "domestic-v1",
) -> Path:
    runtime = {}
    if coverage is not None:
        runtime["flaggems_dispatch"] = {
            "status": "passed",
            "active_coverage": coverage,
            "configured_coverage": coverage,
            "missing_ops": [],
        }
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "precision": {"requested": "bf16", "effective": "mixed_precision"},
                "measurement": {
                    "e2e_seconds": seconds,
                    "performance_eligible": True,
                    "finite": True,
                    "useful_flops": 120.0e12,
                },
                "workload": {
                    "model_execution_profile": {
                        "name": profile,
                        "platform": "example",
                    }
                },
                "runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    return path


def _diagnostic(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "measured",
                "lead_count": 4,
                "implementation_parity": {"finite": True, "relative_l2": 0.01},
                "worst": {"normalized_max_abs": 2.0, "normalized_rmse": 0.2},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_summarizes_median_e2e_and_path_matched_mfu(tmp_path: Path) -> None:
    native = [
        _report(tmp_path / f"native-{index}.json", seconds)
        for index, seconds in enumerate((4.0, 2.0, 3.0), start=1)
    ]
    flaggems = [
        _report(tmp_path / f"fg-{index}.json", seconds, coverage="7/7")
        for index, seconds in enumerate((3.0, 2.5, 2.0), start=1)
    ]
    payload = {
        "schema": "graphcast.bf16_inference_campaign_input.v1",
        "platforms": [
            {
                "platform": "example",
                "bf16_roof_tflops": 100.0,
                "expected_model_execution_profile": "domestic-v1",
                "expected_flaggems_coverage": "7/7",
                "core_flaggems_coverage": "7/7",
                "diagnostic_reports": {
                    "native": str(_diagnostic(tmp_path / "native-diagnostic.json")),
                    "flaggems": str(_diagnostic(tmp_path / "fg-diagnostic.json")),
                },
                "native_reports": [str(path) for path in native],
                "flaggems_reports": [str(path) for path in flaggems],
            }
        ],
    }

    result = MODULE.summarize(payload)

    native_result, flaggems_result = result["platforms"][0]["modes"]
    assert native_result["median_e2e_seconds"] == 3.0
    assert native_result["effective_tflops"] == pytest.approx(40.0)
    assert native_result["mfu"] == pytest.approx(0.4)
    assert flaggems_result["median_e2e_seconds"] == 2.5
    assert flaggems_result["operator_coverage"] == "7/7"
    assert flaggems_result["dispatch_coverage"] == "7/7"
    assert result["platforms"][0]["model_execution_profile"] == "domestic-v1"
    assert result["useful_flops"] == 120.0e12


def test_rejects_unverified_flaggems_coverage(tmp_path: Path) -> None:
    reports = [
        _report(tmp_path / f"fg-{index}.json", 2.0, coverage="6/7")
        for index in range(3)
    ]
    diagnostic = _diagnostic(tmp_path / "diagnostic.json")
    payload = {
        "schema": "graphcast.bf16_inference_campaign_input.v1",
        "platforms": [
            {
                "platform": "example",
                "bf16_roof_tflops": 100.0,
                "expected_model_execution_profile": "domestic-v1",
                "expected_flaggems_coverage": "7/7",
                "diagnostic_reports": {
                    "native": str(diagnostic),
                    "flaggems": str(diagnostic),
                },
                "native_reports": [],
                "flaggems_reports": [str(path) for path in reports],
            }
        ],
    }

    with pytest.raises(ValueError, match="coverage"):
        MODULE.summarize(payload)


def test_rejects_mismatched_model_execution_profile(tmp_path: Path) -> None:
    reports = [
        _report(tmp_path / f"native-{index}.json", 2.0, profile="baseline")
        for index in range(3)
    ]
    diagnostic = _diagnostic(tmp_path / "diagnostic.json")
    payload = {
        "schema": "graphcast.bf16_inference_campaign_input.v1",
        "platforms": [
            {
                "platform": "example",
                "bf16_roof_tflops": 100.0,
                "expected_model_execution_profile": "domestic-v1",
                "diagnostic_reports": {"native": str(diagnostic)},
                "native_reports": [str(path) for path in reports],
                "flaggems_reports": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="model execution profile"):
        MODULE.summarize(payload)
