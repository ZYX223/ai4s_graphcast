from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from tests.test_inference_reference import case_spec, frozen, reference_run, runs
from tests.test_operational_40step_runner import CountingModel, _prepared
from scripts.inference import diagnose_inference as diagnostic
from scripts.inference import inference_reference as reference
from scripts.inference.run_operational_40step_inference import run_tensor_resident_rollout


def runner():
    return importlib.import_module("scripts.inference.run_inference_reference")


def test_torch_arguments_lock_native_strict_baseline_and_arbitrary_steps(tmp_path):
    case_path = frozen(tmp_path)
    case = reference.load_case(case_path)
    args = runner().torch_arguments(case_path, case, tmp_path / "run", "cuda:2")
    options = dict(zip(args[::2], args[1::2]))
    assert options["--steps"] == "2"
    assert options["--step-hours"] == "6"
    assert options["--precision"] == "fp32"
    assert options["--model-execution-profile"] == "baseline"
    assert options["--platform"] == "h100"
    assert options["--device"] == "cuda:2"
    assert options["--rollout-forcings"] == str(case_path / "assets/rollout_forcings.nc")
    assert "--trajectory-output" not in options


def test_torch_arguments_satisfy_rollout_warmup_without_advancing_state(tmp_path):
    case_path = frozen(tmp_path)
    case = reference.load_case(case_path)
    args = runner().torch_arguments(case_path, case, tmp_path / "run", "cuda:0")
    options = dict(zip(args[::2], args[1::2]))
    model = CountingModel([])

    result = run_tensor_resident_rollout(
        model,
        _prepared(2),
        steps=2,
        warmup_calls=int(options["--warmup-calls"]),
        synchronize=lambda: None,
    )

    assert model.calls == 3
    assert result.final_output.item() == 2.0


def test_nonfinal_reference_lead_requests_actual_trajectory(tmp_path):
    case_path = frozen(tmp_path)
    case = reference.load_case(case_path)
    case["lead_hours"] = 6
    args = runner().torch_arguments(case_path, case, tmp_path / "run", "cuda:0")
    options = dict(zip(args[::2], args[1::2]))
    assert options["--trajectory-output"] == str(tmp_path / "run/prediction.nc")
    assert options["--output"] == str(tmp_path / "run/final_prediction.nc")


def test_reference_environment_does_not_inherit_domestic_optimization_or_tf32(tmp_path):
    env = runner().reference_environment({"PATH": "/usr/bin", "GRAPHCAST_FLAGOS_MODE": "flaggems",
        "GRAPHCAST_REUSE_AGGREGATION_BUFFER": "1", "GRAPHCAST_EDGE_INPUT_PROJECTION": "factorized",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": "1"})
    assert env["GRAPHCAST_FLAGOS_MODE"] == "off"
    assert "GRAPHCAST_REUSE_AGGREGATION_BUFFER" not in env
    assert "GRAPHCAST_EDGE_INPUT_PROJECTION" not in env
    assert env["NVIDIA_TF32_OVERRIDE"] == "0"
    assert env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "0"
    assert env["PATH"] == "/usr/bin"


def test_official_source_checker_rejects_wrong_checkout_without_importing_jax(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    with pytest.raises(ValueError, match="official"):
        runner().verify_official_source(source)


def test_checkpoint_config_mismatch_rejected_on_cpu(tmp_path):
    spec = case_spec(tmp_path)
    with pytest.raises(ValueError, match="checkpoint/config"):
        reference.check_checkpoint_config(Path(spec["assets"]["jax_weights"]),
                                          {"model_config": {"mesh_size": 5}})


def test_raw_report_conversion_requires_observed_fp32_and_baseline(tmp_path):
    report = {
        "status": "passed", "runtime": {"flagos_mode": "off", "versions": {"torch": "fixture"}},
        "precision": {"effective": "strict_fp32", "float32_math_mode": "ieee_fp32",
            "autocast": {"enabled": False},
            **{name: "torch.float32" for name in ("parameter_dtype", "state_dtype", "forcing_dtype",
                "model_input_dtype", "model_output_dtype", "feedback_dtype", "final_output_dtype")}},
        "workload": {"edge_input_projection": "concatenated", "mesh2grid_full_grid_fast_path": False,
                     "index_add_execution": "inline", "model_execution_profile": {"name": "baseline"}},
    }
    assert runner().validate_torch_report(report) is None
    report["precision"]["model_output_dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError, match="FP32"):
        runner().validate_torch_report(report)


def test_qualified_export_roundtrips_through_real_candidate_diagnostic(tmp_path):
    case = frozen(tmp_path)
    jax, torch = runs(case, tmp_path, torch_scale=1.04)
    output = tmp_path / "qualified"
    reference.qualify(case, jax, torch, output)
    # Use the real reference prediction as a synthetic candidate, with a real
    # production report schema and frozen asset identities.
    candidate = torch[0]
    meta = json.loads((output / "baseline.json").read_text())
    reference.write_json(candidate / "report.json", {
        "schema": "graphcast.operational_rollout.v2", "status": "passed", "source": meta["source"],
        "workload": {"steps": 2, "step_hours": 6}, "runtime": {"flagos_mode": "off"},
        "artifacts": {"prediction": {"sha256": diagnostic.sha256(candidate / "prediction.nc")}},
    })
    result = diagnostic.diagnose_run(candidate, output / "baseline.npz", meta)
    assert result["acceptance_passed"] is True
    assert result["comparisons"][0]["overall_mean_percent"] == pytest.approx(4.)
    meta["qualification"]["repeatability_percent"] = 1.
    with pytest.raises(ValueError, match="numerical evidence"):
        diagnostic.diagnose_run(candidate, output / "baseline.npz", meta)
    meta["qualification"]["repeatability_percent"] = .1
    meta["qualification"]["status"] = "unqualified"
    assert diagnostic.diagnose_run(candidate, output / "baseline.npz", meta)["status"] == "unqualified_reference"


def test_nonfinite_reference_cannot_be_certified(tmp_path):
    case = frozen(tmp_path)
    jax, torch = runs(case, tmp_path, torch_scale=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        reference.qualify(case, jax, torch, tmp_path / "baseline")


def test_cli_preflight_works_in_cpu_process_without_creating_device_results(tmp_path):
    case = frozen(tmp_path)
    out = tmp_path / "not-a-run"
    script = Path(__file__).resolve().parents[1] / "scripts/inference/run_inference_reference.py"
    result = subprocess.run([sys.executable, str(script), "--case", str(case), "--engine", "torch",
                             "--device", "cuda:0", "--output-dir", str(out), "--check-only"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "no device run" in result.stdout
    assert not out.exists()
