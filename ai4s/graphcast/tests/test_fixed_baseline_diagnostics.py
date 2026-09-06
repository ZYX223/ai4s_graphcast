from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inference/diagnose_inference.py"
NAMES = (
    "u_component_of_wind", "v_component_of_wind", "geopotential",
    "temperature", "specific_humidity", "vertical_velocity",
)
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
KEYS = ("jax_fp32_no_bf16", "pytorch_gpu")


def module():
    assert SCRIPT.is_file(), "fixed-baseline inference acceptance entry point is missing"
    spec = importlib.util.spec_from_file_location("fixed_baseline_diagnostics", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, *, lead=240, scale=1.0):
    baseline = tmp_path / "baseline.npz"
    fields = dict(
        baseline_keys=np.asarray(KEYS), baseline_titles=np.asarray(["JAX FP32", "Torch GPU"]),
        variable_names=np.asarray(NAMES), variable_labels=np.asarray(list("UVPTQW")),
        level=np.asarray(LEVELS), lat=np.asarray([-45., 45.]), lon=np.asarray([0., 180.]),
        lead_hour=np.asarray(lead), time_index=np.asarray(lead // 6 - 1),
    )
    # Mixed signs catch abs(mean(reference)); unequal magnitudes catch pointwise MAPE.
    values = np.broadcast_to(np.asarray([[-10., 30.], [-10., 30.]]), (13, 2, 2)).copy()
    for key in KEYS:
        for name in NAMES:
            fields[f"{key}__{name}"] = values.copy()
    np.savez(baseline, **fields)
    init = np.datetime64("2021-01-01T00:00:00", "ns")
    candidate = xr.Dataset(
        {name: (("batch", "time", "level", "lat", "lon"),
                (values * scale)[None, None], {"units": "fixture-unit"}) for name in NAMES},
        coords={"batch": [0], "time": np.asarray([lead], dtype="timedelta64[h]"),
                "level": LEVELS, "lat": [-45., 45.], "lon": [0., 180.],
                "datetime": (("batch", "time"), np.asarray([[init + np.timedelta64(lead, "h")]]))},
    )
    run = tmp_path / "run"
    run.mkdir()
    candidate.to_netcdf(run / "prediction.nc")
    source = {name: "a" * 64 for name in (
        "weights_sha256", "inputs_sha256", "rollout_forcings_sha256", "target_template_sha256"
    )}
    source["normalization_stats"] = {
        name: {"sha256": "b" * 64} for name in
        ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")
    }
    report = {
        "schema": "graphcast.operational_rollout.v2", "status": "passed", "source": source,
        "workload": {"steps": lead // 6, "step_hours": 6, "model_execution_profile": "domestic-v1"},
        "precision": {"requested": "tf32", "float32_math_mode": "tf32"},
        "runtime": {"chip": "C550", "flagos_mode": "flaggems", "flaggems_dispatch": {
            "counts": {name: 3 for name in ("addmm", "cat", "native_layer_norm", "index_add_", "silu", "index", "add")}
        }},
        "artifacts": {"prediction": {"sha256": hashlib.sha256((run / "prediction.nc").read_bytes()).hexdigest()}},
    }
    (run / "report.json").write_text(json.dumps(report))
    metadata = {
        "schema": "graphcast.fixed_baseline.v1", "provenance_status": "verified",
        "provenance_evidence": "synthetic CPU fixture; not a weather baseline",
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "initialization_time": "2021-01-01T00:00:00", "lead_hours": lead,
        "source": source, "units": {name: "fixture-unit" for name in NAMES},
    }
    manifest = tmp_path / "baseline.json"
    manifest.write_text(json.dumps(metadata))
    return baseline, manifest, run, candidate


def evaluate(tmp_path, *, lead=240, scale=1.0):
    baseline, manifest, run, _ = fixture(tmp_path, lead=lead, scale=scale)
    return module().diagnose_run(run, baseline, json.loads(manifest.read_text()))


@pytest.mark.parametrize("scale,expected,status", [(1., 0., "passed"), (1.04, 4., "passed"), (1.05, 5., "failed")])
def test_fixed_baseline_percentage_and_strict_boundary(tmp_path, scale, expected, status):
    result = evaluate(tmp_path, scale=scale)
    assert result["status"] == status
    assert result["core_operator_coverage"] == "7/7"
    for comparison in result["comparisons"]:
        assert comparison["overall_mean_percent"] == pytest.approx(expected)
        assert comparison["z500_percent"] == pytest.approx(expected)
        assert len(comparison["layers"]) == 78


def rewrite_candidate(run, candidate):
    candidate.to_netcdf(run / "prediction.nc", mode="w")
    report = json.loads((run / "report.json").read_text())
    report["artifacts"]["prediction"]["sha256"] = hashlib.sha256((run / "prediction.nc").read_bytes()).hexdigest()
    (run / "report.json").write_text(json.dumps(report))


def test_z500_fails_independently_of_small_overall_mean(tmp_path):
    baseline, manifest, run, candidate = fixture(tmp_path)
    candidate["geopotential"].loc[{"level": 500}] *= 1.1
    rewrite_candidate(run, candidate)
    result = module().diagnose_run(run, baseline, json.loads(manifest.read_text()))
    assert result["status"] == "failed"
    assert result["comparisons"][0]["overall_mean_percent"] == pytest.approx(10 / 78)
    assert result["comparisons"][0]["z500_percent"] == pytest.approx(10)


def test_worst_layer_is_not_a_separate_acceptance_gate(tmp_path):
    baseline, manifest, run, candidate = fixture(tmp_path)
    candidate["vertical_velocity"].loc[{"level": 500}] *= 1.2
    rewrite_candidate(run, candidate)
    result = module().diagnose_run(run, baseline, json.loads(manifest.read_text()))
    assert result["status"] == "passed"
    assert result["comparisons"][0]["overall_mean_percent"] == pytest.approx(20 / 78)


def test_formula_is_ratio_of_means_not_mean_of_pointwise_ratios(tmp_path):
    baseline, manifest, run, candidate = fixture(tmp_path)
    for name in NAMES:
        candidate[name] += 1
    rewrite_candidate(run, candidate)
    result = module().diagnose_run(run, baseline, json.loads(manifest.read_text()))
    assert result["comparisons"][0]["overall_mean_percent"] == 5.0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_output_cannot_be_skipped(tmp_path, value):
    baseline, manifest, run, candidate = fixture(tmp_path)
    candidate["temperature"].values[..., 0, 0] = value
    rewrite_candidate(run, candidate)
    result = module().diagnose_run(run, baseline, json.loads(manifest.read_text()))
    assert result["status"] == "failed_nonfinite"
    json.dumps(result, allow_nan=False)


def test_non_40_step_baseline_selects_by_lead_time(tmp_path):
    assert evaluate(tmp_path, lead=12)["status"] == "passed"


@pytest.mark.parametrize("change", ["lead", "initialization", "coordinates", "units", "variable", "batch"])
def test_mismatched_candidate_is_not_comparable(tmp_path, change):
    baseline, manifest, run, candidate = fixture(tmp_path)
    if change == "lead":
        candidate = candidate.assign_coords(time=np.asarray([6], dtype="timedelta64[h]"))
    elif change == "initialization":
        candidate["datetime"] += np.timedelta64(24, "h")
    elif change == "coordinates":
        candidate = candidate.assign_coords(lat=[-44., 45.])
    elif change == "units":
        candidate["temperature"].attrs["units"] = "wrong"
    elif change == "variable":
        candidate = candidate.drop_vars("temperature")
    else:
        candidate = xr.concat([candidate, candidate.assign_coords(batch=[1])], dim="batch")
    rewrite_candidate(run, candidate)
    with pytest.raises(ValueError):
        module().diagnose_run(run, baseline, json.loads(manifest.read_text()))


@pytest.mark.parametrize("change", ["baseline_hash", "weights", "prediction_hash", "stats"])
def test_stale_or_unrelated_evidence_is_rejected(tmp_path, change):
    baseline, manifest, run, _ = fixture(tmp_path)
    meta = json.loads(manifest.read_text())
    if change == "baseline_hash":
        meta["baseline_sha256"] = "f" * 64
    elif change == "weights":
        meta["source"]["weights_sha256"] = "f" * 64
    elif change == "stats":
        meta["source"]["normalization_stats"]["stddev_by_level.nc"]["sha256"] = "f" * 64
    else:
        with (run / "prediction.nc").open("ab") as stream:
            stream.write(b"unbound mutation")
    with pytest.raises(ValueError):
        module().diagnose_run(run, baseline, meta)


def test_unverified_baseline_can_measure_but_not_pass(tmp_path):
    baseline, manifest, run, _ = fixture(tmp_path)
    meta = json.loads(manifest.read_text())
    meta.update(provenance_status="unverified", source={}, initialization_time=None)
    result = module().diagnose_run(run, baseline, meta)
    assert result["status"] == "unverified_reference"
    assert result["acceptance_passed"] is False
    assert result["reference_qualified"] is None  # Legacy NPZ never underwent the new qualification.
    assert result["comparisons"][0]["overall_mean_percent"] == 0.0


def test_cli_batches_runs_and_preserves_detailed_metrics(tmp_path):
    baseline, manifest, run, _ = fixture(tmp_path)
    output = tmp_path / "diagnostics"
    rc = module().main(["--run-dir", str(run), "--baseline", str(baseline),
                        "--baseline-metadata", str(manifest), "--output-dir", str(output)])
    assert rc == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["runs"][0]["status"] == "passed"
    assert summary["baseline"]["sha256"]
    assert len((output / "details.csv").read_text().splitlines()) == 157
    assert module().main(["--run-dir", str(run), "--baseline", str(baseline),
                          "--baseline-metadata", str(manifest), "--output-dir", str(output)]) == 2


def test_cli_bad_input_has_nonzero_exit_and_no_false_pass(tmp_path):
    baseline, manifest, run, _ = fixture(tmp_path)
    (run / "prediction.nc").unlink()
    output = tmp_path / "diagnostics"
    assert module().main(["--run-dir", str(run), "--baseline", str(baseline),
                          "--baseline-metadata", str(manifest), "--output-dir", str(output)]) == 2
    assert json.loads((output / "summary.json").read_text())["runs"][0]["status"] == "invalid_inputs"


@pytest.mark.parametrize("change", ["zero_denominator", "missing_reference"])
def test_invalid_reference_content_cannot_pass(tmp_path, change):
    baseline, manifest, run, _ = fixture(tmp_path)
    with np.load(baseline) as archive:
        fields = dict(archive)
    if change == "zero_denominator":
        fields["jax_fp32_no_bf16__temperature"][0] = 0
    else:
        fields["baseline_keys"] = np.asarray(["jax_fp32_no_bf16"])
    np.savez(baseline, **fields)
    metadata = json.loads(manifest.read_text())
    metadata["baseline_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module().diagnose_run(run, baseline, metadata)


@pytest.mark.parametrize("field,value", [("steps", 0), ("steps", True), ("steps", 41), ("step_hours", -6)])
def test_final_output_horizon_must_match_runner_report(tmp_path, field, value):
    baseline, manifest, run, _ = fixture(tmp_path)
    report = json.loads((run / "report.json").read_text())
    report["workload"][field] = value
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module().diagnose_run(run, baseline, json.loads(manifest.read_text()))


def test_failed_runner_cannot_be_promoted_to_acceptance(tmp_path):
    baseline, manifest, run, _ = fixture(tmp_path)
    report = json.loads((run / "report.json").read_text())
    report["status"] = "failed_nonfinite"
    (run / "report.json").write_text(json.dumps(report))
    result = module().diagnose_run(run, baseline, json.loads(manifest.read_text()))
    assert result["status"] == "failed_nonfinite"
    assert not result["acceptance_passed"]


def test_core_coverage_counts_observed_not_requested_ops(tmp_path):
    baseline, manifest, run, _ = fixture(tmp_path)
    report = json.loads((run / "report.json").read_text())
    report["runtime"]["flaggems_dispatch"]["counts"]["addmm"] = 0
    (run / "report.json").write_text(json.dumps(report))
    result = module().diagnose_run(run, baseline, json.loads(manifest.read_text()))
    assert result["status"] == "passed"
    assert result["core_operator_coverage"] == "6/7"


def test_v2_provenance_alone_cannot_promote_unqualified_reference(tmp_path):
    baseline, manifest, run, _ = fixture(tmp_path)
    metadata = json.loads(manifest.read_text())
    metadata["source"]["config_sha256"] = "c" * 64
    report = json.loads((run / "report.json").read_text())
    report["source"]["config_sha256"] = "c" * 64
    (run / "report.json").write_text(json.dumps(report))
    metadata.update(schema="graphcast.fixed_baseline.v2", qualification={"status": "unqualified"})
    result = module().diagnose_run(run, baseline, metadata)
    assert result["provenance_verified"] is True
    assert result["status"] == "unqualified_reference"
    assert result["acceptance_passed"] is False
    assert result["comparisons"][0]["overall_mean_percent"] == 0.
