"""Tiny CPU fixtures test evidence rules, not weather forecast accuracy."""
from __future__ import annotations

import importlib
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from scripts.inference import diagnose_inference as diagnostic


def api():
    return importlib.import_module("scripts.inference.inference_reference")


def write_json(path, value):
    path.write_text(json.dumps(value))


def case_spec(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    init = np.datetime64("2021-01-01T06:00", "ns")
    def dataset(leads):
        time = np.asarray(leads, dtype="timedelta64[h]")
        return xr.Dataset(
            {name: (("batch", "time", "level", "lat", "lon"),
                    np.full((1, len(time), 13, 2, 2), 10, dtype=np.float32),
                    {"units": "fixture-unit"}) for name in diagnostic.VARIABLES},
            coords={"batch": [0], "time": time, "level": diagnostic.LEVELS,
                    "lat": [-45., 45.], "lon": [0., 180.],
                    "datetime": (("batch", "time"), (init + time)[None])},
        )
    inputs, forcing, target = dataset([-6, 0]), dataset([6, 12]), dataset([6])
    forcing = forcing.rename({name: f"forcing_{index}" for index, name in enumerate(forcing)})
    # Validate real NPZ checkpoint config without importing JAX.
    model = {"resolution": .25, "mesh_size": 6, "latent_size": 512,
             "gnn_msg_steps": 16, "hidden_layers": 1,
             "radius_query_fraction_edge_length": .6}
    task = {"input_duration": "12h", "pressure_levels": diagnostic.LEVELS,
            "input_variables": list(inputs), "forcing_variables": list(forcing),
            "target_variables": list(target)}
    checkpoint = {f"model_config:{key}": np.asarray(val) for key, val in model.items()}
    for key, value in task.items():
        if isinstance(value, list):
            checkpoint.update({f"task_config:{key}:{i}": np.asarray(v) for i, v in enumerate(value)})
        else:
            checkpoint[f"task_config:{key}"] = np.asarray(value)
    np.savez(raw / "jax_weights.npz", **checkpoint)
    (raw / "weights.pt").write_bytes(b"synthetic converted weights, not executable")
    (raw / "converter.py").write_text("# Synthetic fixture conversion identity\n")
    config = {"source": {"checkpoint_sha256": diagnostic.sha256(raw / "jax_weights.npz")},
              "model_config": model, "task_config": task}
    (raw / "config.yaml").write_text(yaml.safe_dump(config))
    for name, ds in (("inputs", inputs), ("rollout_forcings", forcing), ("target_template", target)):
        ds.to_netcdf(raw / f"{name}.nc")
    stats = raw / "stats"
    stats.mkdir()
    for name in diagnostic.STATS_FILES:
        target.isel(batch=0, time=0, drop=True).to_netcdf(stats / name)
    write_json(raw / "conversion.json", {
        "schema": "graphcast.checkpoint_conversion_audit.v1", "status": "passed",
        "parameter_count": 1, "mapping": {"fixture.weight": {"jax_key": "fixture"}},
        "jax_weights_sha256": diagnostic.sha256(raw / "jax_weights.npz"),
        "torch_weights_sha256": diagnostic.sha256(raw / "weights.pt"),
        "converter_sha256": diagnostic.sha256(raw / "converter.py"),
        "description": "synthetic CPU fixture; never a publishable weather baseline",
    })
    return {
        "steps": 2, "step_hours": 6, "lead_hours": 12,
        "initialization_time": str(init), "repeatability_percent": .1,
        "assets": {"config": str(raw / "config.yaml"), "jax_weights": str(raw / "jax_weights.npz"),
                   "weights": str(raw / "weights.pt"), "converter": str(raw / "converter.py"),
                   "conversion": str(raw / "conversion.json"),
                   **{n: str(raw / f"{n}.nc") for n in ("inputs", "rollout_forcings", "target_template")},
                   **{f"stats/{n}": str(stats / n) for n in diagnostic.STATS_FILES}},
    }


def frozen(tmp_path):
    spec = case_spec(tmp_path)
    dest = tmp_path / "case"
    api().freeze_case(spec, dest)
    return dest


def reference_run(case_path, path, role, *, scale=1.):
    case = api().load_case(case_path)
    path.mkdir()
    with xr.open_dataset(case_path / "assets/target_template.nc", decode_timedelta=True) as template:
        lead = np.timedelta64(case["lead_hours"], "h")
        ds = template.load().assign_coords(time=[lead])
        ds = ds.assign_coords(datetime=(("batch", "time"),
            [[np.datetime64(case["initialization_time"], "ns") + lead]]))
        for name in ds:
            ds[name] *= scale
        ds.to_netcdf(path / "prediction.nc")
    report = {
        "schema": "graphcast.inference_reference_run.v1", "status": "passed",
        "role": role, "execution_id": str(uuid.uuid4()),
        "case_sha256": diagnostic.sha256(case_path / "manifest.json"),
        "source": api().case_source(case), "runner_sha256": "c" * 64,
        "engine_source": {"commit": api().OFFICIAL_COMMIT, "tree_sha256": "d" * 64,
                          "verified_official": True} if role == diagnostic.REFERENCE_KEYS[0]
                         else {"tree_sha256": "e" * 64},
        "runtime": {"device_name": "NVIDIA H100 80GB HBM3", "device_type": "cuda",
                    "versions": {"framework": "fixture"}, "flagos_mode": "off"},
        "precision": {"effective": "strict_fp32", "parameter_dtype": "float32",
                      "data_dtype": "float32", "matmul_precision": "highest",
                      "autocast_enabled": False},
        "workload": {"steps": case["steps"], "step_hours": case["step_hours"],
                     "model_profile": "baseline", "compile": False},
        "artifacts": {"prediction": {"sha256": diagnostic.sha256(path / "prediction.nc")}},
    }
    write_json(path / "reference_report.json", report)
    return path


def runs(case, tmp_path, *, torch_scale=1., repeat_scale=1.):
    jax = [reference_run(case, tmp_path / f"jax-{i}", diagnostic.REFERENCE_KEYS[0]) for i in range(2)]
    torch = [reference_run(case, tmp_path / f"torch-{i}", diagnostic.REFERENCE_KEYS[1],
                           scale=torch_scale * (repeat_scale if i else 1.)) for i in range(2)]
    return jax, torch


def test_case_copies_identical_inputs_and_checks_tampering(tmp_path):
    spec = case_spec(tmp_path)
    dest = tmp_path / "case"
    manifest = api().freeze_case(spec, dest)
    assert manifest["steps"] == 2
    assert manifest["lead_hours"] == 12
    assert api().load_case(dest)["assets"]["rollout_forcings"]["sha256"] == diagnostic.sha256(
        Path(spec["assets"]["rollout_forcings"]))
    with pytest.raises(FileExistsError):
        api().freeze_case(spec, dest)
    (dest / "assets/weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        api().load_case(dest)


@pytest.mark.parametrize("field,value", [("steps", 0), ("steps", 3), ("step_hours", 3),
    ("lead_hours", 9), ("initialization_time", "2021-01-02T06:00"),
    ("repeatability_percent", .2), ("repeatability_percent", 5.), ("repeatability_percent", float("nan"))])
def test_case_rejects_inconsistent_time_and_guard(tmp_path, field, value):
    spec = case_spec(tmp_path)
    spec[field] = value
    with pytest.raises(ValueError):
        api().freeze_case(spec, tmp_path / "case")


def test_case_requires_matching_original_and_conversion_lineage(tmp_path):
    spec = case_spec(tmp_path)
    record = Path(spec["assets"]["conversion"])
    value = json.loads(record.read_text())
    value["jax_weights_sha256"] = "f" * 64
    write_json(record, value)
    with pytest.raises(ValueError, match="conversion"):
        api().freeze_case(spec, tmp_path / "case")


def test_case_refuses_unverified_conversion_attestation(tmp_path):
    spec = case_spec(tmp_path)
    record = Path(spec["assets"]["conversion"])
    value = json.loads(record.read_text())
    value["status"] = "not_run"
    write_json(record, value)
    with pytest.raises(ValueError, match="conversion"):
        api().freeze_case(spec, tmp_path / "case")


def test_parameterized_export_uses_unchanged_formula(tmp_path):
    case = frozen(tmp_path)
    jax, torch = runs(case, tmp_path, torch_scale=1.04)
    out = tmp_path / "baseline"
    result = api().qualify(case, jax, torch, out)
    assert result["status"] == "qualified"
    assert result["cross_reference"]["overall_mean_percent"] == pytest.approx(4.)
    assert result["cross_reference"]["z500_percent"] == pytest.approx(4.)
    with np.load(out / "baseline.npz", allow_pickle=False) as archive:
        assert archive["lead_hour"] == 12
        assert archive["time_index"] == 1
        assert archive["pytorch_gpu__temperature"].shape == (13, 2, 2)
    metadata = json.loads((out / "baseline.json").read_text())
    assert metadata["provenance_status"] == "verified"
    assert metadata["qualification"]["status"] == "qualified"
    assert metadata["source"]["config_sha256"]


@pytest.mark.parametrize("scale,repeat", [(1.06, 1.), (1., 1.002)])
def test_failed_agreement_or_repeatability_never_qualifies(tmp_path, scale, repeat):
    case = frozen(tmp_path)
    jax, torch = runs(case, tmp_path, torch_scale=scale, repeat_scale=repeat)
    out = tmp_path / "baseline"
    result = api().qualify(case, jax, torch, out)
    assert result["status"] == "unqualified"
    assert (out / "baseline.npz").exists()  # Keep failed evidence for investigation.
    assert json.loads((out / "baseline.json").read_text())["qualification"]["status"] == "unqualified"


@pytest.mark.parametrize("mutation", ["duplicate", "bf16", "native_mode", "profile", "hardware", "case", "source", "official", "prediction"])
def test_untrusted_run_evidence_cannot_create_reference(tmp_path, mutation):
    case = frozen(tmp_path)
    jax, torch = runs(case, tmp_path)
    target = jax[1] if mutation == "official" else torch[1]
    report_path = target / "reference_report.json"
    report = json.loads(report_path.read_text())
    if mutation == "duplicate":
        report["execution_id"] = json.loads((torch[0] / "reference_report.json").read_text())["execution_id"]
    elif mutation == "bf16":
        report["precision"]["effective"] = "mixed_precision"
    elif mutation == "native_mode":
        report["runtime"]["flagos_mode"] = "flaggems"
    elif mutation == "profile":
        report["workload"]["model_profile"] = "domestic-v1"
    elif mutation == "hardware":
        report["runtime"]["device_name"] = "NVIDIA A100"
    elif mutation == "case":
        report["case_sha256"] = "f" * 64
    elif mutation == "source":
        report["source"]["weights_sha256"] = "f" * 64
    elif mutation == "official":
        report["engine_source"]["verified_official"] = False
    else:
        with (target / "prediction.nc").open("ab") as stream:
            stream.write(b"tampered")
    write_json(report_path, report)
    with pytest.raises(ValueError):
        api().qualify(case, jax, torch, tmp_path / "baseline")


def test_insufficient_runs_not_qualified(tmp_path):
    case = frozen(tmp_path)
    jax, torch = runs(case, tmp_path)
    with pytest.raises(ValueError, match="two"):
        api().qualify(case, jax[:1], torch, tmp_path / "baseline")
