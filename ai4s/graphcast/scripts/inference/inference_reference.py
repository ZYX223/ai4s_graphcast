#!/usr/bin/env python3
"""Freeze a shared inference case and qualify strict JAX/Torch references (CPU)."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.inference import diagnose_inference as diagnostic  # noqa: E402

OFFICIAL_COMMIT = "08cf73625c9d12bd9aaa038868bcb2fe488f2a22"  # GraphCast v0.2, not WeatherNext2.
ASSETS = {
    "config": "assets/config.yaml", "weights": "assets/weights.pt",
    "jax_weights": "assets/jax_weights.npz", "converter": "assets/converter.py",
    "conversion": "assets/conversion.json", "inputs": "assets/inputs.nc",
    "rollout_forcings": "assets/rollout_forcings.nc", "target_template": "assets/target_template.nc",
    **{f"stats/{name}": f"assets/stats/{name}" for name in diagnostic.STATS_FILES},
}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_checkpoint_config(path: Path, config: dict) -> None:
    """Inspect official checkpoint metadata without a JAX import or unpickling."""
    with np.load(path, allow_pickle=False) as archive:
        for group in ("model_config", "task_config"):
            names = {key.split(":")[1] for key in archive.files if key.startswith(group + ":")}
            if not names:
                raise ValueError(f"checkpoint is missing {group}")
            for name in names:
                prefix = f"{group}:{name}"
                actual = (archive[prefix].item() if prefix in archive else
                          [archive[f"{prefix}:{i}"].item() for i in range(
                              sum(key.startswith(prefix + ":") for key in archive.files))])
                if config.get(group, {}).get(name) != actual:
                    raise ValueError(f"checkpoint/config mismatch: {group}/{name}")


def _validate_case(spec: dict, paths: dict[str, Path]) -> dict:
    for name in ("steps", "step_hours", "lead_hours"):
        if type(spec.get(name)) is not int or spec[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if spec["lead_hours"] % spec["step_hours"] or spec["lead_hours"] > spec["steps"] * spec["step_hours"]:
        raise ValueError("selected lead must be a rollout timestep")
    tolerance = spec.get("repeatability_percent", .1)
    if not np.isfinite(tolerance) or not 0 < tolerance <= diagnostic.MAX_REPEATABILITY_PERCENT:
        raise ValueError("repeatability_percent must be finite, positive and at most 0.1")
    init = np.datetime64(spec["initialization_time"], "ns")
    if np.isnat(init):
        raise ValueError("initialization_time must be known")
    config = yaml.safe_load(paths["config"].read_text())
    diagnostic._check_digest(diagnostic.sha256(paths["jax_weights"]),
                             config["source"].get("checkpoint_sha256"), "original checkpoint")
    check_checkpoint_config(paths["jax_weights"], config)
    conversion = json.loads(paths["conversion"].read_text())
    if (not conversion.get("description") or conversion.get("status") != "passed"
            or conversion.get("schema") != "graphcast.checkpoint_conversion_audit.v1"
            or not conversion.get("mapping") or type(conversion.get("parameter_count")) is not int
            or conversion["parameter_count"] != len(conversion["mapping"])):
        raise ValueError("conversion requires a completed exact parameter audit")
    for name, field in (("jax_weights", "jax_weights_sha256"), ("weights", "torch_weights_sha256"),
                        ("converter", "converter_sha256")):
        diagnostic._check_digest(diagnostic.sha256(paths[name]), conversion.get(field), "conversion " + field)
    with ExitStack() as stack:
        datasets = {key: stack.enter_context(xr.open_dataset(paths[key], decode_timedelta=True))
                    for key in ("inputs", "rollout_forcings", "target_template")}
        target = datasets["target_template"]
        units = {name: target[name].attrs.get("units") for name in diagnostic.VARIABLES}
        if not all(units.values()):
            raise ValueError("target requires physical units for all diagnostic variables")
        required_vars = dict(zip(datasets, ("input_variables", "forcing_variables", "target_variables")))
        for key, ds in datasets.items():
            expected = ([-spec["step_hours"], 0] if key == "inputs" else
                        [spec["step_hours"]] if key == "target_template" else
                        np.arange(1, spec["steps"] + 1) * spec["step_hours"])
            times = np.asarray(expected, dtype="timedelta64[h]").astype("timedelta64[ns]")
            if ds.sizes.get("batch") != 1 or not np.array_equal(ds.time.values, times):
                raise ValueError(f"{key} must match the declared input/lead sequence")
            if "datetime" not in ds.coords or not np.array_equal(ds.datetime.values, (init + times)[None]):
                raise ValueError(f"{key} datetime differs from initialization plus lead")
            if set(ds.data_vars) != set(config["task_config"][required_vars[key]]):
                raise ValueError(f"{key} variables differ from checkpoint task_config")
            for coordinate in ("lat", "lon", "level"):
                if coordinate not in ds.coords:
                    if key == "rollout_forcings" and coordinate == "level":
                        continue
                    raise ValueError(f"{key} missing {coordinate}")
                values = np.asarray(ds[coordinate])
                if (not np.array_equal(values, target[coordinate]) or not np.isfinite(values).all()
                        or np.unique(values).size != values.size):
                    raise ValueError(f"{key} {coordinate} differs from target grid")
            for name, variable in ds.data_vars.items():
                if variable.dtype != np.float32:
                    raise ValueError(f"{key}/{name} must be stored in FP32 before freezing")
                # Template values are not truth and can legitimately be NaN.
                if key != "target_template" and not np.isfinite(variable.values).all():
                    raise ValueError(f"non-finite {key}/{name}")
        if target.level.values.tolist() != diagnostic.LEVELS:
            raise ValueError("the handoff metric requires its 13 pressure levels")
        coordinates = {key: target[key].values.tolist() for key in ("lat", "lon", "level")}
    return {"units": units, "coordinates": coordinates, "task_config": config["task_config"],
            "conversion": conversion, "repeatability_percent": tolerance}


def freeze_case(spec: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    paths = {name: Path(spec["assets"][name]).resolve(strict=True) for name in ASSETS}
    digests = {name: diagnostic.sha256(path) for name, path in paths.items()}
    details = _validate_case(spec, paths)
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    for name, relative in ASSETS.items():
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], dest)
        diagnostic._check_digest(diagnostic.sha256(dest), digests[name], name)
        diagnostic._check_digest(diagnostic.sha256(paths[name]), digests[name], name)
        assets[name] = {"path": relative, "sha256": digests[name], "original_path": str(paths[name])}
    manifest = {"schema": "graphcast.inference_case.v1", "official_commit": OFFICIAL_COMMIT,
                **{key: spec[key] for key in ("steps", "step_hours", "lead_hours", "initialization_time")},
                **details, "assets": assets, "qualification_status": "not_run"}
    write_json(output / "manifest.json", manifest)  # Terminal marker, only after complete copy.
    return manifest


def load_case(path: Path) -> dict:
    case = json.loads((path / "manifest.json").read_text())
    if case.get("schema") != "graphcast.inference_case.v1" or case.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("unsupported case schema or official source pin")
    tolerance = case.get("repeatability_percent", 0)
    if not np.isfinite(tolerance) or not 0 < tolerance <= diagnostic.MAX_REPEATABILITY_PERCENT:
        raise ValueError("case cannot loosen the 0.1% repeatability guard")
    for name, relative in ASSETS.items():
        entry = case["assets"][name]
        if entry.get("path") != relative or not (path / relative).resolve().is_relative_to(path.resolve()):
            raise ValueError(f"case asset escapes the bundle: {name}")
        diagnostic._check_digest(diagnostic.sha256(path / relative), entry.get("sha256"), name)
    return case


def case_source(case: dict) -> dict:
    return {**{f"{name}_sha256": case["assets"][name]["sha256"] for name in
               ("config", "weights", "inputs", "rollout_forcings", "target_template", "jax_weights")},
            "normalization_stats": {name: {"sha256": case["assets"][f"stats/{name}"]["sha256"]}
                                    for name in diagnostic.STATS_FILES}}


def _read_run(path: Path, role: str, case_path: Path, case: dict) -> tuple[dict, dict]:
    report_path = path / "reference_report.json"
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "graphcast.inference_reference_run.v1" or report.get("status") != "passed"
            or report.get("role") != role or not report.get("execution_id")
            or report.get("case_sha256") != diagnostic.sha256(case_path / "manifest.json")
            or report.get("source") != case_source(case)):
        raise ValueError("reference run identity/source is not bound to this case")
    precision = report.get("precision", {})
    if any(precision.get(key) != expected for key, expected in {
        "effective": "strict_fp32", "parameter_dtype": "float32", "data_dtype": "float32",
        "matmul_precision": "highest", "autocast_enabled": False,
    }.items()):
        raise ValueError("reference execution is not strict FP32")
    workload, runtime = report.get("workload", {}), report.get("runtime", {})
    if (workload.get("steps") != case["steps"] or workload.get("step_hours") != case["step_hours"]
            or workload.get("model_profile") != "baseline" or workload.get("compile") is not False
            or runtime.get("flagos_mode") != "off"):
        raise ValueError("reference must use the unoptimized Native baseline")
    if role == "pytorch_gpu" and ("H100" not in runtime.get("device_name", "") or runtime.get("device_type") != "cuda"):
        raise ValueError("Torch reference must be run on H100")
    engine = report.get("engine_source", {})
    if (not engine.get("tree_sha256") or not report.get("runner_sha256") or not runtime.get("versions")
            or (role == "jax_fp32_no_bf16" and
                (engine.get("commit") != OFFICIAL_COMMIT or engine.get("verified_official") is not True))):
        raise ValueError("unverified engine source")
    prediction_path = path / "prediction.nc"
    digest = diagnostic.sha256(prediction_path)
    diagnostic._check_digest(digest, report["artifacts"]["prediction"].get("sha256"), "prediction")
    archive = {**{name: np.asarray(value) for name, value in case["coordinates"].items()},
               "lead_hour": np.asarray(case["lead_hours"])}
    with xr.open_dataset(prediction_path, decode_timedelta=True) as ds:
        frame = diagnostic._select_prediction(ds, archive, case, True, case["steps"] * case["step_hours"])
        fields = {}
        for name in diagnostic.VARIABLES:
            values = np.asarray(frame[name].transpose("level", "latitude", "longitude"))
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"non-finite or non-FP32 reference prediction: {name}")
            fields[name] = values.copy()
    evidence = {"path": str(path.resolve()), "report_sha256": diagnostic.sha256(report_path),
                "prediction_sha256": digest, "report": report}
    return fields, evidence


def _as_frame(fields: dict, case: dict) -> xr.Dataset:
    return xr.Dataset({name: (("level", "latitude", "longitude"), value) for name, value in fields.items()},
                      coords={"level": case["coordinates"]["level"],
                              "latitude": case["coordinates"]["lat"], "longitude": case["coordinates"]["lon"]})


def qualify(case_path: Path, jax_runs: list[Path], torch_runs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if len(jax_runs) < 2 or len(torch_runs) < 2:
        raise ValueError("at least two independent runs of each engine are required")
    case = load_case(case_path)
    case_digest = diagnostic.sha256(case_path / "manifest.json")
    groups = [_read_run(path, role, case_path, case) for role, paths in
              zip(diagnostic.REFERENCE_KEYS, (jax_runs, torch_runs)) for path in paths]
    ids = [evidence["report"]["execution_id"] for _, evidence in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate executions are not repeatability evidence")
    for group in (groups[:len(jax_runs)], groups[len(jax_runs):]):
        anchor = group[0][1]["report"]
        for _, item in group[1:]:
            if any(item["report"][key] != anchor[key] for key in ("engine_source", "runtime", "precision", "workload", "runner_sha256")):
                raise ValueError("repeat runs must use the same engine source, environment and policy")
    archive = {"baseline_keys": np.asarray(diagnostic.REFERENCE_KEYS),
               "baseline_titles": np.asarray(["Official JAX strict FP32", "H100 Torch Native eager strict FP32"]),
               "variable_names": np.asarray(list(diagnostic.VARIABLES)),
               "variable_labels": np.asarray(list(diagnostic.VARIABLES.values())),
               "lead_hour": np.asarray(case["lead_hours"]),
               "time_index": np.asarray(case["lead_hours"] // case["step_hours"] - 1),
               **{name: np.asarray(value) for name, value in case["coordinates"].items()}}
    for key, (fields, _) in zip(diagnostic.REFERENCE_KEYS, (groups[0], groups[len(jax_runs)])):
        archive.update({f"{key}__{name}": value for name, value in fields.items()})
    comparisons = [diagnostic._compare(_as_frame(fields, case), archive) for fields, _ in groups]
    cross = comparisons[len(jax_runs)][0]
    repeats = []
    for i, rows in enumerate(comparisons):
        own = rows[0 if i < len(jax_runs) else 1]
        repeats.append({"execution_id": ids[i], "overall_mean_percent": own["overall_mean_percent"],
                        "z500_percent": own["z500_percent"]})
    stable = all(max(row["overall_mean_percent"], row["z500_percent"]) < case["repeatability_percent"] for row in repeats)
    agreed = all(row[0]["metrics_within_limits"] for row in comparisons[len(jax_runs):])
    result = {"schema": "graphcast.reference_qualification.v1", "status": "qualified" if stable and agreed else "unqualified",
              "cross_reference": cross, "repeatability": repeats, "repeatability_percent": case["repeatability_percent"],
              "repeatability_passed": stable, "cross_reference_passed": agreed,
              "case_sha256": case_digest, "scope": "one_initialization_and_selected_lead_only",
              "weather_skill": "not_evaluated"}
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "baseline.npz", **archive)
    result["baseline_sha256"] = diagnostic.sha256(output / "baseline.npz")
    result["runs"] = []
    # Preserve raw evidence; never replace the historical NPZ or source runs.
    for index, (_, evidence) in enumerate(groups):
        dest = output / "references" / str(index)
        dest.mkdir(parents=True)
        for filename, digest_key in (("prediction.nc", "prediction_sha256"), ("reference_report.json", "report_sha256")):
            source = Path(evidence["path"]) / filename
            shutil.copyfile(source, dest / filename)
            diagnostic._check_digest(diagnostic.sha256(dest / filename), evidence[digest_key], filename)
            diagnostic._check_digest(diagnostic.sha256(source), evidence[digest_key], filename)
        result["runs"].append({**{key: value for key, value in evidence.items() if key != "report"},
                               "archived_path": str(dest.relative_to(output))})
    load_case(case_path)
    diagnostic._check_digest(diagnostic.sha256(case_path / "manifest.json"), case_digest, "case manifest")
    write_json(output / "qualification.json", result)
    write_json(output / "baseline.json", {
        "schema": "graphcast.fixed_baseline.v2", "provenance_status": "verified",
        "provenance_evidence": "Frozen case and bound independent execution reports; conversion identity recorded, numerical agreement qualified separately.",
        "baseline_sha256": result["baseline_sha256"], "source": case_source(case),
        **{key: case[key] for key in ("initialization_time", "lead_hours", "units")},
        "case_sha256": case_digest, "qualification": result,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="copy one shared case from a JSON asset specification")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    qualify_parser = sub.add_parser("qualify", help="qualify completed independent strict references")
    qualify_parser.add_argument("--case", type=Path, required=True)
    qualify_parser.add_argument("--jax-run", type=Path, action="append", required=True)
    qualify_parser.add_argument("--torch-run", type=Path, action="append", required=True)
    qualify_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            freeze_case(json.loads(args.spec.read_text()), args.output_dir)
            print("case frozen; reference qualification not run")
            return 0
        result = qualify(args.case, args.jax_run, args.torch_run, args.output_dir)
        print(result["status"])
        return 0 if result["status"] == "qualified" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"reference evidence error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
