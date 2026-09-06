"""Tests for lossless, declarative completion of legacy NetCDF metadata."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def api():
    script = Path(__file__).parents[1] / "tools/inference_reference/complete_metadata.py"
    spec = importlib.util.spec_from_file_location("complete_reference_metadata", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_asset(tmp_path: Path, *, existing_metadata: bool = False) -> tuple[Path, dict]:
    path = tmp_path / "legacy.nc"
    values = np.asarray([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    dataset = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), values),
            "station_id": (("lat",), np.asarray([7, 9], dtype=np.int16)),
        },
        coords={
            "batch": ("batch", np.asarray([3], dtype=np.int8)),
            "time": ("time", np.asarray([-6, 0], dtype="timedelta64[h]")),
            "lat": ("lat", np.asarray([-45.0, 45.0], dtype=np.float32)),
            "lon": ("lon", np.asarray([0.0, 180.0], dtype=np.float64)),
        },
        attrs={"legacy_label": "untouched"},
    )
    datetimes = [["2021-01-01T00:00:00", "2021-01-01T06:00:00"]]
    if existing_metadata:
        dataset = dataset.assign_coords(
            datetime=(("batch", "time"), np.asarray(datetimes, dtype="datetime64[ns]"))
        )
        dataset["temperature"].attrs["units"] = "K"
    dataset.to_netcdf(path)
    entry = {
        "path": str(path),
        "sha256": sha256(path),
        "datetimes": datetimes,
        "units": {"temperature": "K", "lat": "degrees_north"},
        "provenance": {
            "time_basis": "reviewed case sidecar supplied by data owner",
            "unit_basis": "reviewed variable manifest supplied by data owner",
            "qualification": "declarative metadata only; upstream weather provenance not authenticated",
        },
    }
    return path, entry


def assert_same_array(left: xr.DataArray, right: xr.DataArray) -> None:
    assert right.dims == left.dims
    assert right.dtype == left.dtype
    np.testing.assert_equal(right.values, left.values)


def test_completes_declared_metadata_without_changing_original_arrays(tmp_path):
    source, entry = legacy_asset(tmp_path)
    output = tmp_path / "completed"

    result = api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    completed = output / "inputs.nc"
    assert result == json.loads((output / "metadata_audit.json").read_text())
    assert result["schema"] == "graphcast.metadata_completion_audit.v1"
    assert result["status"] == "passed"
    assert result["assets"]["inputs.nc"]["source_sha256"] == sha256(source)
    assert result["assets"]["inputs.nc"]["completed_sha256"] == sha256(completed)
    assert result["assets"]["inputs.nc"]["declarations"] == {
        "datetimes": entry["datetimes"],
        "units": entry["units"],
        "provenance": entry["provenance"],
    }
    with xr.open_dataset(source, decode_times=False, decode_timedelta=False) as before, xr.open_dataset(
        completed, decode_times=False, decode_timedelta=False
    ) as after:
        for name in before.variables:
            assert name in after.variables
            assert_same_array(before[name], after[name])
        assert after["temperature"].attrs["units"] == "K"
        assert after["lat"].attrs["units"] == "degrees_north"
        assert after["datetime"].dims == ("batch", "time")
        np.testing.assert_array_equal(
            xr.decode_cf(after)["datetime"].values,
            np.asarray(entry["datetimes"], dtype="datetime64[ns]"),
        )
        assert after.attrs["legacy_label"] == "untouched"
        assert after.attrs["metadata_completion_source_sha256"] == sha256(source)
        bound = json.loads(after.attrs["metadata_completion_provenance"])
        assert bound == entry["provenance"]
        assert "ERA5" not in after.attrs["metadata_completion_provenance"]


def test_accepts_matching_existing_metadata(tmp_path):
    _, entry = legacy_asset(tmp_path, existing_metadata=True)
    output = tmp_path / "completed"

    api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    with xr.open_dataset(output / "inputs.nc") as completed:
        assert completed["datetime"].dims == ("batch", "time")
        assert completed["temperature"].attrs["units"] == "K"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda entry: entry.update(sha256="0" * 64), "hash"),
        (lambda entry: entry.update(datetimes=[["2021-01-01T00:00:00"]]), "datetimes"),
        (lambda entry: entry.update(units={"unknown": "K"}), "unknown"),
        (lambda entry: entry.update(units={"temperature": ""}), "nonempty"),
        (lambda entry: entry.update(provenance={}), "provenance"),
    ],
)
def test_rejects_invalid_declarations_without_terminal_audit(tmp_path, mutation, match):
    _, entry = legacy_asset(tmp_path)
    mutation(entry)
    output = tmp_path / "completed"

    with pytest.raises(ValueError, match=match):
        api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert not output.exists()


def test_rejects_conflicting_existing_units_and_datetimes(tmp_path):
    _, entry = legacy_asset(tmp_path, existing_metadata=True)
    output = tmp_path / "completed"
    conflicting_units = json.loads(json.dumps(entry))
    conflicting_units["units"]["temperature"] = "degC"
    with pytest.raises(ValueError, match="conflicting units"):
        api().complete_metadata({"assets": {"inputs.nc": conflicting_units}}, output)

    conflicting_time = json.loads(json.dumps(entry))
    conflicting_time["datetimes"][0][1] = "2021-01-02T06:00:00"
    with pytest.raises(ValueError, match="existing datetime"):
        api().complete_metadata({"assets": {"inputs.nc": conflicting_time}}, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "datetimes",
    [
        [[0, 1]],
        [["2500-01-01T00:00:00", "2500-01-01T06:00:00"]],
        [["2021-01-01T00:00:00.1234567891", "2021-01-01T06:00:00"]],
    ],
)
def test_rejects_non_string_or_non_roundtrippable_datetimes(tmp_path, datetimes):
    _, entry = legacy_asset(tmp_path)
    entry["datetimes"] = datetimes
    output = tmp_path / "completed"

    with pytest.raises(ValueError, match="datetimes"):
        api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "datetimes",
    [
        [["1600-01-01T00:00:00.123456789", "1600-01-01T06:00:00.123456789"]],
        [["2500-01-01T00:00:00.123456789", "2500-01-01T06:00:00.123456789"]],
    ],
)
def test_rejects_nine_digit_datetimes_outside_nanosecond_range(tmp_path, datetimes):
    _, entry = legacy_asset(tmp_path)
    entry["datetimes"] = datetimes
    output = tmp_path / "completed"

    with pytest.raises(ValueError, match="datetimes"):
        api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "provenance",
    [
        {"time_basis": "reviewed time evidence"},
        {"unit_basis": "reviewed unit evidence"},
        {"time_basis": " ", "unit_basis": "reviewed unit evidence"},
    ],
)
def test_requires_nonempty_time_and_unit_provenance_evidence(tmp_path, provenance):
    _, entry = legacy_asset(tmp_path)
    entry["provenance"] = provenance
    output = tmp_path / "completed"

    with pytest.raises(ValueError, match="(time_basis|unit_basis)"):
        api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert not output.exists()


def test_rejects_datetime_changed_during_serialization(tmp_path, monkeypatch):
    _, entry = legacy_asset(tmp_path)
    output = tmp_path / "completed"
    module = api()
    assert_unchanged = module._assert_original_arrays_unchanged

    def assert_then_corrupt(source, completed_path):
        assert_unchanged(source, completed_path)
        with xr.open_dataset(completed_path) as opened:
            corrupted = opened.load().assign_coords(datetime=opened.datetime + np.timedelta64(1, "h"))
        replacement = completed_path.with_suffix(".replacement.nc")
        corrupted.to_netcdf(replacement)
        replacement.replace(completed_path)

    monkeypatch.setattr(module, "_assert_original_arrays_unchanged", assert_then_corrupt)
    with pytest.raises(RuntimeError, match="serialized datetime"):
        module.complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert not output.exists()


def test_rejects_existing_destination(tmp_path):
    _, entry = legacy_asset(tmp_path)
    output = tmp_path / "completed"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing")

    with pytest.raises(FileExistsError):
        api().complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert marker.read_text() == "existing"
    assert not (output / "metadata_audit.json").exists()


def test_rejects_source_mutated_after_initial_hash_check(tmp_path, monkeypatch):
    source, entry = legacy_asset(tmp_path)
    output = tmp_path / "completed"
    module = api()
    validate_entry = module._validate_entry

    def validate_then_mutate(name, declaration):
        validated = validate_entry(name, declaration)
        with source.open("ab") as stream:
            stream.write(b"changed after validation")
        return validated

    monkeypatch.setattr(module, "_validate_entry", validate_then_mutate)
    with pytest.raises(ValueError, match="hash"):
        module.complete_metadata({"assets": {"inputs.nc": entry}}, output)

    assert not output.exists()


def test_cli_reads_json_spec_and_writes_completed_directory(tmp_path):
    _, entry = legacy_asset(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"assets": {"inputs.nc": entry}}))
    output = tmp_path / "completed"
    script = Path(__file__).parents[1] / "tools/inference_reference/complete_metadata.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--spec", str(spec), "--output-dir", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "passed"
    assert (output / "inputs.nc").is_file()
    assert (output / "metadata_audit.json").is_file()
