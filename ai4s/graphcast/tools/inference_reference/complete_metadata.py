#!/usr/bin/env python3
"""Complete explicitly declared metadata on legacy NetCDF assets, losslessly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import xarray as xr

SCHEMA = "graphcast.metadata_completion_audit.v1"
_BOUND_ATTRIBUTES = (
    "metadata_completion_source_sha256",
    "metadata_completion_provenance",
)
_DATETIME_PATTERN = re.compile(
    r"^(?P<whole>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d{1,9}))?$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _check_source_hash(name: str, source: Path, expected_hash: str) -> None:
    if _sha256(source) != expected_hash:
        raise ValueError(f"asset {name!r} hash differs from declared sha256")


def _parse_datetimes(name: str, declaration: object, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(declaration, dtype=object)
    if raw.shape != shape:
        raise ValueError(f"asset {name!r} datetimes must have batch/time shape {shape}")
    parsed = []
    for value in raw.flat:
        match = _DATETIME_PATTERN.fullmatch(value) if isinstance(value, str) else None
        if match is None:
            raise ValueError(f"asset {name!r} datetimes must be ISO timestamp strings")
        try:
            native = np.datetime64(value)
            nanoseconds = native.astype("datetime64[ns]")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"asset {name!r} datetimes contain an unsupported timestamp") from error
        if np.isnat(nanoseconds) or nanoseconds.astype(native.dtype) != native:
            raise ValueError(f"asset {name!r} datetimes must round-trip exactly at nanosecond precision")
        normalized = match["whole"] + "." + (match["fraction"] or "").ljust(9, "0")
        if np.datetime_as_string(nanoseconds, unit="ns") != normalized:
            raise ValueError(f"asset {name!r} datetimes exceed the nanosecond timestamp range")
        parsed.append(nanoseconds)
    return np.asarray(parsed, dtype="datetime64[ns]").reshape(shape)


def _validate_entry(name: str, entry: object) -> tuple[Path, str, np.ndarray, dict, dict]:
    if not isinstance(entry, dict):
        raise ValueError(f"asset {name!r} entry must be an object")
    source = Path(entry.get("path", ""))
    if not source.is_file():
        raise ValueError(f"asset {name!r} path is not a file")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str):
        raise ValueError(f"asset {name!r} hash differs from declared sha256")
    expected_hash = expected_hash.lower()
    _check_source_hash(name, source, expected_hash)
    units = entry.get("units")
    if not isinstance(units, dict):
        raise ValueError(f"asset {name!r} units must be an object")
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(f"asset {name!r} provenance must be a nonempty object")
    for field in ("time_basis", "unit_basis"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ValueError(f"asset {name!r} provenance {field} must be a nonempty string")
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"asset {name!r} provenance must be finite JSON") from error

    with xr.open_dataset(source, decode_times=False, decode_timedelta=False) as opened:
        dataset = opened.load()
    if "batch" not in dataset.sizes or "time" not in dataset.sizes:
        raise ValueError(f"asset {name!r} requires existing batch and time dimensions")
    expected_shape = (dataset.sizes["batch"], dataset.sizes["time"])
    datetimes = _parse_datetimes(name, entry.get("datetimes"), expected_shape)

    unknown = set(units).difference(dataset.variables)
    if unknown:
        raise ValueError(f"asset {name!r} units contain unknown names: {sorted(unknown)}")
    for field, value in units.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"asset {name!r} unit for {field!r} must be nonempty")
        existing = dataset[field].attrs.get("units")
        if existing is not None and existing != value:
            raise ValueError(f"asset {name!r}/{field} has conflicting units")

    if "datetime" in dataset.variables:
        if dataset["datetime"].dims != ("batch", "time"):
            raise ValueError(f"asset {name!r} existing datetime must have batch/time dimensions")
        with xr.open_dataset(source, decode_times=True, decode_timedelta=False) as decoded:
            existing_datetimes = np.asarray(decoded["datetime"].values, dtype="datetime64[ns]")
        if not np.array_equal(existing_datetimes, datetimes):
            raise ValueError(f"asset {name!r} existing datetime differs from declaration")
    for attribute in _BOUND_ATTRIBUTES:
        if attribute in dataset.attrs:
            raise ValueError(f"asset {name!r} already has reserved attribute {attribute!r}")
    return source, expected_hash, datetimes, units, provenance


def _assert_original_arrays_unchanged(source: Path, completed: Path) -> None:
    with xr.open_dataset(source, decode_times=False, decode_timedelta=False) as before, xr.open_dataset(
        completed, decode_times=False, decode_timedelta=False
    ) as after:
        for name, original in before.variables.items():
            if name not in after.variables or after[name].dims != original.dims:
                raise RuntimeError(f"serialization changed original variable {name!r}")
            if after[name].dtype != original.dtype:
                raise RuntimeError(f"serialization changed dtype of original variable {name!r}")
            if not np.array_equal(after[name].values, original.values, equal_nan=True):
                raise RuntimeError(f"serialization changed values of original variable {name!r}")


def _assert_serialized_datetimes(completed: Path, declared: np.ndarray) -> None:
    with xr.open_dataset(completed, decode_times=True, decode_timedelta=False) as dataset:
        if "datetime" not in dataset.coords or dataset["datetime"].dims != ("batch", "time"):
            raise RuntimeError("serialized datetime does not have batch/time coordinate form")
        actual = np.asarray(dataset["datetime"].values, dtype="datetime64[ns]")
    if not np.array_equal(actual, declared):
        raise RuntimeError("serialized datetime differs from declaration")


def complete_metadata(spec: dict, output: Path) -> dict:
    """Apply declared datetime/unit metadata and return the terminal audit."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    if not isinstance(spec, dict) or not isinstance(spec.get("assets"), dict) or not spec["assets"]:
        raise ValueError("spec assets must be a nonempty object")

    validated: dict[str, tuple[Path, str, np.ndarray, dict, dict]] = {}
    for name, entry in spec["assets"].items():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("asset names must be nonempty relative paths")
        validated[name] = _validate_entry(name, entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        audit = {"schema": SCHEMA, "status": "passed", "assets": {}}
        for name, (source, expected_hash, datetimes, units, provenance) in validated.items():
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _check_source_hash(name, source, expected_hash)
            with xr.open_dataset(source, decode_times=False, decode_timedelta=False) as opened:
                dataset = opened.load()
            _check_source_hash(name, source, expected_hash)
            if "datetime" not in dataset.variables:
                dataset = dataset.assign_coords(datetime=(("batch", "time"), datetimes))
            for field, unit in units.items():
                dataset[field].attrs["units"] = unit
            dataset.attrs["metadata_completion_source_sha256"] = expected_hash
            dataset.attrs["metadata_completion_provenance"] = json.dumps(
                provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            dataset.to_netcdf(destination)
            _assert_original_arrays_unchanged(source, destination)
            _assert_serialized_datetimes(destination, datetimes)
            _check_source_hash(name, source, expected_hash)
            declarations = {
                "datetimes": spec["assets"][name]["datetimes"],
                "units": units,
                "provenance": provenance,
            }
            audit["assets"][name] = {
                "source_path": str(source.resolve()),
                "source_sha256": expected_hash,
                "completed_sha256": _sha256(destination),
                "declarations": declarations,
            }
        _write_json(temporary / "metadata_audit.json", audit)
        temporary.replace(output)
        return audit
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON metadata declaration")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    complete_metadata(json.loads(args.spec.read_text()), args.output_dir)
    print((args.output_dir / "metadata_audit.json").read_text(), end="")


if __name__ == "__main__":
    main()
