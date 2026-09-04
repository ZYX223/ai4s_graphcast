from __future__ import annotations

import importlib.util
from pathlib import Path

import netCDF4 as nc
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tools" / "bf16_inference" / "hash_netcdf_semantics.py"
SPEC = importlib.util.spec_from_file_location("hash_netcdf_semantics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, *, compressed: bool, delta: float = 0.0) -> None:
    with nc.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("x", 3)
        variable = dataset.createVariable("forcing", "f4", ("time", "x"), zlib=compressed)
        variable[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
        variable[1, 2] += delta
        dataset.history = "encoding metadata is intentionally ignored"


def test_semantic_hash_ignores_netcdf_encoding(tmp_path: Path) -> None:
    left = tmp_path / "left.nc"
    right = tmp_path / "right.nc"
    _write(left, compressed=False)
    _write(right, compressed=True)

    assert left.read_bytes() != right.read_bytes()
    assert MODULE.semantic_manifest(left)["semantic_sha256"] == MODULE.semantic_manifest(right)[
        "semantic_sha256"
    ]


def test_semantic_hash_changes_with_values(tmp_path: Path) -> None:
    left = tmp_path / "left.nc"
    right = tmp_path / "right.nc"
    _write(left, compressed=False)
    _write(right, compressed=False, delta=1.0)

    assert MODULE.semantic_manifest(left)["semantic_sha256"] != MODULE.semantic_manifest(right)[
        "semantic_sha256"
    ]
