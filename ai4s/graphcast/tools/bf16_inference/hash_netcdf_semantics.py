#!/usr/bin/env python3
"""Hash NetCDF variable layout and values while ignoring container encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np


def _update(digest: Any, value: object) -> None:
    digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")


def _canonical_bytes(value: object) -> tuple[bytes, bytes]:
    masked = np.ma.asarray(value)
    mask = np.ascontiguousarray(np.ma.getmaskarray(masked), dtype=np.bool_)
    array = np.ascontiguousarray(np.ma.filled(masked, 0))
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
        array = array.copy()
        array[np.isnan(array)] = np.nan
    return mask.tobytes(), array.tobytes()


def semantic_manifest(path: Path) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    dataset_digest = hashlib.sha256()
    with nc.Dataset(path) as dataset:
        dimensions = {
            name: {"size": len(dimension), "unlimited": dimension.isunlimited()}
            for name, dimension in sorted(dataset.dimensions.items())
        }
        _update(dataset_digest, dimensions)
        for name in sorted(dataset.variables):
            variable = dataset.variables[name]
            subject = {
                "name": name,
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "dtype": np.dtype(variable.dtype).str,
            }
            variable_digest = hashlib.sha256()
            _update(variable_digest, subject)
            if "time" in variable.dimensions:
                axis = variable.dimensions.index("time")
                selections = []
                for index in range(variable.shape[axis]):
                    selection = [slice(None)] * len(variable.shape)
                    selection[axis] = index
                    selections.append(tuple(selection))
            else:
                selections = [(slice(None),) * len(variable.shape)]
            for selection in selections:
                mask_bytes, value_bytes = _canonical_bytes(variable[selection])
                variable_digest.update(mask_bytes)
                variable_digest.update(value_bytes)
            record = {**subject, "sha256": variable_digest.hexdigest()}
            variables.append(record)
            _update(dataset_digest, record)
    return {
        "schema": "graphcast.netcdf_semantic_identity.v1",
        "path": str(path.resolve()),
        "dimensions": dimensions,
        "variables": variables,
        "semantic_sha256": dataset_digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("input", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(semantic_manifest(args.input), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
