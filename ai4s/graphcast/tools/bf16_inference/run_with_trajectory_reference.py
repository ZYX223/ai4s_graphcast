#!/usr/bin/env python3
"""Run GraphCast while reducing each lead against an immutable trajectory reference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_reference_dataset(path: Path) -> xr.Dataset:
    # Lead time is used only as an ordered axis here.  Avoid decoding its
    # serialization metadata because older platform xarray releases reject
    # files written by newer xarray when both ``units`` and ``dtype`` remain.
    return xr.open_dataset(path, decode_timedelta=False)


def _load_runner(project_root: Path) -> Any:
    for path in (project_root, project_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runner_path = project_root / "scripts/inference/run_operational_40step_inference.py"
    spec = importlib.util.spec_from_file_location("graphcast_rollout_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load rollout runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrajectoryReferenceWriter:
    """Drop-in trajectory sink that stores metrics instead of duplicate grid tensors."""

    def __init__(
        self,
        output: Path,
        *,
        steps: int,
        step_hours: int,
        valid_times: xr.DataArray | None = None,
    ) -> None:
        reference_path = Path(os.environ["GRAPHCAST_TRAJECTORY_REFERENCE"])
        stddev_path = Path(os.environ["GRAPHCAST_TRAJECTORY_STDDEV"])
        if not reference_path.is_file() or not stddev_path.is_file():
            raise FileNotFoundError(reference_path if not reference_path.is_file() else stddev_path)
        self.output = output
        self.steps = steps
        self.step_hours = step_hours
        self._reference_path = reference_path
        self._stddev_path = stddev_path
        self._reference_sha256 = _sha256(reference_path)
        self._stddev_sha256 = _sha256(stddev_path)
        self._reference = _open_reference_dataset(reference_path)
        self._stddev = xr.open_dataset(stddev_path)
        if int(self._reference.sizes.get("lead_time", 0)) != steps:
            raise ValueError("reference trajectory lead count differs from requested steps")
        if valid_times is None or int(valid_times.size) != steps:
            raise ValueError("candidate valid times differ from requested steps")
        self._next_step = 0
        self._records: list[dict[str, Any]] = []
        self._values = 0
        self._sum_abs = 0.0
        self._sum_sq = 0.0
        self._reference_sum_sq = 0.0
        self._max_abs = 0.0
        self._all_finite = True
        self._allclose = True
        self._temporary: Path | None = None

    @staticmethod
    def _candidate_array(array: xr.DataArray) -> xr.DataArray:
        for dimension in ("time", "lead_time"):
            if dimension in array.dims:
                if int(array.sizes[dimension]) != 1:
                    raise ValueError("candidate prediction must contain one lead")
                array = array.isel({dimension: 0}, drop=True)
        return array

    def append(self, step_index: int, prediction: xr.Dataset) -> None:
        if step_index != self._next_step or step_index >= self.steps:
            raise ValueError("candidate leads must be compared once in order")
        if set(prediction.data_vars) != set(self._reference.data_vars):
            raise ValueError("candidate and reference variables differ")
        for name in sorted(prediction.data_vars):
            reference = self._reference[name].isel(lead_time=step_index).load()
            candidate = self._candidate_array(prediction[name])
            if set(candidate.dims) != set(reference.dims):
                raise ValueError(f"candidate and reference dimensions differ for {name!r}")
            candidate = candidate.transpose(*reference.dims)
            for dimension in reference.dims:
                if dimension in reference.coords and dimension in candidate.coords:
                    if not np.array_equal(reference[dimension], candidate[dimension]):
                        raise ValueError(
                            f"candidate and reference coordinates differ for {name!r}: {dimension}"
                        )
            levels: list[Any | None] = (
                list(reference["level"].values) if "level" in reference.dims else [None]
            )
            if name not in self._stddev:
                raise ValueError(f"normalization stddev is missing {name!r}")
            for level in levels:
                selection = {"level": level} if level is not None else {}
                reference_record = reference.sel(selection)
                candidate_record = candidate.sel(selection)
                scale = self._stddev[name]
                if level is not None and "level" in scale.dims:
                    scale = scale.sel(level=level)
                scale_values = np.asarray(scale.values, dtype=np.float64)
                if not np.isfinite(scale_values).all() or np.any(scale_values <= 0):
                    raise ValueError(f"normalization stddev is invalid for {name!r}")
                reference_values = np.asarray(reference_record.values, dtype=np.float64)
                candidate_values = np.asarray(candidate_record.values, dtype=np.float64)
                difference = candidate_values - reference_values
                finite = bool(
                    np.isfinite(reference_values).all()
                    and np.isfinite(candidate_values).all()
                    and np.isfinite(difference).all()
                )
                absolute = np.abs(difference)
                normalized = difference / scale_values
                values = int(difference.size)
                sum_abs = float(np.sum(absolute, dtype=np.float64))
                sum_sq = float(np.sum(np.square(difference), dtype=np.float64))
                reference_sum_sq = float(
                    np.sum(np.square(reference_values), dtype=np.float64)
                )
                max_abs = float(np.max(absolute))
                record_allclose = bool(
                    np.allclose(candidate_values, reference_values, atol=1.0e-5, rtol=1.0e-5)
                )
                self._records.append(
                    {
                        "variable": name,
                        "level": int(level) if level is not None else None,
                        "lead_hours": float((step_index + 1) * self.step_hours),
                        "finite": finite,
                        "normalized_max_abs": float(np.max(np.abs(normalized))),
                        "normalized_rmse": float(np.sqrt(np.mean(np.square(normalized)))),
                        "parity_max_abs": max_abs,
                        "parity_mean_abs": sum_abs / values,
                        "parity_rmse": math.sqrt(sum_sq / values),
                        "parity_relative_l2": (
                            math.sqrt(sum_sq / reference_sum_sq)
                            if reference_sum_sq > 0
                            else None
                        ),
                        "parity_allclose": record_allclose,
                    }
                )
                self._values += values
                self._sum_abs += sum_abs
                self._sum_sq += sum_sq
                self._reference_sum_sq += reference_sum_sq
                self._max_abs = max(self._max_abs, max_abs)
                self._all_finite = self._all_finite and finite
                self._allclose = self._allclose and record_allclose
        self._next_step += 1

    def seal(self) -> Path:
        if self._next_step != self.steps:
            self.abort()
            raise RuntimeError(
                f"trajectory comparison is incomplete: {self._next_step} of {self.steps} leads"
            )
        if self._temporary is not None:
            return self._temporary
        if _sha256(self._reference_path) != self._reference_sha256:
            raise RuntimeError("reference trajectory changed during comparison")
        if _sha256(self._stddev_path) != self._stddev_sha256:
            raise RuntimeError("normalization stddev changed during comparison")
        payload = {
            "schema": "graphcast.streaming_trajectory_diagnostics.v1",
            "status": "measured" if self._all_finite else "failed_nonfinite",
            "methodology": {
                "comparison": "candidate_minus_native_fp32_at_matching_autoregressive_lead",
                "normalization": "graphcast_stddev_by_level",
                "weather_skill": "not_measured_without_time_matched_truth_and_climatology",
            },
            "inputs": {
                "reference": {
                    "path": str(self._reference_path.resolve()),
                    "sha256": self._reference_sha256,
                },
                "stddev": {
                    "path": str(self._stddev_path.resolve()),
                    "sha256": self._stddev_sha256,
                },
            },
            "lead_count": self.steps,
            "records": self._records,
            "implementation_parity": {
                "finite": self._all_finite,
                "values": self._values,
                "max_abs": self._max_abs,
                "mean_abs": self._sum_abs / self._values,
                "rmse": math.sqrt(self._sum_sq / self._values),
                "relative_l2": (
                    math.sqrt(self._sum_sq / self._reference_sum_sq)
                    if self._reference_sum_sq > 0
                    else None
                ),
                "allclose": self._allclose,
                "atol": 1.0e-5,
                "rtol": 1.0e-5,
            },
            "worst": {
                "normalized_max_abs": max(
                    record["normalized_max_abs"] for record in self._records
                ),
                "normalized_rmse": max(
                    record["normalized_rmse"] for record in self._records
                ),
            },
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._temporary = Path(temporary_name)
        self._reference.close()
        self._stddev.close()
        return self._temporary

    def publish(self) -> None:
        temporary = self.seal()
        os.replace(temporary, self.output)
        self._temporary = None

    def abort(self) -> None:
        self._reference.close()
        self._stddev.close()
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None


def main() -> int:
    project_root = Path(os.environ["PROJECT_ROOT"])
    runner = _load_runner(project_root)
    runner.TrajectoryNetCDFWriter = TrajectoryReferenceWriter
    return int(runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
