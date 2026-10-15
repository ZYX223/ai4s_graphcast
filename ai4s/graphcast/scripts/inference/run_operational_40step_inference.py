#!/usr/bin/env python3
"""Run GraphCast as a device-resident autoregressive rollout of any positive length."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = GRAPHCAST_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    load_normalization_stats,
)
from graphcast_compat.autoregressive_inference import (  # noqa: E402
    advance_device_rollout,
    normalized_output_to_prediction,
    prepare_device_rollout,
)
from graphcast_compat.backend import (  # noqa: E402
    configure_float32_math,
    memory_text,
    require_complete_flaggems_dispatch,
    reset_peak_memory,
    resolve_runtime_backend,
    use_prevalidated_flaggems_index_add,
)
from graphcast_compat.backend import (  # noqa: E402
    synchronize as synchronize_runtime,
)
from graphcast_compat.fusion_candidate_profile import (  # noqa: E402
    PostLayerNormResidualRecorder,
)
from graphcast_compat.google_small import (  # noqa: E402
    _environment_flag,
    require_post_layer_norm_residual,
)
from graphcast_compat.inference_precision import (  # noqa: E402
    InferencePrecisionPolicy,
    resolve_inference_precision,
    run_precision_forward,
)
from graphcast_compat.normalization import REQUIRED_STATS_FILES  # noqa: E402

FORWARD_FLOPS = 28_857_094_475_776


@dataclass(frozen=True)
class RolloutDtypeEvidence:
    state: str
    forcing: str
    model_input: str
    model_output: str
    feedback: str
    final_output: str


@dataclass(frozen=True)
class RolloutMeasurement:
    steps: int
    warmup_calls: int
    e2e_seconds: float
    final_output: torch.Tensor
    finite: bool
    dtype_evidence: RolloutDtypeEvidence


class TrajectoryNetCDFWriter:
    """Stream physical predictions into one terminally published NetCDF file."""

    def __init__(
        self,
        output: Path,
        *,
        steps: int,
        step_hours: int,
        valid_times: xr.DataArray | None = None,
    ) -> None:
        if steps <= 0 or step_hours <= 0:
            raise ValueError("trajectory steps and step hours must be positive")
        self.output = output
        self.steps = steps
        self.step_hours = step_hours
        self.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.stem}.", suffix=".tmp.nc", dir=output.parent
        )
        os.close(descriptor)
        os.unlink(temporary)
        self._temporary = Path(temporary)
        self._dataset: Any | None = None
        self._next_step = 0
        self._signature: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] | None = None
        self._coordinate_values: dict[str, np.ndarray] = {}
        self._valid_times = valid_times
        if valid_times is None:
            raise ValueError("diagnostic trajectories require forcing valid times")
        time_dimensions = [
            dimension for dimension in ("time", "lead_time") if dimension in valid_times.dims
        ]
        if len(time_dimensions) != 1 or valid_times.sizes[time_dimensions[0]] != steps:
            raise ValueError("valid times must contain exactly one value per trajectory lead")
        if not np.issubdtype(valid_times.dtype, np.datetime64):
            raise ValueError("valid times must use a datetime64 dtype")
        self._valid_time_dimension = time_dimensions[0]
        reference_dimensions = tuple(
            dimension for dimension in valid_times.dims if dimension != self._valid_time_dimension
        )
        if reference_dimensions not in ((), ("batch",)):
            raise ValueError("valid times may vary only over batch and lead")
        ordered_valid_times = np.moveaxis(
            np.asarray(valid_times.values),
            valid_times.dims.index(self._valid_time_dimension),
            -1,
        )
        lead_offsets = np.arange(1, steps + 1, dtype=np.int64) * np.timedelta64(step_hours, "h")
        references = ordered_valid_times - lead_offsets
        if not np.all(references == references[..., :1]):
            raise ValueError(
                "valid times must be regular autoregressive leads from one initialization"
            )
        self._reference_dimensions = reference_dimensions
        self._reference_values = references[..., 0]

    @staticmethod
    def _lead_dimension(prediction: xr.Dataset) -> str | None:
        for name in ("time", "lead_time"):
            if name in prediction.dims:
                if prediction.sizes[name] != 1:
                    raise ValueError("each trajectory prediction must contain one lead")
                return name
        return None

    @staticmethod
    def _copy_attributes(destination: Any, attributes: dict[str, Any]) -> None:
        safe = {
            name: value
            for name, value in attributes.items()
            if name != "_FillValue" and isinstance(value, (str, int, float, np.number))
        }
        if safe:
            destination.setncatts(safe)

    def _initialize(self, prediction: xr.Dataset, lead_dimension: str | None) -> None:
        import netCDF4

        dataset = netCDF4.Dataset(self._temporary, mode="w", format="NETCDF4")
        dataset.setncatts(
            {
                "schema": "graphcast.inference_trajectory.v1",
                "steps": self.steps,
                "step_hours": self.step_hours,
                "performance_eligible": 0,
                "diagnostic_output_d2h_each_step": 1,
            }
        )
        dataset.createDimension("lead_time", self.steps)
        lead_coordinate = dataset.createVariable("lead_time", "i8", ("lead_time",))
        lead_coordinate[:] = np.arange(1, self.steps + 1, dtype=np.int64) * self.step_hours
        lead_coordinate.setncatts(
            {
                "units": "hours",
                "dtype": "timedelta64[ns]",
                "standard_name": "forecast_period",
            }
        )

        for dimension, size in prediction.sizes.items():
            if dimension != lead_dimension:
                dataset.createDimension(dimension, int(size))
        for name, coordinate in prediction.coords.items():
            if name == lead_dimension or lead_dimension in coordinate.dims:
                continue
            if coordinate.ndim != 1 or coordinate.dims != (name,):
                continue
            values = np.asarray(coordinate.values)
            if values.dtype.kind not in "biuf":
                continue
            variable = dataset.createVariable(name, values.dtype, (name,))
            variable[:] = values
            self._copy_attributes(variable, coordinate.attrs)
            self._coordinate_values[name] = values.copy()

        if "batch" not in dataset.dimensions:
            raise ValueError("diagnostic trajectory predictions require a batch dimension")
        if self._reference_dimensions == ():
            if len(dataset.dimensions["batch"]) != 1:
                raise ValueError("one-dimensional valid times require a single prediction batch")
            reference_values = np.asarray([self._reference_values])
            valid_time_values = np.asarray(self._valid_times.values).reshape(1, self.steps)
        else:
            if self._valid_times.sizes["batch"] != len(dataset.dimensions["batch"]):
                raise ValueError("valid-time batch size differs from prediction")
            if "batch" in self._valid_times.coords and "batch" in prediction.coords:
                if not np.array_equal(
                    np.asarray(self._valid_times["batch"].values),
                    np.asarray(prediction["batch"].values),
                ):
                    raise ValueError("valid-time batch labels differ from prediction")
            reference_values = np.asarray(self._reference_values)
            valid_time_values = np.moveaxis(
                np.asarray(self._valid_times.values),
                self._valid_times.dims.index(self._valid_time_dimension),
                -1,
            )
        forecast_reference_time = dataset.createVariable(
            "forecast_reference_time", "i8", ("batch",)
        )
        forecast_reference_time[:] = reference_values.astype("datetime64[s]").astype(np.int64)
        forecast_reference_time.setncatts(
            {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
                "standard_name": "forecast_reference_time",
            }
        )
        valid_time = dataset.createVariable("valid_time", "i8", ("batch", "lead_time"))
        valid_time[:] = valid_time_values.astype("datetime64[s]").astype(np.int64)
        valid_time.setncatts(
            {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
                "standard_name": "time",
            }
        )

        signature: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
        for name, array in prediction.data_vars.items():
            dimensions = tuple(
                "lead_time" if dimension == lead_dimension else dimension
                for dimension in array.dims
            )
            if lead_dimension is None:
                dimensions = ("lead_time", *dimensions)
            dataset.createVariable(
                name,
                np.asarray(array.values).dtype,
                dimensions,
                zlib=True,
                complevel=1,
            )
            self._copy_attributes(dataset.variables[name], array.attrs)
            existing = str(array.attrs.get("coordinates", "")).split()
            dataset.variables[name].setncattr(
                "coordinates",
                " ".join([*existing, "valid_time", "forecast_reference_time"]),
            )
            shape = tuple(
                int(size)
                for dimension, size in zip(array.dims, array.shape)
                if dimension != lead_dimension
            )
            signature[name] = (array.dims, shape)
        self._dataset = dataset
        self._signature = signature

    def _validate(self, prediction: xr.Dataset, lead_dimension: str | None) -> None:
        assert self._signature is not None
        if set(prediction.data_vars) != set(self._signature):
            raise ValueError("trajectory variables changed between leads")
        for name, (dimensions, shape) in self._signature.items():
            array = prediction[name]
            current_shape = tuple(
                int(size)
                for dimension, size in zip(array.dims, array.shape)
                if dimension != lead_dimension
            )
            if array.dims != dimensions or current_shape != shape:
                raise ValueError(f"trajectory layout changed for {name!r}")
        for name, expected in self._coordinate_values.items():
            if name not in prediction.coords or not np.array_equal(
                np.asarray(prediction[name].values), expected
            ):
                raise ValueError(f"trajectory coordinate changed between leads: {name}")

    def append(self, step_index: int, prediction: xr.Dataset) -> None:
        if step_index != self._next_step or step_index >= self.steps:
            raise ValueError("trajectory leads must be appended once in order")
        lead_dimension = self._lead_dimension(prediction)
        if self._dataset is None:
            self._initialize(prediction, lead_dimension)
        else:
            self._validate(prediction, lead_dimension)
        assert self._dataset is not None
        for name, array in prediction.data_vars.items():
            values = np.asarray(array.values)
            if lead_dimension is None:
                target = (step_index, *([slice(None)] * values.ndim))
            else:
                source_axis = array.dims.index(lead_dimension)
                values = np.take(values, 0, axis=source_axis)
                target = tuple(
                    step_index if dimension == lead_dimension else slice(None)
                    for dimension in array.dims
                )
            self._dataset.variables[name][target] = values
        self._dataset.sync()
        self._next_step += 1

    def seal(self) -> Path:
        if self._next_step != self.steps or self._dataset is None:
            self.abort()
            raise RuntimeError(
                f"trajectory is incomplete: wrote {self._next_step} of {self.steps} leads"
            )
        self._dataset.close()
        self._dataset = None
        return self._temporary

    def publish(self) -> None:
        if self._dataset is not None:
            self.seal()
        os.replace(self._temporary, self.output)

    def finalize(self) -> None:
        self.seal()
        self.publish()

    def abort(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        self._temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ModelExecutionProfile:
    name: str
    platform: str
    runtime_backend: str
    edge_input_projection: str
    grid2mesh_edge_chunk_size: int | None
    mesh2grid_edge_chunk_size: int | None
    mesh2grid_full_grid_fast_path: bool
    index_add_execution: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "platform": self.platform,
            "runtime_backend": self.runtime_backend,
            "configuration_overrides": {
                "edge_input_projection": self.edge_input_projection,
                "grid2mesh_edge_chunk_size": self.grid2mesh_edge_chunk_size,
                "mesh2grid_edge_chunk_size": self.mesh2grid_edge_chunk_size,
                "mesh2grid_full_grid_fast_path": self.mesh2grid_full_grid_fast_path,
                "index_add_execution": self.index_add_execution,
            },
        }


DOMESTIC_PLATFORMS = frozenset(("c550", "bw3000", "s5000", "910c", "810e"))
PLATFORM_RUNTIME_BACKENDS = {
    "h100": "nvidia",
    "c550": "metax",
    "bw3000": "hygon",
    "s5000": "mthreads",
    "910c": "ascend",
    "810e": "ptg",
}


def resolve_model_execution_profile(platform: str, name: str) -> ModelExecutionProfile:
    if platform == "h100" and name == "baseline":
        return ModelExecutionProfile(
            name=name,
            platform=platform,
            runtime_backend=PLATFORM_RUNTIME_BACKENDS[platform],
            edge_input_projection="concatenated",
            grid2mesh_edge_chunk_size=None,
            mesh2grid_edge_chunk_size=None,
            mesh2grid_full_grid_fast_path=False,
            index_add_execution="inline",
        )
    if platform in DOMESTIC_PLATFORMS and name == "domestic-v1":
        edge_chunk_size = 524288 if platform == "bw3000" else 131072
        return ModelExecutionProfile(
            name=name,
            platform=platform,
            runtime_backend=PLATFORM_RUNTIME_BACKENDS[platform],
            edge_input_projection="factorized",
            grid2mesh_edge_chunk_size=edge_chunk_size,
            mesh2grid_edge_chunk_size=edge_chunk_size,
            mesh2grid_full_grid_fast_path=True,
            index_add_execution=(
                "prevalidated-static"
                if platform in {"bw3000", "s5000"}
                else "inline"
            ),
        )
    raise ValueError(f"model execution profile {name!r} is not valid for platform {platform!r}")


def validate_runtime_platform(platform: str, backend: str) -> None:
    expected = PLATFORM_RUNTIME_BACKENDS[platform]
    if backend != expected:
        raise RuntimeError(
            f"platform {platform!r} requires runtime backend {expected!r}, found {backend!r}"
        )


def calculate_throughput(
    *,
    forward_flops: int,
    steps: int,
    e2e_seconds: float,
    peak_tflops_per_device: float,
    device_count: int,
) -> dict[str, int | float | None]:
    if forward_flops <= 0 or steps <= 0:
        raise ValueError("forward FLOPs and steps must be positive")
    if not math.isfinite(e2e_seconds) or e2e_seconds <= 0:
        raise ValueError("E2E seconds must be finite and positive")
    if device_count <= 0:
        raise ValueError("device count must be positive")
    useful_flops = int(forward_flops) * int(steps)
    effective_tflops = useful_flops / e2e_seconds / 1.0e12
    mfu = None
    if peak_tflops_per_device > 0:
        mfu = effective_tflops / (peak_tflops_per_device * device_count)
    return {
        "useful_flops": useful_flops,
        "effective_tflops": effective_tflops,
        "mfu": mfu,
    }


def dataset_is_finite(dataset: xr.Dataset) -> bool:
    return bool(dataset.data_vars) and all(
        np.isfinite(np.asarray(variable.values)).all() for variable in dataset.data_vars.values()
    )


def validate_normalization_stats(
    stats: Any,
    *,
    inputs: xr.Dataset,
    forcings: xr.Dataset,
    target_template: xr.Dataset,
) -> None:
    standard_variables = (
        set(inputs.data_vars) | set(forcings.data_vars) | set(target_template.data_vars)
    )
    residual_variables = set(inputs.data_vars) & set(target_template.data_vars)
    required_variables = {
        "mean_by_level": standard_variables,
        "stddev_by_level": standard_variables,
        "diffs_stddev_by_level": residual_variables,
    }
    for name, variables in required_variables.items():
        dataset = getattr(stats, name)
        if not dataset.data_vars:
            raise ValueError(f"{name} requires a non-empty variable set")
        missing = sorted(variables - set(dataset.data_vars))
        if missing:
            raise ValueError(f"{name} is missing required variables: {', '.join(missing)}")
        for variable_name in sorted(variables):
            variable = dataset[variable_name]
            values = np.asarray(variable.values)
            if not np.isfinite(values).all():
                qualifier = "positive finite" if name != "mean_by_level" else "finite"
                raise ValueError(
                    f"{name} variable {variable_name!r} must contain {qualifier} values"
                )
            if name != "mean_by_level" and np.any(values <= 0.0):
                raise ValueError(
                    f"{name} variable {variable_name!r} must contain positive finite values"
                )


def run_tensor_resident_rollout(
    model: torch.nn.Module,
    prepared: Any,
    *,
    steps: int,
    warmup_calls: int,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
    precision_policy: InferencePrecisionPolicy | None = None,
    device_type: str | None = None,
    trajectory_callback: Callable[[int, torch.Tensor], None] | None = None,
) -> RolloutMeasurement:
    """Warm up outside the timer, then measure all dependent rollout steps."""

    if steps <= 0:
        raise ValueError(f"rollout steps must be positive, got {steps}")
    if int(prepared.steps) != steps:
        raise ValueError(
            f"requested {steps} rollout steps but received {prepared.steps} forcing frames"
        )
    if warmup_calls < 1:
        raise ValueError("at least one warmup call is required")

    policy = precision_policy or resolve_inference_precision("fp32")
    state = prepared.initial_state.clone()
    resolved_device_type = device_type or state.device.type
    first_forcing = prepared.forcing_features[0]
    model_input_dtype: str | None = None
    model_output_dtype: str | None = None
    feedback_dtype: str | None = None

    def observe_forward(result: Any) -> None:
        nonlocal model_input_dtype, model_output_dtype, feedback_dtype
        if model_input_dtype is not None and model_input_dtype != result.model_input_dtype:
            raise RuntimeError("model input dtype changed during rollout")
        if model_output_dtype is not None and model_output_dtype != result.model_output_dtype:
            raise RuntimeError("model output dtype changed during rollout")
        model_input_dtype = result.model_input_dtype
        model_output_dtype = result.model_output_dtype
        feedback_dtype = str(result.prediction.dtype)

    with torch.inference_mode():
        for _ in range(warmup_calls):
            warmup = run_precision_forward(
                model,
                state,
                first_forcing,
                policy=policy,
                device_type=resolved_device_type,
            )
            observe_forward(warmup)
            if tuple(warmup.prediction.shape[-1:]) != (int(prepared.output_channels),):
                raise RuntimeError("warmup output does not match the configured output channels")
            del warmup
        synchronize()

        started = clock()
        normalized_output: torch.Tensor | None = None
        final_output: torch.Tensor | None = None
        finite = True
        for step_index in range(steps):
            forcing = prepared.forcing_features[step_index]
            forward = run_precision_forward(
                model,
                state,
                forcing,
                policy=policy,
                device_type=resolved_device_type,
            )
            observe_forward(forward)
            prediction = forward.prediction
            final_step = step_index + 1 == steps
            normalized_output = advance_device_rollout(
                state,
                forcing,
                prediction,
                prepared,
                advance_state=not final_step,
                return_absolute_output=final_step or trajectory_callback is not None,
            )
            if normalized_output is not None and trajectory_callback is not None:
                finite = finite and bool(torch.isfinite(normalized_output).all().item())
                captured = normalized_output.detach().to(device="cpu", dtype=torch.float32)
                trajectory_callback(step_index, captured)
                if final_step:
                    final_output = captured
            del forward, prediction
        if normalized_output is None:
            raise RuntimeError("the final rollout step did not produce an absolute output")
        finite = finite and bool(torch.isfinite(normalized_output).all().item())
        if final_output is None:
            final_output = normalized_output.detach().to(device="cpu", dtype=torch.float32)
        synchronize()
        e2e_seconds = clock() - started

    return RolloutMeasurement(
        steps=steps,
        warmup_calls=warmup_calls,
        e2e_seconds=e2e_seconds,
        final_output=final_output,
        finite=finite,
        dtype_evidence=RolloutDtypeEvidence(
            state=str(state.dtype),
            forcing=str(prepared.forcing_features.dtype),
            model_input=str(model_input_dtype),
            model_output=str(model_output_dtype),
            feedback=str(feedback_dtype),
            final_output=str(final_output.dtype),
        ),
    )


def precision_metadata(
    policy: InferencePrecisionPolicy,
    measurement: RolloutMeasurement,
    *,
    model: torch.nn.Module,
    device_type: str,
    float32_math_mode: str,
) -> dict[str, Any]:
    parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if len(parameter_dtypes) != 1:
        raise RuntimeError(f"model parameters require one dtype, found {sorted(parameter_dtypes)}")
    evidence = measurement.dtype_evidence
    return {
        "requested": policy.requested,
        "effective": policy.effective,
        "autocast": {
            "enabled": policy.autocast_dtype is not None,
            "device_type": device_type,
            "dtype": str(policy.autocast_dtype) if policy.autocast_dtype is not None else None,
        },
        "parameter_dtype": next(iter(parameter_dtypes)),
        "state_dtype": evidence.state,
        "forcing_dtype": evidence.forcing,
        "model_input_dtype": evidence.model_input,
        "model_output_dtype": evidence.model_output,
        "feedback_dtype": evidence.feedback,
        "final_output_dtype": evidence.final_output,
        "float32_math_mode": float32_math_mode,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if (
        isinstance(payload, dict)
        and payload
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    ):
        return payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            value = payload.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
            ):
                return value
    raise ValueError("checkpoint does not contain a supported model state dictionary")


def stamp_final_prediction_time(
    prediction: xr.Dataset,
    *,
    steps: int,
    step_hours: int,
    valid_times: xr.DataArray | None,
) -> xr.Dataset:
    """Label the final physical output, not the reusable one-step template."""
    if steps <= 0 or step_hours <= 0 or prediction.sizes.get("time") != 1:
        raise ValueError("final prediction requires one frame and a positive rollout horizon")
    result = prediction.drop_vars(
        [
            name
            for name in ("datetime", "valid_time", "forecast_reference_time")
            if name in prediction
        ]
    ).assign_coords(time=[np.timedelta64(steps * step_hours, "h")])
    if valid_times is None:
        # Missing initialization evidence must remain missing, never copied from the template.
        return result
    if "time" not in valid_times.dims or valid_times.sizes["time"] != steps:
        raise ValueError("forcing valid times do not cover the rollout horizon")
    if valid_times.dims == ("time",):
        values = np.asarray(valid_times)[None, :]
    elif set(valid_times.dims) == {"batch", "time"}:
        values = np.asarray(valid_times.transpose("batch", "time"))
    else:
        raise ValueError("forcing valid times require time and optional batch dimensions")
    if (
        values.shape[0] != prediction.sizes.get("batch")
        or not np.issubdtype(values.dtype, np.datetime64)
        or np.isnat(values).any()
    ):
        raise ValueError("forcing valid-time batch or dtype is invalid")
    leads = np.arange(1, steps + 1) * np.timedelta64(step_hours, "h")
    initializations = values - leads
    if not np.all(initializations == initializations[:, :1]):
        raise ValueError("forcing valid times do not describe a regular autoregressive rollout")
    return result.assign_coords(
        datetime=(("batch", "time"), values[:, -1:]),
        forecast_reference_time=("batch", initializations[:, 0]),
    )


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish one JSON artifact for diagnostic-tool compatibility."""

    os.replace(_stage_json(path, payload), path)


def _stage_netcdf(path: Path, dataset: xr.Dataset) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        dataset.to_netcdf(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _require_distinct_paths(paths: dict[str, Path]) -> None:
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        canonical = path.resolve()
        if canonical in seen:
            raise ValueError(
                f"runner input and output paths must be distinct: {seen[canonical]} and {label}"
            )
        seen[canonical] = label


def _backup_existing_outputs(paths: list[Path]) -> dict[Path, Path]:
    backups: dict[Path, Path] = {}
    try:
        for path in paths:
            if not path.exists():
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".backup", dir=path.parent
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(path, backup)
            backups[path] = backup
    except BaseException:
        for original, backup in reversed(tuple(backups.items())):
            os.replace(backup, original)
        raise
    return backups


def _restore_output_backups(backups: dict[Path, Path]) -> None:
    for original, backup in reversed(tuple(backups.items())):
        original.unlink(missing_ok=True)
        os.replace(backup, original)


def build_model_for_run(
    config: GoogleSmallConfig,
    *,
    edge_input_projection: str,
    mesh2grid_full_grid_fast_path: bool = False,
    index_add_execution: str = "inline",
    fuse_post_layer_norm_residual: bool | None = None,
) -> GoogleSmallCompatibleModel:
    return GoogleSmallCompatibleModel(
        config,
        edge_input_projection_mode=edge_input_projection,
        mesh2grid_full_grid_fast_path=mesh2grid_full_grid_fast_path,
        index_add_execution=index_add_execution,
        fuse_post_layer_norm_residual=fuse_post_layer_norm_residual,
    )


def post_layer_norm_residual_source_audit(
    requested: bool,
    *,
    flagos_mode: str,
) -> dict[str, Any]:
    """Validate the public fusion source before inference; never touch tensors."""
    audit = {"requested": requested, "effective": False}
    if not requested and flagos_mode != "flaggems":
        return audit
    if flagos_mode != "flaggems":
        raise RuntimeError("post_layer_norm_residual fusion requires flaggems mode")
    fused_op = require_post_layer_norm_residual() if requested else None
    import flag_gems

    package_file = Path(flag_gems.__file__).resolve()
    symbol_file = Path(inspect.getfile(fused_op)).resolve() if requested else None
    selected_root = os.environ.get("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", "").strip()
    package_parent = package_file.parent.parent
    root = (
        Path(selected_root).expanduser().resolve()
        if selected_root
        else package_parent.parent
        if package_parent.name == "src"
        else package_parent
    )
    for path in (package_file, symbol_file):
        if path is not None and not path.is_relative_to(root):
            raise RuntimeError(f"fusion source outside selected FlagGems tree: {path}")
    revision = None
    # An archive/site install has no verifiable Git revision. Do not mistake a
    # surrounding repository's HEAD for the selected FlagGems source revision.
    if (root / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    return {
        **audit,
        "effective": requested,
        "module": fused_op.__module__ if requested else None,
        "file": str(symbol_file) if symbol_file else None,
        "sha256": _sha256(symbol_file) if symbol_file else None,
        "flaggems_module_file": str(package_file),
        "flaggems_source_root": str(root),
        "flaggems_revision": revision,
        "revision_status": "git_head" if revision else "unavailable_no_git_metadata",
    }


def apply_edge_chunk_overrides(
    config: GoogleSmallConfig,
    *,
    grid2mesh: int | None,
    mesh2grid: int | None,
) -> GoogleSmallConfig:
    for value in (grid2mesh, mesh2grid):
        if value is not None and value <= 0:
            raise ValueError("edge chunk size must be positive")
    return replace(
        config,
        grid2mesh_edge_chunk_size=(
            grid2mesh if grid2mesh is not None else config.grid2mesh_edge_chunk_size
        ),
        mesh2grid_edge_chunk_size=(
            mesh2grid if mesh2grid is not None else config.mesh2grid_edge_chunk_size
        ),
    )


def model_execution_metadata(
    model: GoogleSmallCompatibleModel,
    *,
    profile: ModelExecutionProfile | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "edge_input_projection": model.edge_input_projection_mode,
        "mesh2grid_full_grid_fast_path": model.mesh2grid_full_grid_fast_path,
        "index_add_execution": model.index_add_execution,
        "grid2mesh_edge_chunk_size": model.config.grid2mesh_edge_chunk_size,
        "mesh2grid_edge_chunk_size": model.config.mesh2grid_edge_chunk_size,
    }
    if profile is not None:
        metadata["model_execution_profile"] = profile.to_metadata()
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--rollout-forcings", type=Path, required=True)
    parser.add_argument("--target-template", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--platform",
        choices=("h100", "c550", "bw3000", "s5000", "910c", "810e"),
        required=True,
    )
    parser.add_argument(
        "--model-execution-profile",
        dest="model_execution_profile_name",
        choices=("baseline", "domestic-v1"),
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="fp32")
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of dependent autoregressive leads (default: 40)",
    )
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--forward-flops", type=int, default=FORWARD_FLOPS)
    parser.add_argument("--peak-tflops-per-device", type=float, default=0.0)
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        help="Diagnostic-only NetCDF containing every autoregressive lead",
    )
    parser.add_argument(
        "--post-layernorm-residual-profile",
        action="store_true",
        help="Record post-LayerNorm residual candidate metadata and event timings",
    )
    parser.add_argument("--post-layernorm-residual-profile-path", type=Path)
    parser.add_argument(
        "--fuse-post-layer-norm-residual",
        action=argparse.BooleanOptionalAction,
        default=_environment_flag("GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL"),
        help="Use the FlagGems public fusion API for single-consumer node updates",
    )
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.model_execution_profile = resolve_model_execution_profile(
            args.platform, args.model_execution_profile_name
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        args.post_layernorm_residual_profile_path is not None
        and not args.post_layernorm_residual_profile
    ):
        parser.error(
            "--post-layernorm-residual-profile-path requires --post-layernorm-residual-profile"
        )
    if args.post_layernorm_residual_profile and args.post_layernorm_residual_profile_path is None:
        args.post_layernorm_residual_profile_path = args.report.with_name(
            f"{args.report.stem}.post-layernorm-residual-profile.json"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.step_hours <= 0:
        raise ValueError("step hours must be positive")
    if not math.isfinite(args.peak_tflops_per_device) or args.peak_tflops_per_device < 0:
        raise ValueError("peak TFLOPS per device must be finite and nonnegative")
    outputs = [args.output, args.report]
    if args.trajectory_output is not None:
        outputs.append(args.trajectory_output)
    if args.post_layernorm_residual_profile:
        outputs.append(args.post_layernorm_residual_profile_path)
    input_paths = {
        "config": args.config,
        "weights": args.weights,
        "inputs": args.inputs,
        "rollout_forcings": args.rollout_forcings,
        "target_template": args.target_template,
        **{f"stats/{name}": args.stats_dir / name for name in REQUIRED_STATS_FILES},
    }
    output_paths = {"prediction": args.output, "report": args.report}
    if args.trajectory_output is not None:
        output_paths["trajectory"] = args.trajectory_output
    if args.post_layernorm_residual_profile:
        output_paths["post_layernorm_residual_profile"] = args.post_layernorm_residual_profile_path
    _require_distinct_paths({**input_paths, **output_paths})
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"report already exists: {args.report}")
    if (
        args.post_layernorm_residual_profile
        and args.post_layernorm_residual_profile_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            "post-LayerNorm residual profile already exists: "
            f"{args.post_layernorm_residual_profile_path}"
        )
    if (
        args.trajectory_output is not None
        and args.trajectory_output.exists()
        and not args.overwrite
    ):
        raise FileExistsError(f"trajectory output already exists: {args.trajectory_output}")
    model_execution_profile = args.model_execution_profile
    runtime = None
    if args.fuse_post_layer_norm_residual:
        # Reject an unusable fusion source before opening large model assets.
        # Unfused runs retain their existing asset-first validation order.
        runtime = resolve_runtime_backend(
            args.device,
            expected_profile=model_execution_profile.runtime_backend,
        )
        validate_runtime_platform(args.platform, runtime.profile.name)
        fusion_source_audit = post_layer_norm_residual_source_audit(
            args.fuse_post_layer_norm_residual,
            flagos_mode=runtime.flagos.mode,
        )
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_identities = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in input_paths.items()
    }

    if runtime is None:
        runtime = resolve_runtime_backend(
            args.device,
            expected_profile=model_execution_profile.runtime_backend,
        )
        validate_runtime_platform(args.platform, runtime.profile.name)
        fusion_source_audit = post_layer_norm_residual_source_audit(
            args.fuse_post_layer_norm_residual,
            flagos_mode=runtime.flagos.mode,
        )
    precision_policy = resolve_inference_precision(args.precision)
    float32_math_mode = configure_float32_math(
        precision_policy.fast_float32_math,
        runtime.profile.device_type,
    )
    config = apply_edge_chunk_overrides(
        GoogleSmallConfig.from_yaml(args.config),
        grid2mesh=model_execution_profile.grid2mesh_edge_chunk_size,
        mesh2grid=model_execution_profile.mesh2grid_edge_chunk_size,
    )
    inputs = xr.open_dataset(args.inputs).load()
    forcings = xr.open_dataset(args.rollout_forcings).load()
    target_template = xr.open_dataset(args.target_template).load()
    stats = load_normalization_stats(args.stats_dir)
    validate_normalization_stats(
        stats,
        inputs=inputs,
        forcings=forcings,
        target_template=target_template,
    )

    model = build_model_for_run(
        config,
        edge_input_projection=model_execution_profile.edge_input_projection,
        mesh2grid_full_grid_fast_path=(model_execution_profile.mesh2grid_full_grid_fast_path),
        index_add_execution=model_execution_profile.index_add_execution,
        fuse_post_layer_norm_residual=args.fuse_post_layer_norm_residual,
    )
    model.load_state_dict(
        _select_state_dict(torch.load(args.weights, map_location="cpu")),
        strict=True,
    )
    model.init_graph(
        latitudes=inputs.coords["lat"].values,
        longitudes=inputs.coords["lon"].values,
    )
    model.to(runtime.device)
    model.eval()
    static_index_add_contract = None
    if model_execution_profile.index_add_execution == "prevalidated-static":
        static_index_add_contract = model.prevalidate_static_index_add()
    prepared = prepare_device_rollout(
        inputs,
        target_template,
        forcings,
        config=config,
        stats=stats,
        device=runtime.device,
    )

    reset_peak_memory(runtime)
    static_runtime_evidence: dict[str, Any] = {"status": "not_requested"}
    trajectory_writer = (
        TrajectoryNetCDFWriter(
            args.trajectory_output,
            steps=args.steps,
            step_hours=args.step_hours,
            valid_times=forcings.coords.get("datetime"),
        )
        if args.trajectory_output is not None
        else None
    )
    trajectory_physical_finite = True

    def capture_trajectory(step_index: int, normalized_output: torch.Tensor) -> None:
        nonlocal trajectory_physical_finite
        assert trajectory_writer is not None
        prediction = normalized_output_to_prediction(normalized_output, target_template, stats)
        trajectory_physical_finite = trajectory_physical_finite and dataset_is_finite(prediction)
        trajectory_writer.append(
            step_index,
            prediction,
        )

    staged_prediction: Path | None = None
    staged_report: Path | None = None
    staged_candidate_profile: Path | None = None
    publication_started = False
    output_backups: dict[Path, Path] = {}
    try:
        if args.post_layernorm_residual_profile:
            args.post_layernorm_residual_profile_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, staged_profile_name = tempfile.mkstemp(
                prefix=f".{args.post_layernorm_residual_profile_path.name}.",
                suffix=".tmp.json",
                dir=args.post_layernorm_residual_profile_path.parent,
            )
            os.close(descriptor)
            staged_candidate_profile = Path(staged_profile_name)
        candidate_recorder = PostLayerNormResidualRecorder(staged_candidate_profile)
        if candidate_recorder.enabled:
            for graph_net in (model.grid2mesh_gnn, model.mesh_gnn, model.mesh2grid_gnn):
                graph_net.post_layer_norm_residual_observer = candidate_recorder.region
        trajectory_callback = capture_trajectory if trajectory_writer is not None else None
        if static_index_add_contract is None:
            measurement = run_tensor_resident_rollout(
                model,
                prepared,
                steps=args.steps,
                warmup_calls=args.warmup_calls,
                synchronize=lambda: synchronize_runtime(runtime),
                precision_policy=precision_policy,
                device_type=runtime.device.type,
                trajectory_callback=trajectory_callback,
            )
        else:
            trusted_scope = (
                nullcontext({"status": "not_applicable_native"})
                if runtime.flagos.mode == "off"
                else use_prevalidated_flaggems_index_add(runtime)
            )
            with trusted_scope as runtime_evidence:
                measurement = run_tensor_resident_rollout(
                    model,
                    prepared,
                    steps=args.steps,
                    warmup_calls=args.warmup_calls,
                    synchronize=lambda: synchronize_runtime(runtime),
                    precision_policy=precision_policy,
                    device_type=runtime.device.type,
                    trajectory_callback=trajectory_callback,
                )
            model.verify_static_index_add(static_index_add_contract)
            static_runtime_evidence = runtime_evidence

        candidate_profile = candidate_recorder.stop(
            metadata={
                "platform": args.platform,
                "runtime_backend": runtime.profile.name,
                "model_execution_profile": model_execution_profile.name,
                "steps": args.steps,
                "warmup_calls": args.warmup_calls,
            }
        )

        dispatch_audit = require_complete_flaggems_dispatch(runtime)
        if (
            post_layer_norm_residual_source_audit(
                args.fuse_post_layer_norm_residual,
                flagos_mode=runtime.flagos.mode,
            )
            != fusion_source_audit
        ):
            raise RuntimeError("post_layer_norm_residual source changed during inference")
        fused_calls = {
            name: getattr(model, name).post_layer_norm_residual_calls
            for name in ("grid2mesh_gnn", "mesh_gnn", "mesh2grid_gnn")
        }
        prediction = normalized_output_to_prediction(
            measurement.final_output,
            target_template,
            stats,
        )
        prediction = stamp_final_prediction_time(
            prediction,
            steps=args.steps,
            step_hours=args.step_hours,
            valid_times=forcings.coords.get("datetime"),
        )
        physical_finite = dataset_is_finite(prediction) and trajectory_physical_finite
        for name, path in input_paths.items():
            if _sha256(path) != source_identities[name]["sha256"]:
                raise RuntimeError(f"runner input changed during inference: {name}")
        staged_prediction = _stage_netcdf(args.output, prediction)
        staged_trajectory = None
        if trajectory_writer is not None:
            staged_trajectory = trajectory_writer.seal()
        metrics = calculate_throughput(
            forward_flops=args.forward_flops,
            steps=args.steps,
            e2e_seconds=measurement.e2e_seconds,
            peak_tflops_per_device=args.peak_tflops_per_device,
            device_count=args.device_count,
        )
        passed = measurement.finite and physical_finite
        performance_eligible = (
            trajectory_writer is None and not candidate_recorder.enabled and passed
        )
        report = {
            "schema": "graphcast.operational_rollout.v2",
            "status": "passed" if passed else "failed_nonfinite",
            "source": {
                "post_layer_norm_residual_fusion": fusion_source_audit,
                "config": source_identities["config"]["path"],
                "config_sha256": source_identities["config"]["sha256"],
                "weights": source_identities["weights"]["path"],
                "weights_sha256": source_identities["weights"]["sha256"],
                "inputs_sha256": source_identities["inputs"]["sha256"],
                "rollout_forcings_sha256": source_identities["rollout_forcings"]["sha256"],
                "target_template_sha256": source_identities["target_template"]["sha256"],
                "normalization_stats": {
                    name: source_identities[f"stats/{name}"] for name in REQUIRED_STATS_FILES
                },
            },
            "runtime": {
                **runtime.to_metadata(),
                "post_layer_norm_residual_dispatch": {
                    "api_calls": sum(fused_calls.values()),
                    "api_calls_by_graph_net": fused_calls,
                    "scope": "warmup_and_measured_rollout",
                    "evidence": "public_api_returned_including_possible_fallback",
                    "synchronizes": False,
                },
                "flaggems_dispatch": (
                    dispatch_audit.to_metadata()
                    if dispatch_audit is not None
                    else {"status": "not_requested"}
                ),
            },
            "precision": precision_metadata(
                precision_policy,
                measurement,
                model=model,
                device_type=runtime.device.type,
                float32_math_mode=float32_math_mode,
            ),
            "workload": {
                "steps": measurement.steps,
                "step_hours": args.step_hours,
                "warmup_calls": measurement.warmup_calls,
                "warmup_excluded_from_e2e": True,
                "final_output_d2h_included_in_e2e": True,
                "device_count": args.device_count,
                **model_execution_metadata(model, profile=model_execution_profile),
                "static_index_add": (
                    {
                        **static_index_add_contract.to_metadata(),
                        "runtime_trusted_scope": static_runtime_evidence,
                    }
                    if static_index_add_contract is not None
                    else {"status": "not_requested"}
                ),
                "post_layernorm_residual_profile": {
                    "enabled": candidate_recorder.enabled,
                    "path": (
                        str(args.post_layernorm_residual_profile_path.resolve())
                        if candidate_recorder.enabled
                        else None
                    ),
                },
            },
            "measurement": {
                "e2e_seconds": measurement.e2e_seconds,
                "forward_flops_per_step": args.forward_flops,
                **metrics,
                "peak_tflops_per_device": args.peak_tflops_per_device,
                "normalized_output_finite": measurement.finite,
                "physical_output_finite": physical_finite,
                "finite": passed,
                "memory": memory_text(runtime),
                "performance_eligible": performance_eligible,
            },
            "diagnostics": {
                "trajectory": (
                    str(args.trajectory_output.resolve())
                    if args.trajectory_output is not None
                    else None
                ),
                "all_leads_captured": trajectory_writer is not None,
                "performance_eligible": performance_eligible,
            },
            "artifacts": {
                "publication": "report_terminal_last",
                "prediction": {
                    "path": str(args.output.resolve()),
                    "sha256": _sha256(staged_prediction),
                },
                "trajectory": (
                    {
                        "path": str(args.trajectory_output.resolve()),
                        "sha256": _sha256(staged_trajectory),
                    }
                    if staged_trajectory is not None
                    else None
                ),
                "post_layernorm_residual_profile": (
                    {
                        "path": str(args.post_layernorm_residual_profile_path.resolve()),
                        "sha256": _sha256(staged_candidate_profile),
                        "groups": len(candidate_profile["groups"]),
                    }
                    if candidate_profile is not None
                    else None
                ),
            },
            "output": str(args.output.resolve()),
        }
        staged_report = _stage_json(args.report, report)
        if args.overwrite:
            output_backups = _backup_existing_outputs(outputs)
        publication_started = True
        os.replace(staged_prediction, args.output)
        staged_prediction = None
        if trajectory_writer is not None:
            trajectory_writer.publish()
        if staged_candidate_profile is not None:
            os.replace(staged_candidate_profile, args.post_layernorm_residual_profile_path)
            staged_candidate_profile = None
        os.replace(staged_report, args.report)
        staged_report = None
        obsolete_backups = tuple(output_backups.values())
        output_backups = {}
        for backup in obsolete_backups:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
    except BaseException:
        if staged_prediction is not None:
            staged_prediction.unlink(missing_ok=True)
        if staged_report is not None:
            staged_report.unlink(missing_ok=True)
        if staged_candidate_profile is not None:
            staged_candidate_profile.unlink(missing_ok=True)
        if trajectory_writer is not None:
            trajectory_writer.abort()
        if publication_started:
            for path in outputs:
                path.unlink(missing_ok=True)
            _restore_output_backups(output_backups)
        raise
    print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
