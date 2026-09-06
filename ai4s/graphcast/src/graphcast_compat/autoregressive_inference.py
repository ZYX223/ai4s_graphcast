"""Device-resident autoregressive inference for GraphCast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xarray as xr

from .data import dataset_to_stacked, grid_node_outputs_to_prediction, select_dataset_variables
from .normalization import normalize_dataset, unnormalize_dataset


@dataclass(frozen=True)
class FeedbackUpdate:
    name: str
    input_previous: slice
    input_current: slice
    output: slice
    residual_to_absolute: torch.Tensor


@dataclass(frozen=True)
class ForcingUpdate:
    name: str
    input_previous: slice
    input_current: slice
    forcing: slice


@dataclass(frozen=True)
class DeviceRolloutInputs:
    initial_state: torch.Tensor
    forcing_features: torch.Tensor
    feedback_updates: tuple[FeedbackUpdate, ...]
    forcing_updates: tuple[ForcingUpdate, ...]
    output_channels: int

    @property
    def steps(self) -> int:
        return int(self.forcing_features.shape[0])


def _to_node_tensor(dataset: xr.Dataset) -> torch.Tensor:
    stacked = dataset_to_stacked(dataset).transpose("batch", "lat", "lon", "channels")
    array = np.asarray(stacked.data, dtype=np.float32)
    batch, lat_count, lon_count, channels = array.shape
    return torch.from_numpy(array.reshape(batch, lat_count * lon_count, channels))


def _variable_slices(dataset: xr.Dataset) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    cursor = 0
    for name in sorted(dataset.data_vars):
        array = dataset[name]
        count = int(
            np.prod(
                [
                    int(array.sizes[dimension])
                    for dimension in array.dims
                    if dimension not in ("batch", "lat", "lon")
                ],
                dtype=np.int64,
            )
        )
        count = max(count, 1)
        slices[name] = slice(cursor, cursor + count)
        cursor += count
    return slices


def _history_slices(
    dataset: xr.Dataset,
    variable_slices: dict[str, slice],
    name: str,
) -> tuple[slice, slice]:
    array = dataset[name]
    channel_dimensions = [
        dimension
        for dimension in array.dims
        if dimension not in ("batch", "lat", "lon")
    ]
    if not channel_dimensions or channel_dimensions[0] != "time":
        raise ValueError(
            f"autoregressive input {name!r} must have time as its first stacked channel dimension"
        )
    if int(array.sizes["time"]) != 2:
        raise ValueError(f"autoregressive input {name!r} must have two frames")
    channels_per_time = int(
        np.prod(
            [int(array.sizes[dimension]) for dimension in channel_dimensions[1:]],
            dtype=np.int64,
        )
    )
    channels_per_time = max(channels_per_time, 1)
    variable_slice = variable_slices[name]
    previous = slice(variable_slice.start, variable_slice.start + channels_per_time)
    current = slice(previous.stop, previous.stop + channels_per_time)
    if current.stop != variable_slice.stop:
        raise ValueError(f"unexpected two-frame layout for {name!r}")
    return previous, current


def prepare_device_rollout(
    inputs: xr.Dataset,
    target_template: xr.Dataset,
    forcings: xr.Dataset,
    *,
    config: Any,
    stats: Any,
    device: torch.device,
) -> DeviceRolloutInputs:
    """Normalize and upload state and all known forcings exactly once."""

    selected_inputs = select_dataset_variables(
        inputs, config.input_variables or None, role="inputs"
    )
    selected_targets = select_dataset_variables(
        target_template, config.target_variables or None, role="targets"
    )
    selected_forcings = select_dataset_variables(
        forcings, config.forcing_variables or None, role="forcings"
    )
    steps = int(selected_forcings.sizes.get("time", 0))
    if steps <= 0:
        raise ValueError("rollout forcings require a non-empty time dimension")
    if int(selected_targets.sizes.get("time", 0)) != 1:
        raise ValueError("target template must contain exactly one time frame")

    dynamic_inputs = {
        name for name, value in selected_inputs.data_vars.items() if "time" in value.dims
    }
    supplied_dynamic = set(selected_targets.data_vars) | set(selected_forcings.data_vars)
    missing_dynamic = sorted(dynamic_inputs - supplied_dynamic)
    if missing_dynamic:
        raise ValueError(
            "time-dependent inputs are neither predicted nor forced: "
            + ", ".join(missing_dynamic)
        )

    norm_inputs = normalize_dataset(
        selected_inputs,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_forcings = normalize_dataset(
        selected_forcings,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    initial_state_cpu = _to_node_tensor(norm_inputs)
    forcing_cpu = torch.stack(
        [
            _to_node_tensor(norm_forcings.isel(time=slice(step, step + 1)))
            for step in range(steps)
        ],
        dim=0,
    )

    input_slices = _variable_slices(norm_inputs)
    target_slices = _variable_slices(selected_targets)
    forcing_slices = _variable_slices(norm_forcings.isel(time=slice(0, 1)))

    feedback_updates: list[FeedbackUpdate] = []
    for name in sorted(selected_targets.data_vars):
        if name not in selected_inputs or "time" not in selected_inputs[name].dims:
            continue
        previous, current = _history_slices(norm_inputs, input_slices, name)
        output = target_slices[name]
        if name not in stats.stddev_by_level or name not in stats.diffs_stddev_by_level:
            raise ValueError(f"normalization stats cannot feed back target {name!r}")
        scale = stats.diffs_stddev_by_level[name] / stats.stddev_by_level[name]
        target_array = selected_targets[name]
        selections = {
            dimension: target_array.coords[dimension]
            for dimension in scale.dims
            if dimension in target_array.coords
        }
        if selections:
            scale = scale.sel(selections)
        scale_values = np.asarray(scale.values, dtype=np.float32).reshape(-1)
        output_count = int(output.stop) - int(output.start)
        if scale_values.size == 1 and output_count > 1:
            scale_values = np.repeat(scale_values, output_count)
        if scale_values.size != output_count:
            raise ValueError(f"feedback scale layout differs for {name!r}")
        feedback_updates.append(
            FeedbackUpdate(
                name=name,
                input_previous=previous,
                input_current=current,
                output=output,
                residual_to_absolute=torch.from_numpy(scale_values).to(device=device),
            )
        )

    forcing_updates: list[ForcingUpdate] = []
    for name in sorted(selected_forcings.data_vars):
        if name not in selected_inputs:
            raise ValueError(f"known forcing {name!r} is missing from inputs")
        previous, current = _history_slices(norm_inputs, input_slices, name)
        forcing_updates.append(
            ForcingUpdate(
                name=name,
                input_previous=previous,
                input_current=current,
                forcing=forcing_slices[name],
            )
        )

    initial_state = initial_state_cpu.to(device=device)
    forcing_features = forcing_cpu.to(device=device)
    output_channels = max(value.stop for value in target_slices.values())
    expected_model_channels = int(config.node_input_dim) - 3
    found_model_channels = int(initial_state.shape[-1] + forcing_features.shape[-1])
    if found_model_channels != expected_model_channels:
        raise ValueError(
            f"device rollout builds {found_model_channels} model channels; expected {expected_model_channels}"
        )
    if output_channels != int(config.output_dim):
        raise ValueError(
            f"device rollout builds {output_channels} output channels; expected {config.output_dim}"
        )
    return DeviceRolloutInputs(
        initial_state=initial_state,
        forcing_features=forcing_features,
        feedback_updates=tuple(feedback_updates),
        forcing_updates=tuple(forcing_updates),
        output_channels=output_channels,
    )


def advance_device_rollout(
    state: torch.Tensor,
    forcing: torch.Tensor,
    prediction: torch.Tensor,
    prepared: DeviceRolloutInputs,
    *,
    advance_state: bool,
    return_absolute_output: bool | None = None,
) -> torch.Tensor | None:
    """Apply residual feedback and forcing updates in normalized tensor space."""

    if return_absolute_output is None:
        return_absolute_output = not advance_state
    normalized_absolute = prediction.clone() if return_absolute_output else None
    for update in prepared.feedback_updates:
        current = state[..., update.input_current]
        predicted = current + prediction[..., update.output] * update.residual_to_absolute
        if normalized_absolute is not None:
            normalized_absolute[..., update.output] = predicted
        if advance_state:
            state[..., update.input_previous].copy_(current)
            state[..., update.input_current].copy_(predicted)
    if advance_state:
        for update in prepared.forcing_updates:
            state[..., update.input_previous].copy_(state[..., update.input_current])
            state[..., update.input_current].copy_(forcing[..., update.forcing])
    return normalized_absolute


def normalized_output_to_prediction(
    normalized_output: torch.Tensor,
    target_template: xr.Dataset,
    stats: Any,
) -> xr.Dataset:
    """Restore the final physical xarray prediction from normalized output."""

    normalized = grid_node_outputs_to_prediction(normalized_output, target_template)
    physical_prediction = unnormalize_dataset(
        normalized,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    for name, template in target_template.data_vars.items():
        physical_prediction[name].attrs = template.attrs.copy()
    return physical_prediction
