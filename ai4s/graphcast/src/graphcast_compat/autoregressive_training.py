"""Differentiable GraphCast autoregressive training in normalized tensor space.

State predictions remain attached while they are shifted into the next
two-frame input window, preserving full backpropagation through rollout time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import xarray as xr

from graphcast import losses as deepmind_losses


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
class AutoregressiveBatch:
    initial_input_features: torch.Tensor
    forcing_features: torch.Tensor
    target_absolute_features: torch.Tensor
    feedback_updates: tuple[FeedbackUpdate, ...]
    forcing_updates: tuple[ForcingUpdate, ...]
    latitude_weights: torch.Tensor
    channel_weights: torch.Tensor
    target_channel_slices: Mapping[str, slice]
    lat_count: int
    lon_count: int
    diagnostic_channel_weights: torch.Tensor | None = None
    sample_weights: torch.Tensor | None = None

    @property
    def rollout_steps(self) -> int:
        return int(self.target_absolute_features.shape[0])


@dataclass(frozen=True)
class AutoregressiveResult:
    loss: torch.Tensor
    lead_losses: torch.Tensor
    per_variable_lead_losses: Mapping[str, torch.Tensor]
    final_input_features: torch.Tensor


def _slice_length(value: slice, *, name: str) -> int:
    if value.step not in (None, 1):
        raise ValueError(f"{name} slice step must be one")
    if value.start is None or value.stop is None:
        raise ValueError(f"{name} slice must have explicit bounds")
    length = int(value.stop) - int(value.start)
    if length <= 0:
        raise ValueError(f"{name} slice must be non-empty")
    return length


def _validate_batch(batch: AutoregressiveBatch) -> None:
    initial = batch.initial_input_features
    forcing = batch.forcing_features
    targets = batch.target_absolute_features
    if initial.ndim != 3:
        raise ValueError("initial_input_features must be [batch, nodes, channels]")
    if forcing.ndim != 4:
        raise ValueError("forcing_features must be [rollout, batch, nodes, channels]")
    if targets.ndim != 4:
        raise ValueError(
            "target_absolute_features must be [rollout, batch, nodes, channels]"
        )
    if targets.shape[:3] != forcing.shape[:3]:
        raise ValueError("target and forcing trajectory dimensions differ")
    if tuple(initial.shape[:2]) != tuple(targets.shape[1:3]):
        raise ValueError("initial input batch/grid differs from trajectory")
    if batch.rollout_steps <= 0:
        raise ValueError("trajectory must contain at least one target")
    if batch.lat_count <= 0 or batch.lon_count <= 0:
        raise ValueError("latitude and longitude counts must be positive")
    if int(initial.shape[1]) != batch.lat_count * batch.lon_count:
        raise ValueError("grid node count does not match latitude/longitude")
    output_channels = int(targets.shape[-1])
    if tuple(batch.latitude_weights.shape) != (batch.lat_count,):
        raise ValueError("latitude_weights shape is invalid")
    if tuple(batch.channel_weights.shape) != (output_channels,):
        raise ValueError("channel_weights shape is invalid")
    if (
        batch.diagnostic_channel_weights is not None
        and tuple(batch.diagnostic_channel_weights.shape) != (output_channels,)
    ):
        raise ValueError("diagnostic_channel_weights shape is invalid")
    if batch.sample_weights is not None:
        if tuple(batch.sample_weights.shape) != (int(initial.shape[0]),):
            raise ValueError("sample_weights shape is invalid")
        weights = batch.sample_weights.detach()
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("sample_weights must be finite and non-negative")

    occupied_inputs: set[int] = set()
    feedback_outputs: set[int] = set()
    for update in batch.feedback_updates:
        previous_count = _slice_length(
            update.input_previous,
            name=f"{update.name}.input_previous",
        )
        current_count = _slice_length(
            update.input_current,
            name=f"{update.name}.input_current",
        )
        output_count = _slice_length(update.output, name=f"{update.name}.output")
        if previous_count != current_count or current_count != output_count:
            raise ValueError(f"feedback channel counts differ for {update.name}")
        if int(update.residual_to_absolute.numel()) != output_count:
            raise ValueError(f"feedback scale count differs for {update.name}")
        scale = update.residual_to_absolute.detach()
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError(f"feedback scales must be finite and positive for {update.name}")
        input_indices = set(
            range(update.input_previous.start, update.input_previous.stop)
        ) | set(range(update.input_current.start, update.input_current.stop))
        if occupied_inputs & input_indices:
            raise ValueError("autoregressive input update slices overlap")
        occupied_inputs.update(input_indices)
        output_indices = set(range(update.output.start, update.output.stop))
        if feedback_outputs & output_indices:
            raise ValueError("autoregressive output update slices overlap")
        feedback_outputs.update(output_indices)

    for update in batch.forcing_updates:
        previous_count = _slice_length(
            update.input_previous,
            name=f"{update.name}.input_previous",
        )
        current_count = _slice_length(
            update.input_current,
            name=f"{update.name}.input_current",
        )
        forcing_count = _slice_length(update.forcing, name=f"{update.name}.forcing")
        if previous_count != current_count or current_count != forcing_count:
            raise ValueError(f"forcing channel counts differ for {update.name}")
        input_indices = set(
            range(update.input_previous.start, update.input_previous.stop)
        ) | set(range(update.input_current.start, update.input_current.stop))
        if occupied_inputs & input_indices:
            raise ValueError("autoregressive input update slices overlap")
        occupied_inputs.update(input_indices)

    for channel_slice in batch.target_channel_slices.values():
        _slice_length(channel_slice, name="target channel")
        if channel_slice.start < 0 or channel_slice.stop > output_channels:
            raise ValueError("target channel slice is outside model output")


def _weighted_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    batch: AutoregressiveBatch,
    grid_chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} differs from "
            f"target shape {tuple(target.shape)}"
        )
    batch_size, grid_nodes, output_channels = prediction.shape
    if grid_nodes != batch.lat_count * batch.lon_count:
        raise ValueError("prediction grid node count is invalid")
    latitude_weights = batch.latitude_weights.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    channel_weights = batch.channel_weights.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    diagnostic_channel_weights = (
        batch.diagnostic_channel_weights.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if batch.diagnostic_channel_weights is not None
        else channel_weights
    )
    sample_weights = (
        batch.sample_weights.to(device=prediction.device, dtype=prediction.dtype)
        if batch.sample_weights is not None
        else torch.ones(batch_size, device=prediction.device, dtype=prediction.dtype)
    )
    normalizer = sample_weights.sum().clamp_min(1.0)
    channel_slices = tuple(batch.target_channel_slices.values())

    if 0 < grid_chunk_size < grid_nodes:
        latitude_by_node = (
            latitude_weights.view(batch.lat_count, 1)
            .expand(batch.lat_count, batch.lon_count)
            .reshape(grid_nodes)
        )
        total_chunks: list[torch.Tensor] = []
        variable_chunks: list[list[torch.Tensor]] = [[] for _ in channel_slices]
        for start in range(0, grid_nodes, grid_chunk_size):
            stop = min(start + grid_chunk_size, grid_nodes)
            squared = (prediction[:, start:stop] - target[:, start:stop]).square()
            weighted = squared * latitude_by_node[start:stop].view(1, -1, 1)
            total_chunks.append(
                (weighted * channel_weights.view(1, 1, output_channels)).sum(dim=(1, 2))
            )
            for index, channel_slice in enumerate(channel_slices):
                variable_chunks[index].append(
                    (
                        weighted[..., channel_slice]
                        * diagnostic_channel_weights[channel_slice].view(1, 1, -1)
                    ).sum(dim=(1, 2))
                )
        per_sample_total = torch.stack(total_chunks).sum(dim=0) / grid_nodes
        per_sample_variables = [
            torch.stack(chunks).sum(dim=0) / grid_nodes for chunks in variable_chunks
        ]
    else:
        squared = (prediction - target).square().reshape(
            batch_size,
            batch.lat_count,
            batch.lon_count,
            output_channels,
        )
        weighted = squared * latitude_weights.view(1, batch.lat_count, 1, 1)
        per_sample_total = (
            weighted * channel_weights.view(1, 1, 1, output_channels)
        ).sum(dim=-1).mean(dim=(1, 2))
        per_sample_variables = []
        for channel_slice in channel_slices:
            values = weighted[..., channel_slice]
            weights = diagnostic_channel_weights[channel_slice]
            per_sample_variables.append(
                (values * weights.view(1, 1, 1, -1)).sum(dim=-1).mean(dim=(1, 2))
            )

    total = (per_sample_total * sample_weights).sum() / normalizer
    variable_losses = [
        (per_sample_variable * sample_weights).sum() / normalizer
        for per_sample_variable in per_sample_variables
    ]
    return total, torch.stack(variable_losses)


def _run_lead(
    model: torch.nn.Module,
    batch: AutoregressiveBatch,
    current_input: torch.Tensor,
    lead: int,
    *,
    loss_grid_chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forcing = batch.forcing_features[lead].to(
        device=current_input.device,
        dtype=current_input.dtype,
    )
    prediction = model(torch.cat((current_input, forcing), dim=-1))
    if not bool(torch.isfinite(prediction.detach()).all()):
        raise FloatingPointError(f"non-finite GraphCast prediction at rollout lead {lead + 1}")
    target = batch.target_absolute_features[lead].to(
        device=prediction.device,
        dtype=prediction.dtype,
    ).clone()
    next_input = current_input.clone()
    for update in batch.feedback_updates:
        current_absolute_view = current_input[..., update.input_current]
        if (
            current_absolute_view.device.type == "musa"
            and current_absolute_view.device != prediction.device
            and not current_absolute_view.is_contiguous()
        ):
            current_absolute_view = current_absolute_view.contiguous()
        current_absolute = current_absolute_view.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        scale = update.residual_to_absolute.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        target[..., update.output] = (
            target[..., update.output] - current_absolute
        ) / scale
        predicted_absolute = current_absolute + prediction[..., update.output] * scale
        next_input[..., update.input_previous] = current_input[..., update.input_current]
        next_input[..., update.input_current] = predicted_absolute.to(
            device=current_input.device,
            dtype=current_input.dtype,
        )
    for update in batch.forcing_updates:
        next_input[..., update.input_previous] = current_input[..., update.input_current]
        next_input[..., update.input_current] = forcing[..., update.forcing]
    lead_loss, variable_losses = _weighted_losses(
        prediction,
        target,
        batch=batch,
        grid_chunk_size=loss_grid_chunk_size,
    )
    if not bool(torch.isfinite(lead_loss.detach())):
        raise FloatingPointError(f"non-finite GraphCast loss at rollout lead {lead + 1}")
    return next_input, lead_loss, variable_losses


def autoregressive_loss(
    model: torch.nn.Module,
    batch: AutoregressiveBatch,
    *,
    checkpoint_segments: int = 0,
    checkpoint_implementation: str = "non_reentrant",
    loss_grid_chunk_size: int = 0,
) -> AutoregressiveResult:
    """Unroll a trajectory and average its losses without detaching feedback."""

    _validate_batch(batch)
    if (
        isinstance(checkpoint_segments, bool)
        or not isinstance(checkpoint_segments, int)
        or checkpoint_segments < 0
    ):
        raise ValueError("checkpoint_segments must be a non-negative integer")
    if checkpoint_implementation not in {"non_reentrant", "reentrant"}:
        raise ValueError("checkpoint_implementation must be non_reentrant or reentrant")
    if (
        isinstance(loss_grid_chunk_size, bool)
        or not isinstance(loss_grid_chunk_size, int)
        or loss_grid_chunk_size < 0
    ):
        raise ValueError("loss_grid_chunk_size must be a non-negative integer")

    rollout_steps = batch.rollout_steps
    segment_count = min(checkpoint_segments, rollout_steps) if checkpoint_segments else 1
    segment_size = math.ceil(rollout_steps / segment_count)
    current_input = batch.initial_input_features
    all_lead_losses = []
    all_variable_losses = []

    def activation_checkpoint(function: Any, value: torch.Tensor):
        if checkpoint_implementation == "reentrant":
            anchor = torch.zeros((), device=value.device, requires_grad=True)
            return torch_checkpoint(
                lambda active_value, _anchor: function(active_value),
                value,
                anchor,
                use_reentrant=True,
            )
        return torch_checkpoint(function, value, use_reentrant=False)

    for start in range(0, rollout_steps, segment_size):
        stop = min(start + segment_size, rollout_steps)

        def run_segment(
            segment_input: torch.Tensor,
            *,
            segment_start: int = start,
            segment_stop: int = stop,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state = segment_input
            lead_losses = []
            variable_losses = []
            for lead in range(segment_start, segment_stop):
                state, lead_loss, per_variable = _run_lead(
                    model,
                    batch,
                    state,
                    lead,
                    loss_grid_chunk_size=loss_grid_chunk_size,
                )
                lead_losses.append(lead_loss)
                variable_losses.append(per_variable)
            return state, torch.stack(lead_losses), torch.stack(variable_losses)

        if checkpoint_segments:
            current_input, segment_losses, segment_variable_losses = activation_checkpoint(
                run_segment,
                current_input,
            )
        else:
            current_input, segment_losses, segment_variable_losses = run_segment(current_input)
        all_lead_losses.append(segment_losses)
        all_variable_losses.append(segment_variable_losses)

    lead_losses = torch.cat(all_lead_losses, dim=0)
    variable_matrix = torch.cat(all_variable_losses, dim=0)
    per_variable = {
        name: variable_matrix[:, index]
        for index, name in enumerate(batch.target_channel_slices)
    }
    return AutoregressiveResult(
        loss=lead_losses.mean(),
        lead_losses=lead_losses,
        per_variable_lead_losses=per_variable,
        final_input_features=current_input,
    )


def _to_node_tensor(dataset: Any, *, device: torch.device) -> torch.Tensor:
    from graphcast_compat.data import dataset_to_stacked

    stacked = dataset_to_stacked(dataset).transpose("batch", "lat", "lon", "channels")
    array = np.asarray(stacked.data, dtype=np.float32)
    batch_size, lat_count, lon_count, channels = array.shape
    return torch.from_numpy(array.reshape(batch_size, lat_count * lon_count, channels)).to(
        device=device
    )


def _variable_slices(dataset: Any) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    cursor = 0
    for name in sorted(dataset.data_vars):
        array = dataset[name]
        channel_count = int(
            np.prod(
                [
                    int(array.sizes[dimension])
                    for dimension in array.dims
                    if dimension not in ("batch", "lat", "lon")
                ],
                dtype=np.int64,
            )
        )
        channel_count = max(channel_count, 1)
        slices[name] = slice(cursor, cursor + channel_count)
        cursor += channel_count
    return slices


def _history_slices(dataset: Any, name: str) -> tuple[slice, slice]:
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
        raise ValueError(f"autoregressive input {name!r} must have two history frames")
    channels_per_time = int(
        np.prod(
            [int(array.sizes[dimension]) for dimension in channel_dimensions[1:]],
            dtype=np.int64,
        )
    )
    channels_per_time = max(channels_per_time, 1)
    variable_slice = _variable_slices(dataset)[name]
    previous = slice(variable_slice.start, variable_slice.start + channels_per_time)
    current = slice(previous.stop, previous.stop + channels_per_time)
    if current.stop != variable_slice.stop:
        raise ValueError(f"unexpected history layout for {name!r}")
    return previous, current


def _normalized_latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    """Use WeatherNext's exact weighting on complete supported latitude grids."""

    latitudes = np.asarray(latitudes)
    if len(latitudes) < 2:
        return np.ones_like(latitudes, dtype=np.float32)
    coordinate = xr.DataArray(
        np.zeros(len(latitudes), dtype=np.float32),
        dims=("lat",),
        coords={"lat": latitudes},
    )
    try:
        return np.asarray(
            deepmind_losses.normalized_latitude_weights(coordinate).data,
            dtype=np.float32,
        )
    except ValueError:
        pass
    delta = abs(float(np.diff(latitudes)[0]))
    if np.any(np.isclose(np.abs(latitudes), 90.0)):
        weights = np.cos(np.deg2rad(latitudes)) * np.sin(np.deg2rad(delta / 2.0))
        pole_value = np.sin(np.deg2rad(delta / 4.0)) ** 2
        weights[np.isclose(np.abs(latitudes), 90.0)] = pole_value
    else:
        weights = np.cos(np.deg2rad(latitudes))
    return (weights / weights.mean()).astype(np.float32)


def _target_channel_weights(
    one_lead_targets: Any,
    *,
    per_variable_weights: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    slices = _variable_slices(one_lead_targets)
    weights = []
    diagnostic_weights = []
    for name in sorted(one_lead_targets.data_vars):
        array = one_lead_targets[name]
        variable_weight = float(per_variable_weights.get(name, 1.0))
        if "level" in array.dims:
            upstream_weights = deepmind_losses.normalized_level_weights(array)
            diagnostic_values = (
                np.asarray(upstream_weights.values, dtype=np.float32)
                / float(array.sizes["level"])
            )
        else:
            diagnostic_values = np.ones(
                _slice_length(slices[name], name=name),
                dtype=np.float32,
            )
        values = variable_weight * diagnostic_values
        weights.append(np.asarray(values, dtype=np.float32).reshape(-1))
        diagnostic_weights.append(np.asarray(diagnostic_values, dtype=np.float32).reshape(-1))
    return np.concatenate(weights), np.concatenate(diagnostic_weights), slices


def prepare_autoregressive_batch(
    inputs: Any,
    targets: Any,
    forcings: Any,
    *,
    config: Any,
    stats: Any,
    input_device: torch.device,
    storage_device: torch.device = torch.device("cpu"),
    per_variable_weights: Mapping[str, float] | None = None,
    sample_weights: Any | None = None,
) -> AutoregressiveBatch:
    """Normalize xarray trajectory data and construct differentiable mappings."""

    from graphcast_compat.data import select_dataset_variables
    from graphcast_compat.normalization import normalize_dataset

    selected_inputs = select_dataset_variables(
        inputs,
        config.input_variables or None,
        role="inputs",
    )
    selected_targets = select_dataset_variables(
        targets,
        config.target_variables or None,
        role="targets",
    )
    selected_forcings = select_dataset_variables(
        forcings,
        config.forcing_variables or None,
        role="forcings",
    )
    rollout_steps = int(selected_targets.sizes.get("time", 0))
    if rollout_steps <= 0:
        raise ValueError("autoregressive targets require a time dimension")
    if int(selected_forcings.sizes.get("time", 0)) != rollout_steps:
        raise ValueError("target and forcing rollout lengths differ")
    norm_inputs = normalize_dataset(
        selected_inputs,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_targets = normalize_dataset(
        selected_targets,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_forcings = normalize_dataset(
        selected_forcings,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    initial_input_features = _to_node_tensor(norm_inputs, device=input_device)
    target_features = torch.stack(
        [
            _to_node_tensor(
                norm_targets.isel(time=slice(lead, lead + 1)),
                device=storage_device,
            )
            for lead in range(rollout_steps)
        ]
    )
    forcing_features = torch.stack(
        [
            _to_node_tensor(
                norm_forcings.isel(time=slice(lead, lead + 1)),
                device=storage_device,
            )
            for lead in range(rollout_steps)
        ]
    )
    one_lead_targets = norm_targets.isel(time=slice(0, 1))
    one_lead_forcings = norm_forcings.isel(time=slice(0, 1))
    target_slices = _variable_slices(one_lead_targets)
    forcing_slices = _variable_slices(one_lead_forcings)

    feedback_updates = []
    for name in sorted(selected_targets.data_vars):
        if name not in selected_inputs or "time" not in selected_inputs[name].dims:
            continue
        previous, current = _history_slices(norm_inputs, name)
        output = target_slices[name]
        if name not in stats.stddev_by_level or name not in stats.diffs_stddev_by_level:
            raise ValueError(f"normalization stats cannot feed back target {name!r}")
        scale = stats.diffs_stddev_by_level[name] / stats.stddev_by_level[name]
        target_array = one_lead_targets[name]
        selections = {
            dimension: target_array.coords[dimension]
            for dimension in scale.dims
            if dimension in target_array.coords
        }
        if selections:
            try:
                scale = scale.sel(selections)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"normalization stats do not cover model coordinates for target {name!r}"
                ) from error
        scale_values = np.asarray(scale.values, dtype=np.float32).reshape(-1)
        expected = _slice_length(output, name=name)
        if scale_values.size == 1 and expected > 1:
            scale_values = np.repeat(scale_values, expected)
        if scale_values.size != expected:
            raise ValueError(f"normalization scale layout differs for target {name!r}")
        feedback_updates.append(
            FeedbackUpdate(
                name=name,
                input_previous=previous,
                input_current=current,
                output=output,
                residual_to_absolute=torch.from_numpy(scale_values),
            )
        )

    forcing_updates = []
    for name in sorted(selected_forcings.data_vars):
        if name not in selected_inputs:
            raise ValueError(f"known forcing {name!r} is missing from model inputs")
        previous, current = _history_slices(norm_inputs, name)
        forcing_updates.append(
            ForcingUpdate(
                name=name,
                input_previous=previous,
                input_current=current,
                forcing=forcing_slices[name],
            )
        )
    channel_weights, diagnostic_channel_weights, target_slices = _target_channel_weights(
        one_lead_targets,
        per_variable_weights=per_variable_weights or {},
    )
    latitudes = np.asarray(inputs.coords["lat"].values, dtype=np.float32)
    longitudes = np.asarray(inputs.coords["lon"].values, dtype=np.float32)
    resolved_sample_weights = (
        torch.as_tensor(sample_weights, dtype=torch.float32, device=storage_device)
        if sample_weights is not None
        else None
    )
    batch = AutoregressiveBatch(
        initial_input_features=initial_input_features,
        forcing_features=forcing_features,
        target_absolute_features=target_features,
        feedback_updates=tuple(feedback_updates),
        forcing_updates=tuple(forcing_updates),
        latitude_weights=torch.from_numpy(_normalized_latitude_weights(latitudes)),
        channel_weights=torch.from_numpy(channel_weights),
        diagnostic_channel_weights=torch.from_numpy(diagnostic_channel_weights),
        target_channel_slices=target_slices,
        lat_count=len(latitudes),
        lon_count=len(longitudes),
        sample_weights=resolved_sample_weights,
    )
    _validate_batch(batch)
    return batch
