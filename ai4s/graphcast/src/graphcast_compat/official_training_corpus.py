"""Multi-shard GraphCast corpus with cross-boundary windows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import xarray as xr

from graphcast_compat.official_training_data import (
    _dataset_nbytes,
    _SIX_HOURS_SECONDS,
    _eligible_reference_indices,
    _finite_dataset,
    _freeze_dataset_arrays,
    _parse_bound,
    _resolve_target_rollout_steps,
    _time_values,
    _validate_six_hour_spacing,
    OfficialTrainingDataset,
    TrainingSample,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingCorpusShard:
    """One NetCDF path consumed by the training loader."""

    path: Path
    relative_path: str


@dataclass(frozen=True)
class TrainingCorpusIndex:
    """Only the paths and dataset role needed by training."""

    path: Path
    name: str
    dataset_kind: str
    training_stage: str
    data_role: str
    shards: tuple[TrainingCorpusShard, ...]


def load_training_corpus_index(path: str | Path) -> TrainingCorpusIndex:
    """Read shard paths directly, ignoring hashes and attestations.

    DeepMind's public loader handles a single xarray dataset rather than a
    multi-file PyTorch corpus. This function is therefore intentionally limited
    to resolving the local shard list; window and batch extraction remains in
    ``OfficialTrainingDataset`` and DeepMind ``data_utils``.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"corpus index is missing: {resolved}")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("corpus index must be an object")
    if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"corpus index schema_version must be {SCHEMA_VERSION}"
        )
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("corpus index name must be non-empty")
    dataset_kind = str(raw.get("dataset_kind", ""))
    training_stage = str(raw.get("training_stage", ""))
    data_role = str(raw.get("data_role", ""))
    allowed = {
        ("era5", "era5_pretrain", "train"),
        ("hres", "hres_finetune", "train"),
        ("hres", "validation", "validation"),
    }
    if (dataset_kind, training_stage, data_role) not in allowed:
        raise ValueError("unsupported corpus dataset/stage/role")
    raw_shards = raw.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("corpus index shards must be non-empty")
    shards = []
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, Mapping):
            raise ValueError("corpus index shard must be an object")
        relative_path = str(raw_shard.get("path", "")).strip()
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("corpus shard path must be relative")
        shards.append(
            TrainingCorpusShard(
                path=(resolved.parent / relative_path).resolve(),
                relative_path=relative_path,
            )
        )
    if len({shard.path for shard in shards}) != len(shards):
        raise ValueError("corpus index contains duplicate shard paths")
    return TrainingCorpusIndex(
        path=resolved,
        name=name,
        dataset_kind=dataset_kind,
        training_stage=training_stage,
        data_role=data_role,
        shards=tuple(shards),
    )


def _normalize_source(source: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in source.dims or "latitude" in source.coords:
        rename["latitude"] = "lat"
    if "longitude" in source.dims or "longitude" in source.coords:
        rename["longitude"] = "lon"
    if rename:
        source = source.rename(rename)
    if "batch" in source.dims:
        if int(source.sizes["batch"]) != 1:
            raise ValueError("official source batch dimension must have size one")
        source = source.isel(batch=0, drop=True)
    return source


class OfficialTrainingCorpus:
    """Expose indexed NetCDF shards as one lazy training dataset.

    Tensor layout conversion comes from the colleague-provided
    ``graphcast_compat.data`` module. Per-window inputs, targets, and forcings
    are extracted by the pinned DeepMind ``data_utils`` implementation through
    ``OfficialTrainingDataset``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        config_path: str | Path,
        rollout_steps: int = 1,
        split_start: str | None = None,
        split_end: str | None = None,
        lat_start: int = 0,
        lat_count: int = 0,
        lon_start: int = 0,
        lon_count: int = 0,
        max_open_shards: int = 2,
        split_cache_max_gib: float = 0.0,
        extraction_cache_max_gib: float = 0.0,
    ):
        if (
            isinstance(max_open_shards, bool)
            or not isinstance(max_open_shards, int)
            or max_open_shards <= 0
        ):
            raise ValueError("max_open_shards must be positive")
        if (
            not math.isfinite(float(split_cache_max_gib))
            or float(split_cache_max_gib) < 0.0
        ):
            raise ValueError(
                "split_cache_max_gib must be finite and non-negative"
            )
        self.manifest = load_training_corpus_index(manifest_path)
        self.path = self.manifest.path
        self.max_open_shards = max_open_shards
        self._open_sources: OrderedDict[int, xr.Dataset] = OrderedDict()
        self._static_cache: OrderedDict[int, xr.Dataset] = OrderedDict()
        self._validated_static_cache: dict[int, xr.Dataset] = {}
        self._dynamic_split_cache: dict[int, xr.Dataset] = {}
        self._dynamic_split_cache_positions: dict[int, dict[int, int]] = {}
        self._split_cache_status: dict[str, Any] = {
            "requested_max_gib": float(split_cache_max_gib),
            "enabled": False,
            "estimated_bytes": 0,
            "loaded_bytes": 0,
            "preload_seconds": 0.0,
            "reason": "disabled",
            "finite_validation": "per_sample",
            "finite_validation_bytes": 0,
            "finite_validation_seconds": 0.0,
        }

        for shard in self.manifest.shards:
            if not shard.path.is_file():
                raise FileNotFoundError(
                    f"corpus shard is missing: {shard.path}"
                )

        template = OfficialTrainingDataset(
            self.manifest.shards[0].path,
            config_path=config_path,
            data_role=self.manifest.data_role,
            dataset_kind=self.manifest.dataset_kind,
            training_stage=self.manifest.training_stage,
            rollout_steps=rollout_steps,
            lat_start=lat_start,
            lat_count=lat_count,
            lon_start=lon_start,
            lon_count=lon_count,
            extraction_cache_max_gib=extraction_cache_max_gib,
        )
        template.close()
        self._dataset = template
        self._dataset.path = self.manifest.path
        self._dataset._source_window = self._source_window

        timestamps = []
        locations = []
        reference_coordinates: dict[str, np.ndarray] | None = None
        reference_shapes: dict[str, tuple[tuple[str, int], ...]] | None = None
        previous_last: np.datetime64 | None = None
        for shard_index, shard in enumerate(self.manifest.shards):
            source = _normalize_source(
                xr.open_dataset(shard.path, cache=False)
            )
            try:
                shard_timestamps = _time_values(source)
                _validate_six_hour_spacing(shard_timestamps)
                if (
                    previous_last is not None
                    and shard_timestamps[0] <= previous_last
                ):
                    raise ValueError(
                        "corpus shards overlap or contain duplicate times"
                    )
                previous_last = shard_timestamps[-1]
                coordinates = {
                    name: np.asarray(source.coords[name].values)
                    for name in ("lat", "lon", "level")
                }
                required_names = sorted(
                    (
                        set(template.input_variables)
                        | set(template.target_variables)
                        | set(template.static_variables)
                    )
                    & set(source.data_vars)
                )
                shapes = {
                    name: tuple(
                        (dimension, int(source[name].sizes[dimension]))
                        for dimension in source[name].dims
                        if dimension not in ("time", "batch")
                    )
                    for name in required_names
                }
                if reference_coordinates is None:
                    reference_coordinates = coordinates
                    reference_shapes = shapes
                else:
                    if any(
                        not np.array_equal(
                            reference_coordinates[name],
                            coordinates[name],
                        )
                        for name in reference_coordinates
                    ):
                        raise ValueError("corpus shard grid drift detected")
                    if shapes != reference_shapes:
                        raise ValueError(
                            "corpus shard variable layout drift detected"
                        )
                timestamps.extend(shard_timestamps.tolist())
                locations.extend(
                    (shard_index, local_index)
                    for local_index in range(len(shard_timestamps))
                )
            finally:
                source.close()
        self.timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
        self._locations = tuple(locations)
        start = _parse_bound(split_start)
        end = _parse_bound(split_end)
        if start is not None and end is not None and start >= end:
            raise ValueError("split_start must be earlier than split_end")
        self._window_indices_by_rollout = {}
        self._excluded_gap_windows_by_rollout = {}
        for active_rollout in range(1, rollout_steps + 1):
            valid, excluded_gaps = _eligible_reference_indices(
                self.timestamps,
                rollout_steps=active_rollout,
                split_start=start,
                split_end=end,
                training_stage=self.manifest.training_stage,
            )
            if not valid:
                raise ValueError(
                    "corpus split contains no complete windows for "
                    f"rollout={active_rollout}"
                )
            self._window_indices_by_rollout[active_rollout] = valid
            self._excluded_gap_windows_by_rollout[
                active_rollout
            ] = excluded_gaps
        self.window_indices = self._window_indices_by_rollout[rollout_steps]
        self.excluded_gap_window_count = (
            self._excluded_gap_windows_by_rollout[rollout_steps]
        )
        self._dataset.timestamps = self.timestamps
        self._dataset.window_indices = self.window_indices
        self._dataset._window_indices_by_rollout = (
            self._window_indices_by_rollout
        )
        self._initialize_split_cache(float(split_cache_max_gib))

    @property
    def shard_count(self) -> int:
        return len(self.manifest.shards)

    @property
    def open_shard_count(self) -> int:
        return len(self._open_sources)

    @property
    def cached_static_shard_count(self) -> int:
        return len(self._static_cache) + len(self._validated_static_cache)

    @property
    def split_cache_status(self) -> dict[str, Any]:
        return dict(self._split_cache_status)

    @property
    def window_count(self) -> int:
        return len(self.window_indices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def _open_shard(self, shard_index: int) -> xr.Dataset:
        if shard_index in self._open_sources:
            source = self._open_sources.pop(shard_index)
            self._open_sources[shard_index] = source
            return source
        source = _normalize_source(
            xr.open_dataset(
                self.manifest.shards[shard_index].path,
                cache=False,
            )
        )
        self._open_sources[shard_index] = source
        while len(self._open_sources) > self.max_open_shards:
            oldest_index, oldest = self._open_sources.popitem(last=False)
            oldest.close()
            self._static_cache.pop(oldest_index, None)
        return source

    def _static_for_shard(
        self,
        shard_index: int,
        source: xr.Dataset,
        names: list[str],
    ) -> xr.Dataset:
        """Load each shard's small static fields at most once per open handle."""

        if shard_index in self._validated_static_cache:
            return self._validated_static_cache[shard_index]
        if shard_index in self._static_cache:
            cached = self._static_cache.pop(shard_index)
            self._static_cache[shard_index] = cached
            return cached
        cached = self._crop(source[names]).load()
        self._static_cache[shard_index] = cached
        return cached

    def _crop(self, dataset: xr.Dataset) -> xr.Dataset:
        selections: dict[str, Any] = {
            "lat": self._dataset._latitude_indices,
            "lon": self._dataset._longitude_indices,
        }
        cropped = dataset.isel(**selections)
        if "level" in cropped.dims:
            cropped = cropped.sel(
                level=self._dataset.pressure_levels.tolist()
            )
        return cropped

    def _initialize_split_cache(self, maximum_gib: float) -> None:
        """Preload only time frames reachable by the selected training split.

        The cache is an execution optimization: sampling order, xarray
        extraction, normalization and all tensors remain unchanged. A zero
        limit preserves the lazy streaming path. If the selected split does
        not fit under the explicit bound, the loader records that fact and
        continues lazily instead of changing the data contract.
        """

        if maximum_gib == 0.0:
            return
        maximum_bytes = int(maximum_gib * 1024**3)
        needed_by_shard: dict[int, dict[int, int]] = {}
        for rollout, references in self._window_indices_by_rollout.items():
            for reference_index in references:
                for global_index in range(
                    reference_index - 1,
                    reference_index + rollout + 1,
                ):
                    shard_index, local_index = self._locations[global_index]
                    needed_by_shard.setdefault(shard_index, {})[
                        global_index
                    ] = local_index

        plans: dict[int, tuple[list[int], dict[int, int], list[str]]] = {}
        estimated_bytes = 0
        for shard_index, global_to_local in sorted(needed_by_shard.items()):
            source = self._open_shard(shard_index)
            dynamic_names = sorted(
                name
                for name in (
                    set(self._dataset.input_variables)
                    | set(self._dataset.target_variables)
                    | set(self._dataset.static_variables)
                )
                if name in source and "time" in source[name].dims
            )
            ordered_pairs = sorted(
                global_to_local.items(),
                key=lambda item: item[1],
            )
            local_indices = [local_index for _, local_index in ordered_pairs]
            positions = {
                global_index: position
                for position, (global_index, _) in enumerate(ordered_pairs)
            }
            candidate = self._crop(
                source[dynamic_names].isel(time=local_indices)
            )
            estimated_bytes += sum(
                int(variable.size) * int(np.dtype(variable.dtype).itemsize)
                for variable in candidate.data_vars.values()
            )
            plans[shard_index] = (local_indices, positions, dynamic_names)

        self._split_cache_status["estimated_bytes"] = estimated_bytes
        if estimated_bytes > maximum_bytes:
            self._split_cache_status["reason"] = "estimated_size_exceeds_limit"
            return

        preload_started = time.perf_counter()
        loaded_bytes = 0
        for shard_index, (
            local_indices,
            positions,
            dynamic_names,
        ) in sorted(plans.items()):
            source = self._open_shard(shard_index)
            cached = self._crop(
                source[dynamic_names].isel(time=local_indices)
            ).load()
            loaded_bytes += sum(
                int(variable.nbytes) for variable in cached.data_vars.values()
            )
            self._dynamic_split_cache[shard_index] = cached
            self._dynamic_split_cache_positions[shard_index] = positions
        self._split_cache_status.update(
            {
                "enabled": True,
                "loaded_bytes": loaded_bytes,
                "preload_seconds": time.perf_counter() - preload_started,
                "reason": "loaded",
            }
        )
        self._validate_cached_split_finite_once()

    def _validate_cached_split_finite_once(self) -> None:
        """Validate cache-owned source values once before optimizer update 1.

        This remains fail-closed, but avoids scanning the same full
        inputs/targets/forcings after every extraction. It is only selected
        when every dynamic frame reachable by the split is resident for this
        corpus object's lifetime. Lazy streaming retains per-sample checks.
        """

        started = time.perf_counter()
        validated_bytes = 0
        for shard_index, dynamic in sorted(
            self._dynamic_split_cache.items()
        ):
            _finite_dataset(
                dynamic,
                role=f"cached dynamic shard {shard_index}",
            )
            validated_bytes += _dataset_nbytes(dynamic)
            _freeze_dataset_arrays(dynamic)

            source = self._open_shard(shard_index)
            static_names = sorted(
                name
                for name in (
                    set(self._dataset.input_variables)
                    | set(self._dataset.target_variables)
                    | set(self._dataset.static_variables)
                )
                if name in source and "time" not in source[name].dims
            )
            if static_names:
                static = self._static_for_shard(
                    shard_index,
                    source,
                    static_names,
                )
                _finite_dataset(
                    static,
                    role=f"cached static shard {shard_index}",
                )
                validated_bytes += _dataset_nbytes(static)
                _freeze_dataset_arrays(static)
                self._validated_static_cache[shard_index] = static
                self._static_cache.pop(shard_index, None)

        for name, values in (
            ("latitude", self._dataset.latitudes),
            ("longitude", self._dataset.longitudes),
            ("pressure level", self._dataset.pressure_levels),
        ):
            if not np.isfinite(np.asarray(values)).all():
                raise ValueError(f"cached {name} coordinate is non-finite")
        if np.isnat(self.timestamps).any():
            raise ValueError("cached time coordinate contains NaT")

        self._dataset._sample_finite_validation = False
        self._split_cache_status.update(
            {
                "finite_validation": "cached_source_once_before_updates",
                "finite_validation_bytes": validated_bytes,
                "finite_validation_seconds": time.perf_counter() - started,
            }
        )

    def _source_window(
        self,
        reference_index: int,
        target_rollout_steps: int,
    ) -> xr.Dataset:
        target_rollout_steps = _resolve_target_rollout_steps(
            target_rollout_steps,
            dataset_rollout_steps=self._dataset.rollout_steps,
        )
        start = reference_index - 1
        stop = reference_index + target_rollout_steps + 1
        requested = list(range(start, stop))
        groups: list[tuple[int, list[tuple[int, int]]]] = []
        for global_index in requested:
            shard_index, local_index = self._locations[global_index]
            if not groups or groups[-1][0] != shard_index:
                groups.append((shard_index, []))
            groups[-1][1].append((global_index, local_index))

        variables = sorted(
            set(self._dataset.input_variables)
            | set(self._dataset.target_variables)
            | set(self._dataset.static_variables)
        )
        dynamic_fragments = []
        static_dataset = None
        for shard_index, positions in groups:
            source = self._open_shard(shard_index)
            dynamic_names = [
                name
                for name in variables
                if name in source and "time" in source[name].dims
            ]
            static_names = [
                name
                for name in variables
                if name in source and "time" not in source[name].dims
            ]
            local_start = positions[0][1]
            local_stop = positions[-1][1] + 1
            if shard_index in self._dynamic_split_cache:
                cache_positions = self._dynamic_split_cache_positions[
                    shard_index
                ]
                fragment = self._dynamic_split_cache[shard_index].isel(
                    time=[
                        cache_positions[global_index]
                        for global_index, _ in positions
                    ]
                ).assign_coords(
                    time=np.asarray(
                        [global_index for global_index, _ in positions],
                        dtype=np.int64,
                    )
                )
            else:
                fragment = self._crop(
                    source[dynamic_names].isel(
                        time=slice(local_start, local_stop)
                    )
                )
                fragment = fragment.assign_coords(
                    time=np.asarray(
                        [position[0] for position in positions],
                        dtype=np.int64,
                    )
                ).load()
            dynamic_fragments.append(fragment)
            if static_dataset is None:
                static_dataset = self._static_for_shard(
                    shard_index,
                    source,
                    static_names,
                )
        dynamic = (
            dynamic_fragments[0]
            if len(dynamic_fragments) == 1
            else xr.concat(
                dynamic_fragments,
                dim="time",
                data_vars="all",
                coords="minimal",
                compat="override",
                join="exact",
            )
        )
        if static_dataset is None:
            return dynamic
        return xr.merge((dynamic, static_dataset), compat="override")

    def sample(
        self,
        *,
        step: int,
        batch_size: int,
        rank: int = 0,
        replicas: int = 1,
        target_rollout_steps: int | None = None,
    ) -> TrainingSample:
        return self._dataset.sample(
            step=step,
            batch_size=batch_size,
            rank=rank,
            replicas=replicas,
            target_rollout_steps=target_rollout_steps,
        )

    def sample_indices(
        self,
        *,
        window_indices: tuple[int, ...] | list[int],
        sample_weights: tuple[float, ...] | list[float],
        rank: int = 0,
        replicas: int = 1,
        target_rollout_steps: int | None = None,
        allow_replacement: bool = False,
    ) -> TrainingSample:
        return self._dataset.sample_indices(
            window_indices=window_indices,
            sample_weights=sample_weights,
            rank=rank,
            replicas=replicas,
            target_rollout_steps=target_rollout_steps,
            allow_replacement=allow_replacement,
        )

    def replacement_indices(
        self,
        *,
        update_index: int,
        batch_size: int,
        seed: int,
        target_rollout_steps: int | None = None,
    ) -> tuple[int, ...]:
        """Draw the official per-update batch without materializing data."""

        return self._dataset.replacement_indices(
            update_index=update_index,
            batch_size=batch_size,
            seed=seed,
            target_rollout_steps=target_rollout_steps,
        )

    def close(self) -> None:
        for source in self._open_sources.values():
            source.close()
        self._open_sources.clear()
        self._static_cache.clear()
        self._validated_static_cache.clear()
        for cached in self._dynamic_split_cache.values():
            cached.close()
        self._dynamic_split_cache.clear()
        self._dynamic_split_cache_positions.clear()

    def __enter__(self) -> "OfficialTrainingCorpus":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
