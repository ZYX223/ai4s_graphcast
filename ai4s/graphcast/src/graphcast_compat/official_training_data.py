"""Lazy autoregressive GraphCast windows from official-style NetCDF datasets.

The source remains open and lazy.  Only the two input frames, requested target
trajectory, known forcings, and requested spatial crop are loaded per sample.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from graphcast import UPSTREAM_COMMIT
from graphcast import data_utils as deepmind_data_utils


_SIX_HOURS_SECONDS = 6 * 60 * 60
_VALIDATION_START = np.datetime64("2022-01-01T00:00:00", "ns")
_TRAINING_START = np.datetime64("2016-01-01T00:00:00", "ns")
_ERA5_PRETRAIN_END = np.datetime64("2018-01-01T00:00:00", "ns")
OFFICIAL_DATA_UTILS_COMMIT = UPSTREAM_COMMIT
OFFICIAL_DATA_UTILS_REFERENCE = (
    "https://github.com/google-deepmind/weathernext/blob/"
    f"{OFFICIAL_DATA_UTILS_COMMIT}/graphcast/data_utils.py"
)


@dataclass(frozen=True)
class TrainingSample:
    """One or more distinct GraphCast training windows."""

    inputs: xr.Dataset
    targets: xr.Dataset
    forcings: xr.Dataset
    window_indices: tuple[int, ...]
    global_window_indices: tuple[int, ...]
    reference_datetimes: tuple[str, ...]
    target_datetimes: tuple[str, ...]
    target_datetimes_by_sample: tuple[tuple[str, ...], ...]
    data_role: str
    dataset_kind: str
    training_stage: str
    rollout_steps: int
    dataset_rollout_steps: int
    global_batch_size: int
    local_batch_size: int
    rank: int
    replicas: int
    sample_weights: tuple[float, ...]
    global_sample_weights: tuple[float, ...]

    @property
    def valid_sample_count(self) -> int:
        return sum(weight > 0.0 for weight in self.sample_weights)

    @property
    def padded_sample_count(self) -> int:
        return self.local_batch_size - self.valid_sample_count

    @property
    def global_valid_sample_count(self) -> int:
        return sum(weight > 0.0 for weight in self.global_sample_weights)

    @property
    def global_padded_sample_count(self) -> int:
        return self.global_batch_size - self.global_valid_sample_count


def _datetime_text(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")


def _parse_bound(value: str | None) -> np.datetime64 | None:
    if value is None or not str(value).strip():
        return None
    parsed = np.datetime64(str(value), "ns")
    if np.isnat(parsed):
        raise ValueError(f"invalid datetime bound: {value}")
    return parsed


def _time_values(source: xr.Dataset) -> np.ndarray:
    if "time" not in source.coords:
        raise ValueError("official training source must have a time coordinate")
    raw_time = np.asarray(source.coords["time"].values)
    if np.issubdtype(raw_time.dtype, np.datetime64):
        return raw_time.astype("datetime64[ns]")

    if "datetime" not in source.coords:
        raise ValueError(
            "numeric source time requires a datetime reference coordinate"
        )
    datetimes = np.asarray(source.coords["datetime"].values)
    datetimes = np.squeeze(datetimes)
    if datetimes.ndim != 1 or len(datetimes) != len(raw_time):
        raise ValueError("datetime coordinate must align with the time dimension")
    datetimes = datetimes.astype("datetime64[ns]")
    if np.isnat(datetimes).any():
        raise ValueError("datetime coordinate contains NaT")
    if len(np.unique(datetimes)) > 1:
        return datetimes

    if np.issubdtype(raw_time.dtype, np.timedelta64):
        offsets = raw_time.astype("timedelta64[ns]")
    elif np.issubdtype(raw_time.dtype, np.number):
        if not np.isfinite(raw_time.astype(np.float64)).all():
            raise ValueError("numeric time coordinate contains non-finite values")
        offsets = np.rint(raw_time.astype(np.float64) * 3600.0).astype(
            "timedelta64[s]"
        )
    else:
        raise ValueError(f"unsupported time coordinate dtype: {raw_time.dtype}")
    return (datetimes[0] + offsets).astype("datetime64[ns]")


def _validate_six_hour_spacing(timestamps: np.ndarray) -> None:
    if len(timestamps) < 3:
        raise ValueError("official training source requires at least three times")
    delta_seconds = np.diff(timestamps).astype("timedelta64[s]").astype(np.int64)
    if not np.all(delta_seconds == _SIX_HOURS_SECONDS):
        raise ValueError(
            "official training source must use contiguous six-hour time spacing"
        )


def _resolve_count(count: int, start: int, total: int, name: str) -> int:
    if start < 0 or start >= total:
        raise ValueError(f"{name}_start is outside the source grid")
    if count == 0:
        count = total - start
    if count <= 0 or start + count > total:
        raise ValueError(f"{name}_count is outside the source grid")
    return count


def _resolve_target_rollout_steps(
    target_rollout_steps: int | None,
    *,
    dataset_rollout_steps: int,
) -> int:
    if target_rollout_steps is None:
        return dataset_rollout_steps
    if (
        isinstance(target_rollout_steps, bool)
        or not isinstance(target_rollout_steps, int)
        or target_rollout_steps <= 0
        or target_rollout_steps > dataset_rollout_steps
    ):
        raise ValueError(
            "target_rollout_steps must be an integer in "
            f"[1, {dataset_rollout_steps}]"
        )
    return target_rollout_steps


def _eligible_reference_indices(
    timestamps: np.ndarray,
    *,
    rollout_steps: int,
    split_start: np.datetime64 | None,
    split_end: np.datetime64 | None,
    training_stage: str,
) -> tuple[tuple[int, ...], int]:
    """Return complete, stage-safe windows for one active rollout horizon."""

    valid = []
    excluded_gap_count = 0
    for reference_index in range(1, len(timestamps) - rollout_steps):
        window_times = timestamps[
            reference_index - 1 : reference_index + rollout_steps + 1
        ]
        deltas = np.diff(window_times).astype(
            "timedelta64[s]"
        ).astype(np.int64)
        if not np.all(deltas == _SIX_HOURS_SECONDS):
            excluded_gap_count += 1
            continue
        reference_time = timestamps[reference_index]
        first_target_time = timestamps[reference_index + 1]
        final_target_time = timestamps[reference_index + rollout_steps]
        if split_start is not None and reference_time < split_start:
            continue
        if split_end is not None and final_target_time >= split_end:
            continue
        if (
            training_stage == "era5_pretrain"
            and final_target_time >= _ERA5_PRETRAIN_END
        ):
            raise ValueError("ERA5 pretraining requires targets before 2018")
        if training_stage == "hres_finetune":
            if final_target_time >= _VALIDATION_START:
                raise ValueError(
                    "2022 and later HRES data is validation-only and cannot "
                    "enter HRES fine-tuning"
                )
            if first_target_time < _TRAINING_START:
                raise ValueError(
                    "HRES operational fine-tuning starts in 2016"
                )
        if (
            training_stage == "validation"
            and first_target_time < _VALIDATION_START
        ):
            raise ValueError(
                "validation role requires 2022 or later targets"
            )
        valid.append(reference_index)
    return tuple(valid), excluded_gap_count


def _finite_dataset(dataset: xr.Dataset, *, role: str) -> None:
    for name, variable in dataset.data_vars.items():
        values = np.asarray(variable.values)
        if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
            raise ValueError(f"{role} variable {name} contains non-finite values")


def _dataset_nbytes(dataset: xr.Dataset) -> int:
    """Count materialized data-variable bytes without double-counting coords."""

    return sum(int(variable.nbytes) for variable in dataset.data_vars.values())


def _freeze_dataset_arrays(dataset: xr.Dataset) -> None:
    """Make cache-owned NumPy arrays read-only when the backend permits it."""

    for variable in dataset.data_vars.values():
        values = np.asarray(variable.values)
        if values.flags.writeable:
            values.setflags(write=False)


class OfficialTrainingDataset:
    """Lazily select deterministic two-input/autoregressive GraphCast windows."""

    def __init__(
        self,
        path: str | Path,
        *,
        config_path: str | Path,
        data_role: str = "train",
        dataset_kind: str | None = None,
        training_stage: str | None = None,
        rollout_steps: int = 1,
        split_start: str | None = None,
        split_end: str | None = None,
        shuffle: bool = False,
        seed: int = 0,
        lat_start: int = 0,
        lat_count: int = 0,
        lon_start: int = 0,
        lon_count: int = 0,
        extraction_cache_max_gib: float = 0.0,
    ):
        if data_role not in {"train", "validation"}:
            raise ValueError("data_role must be train or validation")
        if (
            isinstance(rollout_steps, bool)
            or not isinstance(rollout_steps, int)
            or rollout_steps <= 0
        ):
            raise ValueError("rollout_steps must be a positive integer")
        if (
            not math.isfinite(float(extraction_cache_max_gib))
            or float(extraction_cache_max_gib) < 0.0
        ):
            raise ValueError(
                "extraction_cache_max_gib must be finite and non-negative"
            )
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"official training dataset is missing: {self.path}")
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        task = config["task_config"]
        variables = config["variables"]
        self.input_duration = str(task["input_duration"])
        self.input_variables = tuple(task["input_variables"])
        self.target_variables = tuple(task["target_variables"])
        self.forcing_variables = tuple(task["forcing_variables"])
        self.static_variables = tuple(
            task.get("static_variables", variables["static"])
        )
        self.atmospheric_variables = frozenset(variables["atmospheric"])
        self.pressure_levels = np.asarray(
            task["pressure_levels"],
            dtype=np.int32,
        )
        source = xr.open_dataset(self.path, cache=False)
        source_dataset_kind = source.attrs.get("graphcast_dataset_kind")
        source_training_stage = source.attrs.get("graphcast_training_stage")
        if dataset_kind is None:
            dataset_kind = str(source_dataset_kind or "hres")
        if training_stage is None:
            training_stage = str(
                source_training_stage
                or (
                    "validation"
                    if data_role == "validation"
                    else (
                        "era5_pretrain"
                        if dataset_kind == "era5"
                        else "hres_finetune"
                    )
                )
            )
        allowed_contracts = {
            ("era5", "train", "era5_pretrain"),
            ("hres", "train", "hres_finetune"),
            ("hres", "validation", "validation"),
        }
        if (dataset_kind, data_role, training_stage) not in allowed_contracts:
            source.close()
            raise ValueError(
                "unsupported dataset/stage contract: "
                f"dataset_kind={dataset_kind}, data_role={data_role}, "
                f"training_stage={training_stage}"
            )
        if (
            source_dataset_kind is not None
            and str(source_dataset_kind) != dataset_kind
        ):
            source.close()
            raise ValueError(
                "dataset_kind disagrees with source provenance attribute"
            )
        if (
            source_training_stage is not None
            and str(source_training_stage) != training_stage
        ):
            source.close()
            raise ValueError(
                "training_stage disagrees with source provenance attribute"
            )
        self.data_role = data_role
        self.dataset_kind = dataset_kind
        self.training_stage = training_stage
        self.rollout_steps = rollout_steps
        self.official_data_utils_reference = OFFICIAL_DATA_UTILS_REFERENCE
        self._sample_finite_validation = True
        self._extraction_cache_max_bytes = int(
            float(extraction_cache_max_gib) * 1024**3
        )
        self._extraction_cache: OrderedDict[
            tuple[Any, ...], tuple[xr.Dataset, xr.Dataset, xr.Dataset]
        ] = OrderedDict()
        self._extraction_cache_bytes = 0
        self._extraction_cache_counters = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "skipped_oversize": 0,
        }
        rename = {}
        if "latitude" in source.dims or "latitude" in source.coords:
            rename["latitude"] = "lat"
        if "longitude" in source.dims or "longitude" in source.coords:
            rename["longitude"] = "lon"
        if rename:
            source = source.rename(rename)
        if "batch" in source.dims:
            if int(source.sizes["batch"]) != 1:
                source.close()
                raise ValueError("official source batch dimension must have size one")
            source = source.isel(batch=0, drop=True)
        for coordinate in ("lat", "lon", "level"):
            if coordinate not in source.coords:
                source.close()
                raise ValueError(f"official source is missing coordinate {coordinate}")

        levels = np.asarray(source.coords["level"].values, dtype=np.int32)
        if set(levels.tolist()) != set(self.pressure_levels.tolist()):
            source.close()
            raise ValueError(
                "official source pressure levels do not match the model contract"
            )
        self.timestamps = _time_values(source)
        _validate_six_hour_spacing(self.timestamps)

        source_latitudes = np.asarray(source.coords["lat"].values, dtype=np.float32)
        source_longitudes = np.asarray(source.coords["lon"].values, dtype=np.float32)
        if len(source_latitudes) < 2 or len(source_longitudes) < 2:
            source.close()
            raise ValueError("official source grid is too small")
        latitude_order = np.argsort(source_latitudes)[::-1]
        longitude_order = np.argsort(np.mod(source_longitudes, 360.0))
        lat_count = _resolve_count(
            lat_count, lat_start, len(latitude_order), "lat"
        )
        lon_count = _resolve_count(
            lon_count, lon_start, len(longitude_order), "lon"
        )
        self._latitude_indices = latitude_order[
            lat_start : lat_start + lat_count
        ]
        self._longitude_indices = longitude_order[
            lon_start : lon_start + lon_count
        ]
        self.latitudes = source_latitudes[self._latitude_indices]
        self.longitudes = np.mod(
            source_longitudes[self._longitude_indices], 360.0
        ).astype(np.float32)

        required_source = (
            set(self.input_variables)
            | set(self.target_variables)
            | set(self.static_variables)
        )
        derived = {
            "year_progress_sin",
            "year_progress_cos",
            "day_progress_sin",
            "day_progress_cos",
        }
        required_source -= derived
        missing = sorted(required_source - set(source.data_vars))
        if missing:
            source.close()
            raise ValueError(
                "official source is missing required variables: "
                + ", ".join(missing)
            )

        start = _parse_bound(split_start)
        end = _parse_bound(split_end)
        if start is not None and end is not None and start >= end:
            source.close()
            raise ValueError("split_start must be earlier than split_end")
        self._window_indices_by_rollout: dict[int, tuple[int, ...]] = {}
        for active_rollout in range(1, self.rollout_steps + 1):
            valid, _ = _eligible_reference_indices(
                self.timestamps,
                rollout_steps=active_rollout,
                split_start=start,
                split_end=end,
                training_stage=training_stage,
            )
            if not valid:
                source.close()
                raise ValueError(
                    "split contains no complete GraphCast windows for "
                    f"rollout={active_rollout}"
                )
            order = np.asarray(valid, dtype=np.int64)
            if shuffle:
                order = np.random.default_rng(
                    np.random.SeedSequence([seed, active_rollout])
                ).permutation(order)
            self._window_indices_by_rollout[active_rollout] = tuple(
                int(index) for index in order
            )
        self.window_indices = self._window_indices_by_rollout[
            self.rollout_steps
        ]
        self._source = source

    @property
    def extraction_cache_status(self) -> dict[str, Any]:
        return {
            "enabled": self._extraction_cache_max_bytes > 0,
            "maximum_bytes": self._extraction_cache_max_bytes,
            "current_bytes": self._extraction_cache_bytes,
            "entries": len(self._extraction_cache),
            **self._extraction_cache_counters,
        }

    @property
    def window_count(self) -> int:
        return len(self.window_indices)

    def eligible_window_indices(
        self,
        target_rollout_steps: int | None = None,
    ) -> tuple[int, ...]:
        active_rollout = _resolve_target_rollout_steps(
            target_rollout_steps,
            dataset_rollout_steps=self.rollout_steps,
        )
        return self._window_indices_by_rollout[active_rollout]

    def window_count_for_rollout(self, rollout_steps: int) -> int:
        return len(self.eligible_window_indices(rollout_steps))

    def close(self) -> None:
        self._extraction_cache.clear()
        self._extraction_cache_bytes = 0
        self._source.close()

    def __enter__(self) -> "OfficialTrainingDataset":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _source_window(
        self,
        reference_index: int,
        target_rollout_steps: int,
    ) -> xr.Dataset:
        variables = sorted(
            (
                set(self.input_variables)
                | set(self.target_variables)
                | set(self.static_variables)
            )
            & set(self._source.data_vars)
        )
        window = self._source[variables].isel(
            time=slice(
                reference_index - 1,
                reference_index + target_rollout_steps + 1,
            ),
            lat=self._latitude_indices,
            lon=self._longitude_indices,
        )
        window = window.sel(level=self.pressure_levels.tolist())
        return window.load()

    def _one_window(
        self,
        reference_index: int,
        batch_index: int,
        target_rollout_steps: int,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """Load one window and delegate extraction to DeepMind data_utils."""

        cache_key = None
        if self._extraction_cache_max_bytes > 0:
            target_datetimes = tuple(
                int(
                    self.timestamps[reference_index + lead].astype(
                        "datetime64[ns]"
                    ).astype(np.int64)
                )
                for lead in range(1, target_rollout_steps + 1)
            )
            cache_key = (
                "official_deepmind_extraction_v1",
                int(
                    self.timestamps[reference_index].astype(
                        "datetime64[ns]"
                    ).astype(np.int64)
                ),
                target_datetimes,
                int(target_rollout_steps),
                ("input", self.input_variables),
                ("target", self.target_variables),
                ("forcing", self.forcing_variables),
                self.input_duration,
                tuple(int(value) for value in self.pressure_levels),
                tuple(float(value) for value in self.latitudes),
                tuple(float(value) for value in self.longitudes),
                self.data_role,
                self.dataset_kind,
                self.training_stage,
            )
            cached = self._extraction_cache.pop(cache_key, None)
            if cached is not None:
                self._extraction_cache[cache_key] = cached
                self._extraction_cache_counters["hits"] += 1
                return tuple(
                    dataset.copy(deep=False).assign_coords(
                        batch=np.asarray([batch_index], dtype=np.int32),
                    )
                    for dataset in cached
                )
            self._extraction_cache_counters["misses"] += 1

        window = self._source_window(
            reference_index,
            target_rollout_steps,
        )
        window_timestamps = self.timestamps[
            reference_index - 1 : (
                reference_index + target_rollout_steps + 1
            )
        ]
        source_lead_times = (
            np.arange(target_rollout_steps + 2, dtype=np.int64)
            * np.timedelta64(6, "h")
        )
        # WeatherNext expects a timedelta ``time`` axis and an explicitly
        # batched absolute ``datetime`` coordinate. Xarray otherwise preserves
        # a pre-existing datetime coordinate as one-dimensional when adding
        # ``batch``, while upstream add_derived_vars correctly requires
        # (batch, time).
        official_batch = (
            window.assign_coords(time=source_lead_times)
            .drop_vars("datetime", errors="ignore")
            .expand_dims(
                batch=np.asarray([batch_index], dtype=np.int32),
                axis=0,
            )
            .assign_coords(
                datetime=(
                    ("batch", "time"),
                    window_timestamps[np.newaxis, :],
                )
            )
        )
        inputs_dataset, targets_dataset, forcings_dataset = (
            deepmind_data_utils.extract_inputs_targets_forcings(
                official_batch,
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(
                    int(level) for level in self.pressure_levels
                ),
                input_duration=self.input_duration,
                target_lead_times=slice(
                    "6h",
                    f"{target_rollout_steps * 6}h",
                ),
            )
        )
        if self._sample_finite_validation:
            _finite_dataset(inputs_dataset, role="input")
            _finite_dataset(targets_dataset, role="target")
            _finite_dataset(forcings_dataset, role="forcing")
        if self._extraction_cache_max_bytes > 0:
            assert cache_key is not None
            cache_value = tuple(
                dataset.assign_coords(
                    batch=np.asarray([0], dtype=np.int32)
                )
                for dataset in (
                    inputs_dataset,
                    targets_dataset,
                    forcings_dataset,
                )
            )
            cache_bytes = sum(
                _dataset_nbytes(dataset) for dataset in cache_value
            )
            if cache_bytes <= self._extraction_cache_max_bytes:
                while (
                    self._extraction_cache
                    and self._extraction_cache_bytes + cache_bytes
                    > self._extraction_cache_max_bytes
                ):
                    _, removed = self._extraction_cache.popitem(last=False)
                    self._extraction_cache_bytes -= sum(
                        _dataset_nbytes(dataset) for dataset in removed
                    )
                    self._extraction_cache_counters["evictions"] += 1
                for dataset in cache_value:
                    _freeze_dataset_arrays(dataset)
                self._extraction_cache[cache_key] = cache_value
                self._extraction_cache_bytes += cache_bytes
            else:
                self._extraction_cache_counters["skipped_oversize"] += 1
        return inputs_dataset, targets_dataset, forcings_dataset

    def sample(
        self,
        *,
        step: int,
        batch_size: int,
        rank: int = 0,
        replicas: int = 1,
        target_rollout_steps: int | None = None,
    ) -> TrainingSample:
        if step <= 0:
            raise ValueError("step must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        active_rollout_steps = _resolve_target_rollout_steps(
            target_rollout_steps,
            dataset_rollout_steps=self.rollout_steps,
        )
        available = self.eligible_window_indices(active_rollout_steps)
        if batch_size > len(available):
            raise ValueError(
                "batch_size cannot exceed the number of distinct data windows"
            )
        if (
            isinstance(replicas, bool)
            or not isinstance(replicas, int)
            or replicas <= 0
        ):
            raise ValueError("replicas must be a positive integer")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= replicas
        ):
            raise ValueError("rank must be in [0, replicas)")
        if batch_size % replicas:
            raise ValueError("global batch_size must be divisible by replicas")
        local_batch_size = batch_size // replicas
        start = ((step - 1) * batch_size) % len(available)
        global_positions = [
            available[(start + offset) % len(available)]
            for offset in range(batch_size)
        ]
        return self.sample_indices(
            window_indices=global_positions,
            sample_weights=(1.0,) * batch_size,
            rank=rank,
            replicas=replicas,
            target_rollout_steps=active_rollout_steps,
        )

    def sample_with_replacement(
        self,
        *,
        update_index: int,
        batch_size: int,
        seed: int,
        rank: int = 0,
        replicas: int = 1,
        target_rollout_steps: int | None = None,
    ) -> TrainingSample:
        """Sample a global mini-batch with replacement as in GraphCast §4.4.

        A stateless SeedSequence keyed by ``seed`` and zero-based optimizer
        ``update_index`` makes every batch exactly reproducible after resume
        while keeping all data-parallel ranks on the same global draw.
        """

        if (
            isinstance(update_index, bool)
            or not isinstance(update_index, int)
            or update_index < 0
        ):
            raise ValueError("update_index must be a non-negative integer")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        positions = self.replacement_indices(
            update_index=update_index,
            batch_size=batch_size,
            seed=seed,
            target_rollout_steps=target_rollout_steps,
        )
        return self.sample_indices(
            window_indices=positions,
            sample_weights=(1.0,) * batch_size,
            rank=rank,
            replicas=replicas,
            target_rollout_steps=target_rollout_steps,
            allow_replacement=True,
        )

    def replacement_indices(
        self,
        *,
        update_index: int,
        batch_size: int,
        seed: int,
        target_rollout_steps: int | None = None,
    ) -> tuple[int, ...]:
        """Draw only the official with-replacement indices without loading data."""

        if (
            isinstance(update_index, bool)
            or not isinstance(update_index, int)
            or update_index < 0
        ):
            raise ValueError("update_index must be a non-negative integer")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        generator = np.random.default_rng(
            np.random.SeedSequence([seed, update_index])
        )
        available = np.asarray(
            self.eligible_window_indices(target_rollout_steps),
            dtype=np.int64,
        )
        return tuple(
            int(value)
            for value in generator.choice(
                available,
                size=batch_size,
                replace=True,
            )
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
        """Load an explicit batch using the requested official target horizon."""

        active_rollout_steps = _resolve_target_rollout_steps(
            target_rollout_steps,
            dataset_rollout_steps=self.rollout_steps,
        )
        global_positions = tuple(int(value) for value in window_indices)
        global_weights = tuple(float(value) for value in sample_weights)
        if not global_positions:
            raise ValueError("window_indices must be non-empty")
        if len(global_positions) != len(global_weights):
            raise ValueError(
                "window_indices and sample_weights must have equal length"
            )
        if any(
            not np.isfinite(weight) or weight not in (0.0, 1.0)
            for weight in global_weights
        ):
            raise ValueError("sample_weights must contain only zero or one")
        if not any(weight > 0.0 for weight in global_weights):
            raise ValueError("global batch must contain a valid sample")
        allowed = set(
            self.eligible_window_indices(active_rollout_steps)
        )
        if any(position not in allowed for position in global_positions):
            raise ValueError("window index is outside the dataset split")
        valid_positions = [
            position
            for position, weight in zip(
                global_positions,
                global_weights,
                strict=True,
            )
            if weight > 0.0
        ]
        if (
            len(valid_positions) != len(set(valid_positions))
            and not allow_replacement
        ):
            raise ValueError("valid epoch window indices must be unique")
        batch_size = len(global_positions)
        if (
            isinstance(replicas, bool)
            or not isinstance(replicas, int)
            or replicas <= 0
        ):
            raise ValueError("replicas must be a positive integer")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= replicas
        ):
            raise ValueError("rank must be in [0, replicas)")
        if batch_size % replicas:
            raise ValueError("global batch_size must be divisible by replicas")
        local_batch_size = batch_size // replicas
        local_start = rank * local_batch_size
        positions = global_positions[
            local_start : local_start + local_batch_size
        ]
        weights = global_weights[
            local_start : local_start + local_batch_size
        ]
        input_parts = []
        target_parts = []
        forcing_parts = []
        for batch_index, reference_index in enumerate(positions):
            inputs, targets, forcings = self._one_window(
                reference_index,
                batch_index,
                active_rollout_steps,
            )
            input_parts.append(inputs)
            target_parts.append(targets)
            forcing_parts.append(forcings)
        concat_options = {
            "dim": "batch",
            "data_vars": "all",
            "coords": "minimal",
            "compat": "override",
            "join": "exact",
        }
        inputs = xr.concat(input_parts, **concat_options)
        targets = xr.concat(target_parts, **concat_options)
        forcings = xr.concat(forcing_parts, **concat_options)
        return TrainingSample(
            inputs=inputs,
            targets=targets,
            forcings=forcings,
            window_indices=tuple(positions),
            global_window_indices=tuple(global_positions),
            reference_datetimes=tuple(
                _datetime_text(self.timestamps[index]) for index in positions
            ),
            target_datetimes=tuple(
                _datetime_text(
                    self.timestamps[index + active_rollout_steps]
                )
                for index in positions
            ),
            target_datetimes_by_sample=tuple(
                tuple(
                    _datetime_text(self.timestamps[index + lead])
                    for lead in range(1, active_rollout_steps + 1)
                )
                for index in positions
            ),
            data_role=self.data_role,
            dataset_kind=self.dataset_kind,
            training_stage=self.training_stage,
            rollout_steps=active_rollout_steps,
            dataset_rollout_steps=self.rollout_steps,
            global_batch_size=batch_size,
            local_batch_size=local_batch_size,
            rank=rank,
            replicas=replicas,
            sample_weights=weights,
            global_sample_weights=global_weights,
        )
