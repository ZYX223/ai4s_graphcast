from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from test_training_data_integration import CONFIG_PATH, _write_source


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat.official_training_data import OfficialTrainingDataset


def _dataset(path: Path, **kwargs) -> OfficialTrainingDataset:
    return OfficialTrainingDataset(path, config_path=CONFIG_PATH, **kwargs)


def test_extracts_distinct_batched_windows_and_graphcast_coordinates(tmp_path):
    dataset = _dataset(
        _write_source(tmp_path / "source.nc", start="2021-01-01T00:00:00", time_count=6),
        data_role="train",
        rollout_steps=1,
        lat_start=0,
        lat_count=2,
        lon_start=0,
        lon_count=2,
    )
    try:
        sample = dataset.sample(step=1, batch_size=2)
    finally:
        dataset.close()

    assert sample.window_indices == (1, 2)
    assert sample.reference_datetimes == (
        "2021-01-01T06:00:00",
        "2021-01-01T12:00:00",
    )
    assert sample.target_datetimes == (
        "2021-01-01T12:00:00",
        "2021-01-01T18:00:00",
    )
    assert sample.inputs.sizes["time"] == 2
    assert sample.targets.sizes["time"] == 1
    assert "total_precipitation_6hr" not in sample.inputs
    assert "total_precipitation_6hr" in sample.targets
    assert sample.inputs["temperature"].dims == ("batch", "time", "level", "lat", "lon")
    assert sample.forcings["day_progress_sin"].dims == ("batch", "time", "lon")
    assert all(
        bool(np.isfinite(variable.values).all())
        for variable in (
            *sample.inputs.data_vars.values(),
            *sample.targets.data_vars.values(),
            *sample.forcings.data_vars.values(),
        )
    )


def test_absolute_datetime_source_shuffle_is_seeded_and_epoch_sensitive(tmp_path):
    path = _write_source(tmp_path / "source.nc", start="2017-01-01T00:00:00", time_count=8)
    first = _dataset(path, data_role="train", shuffle=True, seed=17)
    repeated = _dataset(path, data_role="train", shuffle=True, seed=17)
    changed = _dataset(path, data_role="train", shuffle=True, seed=18)
    try:
        first_indices = tuple(first.sample(step=step, batch_size=1).window_indices[0] for step in range(1, 7))
        repeated_indices = tuple(repeated.sample(step=step, batch_size=1).window_indices[0] for step in range(1, 7))
        changed_indices = tuple(changed.sample(step=step, batch_size=1).window_indices[0] for step in range(1, 7))
    finally:
        first.close()
        repeated.close()
        changed.close()

    assert first_indices == repeated_indices
    assert first_indices != changed_indices
    assert set(first_indices) == set(range(1, 7))


def test_active_rollout_is_selected_before_window_materialization(tmp_path):
    dataset = _dataset(
        _write_source(tmp_path / "source.nc", start="2017-01-01T00:00:00", time_count=15),
        data_role="train",
        dataset_kind="era5",
        training_stage="era5_pretrain",
        rollout_steps=12,
    )
    loaded_time_sizes: list[int] = []
    source_window = dataset._source_window

    def tracked_source_window(reference_index: int, target_rollout_steps: int):
        window = source_window(reference_index, target_rollout_steps)
        loaded_time_sizes.append(int(window.sizes["time"]))
        return window

    dataset._source_window = tracked_source_window
    try:
        sample = dataset.sample_indices(
            window_indices=(1,),
            sample_weights=(1.0,),
            target_rollout_steps=2,
        )
    finally:
        dataset.close()

    assert loaded_time_sizes == [4]
    assert sample.dataset_rollout_steps == 12
    assert sample.rollout_steps == 2
    assert sample.targets.sizes["time"] == 2
    assert sample.forcings.sizes["time"] == 2


def test_rank_local_samples_record_global_origins_and_weights(tmp_path):
    dataset = _dataset(
        _write_source(tmp_path / "source.nc", start="2017-01-01T00:00:00", time_count=6),
        data_role="train",
        rollout_steps=1,
    )
    try:
        rank_zero = dataset.sample_indices(
            window_indices=(1, 2),
            sample_weights=(1.0, 0.0),
            rank=0,
            replicas=2,
        )
        rank_one = dataset.sample_indices(
            window_indices=(1, 2),
            sample_weights=(1.0, 0.0),
            rank=1,
            replicas=2,
        )
    finally:
        dataset.close()

    assert rank_zero.window_indices == (1,)
    assert rank_one.window_indices == (2,)
    assert rank_zero.global_window_indices == (1, 2)
    assert rank_one.global_window_indices == (1, 2)
    assert rank_zero.sample_weights == (1.0,)
    assert rank_one.sample_weights == (0.0,)
    assert rank_zero.global_sample_weights == (1.0, 0.0)
    assert rank_one.global_sample_weights == (1.0, 0.0)
    assert rank_zero.global_valid_sample_count == 1
    assert rank_one.global_valid_sample_count == 1
