from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_training_data_integration import CONFIG_PATH, _write_source


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat.official_training_corpus import OfficialTrainingCorpus


def _write_manifest(path: Path, shards: list[Path]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-corpus",
                "dataset_kind": "era5",
                "training_stage": "era5_pretrain",
                "data_role": "train",
                "shards": [{"path": shard.name} for shard in shards],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cross_shard_windows_are_contiguous_and_not_lost(tmp_path):
    first = _write_source(tmp_path / "first.nc", start="2017-01-01T00:00:00", time_count=8)
    second = _write_source(tmp_path / "second.nc", start="2017-01-03T00:00:00", time_count=8)
    manifest = _write_manifest(tmp_path / "corpus.json", [first, second])

    corpus = OfficialTrainingCorpus(
        manifest,
        config_path=CONFIG_PATH,
        rollout_steps=2,
        max_open_shards=1,
    )
    try:
        sample = corpus.sample_indices(window_indices=(7,), sample_weights=(1.0,))
    finally:
        corpus.close()

    assert corpus.shard_count == 2
    assert corpus.window_count == 13
    assert corpus.window_count_for_rollout(1) == 14
    assert corpus.window_count_for_rollout(2) == 13
    assert sample.reference_datetimes == ("2017-01-02T18:00:00",)
    assert sample.target_datetimes_by_sample == (
        ("2017-01-03T00:00:00", "2017-01-03T06:00:00"),
    )
    assert sample.window_indices == (7,)
    assert corpus.manifest.path == manifest.resolve()
    assert corpus.open_shard_count <= 1
    assert corpus.cached_static_shard_count <= 1


def test_cross_shard_active_horizon_preserves_maximum_window_plan(tmp_path):
    first = _write_source(tmp_path / "first.nc", start="2017-01-01T00:00:00", time_count=8)
    second = _write_source(tmp_path / "second.nc", start="2017-01-03T00:00:00", time_count=8)
    corpus = OfficialTrainingCorpus(
        _write_manifest(tmp_path / "corpus.json", [first, second]),
        config_path=CONFIG_PATH,
        rollout_steps=2,
        max_open_shards=1,
    )
    try:
        sample = corpus.sample_indices(
            window_indices=(7,),
            sample_weights=(1.0,),
            target_rollout_steps=1,
        )
    finally:
        corpus.close()

    assert corpus.window_count == 13
    assert sample.dataset_rollout_steps == 2
    assert sample.rollout_steps == 1
    assert sample.target_datetimes_by_sample == (("2017-01-03T00:00:00",),)
    assert sample.targets.sizes["time"] == 1
    assert sample.forcings.sizes["time"] == 1


@pytest.mark.parametrize("first_time_count", (1, 2))
def test_short_first_shard_is_valid_when_global_corpus_completes_window(
    tmp_path,
    first_time_count,
):
    first = _write_source(
        tmp_path / "first.nc",
        start="2017-01-01T00:00:00",
        time_count=first_time_count,
    )
    second = _write_source(
        tmp_path / "second.nc",
        start=np.datetime_as_string(
            np.datetime64("2017-01-01T00:00:00")
            + np.timedelta64(6 * first_time_count, "h"),
            unit="s",
        ),
        time_count=6 - first_time_count,
    )
    corpus = OfficialTrainingCorpus(
        _write_manifest(tmp_path / "corpus.json", [first, second]),
        config_path=CONFIG_PATH,
        rollout_steps=2,
        max_open_shards=1,
    )
    try:
        assert corpus.window_count == 3
        assert corpus.open_shard_count == 0
        sample = corpus.sample_indices(
            window_indices=(1,),
            sample_weights=(1.0,),
        )
    finally:
        corpus.close()

    assert sample.reference_datetimes == ("2017-01-01T06:00:00",)
    assert sample.target_datetimes_by_sample == (
        ("2017-01-01T12:00:00", "2017-01-01T18:00:00"),
    )
    assert sample.inputs.sizes["time"] == 2
    assert sample.targets.sizes["time"] == 2


def test_replacement_sampling_is_fixed_seed_and_records_valid_count(tmp_path):
    shard = _write_source(tmp_path / "source.nc", start="2017-01-01T00:00:00", time_count=8)
    corpus = OfficialTrainingCorpus(
        _write_manifest(tmp_path / "corpus.json", [shard]),
        config_path=CONFIG_PATH,
        rollout_steps=2,
    )
    try:
        positions = corpus.replacement_indices(update_index=3, batch_size=8, seed=27)
        repeated = corpus.replacement_indices(update_index=3, batch_size=8, seed=27)
        sample = corpus.sample_indices(
            window_indices=positions,
            sample_weights=(1.0,) * len(positions),
            target_rollout_steps=1,
            allow_replacement=True,
        )
    finally:
        corpus.close()

    assert positions == repeated
    assert sample.global_window_indices == positions
    assert sample.global_valid_sample_count == 8
    assert len(set(positions)) < len(positions)


def test_close_releases_delegated_extraction_cache_when_dataset_is_retained(tmp_path):
    shard = _write_source(
        tmp_path / "source.nc",
        start="2017-01-01T00:00:00",
        time_count=8,
    )
    corpus = OfficialTrainingCorpus(
        _write_manifest(tmp_path / "corpus.json", [shard]),
        config_path=CONFIG_PATH,
        rollout_steps=2,
        extraction_cache_max_gib=1.0,
    )
    delegated_dataset = corpus._dataset
    corpus.sample_indices(window_indices=(1,), sample_weights=(1.0,))
    assert delegated_dataset.extraction_cache_status["entries"] == 1
    assert delegated_dataset.extraction_cache_status["current_bytes"] > 0

    corpus.close()

    assert delegated_dataset.extraction_cache_status["entries"] == 0
    assert delegated_dataset.extraction_cache_status["current_bytes"] == 0


def test_gap_excludes_only_windows_that_cross_the_gap(tmp_path):
    first = _write_source(tmp_path / "first.nc", start="2017-01-01T00:00:00", time_count=8)
    second = _write_source(tmp_path / "second.nc", start="2017-01-04T00:00:00", time_count=8)
    corpus = OfficialTrainingCorpus(
        _write_manifest(tmp_path / "corpus.json", [first, second]),
        config_path=CONFIG_PATH,
        rollout_steps=2,
    )
    try:
        assert corpus.window_count == 10
        assert corpus.excluded_gap_window_count == 3
    finally:
        corpus.close()


def test_overlapping_shards_fail_closed(tmp_path):
    first = _write_source(tmp_path / "first.nc", start="2017-01-01T00:00:00", time_count=8)
    overlap = _write_source(tmp_path / "overlap.nc", start="2017-01-02T18:00:00", time_count=8)
    manifest = _write_manifest(tmp_path / "corpus.json", [first, overlap])

    with pytest.raises(ValueError, match="overlap|duplicate"):
        OfficialTrainingCorpus(manifest, config_path=CONFIG_PATH, rollout_steps=2)


def test_missing_shard_path_fails_before_training(tmp_path):
    manifest = _write_manifest(tmp_path / "corpus.json", [tmp_path / "missing.nc"])

    with pytest.raises(FileNotFoundError, match="shard is missing"):
        OfficialTrainingCorpus(manifest, config_path=CONFIG_PATH, rollout_steps=2)
