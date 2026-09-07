from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat.epoch_plan import EpochPlan


def test_odd_window_count_has_one_zero_weight_padding_position():
    plan = EpochPlan.create(range(111), epoch=0, seed=27, global_batch_size=2, shuffle=True)

    assert plan.valid_window_count == 111
    assert plan.optimizer_steps == 56
    assert plan.padded_sample_count == 1
    assert len(set(plan.ordered_window_indices)) == 111
    assert sum(batch.valid_count for batch in plan.batches) == 111
    assert sum(batch.padded_count for batch in plan.batches) == 1
    assert plan.batches[-1].sample_weights == (1.0, 0.0)
    assert plan.batches[-1].window_indices[0] == plan.batches[-1].window_indices[1]
    assert plan.duplicate_valid_count == 0
    assert plan.coverage == 1.0


def test_plan_is_reproducible_and_epoch_changes_shuffle():
    first = EpochPlan.create(range(32), epoch=3, seed=9, global_batch_size=4, shuffle=True)
    repeated = EpochPlan.create(range(32), epoch=3, seed=9, global_batch_size=4, shuffle=True)
    next_epoch = EpochPlan.create(range(32), epoch=4, seed=9, global_batch_size=4, shuffle=True)

    assert first.digest == repeated.digest
    assert first.ordered_window_indices == repeated.ordered_window_indices
    assert first.ordered_window_indices != next_epoch.ordered_window_indices


def test_rank_batches_partition_global_batch_and_preserve_tail_mask():
    plan = EpochPlan.create(range(5), epoch=0, seed=0, global_batch_size=2, shuffle=False)

    rank_zero = plan.rank_batch(2, rank=0, replicas=2)
    rank_one = plan.rank_batch(2, rank=1, replicas=2)

    assert rank_zero.window_indices == (4,)
    assert rank_zero.sample_weights == (1.0,)
    assert rank_one.window_indices == (4,)
    assert rank_one.sample_weights == (0.0,)
    assert rank_zero.global_valid_count == 1
    assert rank_one.global_valid_count == 1
    assert rank_zero.local_valid_count == 1
    assert rank_one.local_valid_count == 0


def test_metadata_and_cursor_are_auditable():
    plan = EpochPlan.create(range(7), epoch=2, seed=5, global_batch_size=4, shuffle=False)

    metadata = plan.to_metadata()
    assert metadata["schema_version"] == 1
    assert metadata["epoch"] == 2
    assert metadata["optimizer_steps"] == 2
    assert metadata["valid_window_count"] == 7
    assert metadata["padded_sample_count"] == 1
    assert metadata["digest"] == plan.digest
    plan.validate_cursor(0)
    plan.validate_cursor(plan.optimizer_steps)
    with pytest.raises(ValueError):
        plan.validate_cursor(plan.optimizer_steps + 1)


def test_duplicate_or_invalid_window_contract_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        EpochPlan.create((1, 1), epoch=0, seed=0, global_batch_size=2)
    with pytest.raises(ValueError, match="non-empty"):
        EpochPlan.create((), epoch=0, seed=0, global_batch_size=2)
    with pytest.raises(ValueError, match="positive"):
        EpochPlan.create((1,), epoch=0, seed=0, global_batch_size=0)
