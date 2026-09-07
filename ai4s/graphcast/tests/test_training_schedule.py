from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat.training_schedule import TrainingSchedule


def _published_schedule() -> TrainingSchedule:
    return TrainingSchedule.from_dict(
        {
            "schema_version": 1,
            "name": "published-boundaries",
            "phases": [
                {
                    "name": "warmup",
                    "updates": 1_000,
                    "learning_rate": {"kind": "linear", "start": 0.0, "end": 1.0e-3},
                    "rollout": {"kind": "constant", "value": 1},
                },
                {
                    "name": "one_step",
                    "updates": 299_000,
                    "learning_rate": {
                        "kind": "half_cosine",
                        "start": 1.0e-3,
                        "end": 0.0,
                    },
                    "rollout": {"kind": "constant", "value": 1},
                },
                {
                    "name": "autoregressive",
                    "updates": 11_000,
                    "learning_rate": {"kind": "constant", "value": 3.0e-7},
                    "rollout": {
                        "kind": "increment",
                        "start": 2,
                        "every_updates": 1_000,
                        "maximum": 12,
                    },
                },
            ],
        }
    )


def test_published_phase_and_rollout_boundaries_have_no_off_by_one():
    schedule = _published_schedule()

    assert schedule.total_updates == 311_000
    assert schedule.value_for_update(0).learning_rate == 0.0
    assert schedule.value_for_update(999).learning_rate == pytest.approx(1.0e-3)
    assert schedule.value_for_update(1_000).learning_rate == pytest.approx(1.0e-3)
    assert schedule.value_for_update(299_999).learning_rate == pytest.approx(0.0, abs=1.0e-15)
    assert schedule.value_for_update(300_000).rollout_steps == 2
    assert schedule.value_for_update(301_000).rollout_steps == 3
    assert schedule.value_for_update(310_999).rollout_steps == 12
    with pytest.raises(ValueError, match="outside"):
        schedule.value_for_update(311_000)


def test_schedule_digest_tracks_semantics_and_constant_can_be_unbounded():
    first = TrainingSchedule.constant(name="one", learning_rate=1.0e-4, rollout_steps=12)
    repeated = TrainingSchedule.constant(name="one", learning_rate=1.0e-4, rollout_steps=12)
    changed = TrainingSchedule.constant(name="one", learning_rate=2.0e-4, rollout_steps=12)

    assert first.digest == repeated.digest
    assert first.digest != changed.digest
    assert first.total_updates is None
    assert first.value_for_update(1_000_000).rollout_steps == 12


def test_unbounded_nonfinal_phase_and_zero_rollout_are_rejected():
    value = _published_schedule().to_dict()
    value["phases"][0]["updates"] = None
    with pytest.raises(ValueError, match="final"):
        TrainingSchedule.from_dict(value)
    with pytest.raises(ValueError, match="rollout"):
        TrainingSchedule.constant(name="invalid", learning_rate=1.0e-4, rollout_steps=0)
