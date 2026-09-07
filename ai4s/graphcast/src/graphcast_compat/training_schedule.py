"""Versioned learning-rate and rollout schedules for GraphCast training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    value = _non_negative_integer(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rate(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class ScheduleValue:
    """Resolved schedule values for one zero-based optimizer update."""

    update_index: int
    phase_index: int
    phase_name: str
    phase_update_index: int
    learning_rate: float
    rollout_steps: int


@dataclass(frozen=True)
class _Phase:
    name: str
    updates: int | None
    learning_rate: Mapping[str, Any]
    rollout: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingSchedule:
    """An immutable schedule whose digest is stored in every checkpoint."""

    name: str
    phases: tuple[_Phase, ...]
    canonical: Mapping[str, Any]
    digest: str

    @classmethod
    def constant(
        cls,
        *,
        name: str,
        learning_rate: float,
        rollout_steps: int,
    ) -> "TrainingSchedule":
        return cls.from_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "name": name,
                "phases": [
                    {
                        "name": "constant",
                        "updates": None,
                        "learning_rate": {"kind": "constant", "value": learning_rate},
                        "rollout": {"kind": "constant", "value": rollout_steps},
                    }
                ],
            }
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainingSchedule":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"training schedule is missing: {resolved}")
        return cls.from_dict(json.loads(resolved.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingSchedule":
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"training schedule schema_version must be {SCHEMA_VERSION}")
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("training schedule name must be non-empty")
        raw_phases = value.get("phases")
        if not isinstance(raw_phases, list) or not raw_phases:
            raise ValueError("training schedule phases must be non-empty")

        phases = []
        canonical_phases = []
        for phase_index, raw_phase in enumerate(raw_phases):
            if not isinstance(raw_phase, Mapping):
                raise ValueError("training schedule phase must be an object")
            phase_name = str(raw_phase.get("name", "")).strip()
            if not phase_name:
                raise ValueError("training schedule phase name is required")
            raw_updates = raw_phase.get("updates")
            updates = (
                None
                if raw_updates is None
                else _positive_integer(raw_updates, name="phase updates")
            )
            if updates is None and phase_index != len(raw_phases) - 1:
                raise ValueError("an unbounded phase must be final")
            learning_rate = cls._validate_learning_rate(raw_phase.get("learning_rate"))
            rollout = cls._validate_rollout(raw_phase.get("rollout"))
            phases.append(
                _Phase(
                    name=phase_name,
                    updates=updates,
                    learning_rate=learning_rate,
                    rollout=rollout,
                )
            )
            canonical_phases.append(
                {
                    "name": phase_name,
                    "updates": updates,
                    "learning_rate": dict(learning_rate),
                    "rollout": dict(rollout),
                }
            )
        canonical = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "phases": canonical_phases,
        }
        serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return cls(
            name=name,
            phases=tuple(phases),
            canonical=canonical,
            digest=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _validate_learning_rate(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("learning_rate must be an object")
        kind = str(value.get("kind", ""))
        if kind == "constant":
            return {"kind": kind, "value": _rate(value.get("value"), name="learning rate")}
        if kind in {"linear", "half_cosine"}:
            return {
                "kind": kind,
                "start": _rate(value.get("start"), name="learning rate start"),
                "end": _rate(value.get("end"), name="learning rate end"),
            }
        raise ValueError(f"unsupported learning_rate kind: {kind}")

    @staticmethod
    def _validate_rollout(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("rollout must be an object")
        kind = str(value.get("kind", ""))
        if kind == "constant":
            return {
                "kind": kind,
                "value": _positive_integer(value.get("value"), name="rollout value"),
            }
        if kind == "increment":
            start = _positive_integer(value.get("start"), name="rollout start")
            maximum = _positive_integer(value.get("maximum"), name="rollout maximum")
            if maximum < start:
                raise ValueError("rollout maximum must be at least start")
            return {
                "kind": kind,
                "start": start,
                "every_updates": _positive_integer(
                    value.get("every_updates"),
                    name="rollout every_updates",
                ),
                "maximum": maximum,
            }
        raise ValueError(f"unsupported rollout kind: {kind}")

    @property
    def total_updates(self) -> int | None:
        if any(phase.updates is None for phase in self.phases):
            return None
        return sum(int(phase.updates) for phase in self.phases if phase.updates is not None)

    def value_for_update(self, update_index: int) -> ScheduleValue:
        update_index = _non_negative_integer(update_index, name="update_index")
        remaining = update_index
        for phase_index, phase in enumerate(self.phases):
            if phase.updates is None or remaining < phase.updates:
                return ScheduleValue(
                    update_index=update_index,
                    phase_index=phase_index,
                    phase_name=phase.name,
                    phase_update_index=remaining,
                    learning_rate=self._learning_rate(
                        phase,
                        phase_update_index=remaining,
                    ),
                    rollout_steps=self._rollout_steps(
                        phase,
                        phase_update_index=remaining,
                    ),
                )
            remaining -= phase.updates
        raise ValueError("update_index is outside the finite schedule")

    @staticmethod
    def _phase_progress(phase: _Phase, phase_update_index: int) -> float:
        if phase.updates is None:
            return 0.0
        if phase.updates == 1:
            return 1.0
        return phase_update_index / (phase.updates - 1)

    @classmethod
    def _learning_rate(cls, phase: _Phase, *, phase_update_index: int) -> float:
        spec = phase.learning_rate
        kind = spec["kind"]
        if kind == "constant":
            return float(spec["value"])
        progress = cls._phase_progress(phase, phase_update_index)
        start = float(spec["start"])
        end = float(spec["end"])
        if kind == "linear":
            return start + (end - start) * progress
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    @staticmethod
    def _rollout_steps(phase: _Phase, *, phase_update_index: int) -> int:
        spec = phase.rollout
        if spec["kind"] == "constant":
            return int(spec["value"])
        return min(
            int(spec["maximum"]),
            int(spec["start"]) + phase_update_index // int(spec["every_updates"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.canonical))
