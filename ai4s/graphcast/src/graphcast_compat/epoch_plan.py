"""Deterministic, auditable GraphCast epoch and DP batch planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable


SCHEMA_VERSION = 1


def _integer(value: int, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _shuffle_key(*, seed: int, epoch: int, window_index: int) -> bytes:
    value = f"graphcast-epoch-plan-v1:{seed}:{epoch}:{window_index}"
    return hashlib.sha256(value.encode("ascii")).digest()


@dataclass(frozen=True)
class EpochBatch:
    """One fixed-shape global optimizer batch."""

    epoch: int
    batch_index: int
    window_indices: tuple[int, ...]
    sample_weights: tuple[float, ...]

    @property
    def valid_count(self) -> int:
        return sum(weight > 0.0 for weight in self.sample_weights)

    @property
    def padded_count(self) -> int:
        return len(self.sample_weights) - self.valid_count


@dataclass(frozen=True)
class RankBatch:
    """One DP rank's slice of a global epoch batch."""

    epoch: int
    batch_index: int
    rank: int
    replicas: int
    window_indices: tuple[int, ...]
    sample_weights: tuple[float, ...]
    global_window_indices: tuple[int, ...]
    global_sample_weights: tuple[float, ...]

    @property
    def local_valid_count(self) -> int:
        return sum(weight > 0.0 for weight in self.sample_weights)

    @property
    def local_padded_count(self) -> int:
        return len(self.sample_weights) - self.local_valid_count

    @property
    def global_valid_count(self) -> int:
        return sum(weight > 0.0 for weight in self.global_sample_weights)

    @property
    def global_padded_count(self) -> int:
        return len(self.global_sample_weights) - self.global_valid_count


@dataclass(frozen=True)
class EpochPlan:
    """An immutable once-only ordering plus fixed-shape padded batches."""

    epoch: int
    seed: int
    global_batch_size: int
    shuffle: bool
    ordered_window_indices: tuple[int, ...]
    batches: tuple[EpochBatch, ...]
    digest: str

    @classmethod
    def create(
        cls,
        window_indices: Iterable[int],
        *,
        epoch: int,
        seed: int,
        global_batch_size: int,
        shuffle: bool = False,
    ) -> "EpochPlan":
        epoch = _integer(epoch, name="epoch")
        seed = _integer(seed, name="seed")
        global_batch_size = _integer(
            global_batch_size,
            name="global_batch_size",
            minimum=1,
        )
        values = tuple(window_indices)
        if not values:
            raise ValueError("window_indices must be non-empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            raise ValueError(
                "window_indices must contain non-negative integers"
            )
        if len(set(values)) != len(values):
            raise ValueError("window_indices must be unique")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")

        ordered = (
            tuple(
                sorted(
                    values,
                    key=lambda value: _shuffle_key(
                        seed=seed,
                        epoch=epoch,
                        window_index=value,
                    ),
                )
            )
            if shuffle
            else values
        )
        optimizer_steps = math.ceil(len(ordered) / global_batch_size)
        padded_count = optimizer_steps * global_batch_size - len(ordered)
        padded = ordered + (ordered[-1],) * padded_count
        weights = (1.0,) * len(ordered) + (0.0,) * padded_count
        batches = tuple(
            EpochBatch(
                epoch=epoch,
                batch_index=batch_index,
                window_indices=tuple(
                    padded[
                        batch_index
                        * global_batch_size : (batch_index + 1)
                        * global_batch_size
                    ]
                ),
                sample_weights=tuple(
                    weights[
                        batch_index
                        * global_batch_size : (batch_index + 1)
                        * global_batch_size
                    ]
                ),
            )
            for batch_index in range(optimizer_steps)
        )
        canonical = {
            "schema_version": SCHEMA_VERSION,
            "epoch": epoch,
            "seed": seed,
            "global_batch_size": global_batch_size,
            "shuffle": shuffle,
            "ordered_window_indices": list(ordered),
            "batches": [
                {
                    "window_indices": list(batch.window_indices),
                    "sample_weights": list(batch.sample_weights),
                }
                for batch in batches
            ],
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            epoch=epoch,
            seed=seed,
            global_batch_size=global_batch_size,
            shuffle=shuffle,
            ordered_window_indices=ordered,
            batches=batches,
            digest=digest,
        )

    @property
    def valid_window_count(self) -> int:
        return len(self.ordered_window_indices)

    @property
    def optimizer_steps(self) -> int:
        return len(self.batches)

    @property
    def padded_sample_count(self) -> int:
        return sum(batch.padded_count for batch in self.batches)

    @property
    def duplicate_valid_count(self) -> int:
        valid = [
            window_index
            for batch in self.batches
            for window_index, weight in zip(
                batch.window_indices,
                batch.sample_weights,
                strict=True,
            )
            if weight > 0.0
        ]
        return len(valid) - len(set(valid))

    @property
    def coverage(self) -> float:
        valid = {
            window_index
            for batch in self.batches
            for window_index, weight in zip(
                batch.window_indices,
                batch.sample_weights,
                strict=True,
            )
            if weight > 0.0
        }
        return len(valid) / self.valid_window_count

    def validate_cursor(self, next_batch_index: int) -> None:
        next_batch_index = _integer(
            next_batch_index,
            name="next_batch_index",
        )
        if next_batch_index > self.optimizer_steps:
            raise ValueError("next_batch_index is outside the epoch plan")

    def rank_batch(
        self,
        batch_index: int,
        *,
        rank: int,
        replicas: int,
    ) -> RankBatch:
        batch_index = _integer(batch_index, name="batch_index")
        replicas = _integer(replicas, name="replicas", minimum=1)
        rank = _integer(rank, name="rank")
        if rank >= replicas:
            raise ValueError("rank must be in [0, replicas)")
        if self.global_batch_size % replicas:
            raise ValueError(
                "global batch size must be divisible by replicas"
            )
        if batch_index >= self.optimizer_steps:
            raise ValueError("batch_index is outside the epoch plan")
        batch = self.batches[batch_index]
        local_batch_size = self.global_batch_size // replicas
        start = rank * local_batch_size
        stop = start + local_batch_size
        return RankBatch(
            epoch=self.epoch,
            batch_index=batch_index,
            rank=rank,
            replicas=replicas,
            window_indices=batch.window_indices[start:stop],
            sample_weights=batch.sample_weights[start:stop],
            global_window_indices=batch.window_indices,
            global_sample_weights=batch.sample_weights,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "epoch": self.epoch,
            "seed": self.seed,
            "global_batch_size": self.global_batch_size,
            "shuffle": self.shuffle,
            "optimizer_steps": self.optimizer_steps,
            "valid_window_count": self.valid_window_count,
            "padded_sample_count": self.padded_sample_count,
            "duplicate_valid_count": self.duplicate_valid_count,
            "coverage": self.coverage,
            "digest": self.digest,
        }
