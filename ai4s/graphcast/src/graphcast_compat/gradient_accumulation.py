"""Batch partitioning primitives for sample-weighted gradient accumulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicrobatchPart:
    """One contiguous portion of a global batch."""

    index: int
    batch_slice: slice
    sample_count: int
    sample_weight: float


@dataclass(frozen=True)
class MicrobatchPlan:
    """Immutable mapping from a global batch to sequential micro-batches."""

    global_batch_size: int
    micro_batch_size: int
    parts: tuple[MicrobatchPart, ...]

    @property
    def accumulation_steps(self) -> int:
        return len(self.parts)

    @classmethod
    def create(cls, *, global_batch_size: int, micro_batch_size: int) -> "MicrobatchPlan":
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")

        effective_micro_batch_size = min(global_batch_size, micro_batch_size)
        parts = []
        for index, start in enumerate(range(0, global_batch_size, effective_micro_batch_size)):
            stop = min(start + effective_micro_batch_size, global_batch_size)
            sample_count = stop - start
            parts.append(
                MicrobatchPart(
                    index=index,
                    batch_slice=slice(start, stop),
                    sample_count=sample_count,
                    sample_weight=sample_count / global_batch_size,
                )
            )
        return cls(
            global_batch_size=global_batch_size,
            micro_batch_size=effective_micro_batch_size,
            parts=tuple(parts),
        )
