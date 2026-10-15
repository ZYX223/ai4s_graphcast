"""Profile the proven post-LayerNorm residual candidate without per-call syncs."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .backend import synchronize

EventPairFactory = Callable[[torch.device], tuple[Any, Any] | None]
CurrentStreamFactory = Callable[[torch.device], Any | None]
SynchronizeDevice = Callable[[torch.device], None]


@dataclass
class _TimedPair:
    start: Any
    end: Any
    stream: Any | None
    completed: bool = False


@dataclass
class PostLayerNormResidualGroup:
    """One shape/layout bucket for candidate-region observations."""

    shape: tuple[int, ...]
    x_stride: tuple[int, ...]
    residual_stride: tuple[int, ...]
    normalized_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    dtype: str
    device: str
    eps: float
    count: int = 0
    timed_pairs: list[_TimedPair] = field(default_factory=list, repr=False)

    def to_metadata(self, device_time_total_ms: float) -> dict[str, Any]:
        return {
            "count": self.count,
            "device": self.device,
            "device_time_total_ms": device_time_total_ms,
            "dtype": self.dtype,
            "eps": self.eps,
            "normalized_shape": list(self.normalized_shape),
            "residual_stride": list(self.residual_stride),
            "shape": list(self.shape),
            "timed_count": sum(pair.completed for pair in self.timed_pairs),
            "weight_shape": list(self.weight_shape),
            "x_stride": list(self.x_stride),
        }


class _CandidateRegion:
    def __init__(self, pair: _TimedPair | None) -> None:
        self._pair = pair

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._pair is not None:
            self._pair.end.record(self._pair.stream)
            self._pair.completed = True
        return False


def _default_event_pair_factory(device: torch.device) -> tuple[Any, Any] | None:
    if device.type == "cuda":
        event_type = torch.cuda.Event
    elif device.type == "npu" and hasattr(torch, "npu"):
        event_type = torch.npu.Event
    elif device.type == "musa" and hasattr(torch, "musa"):
        event_type = torch.musa.Event
    else:
        return None
    return event_type(enable_timing=True), event_type(enable_timing=True)


def _default_current_stream_factory(device: torch.device) -> Any | None:
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    if device.type == "npu" and hasattr(torch, "npu"):
        return torch.npu.current_stream(device)
    if device.type == "musa" and hasattr(torch, "musa"):
        return torch.musa.current_stream(device)
    return None


class PostLayerNormResidualRecorder:
    """Aggregate candidate metadata and defer accelerator timing until ``stop``."""

    def __init__(
        self,
        output_path: str | Path | None,
        *,
        event_pair_factory: EventPairFactory | None = None,
        current_stream_factory: CurrentStreamFactory | None = None,
        synchronize_device: SynchronizeDevice | None = None,
    ) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self._event_pair_factory = event_pair_factory or _default_event_pair_factory
        self._current_stream_factory = current_stream_factory or _default_current_stream_factory
        self._synchronize_device = synchronize_device or synchronize
        self._groups: dict[tuple[Any, ...], PostLayerNormResidualGroup] = {}

    @property
    def enabled(self) -> bool:
        return self.output_path is not None

    @property
    def groups(self) -> tuple[PostLayerNormResidualGroup, ...]:
        return tuple(self._groups.values())

    @staticmethod
    def disabled_region() -> Any:
        return nullcontext()

    def _observe(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        normalized_shape: tuple[int, ...],
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> tuple[PostLayerNormResidualGroup | None, _TimedPair | None]:
        del bias
        if not self.enabled:
            return None, None
        if x.device != residual.device:
            raise ValueError("post-LayerNorm residual candidate tensors must use the same device")
        key = (
            tuple(x.shape),
            tuple(x.stride()),
            tuple(residual.stride()),
            tuple(normalized_shape),
            tuple(weight.shape) if weight is not None else (),
            str(x.dtype),
            str(x.device),
            float(eps),
        )
        group = self._groups.get(key)
        if group is None:
            group = PostLayerNormResidualGroup(
                shape=tuple(x.shape),
                x_stride=tuple(x.stride()),
                residual_stride=tuple(residual.stride()),
                normalized_shape=tuple(normalized_shape),
                weight_shape=tuple(weight.shape) if weight is not None else (),
                dtype=str(x.dtype),
                device=str(x.device),
                eps=float(eps),
            )
            self._groups[key] = group
        group.count += 1
        event_pair = self._event_pair_factory(x.device)
        if event_pair is None:
            return group, None
        pair = _TimedPair(*event_pair, self._current_stream_factory(x.device))
        group.timed_pairs.append(pair)
        return group, pair

    def observe(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        normalized_shape: tuple[int, ...],
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> None:
        """Record metadata only; use ``region`` to time an expression."""
        self._observe(x, residual, normalized_shape, weight, bias, eps)

    def region(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        normalized_shape: tuple[int, ...],
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> _CandidateRegion:
        """Return a context manager that encloses the candidate expression."""
        _, pair = self._observe(x, residual, normalized_shape, weight, bias, eps)
        if pair is not None:
            pair.start.record(pair.stream)
        return _CandidateRegion(pair)

    def stop(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Synchronize once per participating device, then resolve saved events."""
        if not self.enabled:
            return None
        timed_groups = [
            group for group in self.groups if any(pair.completed for pair in group.timed_pairs)
        ]
        for device in dict.fromkeys(torch.device(group.device) for group in timed_groups):
            self._synchronize_device(device)
        groups = []
        for group in self.groups:
            total_ms = sum(
                float(pair.start.elapsed_time(pair.end))
                for pair in group.timed_pairs
                if pair.completed
            )
            groups.append(group.to_metadata(total_ms))
        payload = {
            "schema_version": 1,
            "profiler": "post_layernorm_residual_candidate_region",
            "metadata": metadata or {},
            "groups": groups,
        }
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return payload
