"""Runtime accounting for Linear FLOPs replayed by activation checkpointing."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import threading
from typing import Any, Callable, Iterator

import torch
from torch.utils.checkpoint import checkpoint


_ACTIVE_COUNTER: "RecomputeFlopCounter | None" = None
_ACTIVE_COUNTER_LOCK = threading.RLock()
_RECOMPUTE_DEPTH: ContextVar[int] = ContextVar(
    "graphcast_activation_recompute_depth",
    default=0,
)


class RecomputeFlopCounter:
    """Count modeled Linear FLOPs that are actually replayed during backward."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.recompute_flops = 0
            self.recompute_linear_calls = 0

    @contextmanager
    def track(self) -> Iterator["RecomputeFlopCounter"]:
        global _ACTIVE_COUNTER
        with _ACTIVE_COUNTER_LOCK:
            active = _ACTIVE_COUNTER
            if active is not None and active is not self:
                raise RuntimeError(
                    "a different recompute FLOP counter is already active"
                )
            _ACTIVE_COUNTER = self
        try:
            yield self
        finally:
            with _ACTIVE_COUNTER_LOCK:
                _ACTIVE_COUNTER = active

    def add_linear(self, inputs: torch.Tensor, weight: torch.Tensor) -> None:
        if inputs.ndim == 0 or weight.ndim != 2:
            raise ValueError("Linear recompute accounting requires tensor inputs and 2D weights")
        in_features = int(weight.shape[1])
        if int(inputs.shape[-1]) != in_features:
            raise ValueError("Linear recompute input width does not match its weight")
        rows = int(inputs.numel()) // in_features
        out_features = int(weight.shape[0])
        with self._lock:
            self.recompute_flops += 2 * rows * in_features * out_features
            self.recompute_linear_calls += 1


@contextmanager
def _recompute_context() -> Iterator[None]:
    token = _RECOMPUTE_DEPTH.set(_RECOMPUTE_DEPTH.get() + 1)
    try:
        yield
    finally:
        _RECOMPUTE_DEPTH.reset(token)


def _checkpoint_contexts():
    return nullcontext(), _recompute_context()


def tracked_checkpoint(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run non-reentrant checkpointing with an exact recomputation marker."""

    if kwargs.pop("use_reentrant", False):
        raise ValueError("recompute FLOP tracking requires use_reentrant=False")
    if "context_fn" in kwargs:
        raise ValueError("tracked_checkpoint owns context_fn")
    return checkpoint(
        function,
        *args,
        use_reentrant=False,
        context_fn=_checkpoint_contexts,
        **kwargs,
    )


def tracked_reentrant_checkpoint(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a reentrant checkpoint while counting its backward replay.

    The pinned torch_npu runtime cannot safely nest multiple connected
    non-reentrant rollout frames around the model's existing non-reentrant
    stage/chunk checkpoints.  A reentrant outer rollout frame avoids that
    saved-tensor-hook conflict.  Its first invocation runs under ``no_grad``;
    only the grad-enabled backward replay enters the FLOP counter context.
    """

    if "use_reentrant" in kwargs:
        raise ValueError("tracked_reentrant_checkpoint owns use_reentrant")
    if "context_fn" in kwargs:
        raise ValueError(
            "reentrant checkpointing does not accept context_fn"
        )

    def counted_function(*values: Any) -> Any:
        if not torch.is_grad_enabled():
            return function(*values)
        with _recompute_context():
            return function(*values)

    return checkpoint(
        counted_function,
        *args,
        use_reentrant=True,
        **kwargs,
    )


def record_linear_recompute(
    inputs: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Record one Linear only while a tracked checkpoint is being replayed."""

    if _RECOMPUTE_DEPTH.get() <= 0:
        return
    with _ACTIVE_COUNTER_LOCK:
        counter = _ACTIVE_COUNTER
    if counter is not None:
        counter.add_linear(inputs, weight)
