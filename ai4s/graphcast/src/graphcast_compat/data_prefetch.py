"""Bounded CPU preparation for the next official GraphCast update.

The worker in this module owns at most one update request.  It performs only
host-side sampling, xarray extraction, validation, normalization and tensor
packing.  Accelerator transfers stay on the training thread, so vendor
runtimes never receive device calls from the background worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
import threading
import time
from types import TracebackType
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    TYPE_CHECKING,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from graphcast_compat.official_training_data import TrainingSample


RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
KeyT = TypeVar("KeyT", bound=Hashable)


_STOP = object()
_NO_PENDING = object()


@dataclass(frozen=True)
class _WorkerOutcome(Generic[KeyT, ResultT]):
    key: KeyT
    value: ResultT | None = None
    error: BaseException | None = None
    traceback: TracebackType | None = None


class DepthOneCpuPrefetcher(Generic[KeyT, RequestT, ResultT]):
    """Run one host-side producer with exactly one outstanding request.

    ``submit`` fails closed if a request has not yet been consumed.  ``take``
    checks the exact request key before returning and re-raises worker errors
    on the training thread.  ``close`` drains a pending bounded task and joins
    the worker; the daemon flag is an additional process-exit safeguard for a
    third-party C extension that cannot be interrupted from Python.
    """

    def __init__(
        self,
        worker: Callable[[RequestT], ResultT],
        *,
        thread_name: str = "graphcast-data-prefetch",
    ) -> None:
        self._worker = worker
        self._requests: Queue[object] = Queue(maxsize=1)
        self._results: Queue[_WorkerOutcome[KeyT, ResultT]] = Queue(maxsize=1)
        self._pending_key: object = _NO_PENDING
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    @property
    def has_pending(self) -> bool:
        return self._pending_key is not _NO_PENDING

    @property
    def worker_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, key: KeyT, request: RequestT) -> None:
        if self._closed:
            raise RuntimeError("data prefetcher is closed")
        if self.has_pending:
            raise RuntimeError(
                "depth-one data prefetcher already has an outstanding request"
            )
        self._pending_key = key
        self._requests.put((key, request))

    def take(self, expected_key: KeyT) -> ResultT:
        if not self.has_pending:
            raise RuntimeError("data prefetcher has no outstanding request")
        if self._pending_key != expected_key:
            raise RuntimeError(
                "prefetched update key differs from the training cursor: "
                f"expected={expected_key!r} pending={self._pending_key!r}"
            )
        outcome = self._results.get()
        self._pending_key = _NO_PENDING
        if outcome.key != expected_key:
            raise RuntimeError(
                "data prefetch worker returned an unexpected update key: "
                f"expected={expected_key!r} actual={outcome.key!r}"
            )
        if outcome.error is not None:
            raise outcome.error.with_traceback(outcome.traceback)
        return cast(ResultT, outcome.value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.has_pending:
            # A running Python thread cannot be interrupted safely.  The
            # producer is deliberately bounded to one finite data request, so
            # drain it before sending the stop marker and joining.
            self._results.get()
            self._pending_key = _NO_PENDING
        self._requests.put(_STOP)
        self._thread.join()
        if self._thread.is_alive():
            raise RuntimeError("data prefetch worker did not terminate")

    def _run(self) -> None:
        while True:
            queued = self._requests.get()
            if queued is _STOP:
                return
            key, request = cast(tuple[KeyT, RequestT], queued)
            try:
                value = self._worker(request)
            except BaseException as error:
                outcome = _WorkerOutcome[KeyT, ResultT](
                    key=key,
                    error=error,
                    traceback=error.__traceback__,
                )
            else:
                outcome = _WorkerOutcome[KeyT, ResultT](
                    key=key,
                    value=value,
                )
            self._results.put(outcome)

    def __enter__(self) -> "DepthOneCpuPrefetcher[KeyT, RequestT, ResultT]":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class OfficialMicrobatchDataRequest:
    index: int
    window_indices: tuple[int, ...]
    sample_weights: tuple[float, ...]


@dataclass(frozen=True)
class OfficialUpdateDataKey:
    """Full identity checked when a prepared update reaches the consumer."""

    global_step: int
    epoch: int
    batch_index: int
    rollout_steps: int
    rank: int
    replicas: int
    allow_replacement: bool
    microbatches: tuple[OfficialMicrobatchDataRequest, ...]


@dataclass(frozen=True)
class OfficialUpdateDataRequest:
    """Exact data contract for one optimizer update."""

    key: OfficialUpdateDataKey

    @property
    def rollout_steps(self) -> int:
        return self.key.rollout_steps

    @property
    def rank(self) -> int:
        return self.key.rank

    @property
    def replicas(self) -> int:
        return self.key.replicas

    @property
    def allow_replacement(self) -> bool:
        return self.key.allow_replacement

    @property
    def microbatches(self) -> tuple[OfficialMicrobatchDataRequest, ...]:
        return self.key.microbatches


@dataclass(frozen=True)
class PreparedSampleMetadata:
    rollout_steps: int
    valid_sample_count: int
    global_valid_sample_count: int
    global_window_indices: tuple[int, ...]
    global_sample_weights: tuple[float, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PreparedOfficialMicrobatch:
    index: int
    batch: Any
    sample_metadata: PreparedSampleMetadata
    sample_seconds: float
    batch_prepare_seconds: float


@dataclass(frozen=True)
class PreparedOfficialUpdate:
    key: OfficialUpdateDataKey
    rollout_steps: int
    microbatches: tuple[PreparedOfficialMicrobatch, ...]
    worker_seconds: float


class OfficialUpdateCpuProducer:
    """Prepare official update data without issuing accelerator operations."""

    def __init__(
        self,
        *,
        dataset: Any,
        config: Any,
        stats: Any,
        per_variable_weights: Any,
        prepare_batch: Callable[..., Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self._config = config
        self._stats = stats
        self._per_variable_weights = per_variable_weights
        self._prepare_batch = prepare_batch

    def __call__(
        self,
        request: OfficialUpdateDataRequest,
    ) -> PreparedOfficialUpdate:
        worker_started = time.perf_counter()
        prepare_batch = self._prepare_batch
        if prepare_batch is None:
            from graphcast_compat.autoregressive_training import (
                prepare_autoregressive_batch,
            )

            prepare_batch = prepare_autoregressive_batch
        prepared = []
        for microbatch in request.microbatches:
            sample_started = time.perf_counter()
            sample = self._dataset.sample_indices(
                window_indices=microbatch.window_indices,
                sample_weights=microbatch.sample_weights,
                rank=request.rank,
                replicas=request.replicas,
                target_rollout_steps=request.rollout_steps,
                allow_replacement=request.allow_replacement,
            )
            sample_seconds = time.perf_counter() - sample_started
            if sample.rollout_steps != request.rollout_steps:
                raise RuntimeError(
                    "materialized target horizon disagrees with prefetched "
                    "training request"
                )
            prepare_started = time.perf_counter()
            batch = prepare_batch(
                sample.inputs,
                sample.targets,
                sample.forcings,
                config=self._config,
                stats=self._stats,
                # Device runtime calls are intentionally forbidden here.
                input_device="cpu",
                storage_device="cpu",
                per_variable_weights=self._per_variable_weights,
                sample_weights=sample.sample_weights,
            )
            prepare_seconds = time.perf_counter() - prepare_started
            provenance = training_sample_provenance(
                sample,
                rollout_steps=request.rollout_steps,
            )
            sample_metadata = PreparedSampleMetadata(
                rollout_steps=sample.rollout_steps,
                valid_sample_count=sample.valid_sample_count,
                global_valid_sample_count=(
                    sample.global_valid_sample_count
                ),
                global_window_indices=tuple(sample.global_window_indices),
                global_sample_weights=tuple(sample.global_sample_weights),
                provenance=provenance,
            )
            prepared.append(
                PreparedOfficialMicrobatch(
                    index=microbatch.index,
                    batch=batch,
                    sample_metadata=sample_metadata,
                    sample_seconds=sample_seconds,
                    batch_prepare_seconds=prepare_seconds,
                )
            )
            del sample
        return PreparedOfficialUpdate(
            key=request.key,
            rollout_steps=request.rollout_steps,
            microbatches=tuple(prepared),
            worker_seconds=time.perf_counter() - worker_started,
        )


def training_sample_provenance(
    sample: "TrainingSample",
    *,
    rollout_steps: int | None = None,
) -> dict[str, Any]:
    """Materialize lightweight provenance before releasing xarray samples."""

    active_rollout_steps = (
        sample.rollout_steps if rollout_steps is None else rollout_steps
    )
    if not 1 <= active_rollout_steps <= sample.rollout_steps:
        raise ValueError("provenance rollout exceeds the materialized sample")
    return {
        "rank": sample.rank,
        "local_window_indices": list(sample.window_indices),
        "global_window_indices": list(sample.global_window_indices),
        "sample_weights": list(sample.sample_weights),
        "global_sample_weights": list(sample.global_sample_weights),
        "valid_sample_count": sample.valid_sample_count,
        "padded_sample_count": sample.padded_sample_count,
        "global_valid_sample_count": sample.global_valid_sample_count,
        "global_padded_sample_count": sample.global_padded_sample_count,
        "reference_datetimes": list(sample.reference_datetimes),
        "target_datetimes": [
            values[active_rollout_steps - 1]
            for values in sample.target_datetimes_by_sample
        ],
        "target_datetimes_by_sample": [
            list(values[:active_rollout_steps])
            for values in sample.target_datetimes_by_sample
        ],
        "dataset_kind": sample.dataset_kind,
        "training_stage": sample.training_stage,
        "rollout_steps": active_rollout_steps,
        "dataset_rollout_steps": sample.dataset_rollout_steps,
    }
