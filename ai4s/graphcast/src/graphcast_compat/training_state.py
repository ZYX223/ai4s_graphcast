"""Exact-resume state and atomic checkpoint I/O for GraphCast training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 3


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class TrainingCursor:
    """The next optimizer batch to execute after a safe checkpoint."""

    epoch: int
    next_batch_index: int
    global_optimizer_step: int
    epoch_plan_digest: str

    def __post_init__(self) -> None:
        _non_negative_integer(self.epoch, name="epoch")
        _non_negative_integer(
            self.next_batch_index,
            name="next_batch_index",
        )
        _non_negative_integer(
            self.global_optimizer_step,
            name="global_optimizer_step",
        )
        if (
            not isinstance(self.epoch_plan_digest, str)
            or len(self.epoch_plan_digest) != 64
        ):
            raise ValueError("epoch_plan_digest must be a SHA256 string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingCursor":
        return cls(
            epoch=int(value["epoch"]),
            next_batch_index=int(value["next_batch_index"]),
            global_optimizer_step=int(value["global_optimizer_step"]),
            epoch_plan_digest=str(value["epoch_plan_digest"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    sha256: str
    sidecar: Path


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_object_sha256(value: Any) -> str:
    """Deterministically hash nested optimizer/model state objects."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor:")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray:")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(
                item,
                key=lambda candidate: (
                    type(candidate).__name__,
                    repr(candidate),
                ),
            ):
                update(key)
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (tuple, list)):
            digest.update(
                b"tuple[" if isinstance(item, tuple) else b"list["
            )
            for child in item:
                update(child)
            digest.update(b"]")
            return
        if isinstance(item, np.generic):
            update(item.item())
            return
        if item is None:
            digest.update(b"none")
            return
        if isinstance(item, bool):
            digest.update(b"bool:1" if item else b"bool:0")
            return
        if isinstance(item, int):
            digest.update(f"int:{item}".encode("ascii"))
            return
        if isinstance(item, float):
            digest.update(f"float:{item.hex()}".encode("ascii"))
            return
        if isinstance(item, str):
            digest.update(b"str:")
            digest.update(item.encode("utf-8"))
            return
        if isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
            return
        raise TypeError(f"unsupported checkpoint state type: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def capture_rng_state(
    *,
    cuda_device_indices: Sequence[int],
    accelerator_type: str | None = None,
) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU and selected accelerator RNG state.

    ``cuda_device_indices`` remains the public argument and checkpoint key so
    existing H100 checkpoints stay loadable.  ``accelerator_type`` records
    whether those logical indices belong to CUDA or MUSA.
    """

    indices = tuple(int(index) for index in cuda_device_indices)
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("cuda_device_indices must be unique and non-negative")
    resolved_accelerator = accelerator_type or "cuda"
    if indices:
        from graphcast_compat.accelerator import (
            accelerator_is_available,
            get_rng_state,
            resolve_accelerator_type,
        )

        resolved_accelerator = resolve_accelerator_type(accelerator_type)
        if not accelerator_is_available(resolved_accelerator):
            raise RuntimeError(
                f"{resolved_accelerator.upper()} RNG state requested "
                "without an available accelerator"
            )
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "accelerator_type": resolved_accelerator,
        "cuda_device_indices": indices,
        "torch_cuda": {
            str(index): get_rng_state(index, resolved_accelerator)
            for index in indices
        } if indices else {},
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    cuda_device_indices: Sequence[int],
    accelerator_type: str | None = None,
) -> None:
    """Restore a rank-local RNG snapshot on the same physical topology."""

    indices = tuple(int(index) for index in cuda_device_indices)
    recorded = tuple(int(index) for index in state["cuda_device_indices"])
    if recorded != indices:
        raise ValueError(
            "checkpoint CUDA RNG device indices disagree with topology"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    recorded_accelerator = str(
        state.get("accelerator_type", "cuda")
    ).lower()
    requested_accelerator = (
        accelerator_type or recorded_accelerator
    ).lower()
    if recorded_accelerator != requested_accelerator:
        raise ValueError(
            "checkpoint accelerator type disagrees with runtime: "
            f"{recorded_accelerator} != {requested_accelerator}"
        )
    if indices:
        from graphcast_compat.accelerator import (
            accelerator_is_available,
            resolve_accelerator_type,
            set_rng_state,
        )

        requested_accelerator = resolve_accelerator_type(
            requested_accelerator
        )
        if not accelerator_is_available(requested_accelerator):
            raise RuntimeError(
                "cannot restore accelerator RNG state without "
                f"{requested_accelerator.upper()}"
            )
    for index in indices:
        key = str(index)
        if key not in cuda_states:
            raise ValueError(f"checkpoint is missing CUDA RNG state {key}")
        set_rng_state(
            cuda_states[key],
            index,
            requested_accelerator,
        )


def build_training_checkpoint(
    *,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    schedule_state: Mapping[str, Any],
    cursor: TrainingCursor,
    rng_states_by_rank: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not rng_states_by_rank:
        raise ValueError("rng_states_by_rank must be non-empty")
    if "resume_contract" not in metadata:
        raise ValueError("checkpoint metadata requires resume_contract")
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "schedule_state": dict(schedule_state),
        "cursor": cursor.to_dict(),
        "rng_states_by_rank": list(rng_states_by_rank),
        "metadata": dict(metadata),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_save_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    latest_path: str | Path | None = None,
) -> CheckpointRecord:
    """Atomically save a schema-v3 checkpoint and its actual SHA256."""

    resolved = Path(path).expanduser().resolve()
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version must be {SCHEMA_VERSION}"
        )
    cursor = TrainingCursor.from_dict(payload["cursor"])
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f"{resolved.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, resolved)
    _fsync_directory(resolved.parent)
    digest = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    _atomic_write_text(sidecar, f"{digest}  {resolved.name}\n")
    if latest_path is not None:
        latest = Path(latest_path).expanduser().resolve()
        _atomic_write_text(
            latest,
            json.dumps(
                {
                    "checkpoint": str(resolved),
                    "epoch": cursor.epoch,
                    "global_optimizer_step": (
                        cursor.global_optimizer_step
                    ),
                    "next_batch_index": cursor.next_batch_index,
                    "sha256": digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    return CheckpointRecord(
        path=resolved,
        sha256=digest,
        sidecar=sidecar,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"training checkpoint is missing: {resolved}")
    if verify_checksum:
        sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
        if not sidecar.is_file():
            raise ValueError("training checkpoint SHA256 sidecar is missing")
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if not fields or len(fields[0]) != 64:
            raise ValueError("training checkpoint SHA256 sidecar is invalid")
        if sha256_file(resolved) != fields[0]:
            raise ValueError("training checkpoint SHA256 verification failed")
    payload = torch.load(
        resolved,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or int(payload.get("schema_version", -1)) != SCHEMA_VERSION
    ):
        raise ValueError(
            f"training checkpoint must use schema version {SCHEMA_VERSION}"
        )
    for field in (
        "model_state",
        "optimizer_state",
        "schedule_state",
        "cursor",
        "rng_states_by_rank",
        "metadata",
    ):
        if field not in payload:
            raise ValueError(f"training checkpoint is missing {field}")
    TrainingCursor.from_dict(payload["cursor"])
    return payload


def validate_resume_contract(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("training checkpoint metadata is invalid")
    actual = metadata.get("resume_contract")
    if not isinstance(actual, Mapping):
        raise ValueError("training checkpoint resume_contract is invalid")
    for name, expected_value in expected.items():
        if actual.get(name) != expected_value:
            raise ValueError(
                f"resume contract mismatch for {name}: "
                f"checkpoint={actual.get(name)!r}, "
                f"expected={expected_value!r}"
            )
