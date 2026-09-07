"""Minimal device, communication, synchronization, and RNG training adapter.

Precision and FlagGems lifecycle remain owned by :mod:`graphcast_compat.backend`.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch

from . import backend


SUPPORTED_ACCELERATORS = ("cuda", "musa", "npu")


def resolve_accelerator_type(requested: str | None = None) -> str:
    raw = (requested or os.environ.get("GRAPHCAST_ACCELERATOR_TYPE") or "").strip().lower()
    if raw:
        if raw not in SUPPORTED_ACCELERATORS:
            raise ValueError(
                "GRAPHCAST_ACCELERATOR_TYPE must be cuda, musa, or npu, "
                f"got {raw!r}"
            )
        if raw == "musa":
            backend._import_torch_musa()
        elif raw == "npu":
            backend._import_torch_npu()
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if backend._musa_is_available():
        return "musa"
    if backend._npu_is_available():
        return "npu"
    raise RuntimeError("no supported CUDA, MUSA, or NPU accelerator is available")


def accelerator_module(accelerator_type: str | None = None) -> Any:
    resolved = resolve_accelerator_type(accelerator_type)
    if resolved == "cuda":
        return torch.cuda
    if resolved == "musa":
        backend._import_torch_musa()
        return torch.musa
    backend._import_torch_npu()
    return torch.npu


def accelerator_is_available(accelerator_type: str | None = None) -> bool:
    return bool(accelerator_module(accelerator_type).is_available())


def accelerator_device_count(accelerator_type: str | None = None) -> int:
    return int(accelerator_module(accelerator_type).device_count())


def accelerator_device(index: int, accelerator_type: str | None = None) -> torch.device:
    return torch.device(resolve_accelerator_type(accelerator_type), int(index))


def set_accelerator_device(index: int, accelerator_type: str | None = None) -> None:
    accelerator_module(accelerator_type).set_device(int(index))


def default_process_group_backend(accelerator_type: str | None = None) -> str:
    resolved = resolve_accelerator_type(accelerator_type)
    return {"cuda": "nccl", "musa": "mccl", "npu": "hccl"}[resolved]


def synchronize_devices(
    device_indices: Sequence[int],
    accelerator_type: str | None = None,
) -> None:
    module = accelerator_module(accelerator_type)
    for index in device_indices:
        module.synchronize(int(index))


def get_rng_state(index: int, accelerator_type: str | None = None) -> torch.Tensor:
    return accelerator_module(accelerator_type).get_rng_state(int(index)).cpu()


def set_rng_state(
    state: torch.Tensor,
    index: int,
    accelerator_type: str | None = None,
) -> None:
    accelerator_module(accelerator_type).set_rng_state(state.cpu(), device=int(index))
