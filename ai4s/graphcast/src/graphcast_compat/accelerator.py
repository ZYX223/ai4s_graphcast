"""Small accelerator boundary shared by official GraphCast training.

The model and training recipe are intentionally accelerator agnostic.  This
module contains the small set of runtime calls whose spelling differs between
CUDA-compatible devices, torch_musa (MThreads), and torch_npu (Ascend).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch


SUPPORTED_ACCELERATORS = ("cuda", "musa", "npu")


def _import_torch_musa() -> None:
    try:
        import torch_musa  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "MUSA was selected, but torch_musa is not importable"
        ) from error


def _import_torch_npu() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "NPU was selected, but torch_npu is not importable"
        ) from error


def resolve_accelerator_type(requested: str | None = None) -> str:
    """Resolve an accelerator without importing vendor modules unnecessarily."""

    raw = (
        requested
        or os.environ.get("GRAPHCAST_ACCELERATOR_TYPE")
        or ""
    ).strip().lower()
    if raw:
        if raw not in SUPPORTED_ACCELERATORS:
            raise ValueError(
                "GRAPHCAST_ACCELERATOR_TYPE must be cuda, musa, or npu, "
                f"got {raw!r}"
            )
        if raw == "musa":
            _import_torch_musa()
        elif raw == "npu":
            _import_torch_npu()
        return raw
    if torch.cuda.is_available():
        return "cuda"
    try:
        _import_torch_musa()
    except RuntimeError:
        pass
    else:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return "musa"
    try:
        _import_torch_npu()
    except RuntimeError:
        pass
    else:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    raise RuntimeError("no supported CUDA, MUSA, or NPU accelerator is available")


def accelerator_module(accelerator_type: str | None = None) -> Any:
    resolved = resolve_accelerator_type(accelerator_type)
    if resolved == "cuda":
        return torch.cuda
    if resolved == "musa":
        _import_torch_musa()
        return torch.musa
    _import_torch_npu()
    return torch.npu


def accelerator_is_available(accelerator_type: str | None = None) -> bool:
    module = accelerator_module(accelerator_type)
    return bool(module.is_available())


def accelerator_device_count(accelerator_type: str | None = None) -> int:
    return int(accelerator_module(accelerator_type).device_count())


def accelerator_device(
    index: int,
    accelerator_type: str | None = None,
) -> torch.device:
    return torch.device(resolve_accelerator_type(accelerator_type), int(index))


def set_accelerator_device(
    index: int,
    accelerator_type: str | None = None,
) -> None:
    accelerator_module(accelerator_type).set_device(int(index))


def default_process_group_backend(
    accelerator_type: str | None = None,
) -> str:
    resolved = resolve_accelerator_type(accelerator_type)
    return {"cuda": "nccl", "musa": "mccl", "npu": "hccl"}[resolved]


def is_accelerator_collective_backend(backend: str | None) -> bool:
    return str(backend).lower() in {"nccl", "mccl", "hccl"}


def synchronize_devices(
    device_indices: Sequence[int],
    accelerator_type: str | None = None,
) -> None:
    module = accelerator_module(accelerator_type)
    for index in device_indices:
        module.synchronize(int(index))


def reset_peak_memory(
    device_indices: Sequence[int],
    accelerator_type: str | None = None,
) -> None:
    resolved = resolve_accelerator_type(accelerator_type)
    module = accelerator_module(resolved)
    for index in device_indices:
        if resolved in {"musa", "npu"}:
            # torch_musa releases have differed on whether a device argument
            # is accepted.  The explicit context plus no-argument form works
            # on the project-pinned 2.5 runtime and targets the intended card.
            with module.device(int(index)):
                module.reset_peak_memory_stats()
        else:
            module.reset_peak_memory_stats(int(index))


def memory_by_device(
    device_indices: Sequence[int],
    accelerator_type: str | None = None,
) -> dict[str, dict[str, float]]:
    module = accelerator_module(accelerator_type)
    return {
        str(index): {
            "allocated_gib": (
                float(module.max_memory_allocated(int(index))) / 1024**3
            ),
            "reserved_gib": (
                float(module.max_memory_reserved(int(index))) / 1024**3
            ),
        }
        for index in device_indices
    }


def accelerator_environment(
    *,
    participating_device_count: int,
    accelerator_type: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_accelerator_type(accelerator_type)
    module = accelerator_module(resolved)
    devices = []
    for index in range(participating_device_count):
        capability = None
        if hasattr(module, "get_device_capability"):
            try:
                capability = list(module.get_device_capability(index))
            except (AttributeError, RuntimeError, TypeError):
                capability = None
        devices.append(
            {
                "index": index,
                "name": module.get_device_name(index),
                "capability": capability,
            }
        )
    runtime_version = (
        getattr(torch.version, "cuda", None)
        if resolved == "cuda"
        else (
            getattr(torch.version, "musa", None)
            if resolved == "musa"
            else getattr(torch.version, "npu", None)
        )
    )
    return {
        "accelerator_type": resolved,
        "accelerator_runtime": runtime_version,
        "device_count": accelerator_device_count(resolved),
        "devices": devices,
    }


def configure_float32_math(
    tensor_core_enabled: bool,
    accelerator_type: str | None = None,
) -> str:
    """Delegate the resolved accelerator math policy to the target backend."""

    from .backend import configure_float32_math as configure

    return configure(
        tensor_core_enabled,
        resolve_accelerator_type(accelerator_type),
    )


def get_rng_state(
    index: int,
    accelerator_type: str | None = None,
) -> torch.Tensor:
    return accelerator_module(accelerator_type).get_rng_state(int(index)).cpu()


def set_rng_state(
    state: torch.Tensor,
    index: int,
    accelerator_type: str | None = None,
) -> None:
    accelerator_module(accelerator_type).set_rng_state(
        state.cpu(),
        device=int(index),
    )
