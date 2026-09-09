"""Consumed helpers from pinned b7667b4 parallel runner; no training loop."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import torch
from graphcast_compat.accelerator import (
    memory_by_device as accelerator_memory_by_device,
    reset_peak_memory as reset_accelerator_peak_memory,
    synchronize_devices,
    resolve_accelerator_type,
)
from graphcast_compat.backend import configure_float32_math as configure_backend_float32_math
from graphcast_compat.hotspot_profiler import HotspotProfilerConfig
from graphcast_compat.official_training_data import OfficialTrainingDataset

PER_VARIABLE_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_components_sha256(
    values: dict[str, torch.Tensor],
) -> dict[str, str]:
    result = {}
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def differing_component_names(
    rows: list[dict[str, str]],
) -> list[str]:
    names = sorted(set().union(*(row.keys() for row in rows)))
    return [
        name
        for name in names
        if len({row.get(name) for row in rows}) != 1
    ]


def configure_float32_math(tensor_core_enabled: bool) -> str:
    return configure_backend_float32_math(tensor_core_enabled, resolve_accelerator_type())


def synchronize(device_indices: tuple[int, ...]) -> None:
    synchronize_devices(device_indices)


def reset_peak_memory(device_indices: tuple[int, ...]) -> None:
    reset_accelerator_peak_memory(device_indices)


def memory_by_device(device_indices: tuple[int, ...]) -> dict[str, Any]:
    return accelerator_memory_by_device(device_indices)


def profile_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    rank: int,
) -> HotspotProfilerConfig:
    npu_profiler_level = os.environ.get(
        "GRAPHCAST_NPU_PROFILER_LEVEL", "Level0"
    ).strip()
    npu_op_attr_value = os.environ.get(
        "GRAPHCAST_NPU_PROFILER_OP_ATTR", "0"
    ).strip().lower()
    if npu_op_attr_value not in {"0", "1", "false", "true"}:
        raise ValueError(
            "GRAPHCAST_NPU_PROFILER_OP_ATTR must be 0/1/false/true"
        )
    return HotspotProfilerConfig(
        enabled=args.profile_hotspots,
        output_dir=run_dir / f"rank_{rank}/hotspot_profile",
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        repeat=args.profile_repeat,
        top_k=args.profile_top_k,
        # The official 0.25-degree path already stores real-shape validation
        # separately. Retaining every tensor shape/allocation in the profiler
        # adds substantial host/device pressure without improving busy time.
        record_shapes=False,
        profile_memory=False,
        with_stack=args.profile_with_stack,
        with_flops=True,
        staging_dir=(
            Path(args.profile_staging_dir)
            if args.profile_staging_dir
            else None
        ),
        split_active_traces=True,
        npu_profiler_level=npu_profiler_level,
        npu_op_attr=npu_op_attr_value in {"1", "true"},
    )


def validate_formal_dataset(
    dataset: OfficialTrainingDataset,
    *,
    formal: bool,
) -> None:
    if not formal:
        return
    if len(dataset.latitudes) != 721 or len(dataset.longitudes) != 1440:
        raise ValueError(
            "formal dataset must be the full 721 by 1440 0.25-degree grid"
        )
    if dataset.rollout_steps != 12:
        raise ValueError("formal dataset must expose 12 targets")


def validate_metric(metric: dict[str, Any]) -> None:
    for name in (
        "loss",
        "step_seconds",
        "training_flops",
    ):
        value = float(metric[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"MFU self-check failed for {name}")
    for name in ("mfu_sys", "hfu_sys", "mfu_e2e"):
        if metric.get(name) is None:
            continue
        value = float(metric[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"MFU self-check failed for {name}")
        if value > 1.0 + 1.0e-6:
            raise ValueError(f"MFU self-check failed: {name} exceeds one")
    if metric.get("mfu_busy") is not None:
        value = float(metric["mfu_busy"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("MFU self-check failed for mfu_busy")
        if value > 1.0 + 1.0e-6:
            raise ValueError("MFU self-check failed: mfu_busy exceeds one")
