#!/usr/bin/env python3
"""Run PyTorch-compatible GraphCast training loss and gradient collection.

This script is the PyTorch counterpart to ``run_google_jax_training_loss.py``.
It uses the Google-compatible PyTorch model in ``src/graphcast_compat`` and the
converted Google checkpoint. The loss is computed in the same normalized
residual space as Google ``InputsAndResiduals`` and uses the same latitude,
pressure-level, and surface-variable weighting as
``graphcast.losses.weighted_mse_per_level``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    dataset_to_stacked,
    inputs_to_grid_node_features,
    load_normalization_stats,
    predict_with_inputs_and_residuals,
)
from graphcast_compat.backend import (  # noqa: E402
    backend_report_lines,
    configure_flagos_environment,
    resolve_runtime_backend,
)
from graphcast_compat.data import select_dataset_variables  # noqa: E402
from graphcast_compat.normalization import normalize_dataset  # noqa: E402

DEFAULT_CASE_DIR = REPO_ROOT / "cases" / "2021_20210101_day_operational_25km"
DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "training" / "google_graphcast_25km.yaml")
DEFAULT_WEIGHTS = str(
    REPO_ROOT / "outputs" / "checkpoints" / "google_graphcast_operational_25km_compat_state.pt"
)
DEFAULT_INPUTS = str(DEFAULT_CASE_DIR / "inputs.nc")
DEFAULT_TARGETS = str(DEFAULT_CASE_DIR / "targets.nc")
DEFAULT_FORCINGS = str(DEFAULT_CASE_DIR / "forcings.nc")
DEFAULT_STATS_DIR = str(REPO_ROOT / "data" / "raw" / "google_graphcast_stats")
DEFAULT_RUN_DIR = str(
    REPO_ROOT / "outputs" / "training" / "runs" / "operational_25km_pytorch_compat_loss"
)
DEFAULT_FLAGGEMS_CONFIG = REPO_ROOT / "configs/flaggems/graphcast_training.yaml"

PER_VARIABLE_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}
PRESERVED_DIMS = ("batch", "lat", "lon")


def import_torch_npu() -> None:
    try:
        import torch_npu  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError("NPU requested, but torch_npu is not importable") from exc


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_enabled(args: argparse.Namespace) -> bool:
    return args.distributed_backend != "none"


def is_rank0(args: argparse.Namespace) -> bool:
    return (not distributed_enabled(args)) or global_rank() == 0


def init_distributed(args: argparse.Namespace) -> torch.device | None:
    if not distributed_enabled(args):
        return None

    backend = args.distributed_backend.strip().lower()
    local = local_rank()
    if backend == "hccl":
        import_torch_npu()
        torch.npu.set_device(local)
        device = torch.device(f"npu:{local}")
        dist.init_process_group(
            backend="hccl",
            init_method=args.init_method,
        )
        return device
    raise ValueError(f"unsupported distributed backend: {args.distributed_backend}")


def distributed_barrier(args: argparse.Namespace) -> None:
    if distributed_enabled(args) and dist.is_initialized():
        dist.barrier()


def destroy_distributed(args: argparse.Namespace) -> None:
    if distributed_enabled(args) and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        try:
            import_torch_npu()
            if torch.npu.is_available():
                torch.npu.set_device(0)
                return torch.device("npu:0")
        except RuntimeError:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "npu":
        import_torch_npu()
        index = 0 if device.index is None else int(device.index)
        torch.npu.set_device(index)
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu.is_available() is false")
        return torch.device(f"npu:{index}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return device


def dtype_from_text(value: str) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {value}")


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def finite_grad(model: torch.nn.Module) -> tuple[bool, float, float]:
    saw_grad = False
    max_abs = 0.0
    total_abs = 0.0
    count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        saw_grad = True
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            return False, float("nan"), float("nan")
        abs_grad = grad.abs()
        max_abs = max(max_abs, float(abs_grad.max().item()))
        total_abs += float(abs_grad.sum().item())
        count += int(abs_grad.numel())
    return saw_grad, total_abs / max(count, 1), max_abs


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def accelerator_memory(device: torch.device) -> dict[str, float | None]:
    if device.type == "npu":
        synchronize(device)
        return {
            "allocated_gib": torch.npu.max_memory_allocated(device) / 1024**3,
            "reserved_gib": torch.npu.max_memory_reserved(device) / 1024**3,
        }
    if device.type != "cuda":
        return {"allocated_gib": None, "reserved_gib": None}
    synchronize(device)
    return {
        "allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def resolve_count(count: int, start: int, total: int, *, name: str) -> int:
    if start < 0:
        raise ValueError(f"{name}-start must be non-negative")
    if count == 0:
        count = total - start
    if count <= 0:
        raise ValueError(f"{name}-count must be positive or 0 for full remaining range")
    if start + count > total:
        raise ValueError(f"{name} crop exceeds range: {start}+{count}>{total}")
    return count


def crop_dataset(
    dataset: xr.Dataset,
    *,
    lat_start: int,
    lat_count: int,
    lon_start: int,
    lon_count: int,
) -> xr.Dataset:
    return dataset.isel(
        lat=slice(lat_start, lat_start + lat_count),
        lon=slice(lon_start, lon_start + lon_count),
    ).load()


def load_and_crop_datasets(args: argparse.Namespace) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs = xr.open_dataset(args.inputs)
    targets = xr.open_dataset(args.targets)
    forcings = xr.open_dataset(args.forcings)
    lat_count = resolve_count(args.lat_count, args.lat_start, int(inputs.sizes["lat"]), name="lat")
    lon_count = resolve_count(args.lon_count, args.lon_start, int(inputs.sizes["lon"]), name="lon")
    args.lat_count = lat_count
    args.lon_count = lon_count
    return (
        crop_dataset(
            inputs,
            lat_start=args.lat_start,
            lat_count=lat_count,
            lon_start=args.lon_start,
            lon_count=lon_count,
        ),
        crop_dataset(
            targets,
            lat_start=args.lat_start,
            lat_count=lat_count,
            lon_start=args.lon_start,
            lon_count=lon_count,
        ),
        crop_dataset(
            forcings,
            lat_start=args.lat_start,
            lat_count=lat_count,
            lon_start=args.lon_start,
            lon_count=lon_count,
        ),
    )


def normalized_target_residuals(
    inputs: xr.Dataset,
    targets: xr.Dataset,
    *,
    stats,
) -> xr.Dataset:
    data_vars = {}
    for name, target in targets.data_vars.items():
        if name in inputs:
            residual = target - inputs[name].isel(time=-1)
            if name in stats.diffs_stddev_by_level:
                residual = residual / stats.diffs_stddev_by_level[name].astype(residual.dtype)
            data_vars[name] = residual
        else:
            data_vars[name] = normalize_dataset(
                xr.Dataset({name: target}),
                scales=stats.stddev_by_level,
                locations=stats.mean_by_level,
            )[name]
    return xr.Dataset(data_vars)


def stacked_dataset_to_node_tensor(dataset: xr.Dataset, *, device: torch.device) -> torch.Tensor:
    stacked = dataset_to_stacked(dataset).transpose("batch", "lat", "lon", "channels")
    array = np.asarray(stacked.data, dtype=np.float32)
    batch, lat, lon, channels = array.shape
    return torch.from_numpy(array.reshape(batch, lat * lon, channels)).to(device=device)


def normalized_latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    latitudes = np.asarray(latitudes, dtype=np.float64)
    if len(latitudes) < 2:
        return np.ones_like(latitudes, dtype=np.float32)
    delta = abs(float(np.diff(latitudes)[0]))
    if np.any(np.isclose(np.abs(latitudes), 90.0)):
        weights = np.cos(np.deg2rad(latitudes)) * np.sin(np.deg2rad(delta / 2.0))
        pole_value = np.sin(np.deg2rad(delta / 4.0)) ** 2
        weights[np.isclose(np.abs(latitudes), 90.0)] = pole_value
    else:
        weights = np.cos(np.deg2rad(latitudes))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def channel_weights_for_targets(targets: xr.Dataset) -> tuple[np.ndarray, dict[str, slice]]:
    weights = []
    slices = {}
    cursor = 0
    for name in sorted(targets.data_vars.keys()):
        array = targets[name]
        non_preserved = [dim for dim in array.dims if dim not in PRESERVED_DIMS]
        time_count = int(array.sizes.get("time", 1))
        variable_weight = float(PER_VARIABLE_WEIGHTS.get(name, 1.0))
        if "level" in array.dims:
            levels = np.asarray(array.coords["level"].values, dtype=np.float32)
            level_weight = levels / levels.sum()
            repeated = np.tile(level_weight, time_count) / max(time_count, 1)
            var_weights = variable_weight * repeated.astype(np.float32)
        else:
            channel_count = int(
                np.prod([array.sizes[dim] for dim in non_preserved], dtype=np.int64)
            )
            var_weights = np.full(
                channel_count,
                variable_weight / max(time_count, 1),
                dtype=np.float32,
            )
        slices[name] = slice(cursor, cursor + len(var_weights))
        cursor += len(var_weights)
        weights.append(var_weights)
    return np.concatenate(weights, axis=0).astype(np.float32), slices


def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lat_count: int,
    lon_count: int,
    latitude_weights: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    batch, grid_nodes, channels = prediction.shape
    if grid_nodes != lat_count * lon_count:
        raise ValueError(f"expected {lat_count * lon_count} nodes, got {grid_nodes}")
    squared = (prediction - target) ** 2
    squared = squared.reshape(batch, lat_count, lon_count, channels)
    weighted = squared * latitude_weights.view(1, lat_count, 1, 1)
    weighted = weighted * channel_weights.view(1, 1, 1, channels)
    return weighted.sum(dim=-1).mean(dim=(1, 2)).mean()


def per_variable_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lat_count: int,
    lon_count: int,
    latitude_weights: torch.Tensor,
    channel_weights: torch.Tensor,
    channel_slices: dict[str, slice],
) -> dict[str, float]:
    result = {}
    batch, _, channels = prediction.shape
    squared = (prediction.detach() - target.detach()) ** 2
    squared = squared.reshape(batch, lat_count, lon_count, channels)
    for name, channel_slice in channel_slices.items():
        values = squared[..., channel_slice]
        weights = channel_weights[channel_slice]
        weighted = values * latitude_weights.view(1, lat_count, 1, 1)
        weighted = weighted * weights.view(1, 1, 1, -1)
        result[name] = float(weighted.sum(dim=-1).mean(dim=(1, 2)).mean().item())
    return result


def finite_dataset(dataset: xr.Dataset) -> bool:
    return all(bool(np.isfinite(array.values).all()) for array in dataset.data_vars.values())


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def set_train_scope(model: torch.nn.Module, scope: str) -> tuple[int, int, list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"

    if scope == "decoder-grid":
        prefix = "mesh2grid_gnn.decoder_nodes_grid_nodes_mlp"
        for name, parameter in model.named_parameters():
            if name.startswith(prefix):
                parameter.requires_grad = True
    elif scope != "all":
        raise ValueError(f"unsupported train scope: {scope}")

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total, trainable_names


def max_abs_state_diff(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    max_diff = 0.0
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))
        unexpected = sorted(set(right) - set(left))
        raise ValueError(f"state dict keys differ: missing={missing}, unexpected={unexpected}")
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise ValueError(
                f"state dict shape differs for {name}: {left[name].shape} vs {right[name].shape}"
            )
        if left[name].numel():
            max_diff = max(max_diff, float((left[name] - right[name]).abs().max().item()))
    return max_diff


def evaluate_loss(
    model: torch.nn.Module,
    *,
    grid_features: torch.Tensor,
    target_tensor: torch.Tensor,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lat_weights: torch.Tensor,
    channel_weights_t: torch.Tensor,
    channel_slices: dict[str, slice],
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        with autocast_context(device, amp, amp_dtype):
            prediction = model(grid_features)
            loss = weighted_mse_loss(
                prediction,
                target_tensor,
                lat_count=len(latitudes),
                lon_count=len(longitudes),
                latitude_weights=lat_weights,
                channel_weights=channel_weights_t,
            )
    synchronize(device)
    per_var = per_variable_losses(
        prediction,
        target_tensor,
        lat_count=len(latitudes),
        lon_count=len(longitudes),
        latitude_weights=lat_weights,
        channel_weights=channel_weights_t,
        channel_slices=channel_slices,
    )
    if was_training:
        model.train()
    return {
        "loss": float(loss.detach().item()),
        "finite_prediction_tensor": bool(torch.isfinite(prediction.detach()).all().item()),
        "per_variable_loss": per_var,
        "seconds": time.perf_counter() - t0,
    }


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PyTorch-compatible GraphCast Training Loss",
        "",
        f"- status: `{summary['status']}`",
        f"- run dir: `{summary['run_dir']}`",
        f"- config: `{summary['config']}`",
        f"- weights: `{summary['weights']}`",
        f"- inputs: `{summary['inputs']}`",
        f"- targets: `{summary['targets']}`",
        f"- forcings: `{summary['forcings']}`",
        f"- stats dir: `{summary['stats_dir']}`",
        f"- device: `{summary['device']}`",
        f"- distributed backend: `{summary.get('distributed_backend', 'none')}`",
        f"- world size: `{summary.get('world_size', 1)}`",
        f"- amp: `{summary['amp']}`",
        f"- amp dtype: `{summary['amp_dtype']}`",
        f"- compute grad: `{summary['compute_grad']}`",
        f"- train scope: `{summary['train_scope']}`",
        f"- activation checkpointing: `{summary['activation_checkpointing']}`",
        f"- mesh2grid edge chunk size: `{summary['mesh2grid_edge_chunk_size']}`",
    ]
    lines.extend(summary.get("backend_lines", []))
    lines.extend(
        [
            f"- trainable parameters: `{summary['trainable_parameters']}`",
            f"- total parameters: `{summary['total_parameters']}`",
            f"- trainable parameter sample: `{summary['trainable_parameter_sample']}`",
            f"- steps: `{summary['steps']}`",
            f"- lr: `{summary['lr']}`",
            f"- crop: `{summary['crop']}`",
            f"- input shape: `{summary['input_shape']}`",
            f"- target shape: `{summary['target_shape']}`",
            f"- final loss: `{summary['final_loss']}`",
            f"- post-train loss: `{summary['post_train_loss']}`",
            f"- saved checkpoint: `{summary['saved_checkpoint']}`",
            f"- reload validation: `{summary['reload_validation']}`",
            f"- inference validation: `{summary['inference_validation']}`",
            f"- accelerator memory: `{summary['accelerator_memory']}`",
            f"- total seconds: `{summary['total_seconds']:.3f}`",
            f"- metrics jsonl: `{summary['metrics_jsonl']}`",
            f"- summary json: `{summary['summary_json']}`",
            "",
            "## Step Metrics",
            "",
        ]
    )
    for item in summary["metrics"]:
        lines.append(
            "- step `{step}` loss `{loss}` grad_mean_abs `{grad_mean_abs}` "
            "grad_max_abs `{grad_max_abs}` grad_ok `{grad_ok}` "
            "step_seconds `{step_seconds}`".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--forcings", default=DEFAULT_FORCINGS)
    parser.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--compute-grad", action="store_true")
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--distributed-backend",
        choices=("none", "hccl"),
        default="none",
        help="set to hccl when launched with torch.distributed.run on Ascend NPU",
    )
    parser.add_argument("--init-method", default="env://")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-scope", choices=("all", "decoder-grid"), default="all")
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="checkpoint each graph processor step during training to reduce saved activations",
    )
    parser.add_argument(
        "--mesh2grid-edge-chunk-size",
        type=int,
        default=None,
        help="override config model_config.mesh2grid_edge_chunk_size for training",
    )
    parser.add_argument("--lat-start", type=int, default=0)
    parser.add_argument(
        "--lat-count", type=int, default=0, help="0 means full remaining latitude range"
    )
    parser.add_argument("--lon-start", type=int, default=0)
    parser.add_argument(
        "--lon-count", type=int, default=0, help="0 means full remaining longitude range"
    )
    parser.add_argument("--mesh-size", type=int, default=None)
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="optional path for saving the final trained model state_dict",
    )
    parser.add_argument(
        "--checkpoint-metadata",
        default=None,
        help="optional JSON metadata path for --save-checkpoint",
    )
    parser.add_argument(
        "--validation-prediction-output",
        default=None,
        help=(
            "optional NetCDF output path for inference validation; defaults to "
            "<run-dir>/validation_prediction.nc when --save-checkpoint is set"
        ),
    )
    parser.add_argument(
        "--skip-checkpoint-validation",
        action="store_true",
        help="save the checkpoint without strict reload/loss/prediction validation",
    )
    parser.add_argument("--reload-loss-atol", type=float, default=1.0e-6)
    parser.add_argument(
        "--flagos-mode",
        default=os.environ.get("GRAPHCAST_FLAGOS_MODE", "off"),
        help="FlagOS backend mode: off or flaggems",
    )
    parser.add_argument(
        "--flaggems-ops",
        default=os.environ.get("GRAPHCAST_FLAGGEMS_OPS", ""),
        help="comma-separated FlagGems ops to enable",
    )
    parser.add_argument(
        "--flaggems-config",
        default=os.environ.get("GRAPHCAST_FLAGGEMS_CONFIG", str(DEFAULT_FLAGGEMS_CONFIG)),
        help="FlagGems operator profile used when --flaggems-ops is empty",
    )
    parser.add_argument(
        "--flaggems-record-path",
        default=os.environ.get("GRAPHCAST_FLAGGEMS_RECORD_PATH", ""),
        help="FlagGems record log path",
    )
    parser.add_argument(
        "--flaggems-strict",
        default=os.environ.get(
            "GRAPHCAST_STRICT_FLAGOS",
            os.environ.get("GRAPHCAST_FLAGGEMS_STRICT", "1"),
        ),
        help="fail if FlagGems setup fails",
    )
    parser.add_argument(
        "--flaggems-src",
        default=os.environ.get(
            "GRAPHCAST_FLAGGEMS_SOURCE_ROOT",
            os.environ.get("GRAPHCAST_FLAGGEMS_SRC", ""),
        ),
        help="optional FlagGems source path",
    )
    parser.add_argument(
        "--flaggems-variant",
        choices=("installed", "official", "internal"),
        default=os.environ.get("GRAPHCAST_FLAGGEMS_VARIANT", "installed"),
        help="installed package or a runtime prepared under third_party",
    )
    parser.add_argument(
        "--flaggems-runtime-dest",
        default=os.environ.get("FLAGGEMS_RUNTIME_DEST", ""),
        help="root containing prepared official/internal FlagGems trees",
    )
    parser.add_argument(
        "--bishengir-opt-dir",
        default=os.environ.get("GRAPHCAST_BISHENGIR_OPT_DIR", ""),
        help="BishengIR opt binary directory",
    )
    args = parser.parse_args(argv)

    configure_flagos_environment(
        mode=args.flagos_mode,
        ops=args.flaggems_ops,
        config=args.flaggems_config,
        source_root=args.flaggems_src,
        record_path=args.flaggems_record_path,
        strict=args.flaggems_strict,
        bishengir_opt_dir=args.bishengir_opt_dir,
        variant=args.flaggems_variant,
        runtime_dest=args.flaggems_runtime_dest,
    )

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.lr != 0.0 and not args.compute_grad:
        raise ValueError("lr updates require --compute-grad")

    t0 = time.perf_counter()
    distributed_device = init_distributed(args)
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    if is_rank0(args):
        run_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(args)
    if is_rank0(args) and metrics_path.exists():
        metrics_path.unlink()
    distributed_barrier(args)

    device = distributed_device if distributed_device is not None else resolve_device(args.device)
    reset_peak_memory(device)
    amp_dtype = dtype_from_text(args.amp_dtype)

    inputs, targets, forcings = load_and_crop_datasets(args)
    stats = load_normalization_stats(args.stats_dir)
    config = GoogleSmallConfig.from_yaml(args.config)
    if args.mesh_size is not None:
        config = replace(config, mesh_size=args.mesh_size)
    if args.mesh2grid_edge_chunk_size is not None:
        if args.mesh2grid_edge_chunk_size <= 0:
            raise ValueError("--mesh2grid-edge-chunk-size must be positive")
        config = replace(
            config,
            mesh2grid_edge_chunk_size=args.mesh2grid_edge_chunk_size,
        )

    selected_targets = select_dataset_variables(
        targets,
        config.target_variables or None,
        role="targets",
    )
    norm_inputs = normalize_dataset(
        inputs,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_forcings = normalize_dataset(
        forcings,
        scales=stats.stddev_by_level,
        locations=stats.mean_by_level,
    )
    norm_targets = normalized_target_residuals(inputs, selected_targets, stats=stats)

    grid_features = inputs_to_grid_node_features(
        norm_inputs,
        norm_forcings,
        input_variables=config.input_variables or None,
        forcing_variables=config.forcing_variables or None,
    ).to(device=device)
    target_tensor = stacked_dataset_to_node_tensor(norm_targets, device=device)
    channel_weights, channel_slices = channel_weights_for_targets(norm_targets)
    latitudes = np.asarray(inputs.coords["lat"].values, dtype=np.float32)
    longitudes = np.asarray(inputs.coords["lon"].values, dtype=np.float32)
    lat_weights = torch.from_numpy(normalized_latitude_weights(latitudes)).to(device=device)
    channel_weights_t = torch.from_numpy(channel_weights).to(device=device)

    runtime = resolve_runtime_backend(args.device)

    model = GoogleSmallCompatibleModel(
        config,
        activation_checkpointing=args.activation_checkpointing,
    )
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.init_graph(latitudes=latitudes, longitudes=longitudes)
    trainable, total, trainable_names = set_train_scope(model, args.train_scope)
    model.to(device)
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if args.compute_grad and not trainable_parameters:
        raise RuntimeError(f"train scope {args.train_scope!r} selected no parameters")
    train_model: torch.nn.Module = model
    if distributed_enabled(args):
        train_model = DDP(
            model,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    optimizer = (
        torch.optim.SGD(
            [parameter for parameter in train_model.parameters() if parameter.requires_grad],
            lr=args.lr,
        )
        if args.lr != 0.0
        else None
    )

    metrics = []
    for step in range(1, args.steps + 1):
        synchronize(device)
        step_t0 = time.perf_counter()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            train_model.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp, amp_dtype):
            prediction = train_model(grid_features)
            loss = weighted_mse_loss(
                prediction,
                target_tensor,
                lat_count=len(latitudes),
                lon_count=len(longitudes),
                latitude_weights=lat_weights,
                channel_weights=channel_weights_t,
            )
        grad_ok = None
        grad_mean_abs = None
        grad_max_abs = None
        if args.compute_grad:
            loss.backward()
            grad_ok, grad_mean_abs, grad_max_abs = finite_grad(train_model)
            if optimizer is not None:
                optimizer.step()
        synchronize(device)
        per_var = per_variable_losses(
            prediction,
            target_tensor,
            lat_count=len(latitudes),
            lon_count=len(longitudes),
            latitude_weights=lat_weights,
            channel_weights=channel_weights_t,
            channel_slices=channel_slices,
        )
        metric = {
            "step": step,
            "loss": float(loss.detach().item()),
            "grad_ok": grad_ok,
            "grad_mean_abs": grad_mean_abs,
            "grad_max_abs": grad_max_abs,
            "per_variable_loss": per_var,
            "step_seconds": round(time.perf_counter() - step_t0, 3),
        }
        metrics.append(metric)
        if is_rank0(args):
            append_jsonl(metrics_path, metric)
            print(
                "rank={rank} step={step} loss={loss:.8g} "
                "grad_mean_abs={grad_mean_abs} grad_max_abs={grad_max_abs} "
                "grad_ok={grad_ok} step_seconds={step_seconds}".format(
                    rank=global_rank(),
                    **metric,
                ),
                flush=True,
            )

    post_train_eval = evaluate_loss(
        model,
        grid_features=grid_features,
        target_tensor=target_tensor,
        latitudes=latitudes,
        longitudes=longitudes,
        lat_weights=lat_weights,
        channel_weights_t=channel_weights_t,
        channel_slices=channel_slices,
        device=device,
        amp=args.amp,
        amp_dtype=amp_dtype,
    )

    saved_checkpoint = None
    checkpoint_metadata = None
    reload_validation: dict[str, Any] | None = None
    inference_validation: dict[str, Any] | None = None
    validation_ok = True

    if args.save_checkpoint and is_rank0(args):
        checkpoint_path = Path(args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trained_state = state_dict_to_cpu(model)
        torch.save(trained_state, checkpoint_path)
        saved_checkpoint = str(checkpoint_path)

        metadata_path = (
            Path(args.checkpoint_metadata)
            if args.checkpoint_metadata
            else checkpoint_path.with_suffix(checkpoint_path.suffix + ".metadata.json")
        )
        checkpoint_metadata = str(metadata_path)

        if not args.skip_checkpoint_validation:
            reload_model = GoogleSmallCompatibleModel(
                config,
                activation_checkpointing=args.activation_checkpointing,
            )
            reloaded_state = torch.load(checkpoint_path, map_location="cpu")
            load_result = reload_model.load_state_dict(reloaded_state, strict=True)
            reload_model.init_graph(latitudes=latitudes, longitudes=longitudes)
            reload_model.to(device)
            reload_eval = evaluate_loss(
                reload_model,
                grid_features=grid_features,
                target_tensor=target_tensor,
                latitudes=latitudes,
                longitudes=longitudes,
                lat_weights=lat_weights,
                channel_weights_t=channel_weights_t,
                channel_slices=channel_slices,
                device=device,
                amp=args.amp,
                amp_dtype=amp_dtype,
            )
            reloaded_cpu_state = state_dict_to_cpu(reload_model)
            state_max_diff = max_abs_state_diff(trained_state, reloaded_cpu_state)
            reload_loss_abs_diff = abs(reload_eval["loss"] - post_train_eval["loss"])
            reload_validation = {
                "strict_load": True,
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
                "post_train_loss": post_train_eval["loss"],
                "reloaded_loss": reload_eval["loss"],
                "loss_abs_diff": reload_loss_abs_diff,
                "loss_atol": args.reload_loss_atol,
                "loss_match": reload_loss_abs_diff <= args.reload_loss_atol,
                "state_max_abs_diff_after_reload": state_max_diff,
                "finite_prediction_tensor": reload_eval["finite_prediction_tensor"],
                "seconds": reload_eval["seconds"],
            }
            validation_ok = validation_ok and bool(reload_validation["loss_match"])
            validation_ok = validation_ok and bool(reload_validation["finite_prediction_tensor"])

            prediction_output = (
                Path(args.validation_prediction_output)
                if args.validation_prediction_output
                else run_dir / "validation_prediction.nc"
            )
            reload_model.eval()
            synchronize(device)
            infer_t0 = time.perf_counter()
            prediction_ds = predict_with_inputs_and_residuals(
                reload_model,
                inputs,
                selected_targets,
                forcings,
                stats,
            )
            synchronize(device)
            prediction_output.parent.mkdir(parents=True, exist_ok=True)
            prediction_ds.to_netcdf(prediction_output)
            prediction_stacked = dataset_to_stacked(prediction_ds)
            prediction_finite = finite_dataset(prediction_ds)
            inference_validation = {
                "prediction_output": str(prediction_output),
                "finite": prediction_finite,
                "prediction_variables": sorted(prediction_ds.data_vars.keys()),
                "prediction_channels": int(prediction_stacked.sizes["channels"]),
                "seconds": time.perf_counter() - infer_t0,
            }
            validation_ok = validation_ok and prediction_finite

        metadata = {
            "checkpoint": saved_checkpoint,
            "source_weights": args.weights,
            "config": args.config,
            "inputs": args.inputs,
            "targets": args.targets,
            "forcings": args.forcings,
            "stats_dir": args.stats_dir,
            "train_scope": args.train_scope,
            "activation_checkpointing": args.activation_checkpointing,
            "mesh2grid_edge_chunk_size": config.mesh2grid_edge_chunk_size,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_parameter_sample": trainable_names[:8],
            "steps": args.steps,
            "lr": args.lr,
            "amp": args.amp,
            "amp_dtype": args.amp_dtype,
            "crop": {
                "lat_start": args.lat_start,
                "lat_count": args.lat_count,
                "lon_start": args.lon_start,
                "lon_count": args.lon_count,
                "lat_range": [float(latitudes[0]), float(latitudes[-1])],
                "lon_range": [float(longitudes[0]), float(longitudes[-1])],
            },
            "post_train_loss": post_train_eval["loss"],
            "reload_validation": reload_validation,
            "inference_validation": inference_validation,
        }
        json_dump(metadata_path, metadata)

    total_seconds = time.perf_counter() - t0
    metric_status_ok = all(item["grad_ok"] is not False for item in metrics)
    summary = {
        "status": "OK" if metric_status_ok and validation_ok else "FAIL",
        "run_dir": str(run_dir),
        "config": args.config,
        "weights": args.weights,
        "inputs": args.inputs,
        "targets": args.targets,
        "forcings": args.forcings,
        "stats_dir": args.stats_dir,
        "device": str(device),
        "distributed_backend": args.distributed_backend,
        "world_size": world_size() if distributed_enabled(args) else 1,
        "rank": global_rank() if distributed_enabled(args) else 0,
        "local_rank": local_rank() if distributed_enabled(args) else 0,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "compute_grad": args.compute_grad,
        "train_scope": args.train_scope,
        "activation_checkpointing": args.activation_checkpointing,
        "mesh2grid_edge_chunk_size": config.mesh2grid_edge_chunk_size,
        "backend_lines": backend_report_lines(runtime),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_sample": trainable_names[:8],
        "steps": args.steps,
        "lr": args.lr,
        "crop": {
            "lat_start": args.lat_start,
            "lat_count": args.lat_count,
            "lon_start": args.lon_start,
            "lon_count": args.lon_count,
            "lat_range": [float(latitudes[0]), float(latitudes[-1])],
            "lon_range": [float(longitudes[0]), float(longitudes[-1])],
        },
        "input_shape": tuple(int(dim) for dim in grid_features.shape),
        "target_shape": tuple(int(dim) for dim in target_tensor.shape),
        "final_loss": metrics[-1]["loss"],
        "post_train_loss": post_train_eval["loss"],
        "post_train_eval": post_train_eval,
        "saved_checkpoint": saved_checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "reload_validation": reload_validation,
        "inference_validation": inference_validation,
        "metrics": metrics,
        "metrics_jsonl": str(metrics_path),
        "summary_json": str(summary_path),
        "report": str(report_path),
        "accelerator_memory": accelerator_memory(device),
        "total_seconds": total_seconds,
    }
    exit_code = 0 if summary["status"] == "OK" else 1
    if is_rank0(args):
        json_dump(summary_path, summary)
        report_path.write_text(build_report(summary), encoding="utf-8")
        print(build_report(summary))
    distributed_barrier(args)
    destroy_distributed(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
