#!/usr/bin/env python3
"""Train Google Small compatible GraphCast on the real 25km NetCDF case with DDP."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import xarray as xr


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
)
from graphcast_compat.normalization import normalize_dataset  # noqa: E402


DEFAULT_CASE_DIR = REPO_ROOT / "cases" / "2021_20210101_day_operational_25km"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference" / "google_graphcast_operational_25km.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "outputs" / "checkpoints" / "google_graphcast_operational_25km_compat_state.pt"
DEFAULT_STATS_DIR = REPO_ROOT / "data" / "raw" / "google_graphcast_stats"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "training" / "reports" / "flagcx_25km_ddp_train_report.md"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "training" / "checkpoints" / "flagcx_25km_ddp_train.pt"


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    return global_rank() == 0


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


def memory_text(device: torch.device) -> str:
    synchronize(device)
    if device.type == "npu":
        allocated = torch.npu.max_memory_allocated(device) / 1024**3
        reserved = torch.npu.max_memory_reserved(device) / 1024**3
        return f"allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        return f"allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    return "n/a"


def init_distributed(args: argparse.Namespace) -> tuple[torch.device, str]:
    backend = args.distributed_backend.strip().lower()
    rank = global_rank()
    local = local_rank()
    if backend == "flagcx":
        import torch_npu  # noqa: F401, PLC0415
        import flagcx  # noqa: F401, PLC0415

        torch.npu.set_device(local)
        device = torch.device(f"npu:{local}")
        backend_config = "cpu:gloo,npu:flagcx"
    elif backend == "hccl":
        import torch_npu  # noqa: F401, PLC0415

        torch.npu.set_device(local)
        device = torch.device(f"npu:{local}")
        backend_config = "hccl"
    else:
        raise ValueError(f"unsupported --distributed-backend: {args.distributed_backend}")

    dist.init_process_group(
        backend=backend_config,
        rank=rank,
        world_size=world_size(),
        init_method=args.init_method,
    )
    return device, backend_config


def load_real_case(args: argparse.Namespace) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs = xr.open_dataset(args.inputs).load()
    targets = xr.open_dataset(args.target).load()
    forcings = xr.open_dataset(args.forcings).load()
    return inputs, targets, forcings


def target_to_training_tensor(
    *,
    inputs: xr.Dataset,
    targets: xr.Dataset,
    stats,
    target_variables: tuple[str, ...],
) -> torch.Tensor:
    """Build the normalized model-output training target.

    For variables also present in inputs, the model learns residuals normalized by
    `diffs_stddev_by_level`. Variables absent from inputs use mean/std
    normalization, matching `unnormalize_prediction_and_add_input`.
    """

    data_vars = {}
    selected_targets = targets[list(target_variables)]
    for name, target in selected_targets.data_vars.items():
        result = target
        if name in inputs:
            result = result - inputs[name].isel(time=-1)
            if name in stats.diffs_stddev_by_level:
                result = result / stats.diffs_stddev_by_level[name].astype(result.dtype)
        else:
            if name in stats.mean_by_level:
                result = result - stats.mean_by_level[name].astype(result.dtype)
            if name in stats.stddev_by_level:
                result = result / stats.stddev_by_level[name].astype(result.dtype)
        data_vars[name] = result.astype("float32")

    stacked = dataset_to_stacked(xr.Dataset(data_vars))
    stacked = stacked.transpose("batch", "lat", "lon", "channels")
    array = np.asarray(stacked.data, dtype=np.float32)
    batch, n_lat, n_lon, channels = array.shape
    return torch.from_numpy(array.reshape(batch, n_lat * n_lon, channels))


def set_train_scope(model: torch.nn.Module, scope: str) -> tuple[int, int, list[str]]:
    for param in model.parameters():
        param.requires_grad = scope == "all"

    if scope == "decoder-grid":
        prefix = "mesh2grid_gnn.decoder_nodes_grid_nodes_mlp"
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = True
    elif scope != "all":
        raise ValueError(f"unsupported --train-scope: {scope}")

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total, trainable_names


def amp_context(args: argparse.Namespace, device: torch.device):
    if parse_bool(args.amp_bf16):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--stats-dir", default=str(DEFAULT_STATS_DIR))
    parser.add_argument("--inputs", default=str(DEFAULT_CASE_DIR / "inputs.nc"))
    parser.add_argument("--target", default=str(DEFAULT_CASE_DIR / "targets.nc"))
    parser.add_argument("--forcings", default=str(DEFAULT_CASE_DIR / "forcings.nc"))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-scope", choices=("decoder-grid", "all"), default="all")
    parser.add_argument("--distributed-backend", choices=("flagcx", "hccl"), default="flagcx")
    parser.add_argument("--init-method", default="env://")
    parser.add_argument("--amp-bf16", default="0")
    parser.add_argument(
        "--activation-checkpointing",
        default="0",
        help="set 1 to checkpoint each graph processor step during training",
    )
    parser.add_argument(
        "--mesh2grid-edge-chunk-size",
        type=int,
        default=None,
        help="override config model_config.mesh2grid_edge_chunk_size for training",
    )
    parser.add_argument(
        "--safe-exit",
        default="1",
        help="use os._exit(0) after successful FlagCX run to avoid current cleanup abort",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    t0 = time.perf_counter()
    device, backend_config = init_distributed(args)
    rank = global_rank()
    local = local_rank()
    size = world_size()

    inputs, targets, forcings = load_real_case(args)
    config = GoogleSmallConfig.from_yaml(args.config)
    if args.mesh2grid_edge_chunk_size is not None:
        if args.mesh2grid_edge_chunk_size <= 0:
            raise ValueError("--mesh2grid-edge-chunk-size must be positive")
        config = replace(
            config,
            mesh2grid_edge_chunk_size=args.mesh2grid_edge_chunk_size,
        )
    stats = load_normalization_stats(args.stats_dir)

    norm_inputs = normalize_dataset(inputs, stats.stddev_by_level, stats.mean_by_level)
    norm_forcings = normalize_dataset(forcings, stats.stddev_by_level, stats.mean_by_level)
    grid_features = inputs_to_grid_node_features(
        norm_inputs,
        norm_forcings,
        input_variables=config.input_variables or None,
        forcing_variables=config.forcing_variables or None,
    ).to(device)
    target_tensor = target_to_training_tensor(
        inputs=inputs,
        targets=targets,
        stats=stats,
        target_variables=config.target_variables,
    ).to(device)

    activation_checkpointing = parse_bool(args.activation_checkpointing)
    model = GoogleSmallCompatibleModel(
        config,
        activation_checkpointing=activation_checkpointing,
    )
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.init_graph(
        latitudes=inputs.coords["lat"].values,
        longitudes=inputs.coords["lon"].values,
    )
    trainable, total, trainable_names = set_train_scope(model, args.train_scope)
    model.to(device)
    ddp_model = DDP(
        model,
        device_ids=None,
        output_device=None,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        [param for param in ddp_model.parameters() if param.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    reset_peak_memory(device)
    step_losses = []
    train_start = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with amp_context(args, device):
            prediction = ddp_model(grid_features)
            loss = torch.mean((prediction.float() - target_tensor.float()) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in ddp_model.parameters() if param.requires_grad],
            2.0,
        )
        optimizer.step()
        synchronize(device)
        loss_value = float(loss.detach().cpu())
        step_losses.append(loss_value)
        print(
            f"rank={rank} local_rank={local} step={step} "
            f"loss={loss_value:.6f} memory={memory_text(device)}",
            flush=True,
        )

    dist.barrier()
    train_seconds = time.perf_counter() - train_start
    total_seconds = time.perf_counter() - t0

    if is_rank0():
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": ddp_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_scope": args.train_scope,
                "activation_checkpointing": activation_checkpointing,
                "mesh2grid_edge_chunk_size": config.mesh2grid_edge_chunk_size,
                "amp_bf16": parse_bool(args.amp_bf16),
                "step_losses": step_losses,
                "world_size": size,
                "backend": backend_config,
            },
            checkpoint_path,
        )
        report_lines = [
            "# GraphCast 25km Distributed Training Report",
            "",
            f"- backend: `{backend_config}`",
            f"- distributed backend option: `{args.distributed_backend}`",
            f"- world size: `{size}`",
            f"- train scope: `{args.train_scope}`",
            f"- activation checkpointing: `{activation_checkpointing}`",
            f"- mesh2grid edge chunk size: `{config.mesh2grid_edge_chunk_size}`",
            f"- amp bf16: `{parse_bool(args.amp_bf16)}`",
            f"- trainable parameters: `{trainable}`",
            f"- total parameters: `{total}`",
            f"- trainable parameter prefixes/sample: `{trainable_names[:8]}`",
            f"- inputs: `{args.inputs}`",
            f"- target: `{args.target}`",
            f"- forcings: `{args.forcings}`",
            f"- grid feature shape: `{tuple(grid_features.shape)}`",
            f"- target tensor shape: `{tuple(target_tensor.shape)}`",
            f"- steps: `{args.steps}`",
            f"- losses: `{step_losses}`",
            f"- train seconds: `{train_seconds:.3f}`",
            f"- total seconds: `{total_seconds:.3f}`",
            f"- rank0 memory: `{memory_text(device)}`",
            f"- checkpoint: `{checkpoint_path}`",
            "",
            "Overall status: `OK`",
            "",
            "本次训练使用真实 25km NetCDF 算例。默认训练范围为 all，",
            "用于验证多卡 DDP 梯度同步和真实 25km full-scope 训练路径。",
        ]
        write_report(Path(args.report), report_lines)
        print(f"report: {args.report}", flush=True)
        print(f"checkpoint: {checkpoint_path}", flush=True)

    dist.barrier()
    if args.distributed_backend == "flagcx" and parse_bool(args.safe_exit):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
