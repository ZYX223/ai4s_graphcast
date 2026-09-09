#!/usr/bin/env python3
"""Run complete, exactly resumable GraphCast training on compatible devices.

Training tensors use the colleague-provided PyTorch compatibility layer first.
Window extraction calls the pinned DeepMind ``data_utils`` implementation.
Only multi-shard indexing, PyTorch autograd, DP/MP execution, and observability
remain as FlagOS integration code.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from contextlib import nullcontext
import json
import math
import os
import platform
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


# ==================================================
# 0. 全局训练配置
# ==================================================
PLATFORM_LABEL = os.environ.get("GRAPHCAST_PLATFORM_LABEL", "H100")
OFFICIAL_2_TO_12_RECIPE = "official_2_to_12"
CURRICULUM_ROLLOUT_START = 2
CURRICULUM_ROLLOUT_END = 12
CURRICULUM_STAGE_COUNT = (
    CURRICULUM_ROLLOUT_END - CURRICULUM_ROLLOUT_START + 1
)
DEFAULT_WARMUP_UPDATES = 1_000
DEFAULT_TRAINING_UPDATES = 299_000
# The published 2-to-12 autoregressive curriculum used 11,000 updates: 1,000
# consecutive updates at each rollout horizon from 2 through 12.
DEFAULT_FINETUNE_UPDATES = 11_000
# The paper used a global batch of 32.  This heterogeneous-hardware validation
# uses the user-approved batch of 16 while retaining the official
# update-driven, trajectory-with-replacement sampler and full model/grid.
DEFAULT_EFFECTIVE_GLOBAL_BATCH_SIZE = 16
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from graphcast_compat.autoregressive_training import (  # noqa: E402
    autoregressive_loss,
    prepare_autoregressive_batch,
)
from graphcast_compat.accelerator import (  # noqa: E402
    accelerator_device,
    accelerator_environment,
)
from graphcast_compat.distributed_training import (  # noqa: E402
    DistributedContext,
    loss_scale_for_valid_samples,
    merge_profiled_steps,
)
from graphcast_compat.data_prefetch import (  # noqa: E402
    DepthOneCpuPrefetcher,
    OfficialMicrobatchDataRequest,
    OfficialUpdateCpuProducer,
    OfficialUpdateDataKey,
    OfficialUpdateDataRequest,
    PreparedOfficialUpdate,
    training_sample_provenance,
)
from graphcast_compat.epoch_plan import EpochPlan  # noqa: E402
from graphcast_compat.training_backend import (  # noqa: E402
    enable_network_replacements_from_env,
    replacement_resume_policy,
)
from graphcast_compat.gradient_accumulation import (  # noqa: E402
    MicrobatchPlan,
)
from graphcast_compat.graph import (  # noqa: E402
    CANONICAL_GRAPH_TOPOLOGY_VERSION,
    validate_canonical_operational_graph_contract,
)
from graphcast_compat.training_model import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
)
from graphcast_compat.hotspot_profiler import HotspotProfiler  # noqa: E402
from graphcast_compat.normalization import (  # noqa: E402
    load_normalization_stats,
)
from graphcast_compat.official_training_corpus import (  # noqa: E402
    OfficialTrainingCorpus,
)
from graphcast_compat.precision import configure_model_precision  # noqa: E402
from graphcast_compat.profiled_mfu import (  # noqa: E402
    enrich_step_metrics,
)
from graphcast_compat.recompute_flops import RecomputeFlopCounter  # noqa: E402
from graphcast_compat.training_recipe import (  # noqa: E402
    build_graphcast_adamw,
    clip_graphcast_gradients,
    graphcast_optimizer_metadata,
    validate_full_operational_model_config,
)
from graphcast_compat.training_schedule import (  # noqa: E402
    TrainingSchedule,
)
from graphcast_compat.training_state import (  # noqa: E402
    TrainingCursor,
    restore_rng_state,
)
from official_epoch_training_file_validation import (  # noqa: E402
    load_model_only_checkpoint,
    load_resume_checkpoint,
    load_schedule_manifest,
    read_metric_segments,
    save_validated_training_checkpoint,
    validate_initial_replica_state,
    validated_replicated_model_hashes,
    write_immutable_json,
)
from training_support import (  # noqa: E402
    PER_VARIABLE_WEIGHTS,
    append_jsonl,
    configure_float32_math,
    json_dump,
    memory_by_device,
    profile_config,
    reset_peak_memory,
    synchronize,
    validate_formal_dataset,
    validate_metric,
)


# ==================================================
# 1. 命令行参数、进度输出与训练配方
# ==================================================
def _rank_zero_progress(
    event: str,
    *,
    rank: int | None = None,
    **values: Any,
) -> None:
    """Emit an immediately visible progress line from global rank zero."""
    effective_rank = (
        int(os.environ.get("RANK", "0")) if rank is None else rank
    )
    if effective_rank != 0:
        return
    fields = " ".join(f"{name}={value}" for name, value in values.items())
    print(f"{event} {fields}".rstrip(), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=("era5", "hres"),
        required=True,
    )
    parser.add_argument(
        "--training-stage",
        choices=("era5_pretrain", "hres_finetune"),
        required=True,
    )
    parser.add_argument("--training-start", default="")
    parser.add_argument("--training-end", default="")
    parser.add_argument(
        "--split-cache-max-gib",
        type=float,
        default=0.0,
        help=(
            "maximum host GiB used to preload dynamic frames reachable by "
            "the selected training split; 0 preserves lazy NetCDF streaming"
        ),
    )
    parser.add_argument(
        "--data-prefetch-depth",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "host-side update prefetch depth; 0 disables prefetch and 1 "
            "uses one CPU producer with one outstanding update"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=(
            "epoch count for manifest schedules; official_2_to_12 derives "
            "the required epoch count from its update budgets"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EFFECTIVE_GLOBAL_BATCH_SIZE,
        help=(
            "effective global mini-batch per optimizer update; the portable "
            "launcher defaults to 16 while the published GraphCast value "
            "is 32"
        ),
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help=(
            "global batch materialized at once before gradient accumulation; "
            "it must be divisible by DP so each replica receives the same "
            "number of full-grid samples"
        ),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=12,
        help=(
            "maximum corpus trajectory; official_2_to_12 keeps this at 12 "
            "and slices one-step phases from the same window"
        ),
    )
    parser.add_argument("--rollout-checkpoint-segments", type=int, default=12)
    parser.add_argument(
        "--rollout-checkpoint-implementation",
        choices=("non_reentrant", "reentrant"),
        default="non_reentrant",
        help=(
            "outer autoregressive checkpoint implementation; model-internal "
            "stage and chunk checkpoints remain non-reentrant"
        ),
    )
    parser.add_argument(
        "--single-step-rollout-checkpoint",
        choices=("enabled", "disabled"),
        default="enabled",
        help=(
            "retain or skip the redundant outer rollout checkpoint when the "
            "active official phase has exactly one lead; model-internal "
            "activation checkpointing remains enabled"
        ),
    )
    parser.add_argument("--data-parallel-replicas", type=int, default=1)
    parser.add_argument("--mp-devices-per-replica", type=int, default=4)
    parser.add_argument(
        "--gradient-sync",
        choices=("manual", "ddp"),
        default="ddp",
        help=(
            "ddp uses PyTorch DistributedDataParallel; "
            "manual preserves the historical explicit averaging path"
        ),
    )
    parser.add_argument("--mesh-split-step", type=int, default=8)
    parser.add_argument(
        "--mesh-checkpoint-granularity",
        choices=("auto", "stage", "step"),
        default="auto",
        help=(
            "activation-recompute boundary for each Mesh PP partition; "
            "auto preserves the established platform default, while step "
            "lowers peak memory without changing model math"
        ),
    )
    parser.add_argument(
        "--grid2mesh-stage-device",
        choices=("primary", "mesh_first", "mesh_second", "output"),
        default="primary",
        help=(
            "execution-only PP placement for the complete Grid-to-Mesh "
            "module; raw ERA5 inputs are transferred before this stage and "
            "the model graph, parameters and operation order are unchanged"
        ),
    )
    parser.add_argument(
        "--grid2mesh-node-stage-device",
        choices=(
            "stage",
            "primary",
            "mesh_first",
            "mesh_second",
            "output",
        ),
        default="stage",
        help=(
            "execution-only PP placement for the Grid-to-Mesh grid-node "
            "processor; stage keeps it with the complete adapter"
        ),
    )
    parser.add_argument(
        "--grid2mesh-stage-checkpoint",
        choices=("enabled", "disabled"),
        default="enabled",
        help=(
            "place one non-reentrant activation checkpoint around the "
            "complete Grid-to-Mesh stage; disabling only this outer "
            "boundary retains the existing chunk checkpoints"
        ),
    )
    parser.add_argument(
        "--mesh2grid-stage-device",
        choices=("output", "primary", "mesh_first", "mesh_second"),
        default="output",
        help=(
            "execution-only PP placement for the Mesh-to-Grid processor; "
            "the final decoder remains on the output stage and this option "
            "does not change model parameters or tensor math"
        ),
    )
    parser.add_argument(
        "--mesh2grid-node-stage-device",
        choices=(
            "stage",
            "output",
            "primary",
            "mesh_first",
            "mesh_second",
        ),
        default="stage",
        help=(
            "execution-only PP placement for the Mesh-to-Grid grid-node "
            "processor; stage keeps it with the edge processor, while "
            "output streams the existing node chunks to the decoder stage"
        ),
    )
    parser.add_argument(
        "--mesh2grid-decoder-stage-device",
        choices=("output", "primary", "mesh_first", "mesh_second"),
        default="output",
        help=(
            "execution-only PP placement for the final Mesh-to-Grid "
            "decoder; existing grid-node chunks are streamed to this "
            "stage without changing parameters, values, or operation order"
        ),
    )
    parser.add_argument(
        "--mesh2grid-stage-checkpoint",
        choices=("enabled", "disabled"),
        default="disabled",
        help=(
            "place one existing non-reentrant activation checkpoint around "
            "the complete Mesh-to-Grid stage; this preserves tensor math "
            "and parameters while trading recomputation for peak memory"
        ),
    )
    parser.add_argument(
        "--mesh-edge-chunk-size",
        type=int,
        default=0,
        help=(
            "execution-only Mesh edge MLP chunk size; 0 keeps the model "
            "config value and a positive value preserves all edges and "
            "residual states while bounding the temporary concatenation"
        ),
    )
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=0,
        help=(
            "execution-only edge MLP chunk size; 0 keeps the model config "
            "value and a smaller positive value reduces activation peak "
            "without changing model parameters, graph, or training samples"
        ),
    )
    parser.add_argument(
        "--grid-node-chunk-size",
        type=int,
        default=0,
        help=(
            "execution-only regular-grid node MLP chunk size; 0 keeps the "
            "model config value and a smaller positive value reduces "
            "activation peak without changing parameters, graph, or samples"
        ),
    )
    parser.add_argument(
        "--loss-grid-chunk-size",
        type=int,
        default=0,
        help=(
            "execution-only number of full-resolution grid nodes reduced "
            "at once by the weighted training loss; 0 preserves the "
            "single full-grid reduction"
        ),
    )
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument(
        "--tensor-core",
        choices=("enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument("--peak-tflops-per-device", type=float, default=0.0)
    parser.add_argument("--peak-tflops-source", default="unavailable")
    parser.add_argument(
        "--init-mode",
        choices=("random", "checkpoint"),
        required=True,
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--allow-pretrained-era5-init", action="store_true",
        help="Allow explicit model-only pretrained initialization for a new ERA5 run; optimizer and schedule start fresh.",
    )
    parser.add_argument("--resume-training-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--schedule-recipe",
        choices=("manifest", OFFICIAL_2_TO_12_RECIPE),
        default="manifest",
        help=(
            "manifest preserves the existing external schedule; "
            "official_2_to_12 enables the built-in one-step warmup/training "
            "followed by the official 2-to-12 autoregressive curriculum"
        ),
    )
    parser.add_argument(
        "--schedule-manifest",
        default="",
        help="required only when --schedule-recipe=manifest",
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=DEFAULT_WARMUP_UPDATES,
        help=(
            "optimizer updates in the one-step linear warmup phase "
            f"(default: {DEFAULT_WARMUP_UPDATES})"
        ),
    )
    parser.add_argument(
        "--training-updates",
        type=int,
        default=DEFAULT_TRAINING_UPDATES,
        help=(
            "optimizer updates in the one-step half-cosine training phase "
            f"(default: {DEFAULT_TRAINING_UPDATES})"
        ),
    )
    parser.add_argument(
        "--updates",
        "--finetune-updates",
        dest="finetune_updates",
        type=int,
        default=DEFAULT_FINETUNE_UPDATES,
        help=(
            "optimizer updates shared equally across rollout horizons 2 "
            "through 12 in autoregressive fine-tuning; must be divisible "
            f"by {CURRICULUM_STAGE_COUNT} "
            f"(default: {DEFAULT_FINETUNE_UPDATES}); "
            "--finetune-updates is a descriptive alias"
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--save-checkpoint", required=True)
    parser.add_argument("--checkpoint-every-steps", type=int, default=20)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--stop-after-global-step", type=int, default=0)
    parser.add_argument("--validation-corpus-manifest", default="")
    parser.add_argument(
        "--validation-mode",
        choices=("quick", "full"),
        default="full",
    )
    parser.add_argument("--validation-steps", type=int, default=2)
    parser.add_argument("--lat-start", type=int, default=0)
    parser.add_argument("--lat-count", type=int, default=0)
    parser.add_argument("--lon-start", type=int, default=0)
    parser.add_argument("--lon-count", type=int, default=0)
    parser.add_argument("--mesh-size", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--profile-hotspots", action="store_true")
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=4)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--profile-top-k", type=int, default=1000)
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profile-staging-dir", default="")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    # Each DP replica owns one complete model on two or four devices.
    # The global micro-batch may contain one or more samples per replica; the
    # launcher validates divisibility and derives gradient accumulation.
    allowed_topologies = {(1, 2), (2, 2), (1, 4), (2, 4)}
    selected_topology = (
        args.data_parallel_replicas,
        args.mp_devices_per_replica,
    )
    topology_description = " or ".join(
        f"DP={dp}, MP={mp}"
        for dp, mp in sorted(allowed_topologies)
    )
    for name in (
        "epochs",
        "batch_size",
        "micro_batch_size",
        "rollout_steps",
        "data_parallel_replicas",
        "mp_devices_per_replica",
        "checkpoint_every_steps",
        "keep_last",
        "validation_steps",
        "warmup_updates",
        "training_updates",
        "finetune_updates",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.rollout_checkpoint_segments < 0:
        raise ValueError("--rollout-checkpoint-segments must be non-negative")
    if args.mesh_edge_chunk_size < 0:
        raise ValueError("--mesh-edge-chunk-size must be non-negative")
    if args.edge_chunk_size < 0:
        raise ValueError("--edge-chunk-size must be non-negative")
    if args.grid_node_chunk_size < 0:
        raise ValueError("--grid-node-chunk-size must be non-negative")
    if args.loss_grid_chunk_size < 0:
        raise ValueError("--loss-grid-chunk-size must be non-negative")
    if (
        not math.isfinite(args.split_cache_max_gib)
        or args.split_cache_max_gib < 0.0
    ):
        raise ValueError("--split-cache-max-gib must be finite and non-negative")
    if (
        args.gradient_sync == "ddp"
        and args.rollout_checkpoint_implementation == "reentrant"
    ):
        raise ValueError("DDP requires non_reentrant rollout checkpointing")
    if (
        args.grid2mesh_stage_checkpoint == "enabled"
        and not args.activation_checkpoint
    ):
        raise ValueError(
            "--grid2mesh-stage-checkpoint=enabled requires "
            "--activation-checkpoint"
        )
    if (
        args.mesh2grid_stage_checkpoint == "enabled"
        and not args.activation_checkpoint
    ):
        raise ValueError(
            "--mesh2grid-stage-checkpoint=enabled requires "
            "--activation-checkpoint"
        )
    if args.batch_size % args.data_parallel_replicas:
        raise ValueError("global batch size must be divisible by DP replicas")
    if args.micro_batch_size % args.data_parallel_replicas:
        raise ValueError(
            "global micro-batch size must be divisible by DP replicas"
        )
    if args.batch_size % args.micro_batch_size:
        raise ValueError(
            "effective global batch size must be divisible by micro-batch size"
        )
    if args.stop_after_global_step < 0:
        raise ValueError("--stop-after-global-step must be non-negative")
    if args.schedule_recipe == "manifest":
        if not args.schedule_manifest:
            raise ValueError(
                "--schedule-manifest is required when "
                "--schedule-recipe=manifest"
            )
        if args.micro_batch_size != args.batch_size:
            raise ValueError(
                "manifest epoch schedules currently require "
                "--micro-batch-size to equal --batch-size"
            )
    elif args.schedule_manifest:
        raise ValueError(
            "--schedule-manifest cannot be combined with the built-in "
            "official_2_to_12 recipe"
        )
    if (
        args.schedule_recipe == OFFICIAL_2_TO_12_RECIPE
        and args.training_stage != "era5_pretrain"
    ):
        raise ValueError(
            "official_2_to_12 is the ERA5 base-training recipe; "
            "HRES adaptation must use its separately approved schedule"
        )
    if args.schedule_recipe == OFFICIAL_2_TO_12_RECIPE:
        if selected_topology not in allowed_topologies:
            raise ValueError(
                f"the {PLATFORM_LABEL} full-model training topology must be "
                f"{topology_description}"
            )
        if args.rollout_steps != 12:
            raise ValueError(
                "official_2_to_12 requires --rollout-steps=12 because that "
                "is the maximum trajectory stored by the corpus"
            )
        if args.mesh_size is not None:
            raise ValueError(
                "official_2_to_12 forbids --mesh-size overrides; "
                "reduced-budget runs must keep the complete operational model"
            )
        if not args.activation_checkpoint:
            raise ValueError(
                "official_2_to_12 full-model training requires "
                "--activation-checkpoint"
            )
        if args.rollout_checkpoint_segments != args.rollout_steps:
            raise ValueError(
                "official_2_to_12 requires one gradient-checkpoint segment "
                "per maximum rollout step"
            )
        if any(
            value != 0
            for value in (
                args.lat_start,
                args.lat_count,
                args.lon_start,
                args.lon_count,
            )
        ):
            raise ValueError(
                "official_2_to_12 forbids spatial crops; reduced-budget runs "
                "must not reduce the full global model/grid"
            )
    if (
        args.schedule_recipe == OFFICIAL_2_TO_12_RECIPE
        and args.finetune_updates % CURRICULUM_STAGE_COUNT
    ):
        raise ValueError(
            "--updates must be divisible by "
            f"{CURRICULUM_STAGE_COUNT} so rollout horizons "
            f"{CURRICULUM_ROLLOUT_START} through "
            f"{CURRICULUM_ROLLOUT_END} receive equal update budgets"
        )
    if args.allow_pretrained_era5_init and (
        args.init_mode != "checkpoint"
        or args.training_stage != "era5_pretrain"
        or args.resume_training_checkpoint
    ):
        raise ValueError(
            "--allow-pretrained-era5-init requires a new ERA5 checkpoint-initialized run"
        )
    if args.resume_training_checkpoint:
        if args.init_checkpoint:
            raise ValueError(
                "exact resume and model-only initialization are mutually "
                "exclusive"
            )
    elif args.init_mode == "checkpoint" and not args.init_checkpoint:
        raise ValueError(
            "checkpoint initialization requires --init-checkpoint"
        )
    elif args.init_mode == "random" and args.init_checkpoint:
        raise ValueError(
            "--init-checkpoint requires checkpoint initialization"
        )
    if args.training_stage == "era5_pretrain" and args.dataset_kind != "era5":
        raise ValueError("ERA5 pretraining requires an ERA5 corpus")
    if args.training_stage == "hres_finetune" and args.dataset_kind != "hres":
        raise ValueError("HRES fine-tuning requires an HRES corpus")
    if not math.isfinite(args.peak_tflops_per_device) or args.peak_tflops_per_device < 0:
        raise ValueError("--peak-tflops-per-device must be finite and non-negative")
    if args.peak_tflops_per_device > 0 and args.peak_tflops_source in ("", "unavailable", "unknown"):
        raise ValueError("a positive peak requires --peak-tflops-source")
    if args.formal:
        if args.epochs < 1:
            raise ValueError("formal training requires a positive data cycle")
        if args.micro_batch_size < args.data_parallel_replicas:
            raise ValueError(
                "formal training requires at least one sample per DP replica"
            )
        if args.rollout_steps != 12:
            raise ValueError("formal training requires a 12-step rollout")
        if selected_topology not in allowed_topologies:
            raise ValueError(
                f"formal {PLATFORM_LABEL} topology must be "
                f"{topology_description}"
            )
        if any(
            value != 0
            for value in (
                args.lat_start,
                args.lat_count,
                args.lon_start,
                args.lon_count,
            )
        ):
            raise ValueError("formal training forbids a spatial crop")
        if (
            not args.resume_training_checkpoint
            and args.training_stage == "era5_pretrain"
            and args.init_mode != "random"
            and not args.allow_pretrained_era5_init
        ):
            raise ValueError(
                "formal ERA5 pretraining must start from the fixed seed"
            )
        if (
            not args.resume_training_checkpoint
            and args.training_stage == "hres_finetune"
            and args.init_mode != "checkpoint"
        ):
            raise ValueError(
                "formal HRES adaptation requires an ERA5 checkpoint"
            )


def resolve_training_schedule(
    args: argparse.Namespace,
) -> tuple[TrainingSchedule, str]:
    """Resolve the immutable schedule and its provenance string.

    The built-in recipe keeps a 12-step dataset window throughout the run.
    Warmup and one-step training consume only lead 1 from that window; the
    fine-tuning phase consumes 2, 3, ..., 12 leads in sequence, assigning the
    same number of updates to every horizon. This preserves one
    corpus/checkpoint contract while changing the active backpropagation
    horizon at documented phase boundaries.
    """

    if args.schedule_recipe == "manifest":
        path = Path(args.schedule_manifest).expanduser().resolve()
        return load_schedule_manifest(path), str(path)
    updates_per_rollout = (
        args.finetune_updates // CURRICULUM_STAGE_COUNT
    )
    schedule = TrainingSchedule.from_dict(
        {
            "schema_version": 1,
            "name": OFFICIAL_2_TO_12_RECIPE,
            "phases": [
                {
                    "name": "one_step_warmup",
                    "updates": args.warmup_updates,
                    "learning_rate": {
                        "kind": "linear",
                        "start": 0.0,
                        "end": 1.0e-3,
                    },
                    "rollout": {"kind": "constant", "value": 1},
                },
                {
                    "name": "one_step_half_cosine",
                    "updates": args.training_updates,
                    "learning_rate": {
                        "kind": "half_cosine",
                        "start": 1.0e-3,
                        "end": 0.0,
                    },
                    "rollout": {"kind": "constant", "value": 1},
                },
                {
                    "name": "rollout_curriculum_2_to_12",
                    "updates": args.finetune_updates,
                    "learning_rate": {
                        "kind": "constant",
                        "value": 3.0e-7,
                    },
                    "rollout": {
                        "kind": "increment",
                        "start": CURRICULUM_ROLLOUT_START,
                        "every_updates": updates_per_rollout,
                        "maximum": CURRICULUM_ROLLOUT_END,
                    },
                },
            ],
        }
    )
    return schedule, f"builtin:{OFFICIAL_2_TO_12_RECIPE}"


# ==================================================
# 2. 运行目录、状态文件与安全写入
# ==================================================
def _atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _acquire_run_lock(run_dir: Path) -> tuple[Path, str] | None:
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    path = run_dir / ".run.lock"
    value = json.dumps(
        {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at_unix": time.time(),
        },
        sort_keys=True,
    )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        owner = path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"run directory is locked by another/stale process: {owner}"
        ) from error
    try:
        os.write(descriptor, (value + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path, value


def _release_run_lock(lock: tuple[Path, str] | None) -> None:
    if lock is None:
        return
    path, expected = lock
    if not path.exists():
        return
    actual = path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise RuntimeError("run lock ownership changed unexpectedly")
    path.unlink()


def _environment_contract(context: DistributedContext) -> dict[str, Any]:
    accelerator = accelerator_environment(
        participating_device_count=context.participating_device_count,
        accelerator_type=context.accelerator_type,
    )
    return {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": (
            torch.backends.cudnn.version()
            if context.accelerator_type == "cuda"
            and torch.backends.cudnn.is_available()
            else None
        ),
        **accelerator,
        "distributed_backend": context.backend,
        "object_collective_backend": context.object_collective_backend,
    }


def _write_heartbeat(
    run_dir: Path,
    *,
    state: str,
    global_optimizer_step: int,
    epoch: int | None = None,
    next_batch_index: int | None = None,
) -> None:
    _atomic_json_dump(
        run_dir / "heartbeat.json",
        {
            "state": state,
            "global_optimizer_step": global_optimizer_step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "updated_at_unix": time.time(),
        },
    )


def _write_state_marker(run_dir: Path, state: str, **values: Any) -> None:
    for old_state in ("RUNNING", "STOPPED", "COMPLETED", "FAILED"):
        (run_dir / old_state).unlink(missing_ok=True)
    _atomic_json_dump(
        run_dir / state,
        {
            "state": state,
            "updated_at_unix": time.time(),
            **values,
        },
    )


# ==================================================
# 3. 模型、loss 与张量辅助函数
# ==================================================
def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


class EpochAutoregressiveModule(torch.nn.Module):
    """Expose complete loss tensors through one DDP forward."""

    def __init__(
        self,
        model: GoogleSmallCompatibleModel,
        *,
        checkpoint_segments: int,
        checkpoint_single_step: bool,
        checkpoint_implementation: str = "non_reentrant",
        loss_grid_chunk_size: int = 0,
    ):
        super().__init__()
        self.model = model
        self.checkpoint_segments = checkpoint_segments
        self.checkpoint_single_step = checkpoint_single_step
        self.checkpoint_implementation = checkpoint_implementation
        self.loss_grid_chunk_size = loss_grid_chunk_size

    def forward(
        self,
        batch: Any,
        *,
        collect_variable_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_checkpoint_segments = (
            self.checkpoint_segments
            if self.checkpoint_single_step or batch.rollout_steps != 1
            else 0
        )
        result = autoregressive_loss(
            self.model,
            batch,
            checkpoint_segments=active_checkpoint_segments,
            checkpoint_implementation=self.checkpoint_implementation,
            loss_grid_chunk_size=self.loss_grid_chunk_size,
            collect_variable_diagnostics=collect_variable_diagnostics,
        )
        # Per-variable diagnostics are available through autoregressive_loss,
        # but are not needed for the optimizer's weighted scalar loss.
        per_variable = (
            torch.stack(tuple(result.per_variable_lead_losses.values()), dim=1)
            if result.per_variable_lead_losses
            else result.lead_losses.new_empty((batch.rollout_steps, 0))
        )
        return result.loss, result.lead_losses, per_variable


def sample_provenance(
    sample: Any,
    *,
    rollout_steps: int | None = None,
) -> dict[str, Any]:
    return training_sample_provenance(
        sample,
        rollout_steps=rollout_steps,
    )


def merge_microbatch_provenance(
    chunks: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge per-rank provenance from sequential gradient micro-batches."""

    if not chunks:
        raise ValueError("micro-batch provenance must be non-empty")
    rank_count = len(chunks[0])
    if rank_count <= 0 or any(len(chunk) != rank_count for chunk in chunks):
        raise ValueError("micro-batch provenance rank counts differ")
    list_fields = (
        "local_window_indices",
        "global_window_indices",
        "sample_weights",
        "global_sample_weights",
        "reference_datetimes",
        "target_datetimes",
        "target_datetimes_by_sample",
    )
    count_fields = (
        "valid_sample_count",
        "padded_sample_count",
        "global_valid_sample_count",
        "global_padded_sample_count",
    )
    merged = []
    for rank_index in range(rank_count):
        rows = [chunk[rank_index] for chunk in chunks]
        value = dict(rows[0])
        for field in list_fields:
            value[field] = [
                item
                for row in rows
                for item in row[field]
            ]
        for field in count_fields:
            value[field] = sum(int(row[field]) for row in rows)
        merged.append(value)
    return merged


def _weighted_distributed_tensor(
    values: torch.Tensor,
    *,
    local_valid_count: int,
    global_valid_count: int,
    context: DistributedContext,
) -> list[float]:
    return [
        context.global_sum(float(value) * local_valid_count)
        / global_valid_count
        for value in values.detach().cpu().reshape(-1)
    ]


# ==================================================
# 4. 完整验证与样本覆盖率
# ==================================================
def _validation_dataset(
    args: argparse.Namespace,
) -> OfficialTrainingCorpus | None:
    if not args.validation_corpus_manifest:
        return None
    dataset = OfficialTrainingCorpus(
        args.validation_corpus_manifest,
        config_path=args.config,
        rollout_steps=args.rollout_steps,
        lat_start=args.lat_start,
        lat_count=args.lat_count,
        lon_start=args.lon_start,
        lon_count=args.lon_count,
    )
    index = dataset.manifest
    if (
        index.dataset_kind != "hres"
        or index.training_stage != "validation"
        or index.data_role != "validation"
    ):
        dataset.close()
        raise ValueError("validation corpus role is invalid")
    return dataset


def evaluate_validation(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    wrapped_unroll: torch.nn.Module,
    base_model: GoogleSmallCompatibleModel,
    config: GoogleSmallConfig,
    stats: Any,
    input_device: torch.device,
) -> dict[str, Any] | None:
    resolved = _validation_dataset(args)
    if resolved is None:
        return None
    validation = resolved
    validate_formal_dataset(validation, formal=args.formal)
    plan = EpochPlan.create(
        validation.window_indices,
        epoch=0,
        seed=args.seed,
        # Evaluation has no optimizer update and therefore does not need to
        # materialize the training effective global batch. Reuse the bounded
        # global micro-batch so the complete model/grid remains unchanged.
        global_batch_size=args.micro_batch_size,
        shuffle=False,
    )
    step_count = (
        plan.optimizer_steps
        if args.validation_mode == "full"
        else min(args.validation_steps, plan.optimizer_steps)
    )
    rows = []
    variable_names: tuple[str, ...] | None = None
    base_model.eval()
    try:
        with torch.no_grad():
            for batch_index in range(step_count):
                planned = plan.batches[batch_index]
                sample = validation.sample_indices(
                    window_indices=planned.window_indices,
                    sample_weights=planned.sample_weights,
                    rank=context.rank,
                    replicas=context.dp_replicas,
                )
                batch = prepare_autoregressive_batch(
                    sample.inputs,
                    sample.targets,
                    sample.forcings,
                    config=config,
                    stats=stats,
                    input_device=input_device,
                    per_variable_weights=PER_VARIABLE_WEIGHTS,
                    sample_weights=sample.sample_weights,
                )
                if variable_names is None:
                    variable_names = tuple(batch.target_channel_slices)
                synchronize(context.device_indices)
                started = time.perf_counter()
                loss, lead_tensor, variable_tensor = wrapped_unroll(
                    batch, collect_variable_diagnostics=True
                )
                synchronize(context.device_indices)
                seconds = context.global_max_seconds(
                    time.perf_counter() - started
                )
                local_valid = sample.valid_sample_count
                global_valid = sample.global_valid_sample_count
                global_loss = (
                    context.global_sum(
                        float(loss.detach().item()) * local_valid
                    )
                    / global_valid
                )
                lead_losses = _weighted_distributed_tensor(
                    lead_tensor,
                    local_valid_count=local_valid,
                    global_valid_count=global_valid,
                    context=context,
                )
                flattened_variables = _weighted_distributed_tensor(
                    variable_tensor,
                    local_valid_count=local_valid,
                    global_valid_count=global_valid,
                    context=context,
                )
                variable_matrix = np.asarray(
                    flattened_variables,
                    dtype=np.float64,
                ).reshape(args.rollout_steps, -1)
                origins = context.gather_objects(
                    sample_provenance(sample)
                )
                if context.is_primary:
                    rows.append(
                        {
                            "batch_index": batch_index,
                            "loss": global_loss,
                            "lead_losses": lead_losses,
                            "per_variable_lead_losses": {
                                name: variable_matrix[:, index].tolist()
                                for index, name in enumerate(
                                    variable_names or ()
                                )
                            },
                            "seconds": seconds,
                            "valid_sample_count": global_valid,
                            "padded_sample_count": (
                                sample.global_padded_sample_count
                            ),
                            "origins_by_rank": origins,
                        }
                    )
                del batch, loss, lead_tensor, variable_tensor, sample
    finally:
        validation.close()
        base_model.train()
    if not context.is_primary:
        return None
    total_valid = sum(row["valid_sample_count"] for row in rows)
    mean_loss = sum(
        row["loss"] * row["valid_sample_count"] for row in rows
    ) / total_valid
    valid_indices = [
        index
        for row in rows
        for origin in row["origins_by_rank"]
        for index, weight in zip(
            origin["local_window_indices"],
            origin["sample_weights"],
            strict=True,
        )
        if weight > 0.0
    ]
    result = {
        "dataset": str(validation.path),
        "data_role": "validation",
        "mode": args.validation_mode,
        "batching": "bounded_global_micro_batch_without_gradient",
        "global_batch_size": args.micro_batch_size,
        "training_effective_global_batch_size": args.batch_size,
        "optimizer_updates": 0,
        "gradient_enabled": False,
        "steps": rows,
        "mean_loss": float(mean_loss),
        "valid_sample_count": total_valid,
        "padded_sample_count": sum(
            row["padded_sample_count"] for row in rows
        ),
        "unique_window_count": len(set(valid_indices)),
        "duplicate_valid_count": len(valid_indices) - len(set(valid_indices)),
        "coverage": (
            len(set(valid_indices)) / validation.window_count
        ),
        "validation_self_check": (
            math.isfinite(mean_loss)
            and (
                args.validation_mode != "full"
                or (
                    len(set(valid_indices)) == validation.window_count
                    and len(valid_indices) == validation.window_count
                )
            )
        ),
    }
    if not result["validation_self_check"]:
        raise RuntimeError("validation coverage or finite-loss check failed")
    return result


def _coverage_summary(
    metrics: list[dict[str, Any]],
    *,
    epochs: int,
    expected_windows: int,
) -> dict[str, Any]:
    per_epoch = []
    for epoch in range(epochs):
        indices = [
            index
            for row in metrics
            if int(row["epoch"]) == epoch
            for origin in row["origins_by_rank"]
            for index, weight in zip(
                origin["local_window_indices"],
                origin["sample_weights"],
                strict=True,
            )
            if weight > 0.0
        ]
        unique = len(set(indices))
        per_epoch.append(
            {
                "epoch": epoch,
                "valid_sample_count": len(indices),
                "unique_window_count": unique,
                "duplicate_valid_count": len(indices) - unique,
                "coverage": unique / expected_windows,
                "complete": (
                    len(indices) == expected_windows
                    and unique == expected_windows
                ),
            }
        )
    return {
        "expected_windows_per_epoch": expected_windows,
        "epochs": per_epoch,
        "all_epochs_complete": all(row["complete"] for row in per_epoch),
    }


def _replacement_sampling_summary(
    metrics: list[dict[str, Any]],
    *,
    available_windows_by_rollout: dict[int, int],
    global_batch_size: int,
    target_optimizer_updates: int,
) -> dict[str, Any]:
    """Summarize the paper's update-driven, with-replacement sampler."""

    indices = [
        index
        for row in metrics
        for origin in row["origins_by_rank"]
        for index, weight in zip(
            origin["local_window_indices"],
            origin["sample_weights"],
            strict=True,
        )
        if weight > 0.0
    ]
    unique = len(set(indices))
    per_rollout = []
    for rollout_steps, available_count in sorted(
        available_windows_by_rollout.items()
    ):
        rollout_indices = [
            index
            for row in metrics
            if int(row["rollout_steps"]) == rollout_steps
            for origin in row["origins_by_rank"]
            for index, weight in zip(
                origin["local_window_indices"],
                origin["sample_weights"],
                strict=True,
            )
            if weight > 0.0
        ]
        rollout_unique = len(set(rollout_indices))
        per_rollout.append(
            {
                "rollout_steps": rollout_steps,
                "available_window_count": available_count,
                "draw_count": len(rollout_indices),
                "unique_window_count": rollout_unique,
                "duplicate_draw_count": (
                    len(rollout_indices) - rollout_unique
                ),
                "observed_window_fraction": (
                    rollout_unique / available_count
                ),
            }
        )
    expected_draws = len(metrics) * global_batch_size
    per_update_batch_complete = all(
        int(row["valid_sample_count"]) == global_batch_size
        and int(row["padded_sample_count"]) == 0
        and row["training_sampling"]
        == "trajectory_with_replacement_per_optimizer_update"
        for row in metrics
    )
    update_budget_complete = len(metrics) == target_optimizer_updates
    return {
        "sampling_unit": "optimizer_update",
        "replacement": True,
        "global_batch_size": global_batch_size,
        "target_optimizer_updates": target_optimizer_updates,
        "completed_optimizer_updates": len(metrics),
        "expected_draw_count": expected_draws,
        "draw_count": len(indices),
        "unique_window_count": unique,
        "duplicate_draw_count": len(indices) - unique,
        "available_window_count_by_rollout": {
            str(key): value
            for key, value in sorted(
                available_windows_by_rollout.items()
            )
        },
        "per_rollout": per_rollout,
        "union_available_window_count": max(
            available_windows_by_rollout.values()
        ),
        "union_observed_window_fraction": (
            unique / max(available_windows_by_rollout.values())
        ),
        "per_update_batch_complete": per_update_batch_complete,
        "update_budget_complete": update_budget_complete,
        "sampling_self_check": (
            len(indices) == expected_draws
            and per_update_batch_complete
            and update_budget_complete
        ),
    }


def _official_update_data_request(
    *,
    global_step: int,
    epoch: int,
    batch_index: int,
    rollout_steps: int,
    rank: int,
    replicas: int,
    allow_replacement: bool,
    global_positions: tuple[int, ...] | list[int],
    global_sample_weights: tuple[float, ...] | list[float],
    microbatch_plan: MicrobatchPlan,
) -> OfficialUpdateDataRequest:
    """Freeze one update's existing sampling plan for the CPU producer."""

    return OfficialUpdateDataRequest(
        key=OfficialUpdateDataKey(
            global_step=global_step,
            epoch=epoch,
            batch_index=batch_index,
            rollout_steps=rollout_steps,
            rank=rank,
            replicas=replicas,
            allow_replacement=allow_replacement,
            microbatches=tuple(
                OfficialMicrobatchDataRequest(
                    index=part.index,
                    window_indices=tuple(
                        global_positions[part.batch_slice]
                    ),
                    sample_weights=tuple(
                        global_sample_weights[part.batch_slice]
                    ),
                )
                for part in microbatch_plan.parts
            ),
        ),
    )


def _next_official_update_data_request(
    *,
    current_global_step: int,
    current_epoch: int,
    current_batch_index: int,
    current_plan: EpochPlan,
    training_epoch_limit: int,
    planned_optimizer_updates: int,
    target_optimizer_updates: int | None,
    stop_after_global_step: int,
    dataset: OfficialTrainingCorpus,
    schedule: TrainingSchedule,
    seed: int,
    global_batch_size: int,
    microbatch_plan: MicrobatchPlan,
    rank: int,
    replicas: int,
) -> OfficialUpdateDataRequest | None:
    """Derive the next request without advancing the training cursor."""

    if current_global_step >= planned_optimizer_updates:
        return None
    if (
        stop_after_global_step > 0
        and current_global_step >= stop_after_global_step
    ):
        return None
    if current_batch_index + 1 < current_plan.optimizer_steps:
        next_epoch = current_epoch
        next_batch_index = current_batch_index + 1
        next_plan = current_plan
    elif current_epoch + 1 < training_epoch_limit:
        next_epoch = current_epoch + 1
        next_batch_index = 0
        next_plan = EpochPlan.create(
            dataset.window_indices,
            epoch=next_epoch,
            seed=seed,
            global_batch_size=global_batch_size,
            shuffle=True,
        )
    else:
        return None

    # ``value_for_update`` and replacement sampling both take a zero-based
    # optimizer-update index.  The current one-based global step is exactly
    # the next update's zero-based index.
    next_schedule_value = schedule.value_for_update(current_global_step)
    next_rollout_steps = next_schedule_value.rollout_steps
    if target_optimizer_updates is not None:
        next_positions = dataset.replacement_indices(
            update_index=current_global_step,
            batch_size=global_batch_size,
            seed=seed,
            target_rollout_steps=next_rollout_steps,
        )
        next_weights = (1.0,) * global_batch_size
        allow_replacement = True
    else:
        planned = next_plan.batches[next_batch_index]
        next_positions = planned.window_indices
        next_weights = planned.sample_weights
        allow_replacement = False
    return _official_update_data_request(
        global_step=current_global_step + 1,
        epoch=next_epoch,
        batch_index=next_batch_index,
        rollout_steps=next_rollout_steps,
        rank=rank,
        replicas=replicas,
        allow_replacement=allow_replacement,
        global_positions=next_positions,
        global_sample_weights=next_weights,
        microbatch_plan=microbatch_plan,
    )


# ==================================================
# 5. 分布式真实训练主流程
# ==================================================
def train(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    startup_started = time.perf_counter()
    _rank_zero_progress(
        "official_graphcast_init",
        stage="run_contract",
        status="start",
        run_dir=run_dir,
    )
    context: DistributedContext | None = None
    dataset: OfficialTrainingCorpus | None = None
    run_lock: tuple[Path, str] | None = None
    try:
        run_lock = _acquire_run_lock(run_dir)
        schedule, schedule_source = resolve_training_schedule(args)
        first_schedule_value = schedule.value_for_update(0)
        if first_schedule_value.rollout_steps > args.rollout_steps:
            raise ValueError(
                "schedule rollout exceeds the dataset rollout window"
            )
        tensor_core_enabled = args.tensor_core == "enabled"
        math_mode = configure_float32_math(tensor_core_enabled)
        _rank_zero_progress(
            "official_graphcast_init",
            stage="run_contract",
            status="ready",
            elapsed_s=f"{time.perf_counter() - startup_started:.3f}",
        )
        stage_started = time.perf_counter()
        _rank_zero_progress(
            "official_graphcast_init",
            stage="distributed_context",
            status="start",
            data_parallel_replicas=args.data_parallel_replicas,
            model_parallel_devices=args.mp_devices_per_replica,
        )
        context = DistributedContext.from_environment(
            dp_replicas=args.data_parallel_replicas,
            mp_devices_per_replica=args.mp_devices_per_replica,
        )
        serial_musa_backward = (
            context.accelerator_type == "musa"
            and context.mp_devices_per_replica > 1
        )
        prewarmed_collective_devices = (
            context.prewarm_device_collectives()
        )
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="distributed_context",
            status="ready",
            tensor_backend=context.backend,
            object_backend=context.object_collective_backend,
            object_group_expected_peers=context.world_size - 1,
            devices=",".join(str(index) for index in context.device_indices),
            prewarmed_collective_devices=prewarmed_collective_devices,
            backward_execution=(
                "serial_musa" if serial_musa_backward else "runtime_default"
            ),
            elapsed_s=f"{time.perf_counter() - stage_started:.3f}",
        )
        if context.is_primary:
            _write_state_marker(run_dir, "RUNNING")
        context.barrier()
        rank_dir = run_dir / f"rank_{context.rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
            if context.accelerator_type == "cuda":
                torch.backends.cudnn.benchmark = False
        reset_peak_memory(context.device_indices)
        input_device = accelerator_device(
            context.device_indices[0],
            context.accelerator_type,
        )
        mesh_first_device = input_device
        secondary_device = accelerator_device(
            context.device_indices[1],
            context.accelerator_type,
        )
        # MP=2 aliases the four logical stages onto two physical devices.
        output_device = secondary_device
        if args.mp_devices_per_replica == 4:
            mesh_first_device = accelerator_device(
                context.device_indices[1],
                context.accelerator_type,
            )
            secondary_device = accelerator_device(
                context.device_indices[2],
                context.accelerator_type,
            )
            output_device = accelerator_device(
                context.device_indices[3],
                context.accelerator_type,
            )
        flagos_runtime = enable_network_replacements_from_env(
            device=input_device
        )
        if args.formal and flagos_runtime.skipped:
            raise ValueError(
                "formal training cannot skip unknown FlagOS operators: "
                + ",".join(flagos_runtime.skipped)
            )
        operator_mode = (
            "flaggems"
            if flagos_runtime.enabled
            else "torch_native"
        )
        operator_replacement = {
            "mode": operator_mode,
            "enabled": flagos_runtime.enabled,
            "operators": list(flagos_runtime.operators),
            "skipped": list(flagos_runtime.skipped),
            "runtime": flagos_runtime.metadata,
        }
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="operator_replacement",
            status="ready",
            mode=operator_mode,
            operators=(
                ",".join(flagos_runtime.operators)
                if flagos_runtime.operators
                else "none"
            ),
            skipped=(
                ",".join(flagos_runtime.skipped)
                if flagos_runtime.skipped
                else "none"
            ),
        )
        json_dump(
            rank_dir / "topology.json",
            {
                "rank": context.rank,
                "world_size": context.world_size,
                "device_indices": list(context.device_indices),
                "input_device": str(input_device),
                "mesh_first_device": str(mesh_first_device),
                "secondary_device": str(secondary_device),
                "output_device": str(output_device),
                "tensor_collective_backend": context.backend,
                "object_collective_backend": (
                    context.object_collective_backend
                ),
                "operator_replacement": operator_replacement,
            },
        )

        stage_started = time.perf_counter()
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="dataset",
            status="start",
            split_start=args.training_start or "manifest_start",
            split_end=args.training_end or "manifest_end",
            max_rollout=args.rollout_steps,
        )
        dataset = OfficialTrainingCorpus(
            args.corpus_manifest,
            config_path=args.config,
            rollout_steps=args.rollout_steps,
            split_start=args.training_start or None,
            split_end=args.training_end or None,
            lat_start=args.lat_start,
            lat_count=args.lat_count,
            lon_start=args.lon_start,
            lon_count=args.lon_count,
            split_cache_max_gib=args.split_cache_max_gib,
        )
        manifest = dataset.manifest
        if (
            manifest.dataset_kind != args.dataset_kind
            or manifest.training_stage != args.training_stage
            or manifest.data_role != "train"
        ):
            raise ValueError(
                "training corpus role disagrees with "
                "--dataset-kind/--training-stage"
            )
        validate_formal_dataset(dataset, formal=args.formal)
        available_windows_by_rollout = {
            rollout: dataset.window_count_for_rollout(rollout)
            for rollout in range(1, args.rollout_steps + 1)
        }
        microbatch_plan = MicrobatchPlan.create(
            global_batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
        )
        local_batch_size = context.local_batch_size(
            global_batch_size=microbatch_plan.micro_batch_size
        )
        initial_plan = EpochPlan.create(
            dataset.window_indices,
            epoch=0,
            seed=args.seed,
            global_batch_size=args.batch_size,
            shuffle=True,
        )
        target_optimizer_updates = (
            schedule.total_updates
            if args.schedule_recipe == OFFICIAL_2_TO_12_RECIPE
            else None
        )
        if target_optimizer_updates is not None:
            # EpochPlan is retained only as an exact-resume cursor container.
            # Official batches ignore its ordering and are sampled with
            # replacement by optimizer-update index.
            training_epoch_limit = math.ceil(
                target_optimizer_updates / initial_plan.optimizer_steps
            )
            run_mode = "official_2_to_12_updates"
        else:
            training_epoch_limit = args.epochs
            run_mode = "epoch_rehearsal"
        planned_optimizer_updates = (
            target_optimizer_updates
            if target_optimizer_updates is not None
            else training_epoch_limit * initial_plan.optimizer_steps
        )
        schedule_phase_updates = tuple(
            phase["updates"]
            for phase in schedule.to_dict()["phases"]
        )
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="dataset",
            status="ready",
            windows=len(dataset.window_indices),
            windows_by_rollout=",".join(
                f"{rollout}:{count}"
                for rollout, count in (
                    available_windows_by_rollout.items()
                )
            ),
            sampling=(
                "with_replacement"
                if target_optimizer_updates is not None
                else "without_replacement_epoch"
            ),
            global_batch_size=args.batch_size,
            global_micro_batch_size=microbatch_plan.micro_batch_size,
            local_micro_batch_size=local_batch_size,
            gradient_accumulation_steps=(
                microbatch_plan.accumulation_steps
            ),
            planned_updates=planned_optimizer_updates,
            split_cache_enabled=dataset.split_cache_status["enabled"],
            split_cache_loaded_bytes=dataset.split_cache_status[
                "loaded_bytes"
            ],
            split_cache_preload_s=(
                f"{dataset.split_cache_status['preload_seconds']:.6f}"
            ),
            split_cache_reason=dataset.split_cache_status["reason"],
            data_prefetch_depth=args.data_prefetch_depth,
            elapsed_s=f"{time.perf_counter() - stage_started:.3f}",
        )
        resume_contract = {
            "graph_topology_version": CANONICAL_GRAPH_TOPOLOGY_VERSION,
            "training_stage": args.training_stage,
            "dataset_kind": args.dataset_kind,
            "corpus_index": str(manifest.path),
            "config": str(
                Path(args.config).expanduser().resolve()
            ),
            "stats_dir": str(
                Path(args.stats_dir).expanduser().resolve()
            ),
            "validation_corpus_index": (
                str(
                    Path(args.validation_corpus_manifest)
                    .expanduser()
                    .resolve()
                )
                if args.validation_corpus_manifest
                else None
            ),
            "schedule_digest": schedule.digest,
            "schedule_recipe": args.schedule_recipe,
            "schedule_source": schedule_source,
            "target_optimizer_updates": target_optimizer_updates,
            "official_data_utils_reference": (
                dataset.official_data_utils_reference
            ),
            "target_materialization": (
                "two_inputs_plus_active_target_lead_times_before_load"
            ),
            "available_windows_by_rollout": {
                str(key): value
                for key, value in (
                    available_windows_by_rollout.items()
                )
            },
            "precision": "fp32",
            "tensor_core_enabled": tensor_core_enabled,
            "global_batch_size": args.batch_size,
            "global_micro_batch_size": microbatch_plan.micro_batch_size,
            "local_micro_batch_size": local_batch_size,
            "gradient_accumulation_steps": (
                microbatch_plan.accumulation_steps
            ),
            "training_sampling": (
                "trajectory_with_replacement_per_optimizer_update"
                if target_optimizer_updates is not None
                else "deterministic_without_replacement_epoch"
            ),
            "data_parallel_replicas": context.dp_replicas,
            "mp_devices_per_replica": context.mp_devices_per_replica,
            "gradient_sync": args.gradient_sync,
            "operator_replacement": replacement_resume_policy(operator_replacement),
            "rollout_steps": args.rollout_steps,
            "rollout_checkpoint_segments": (
                args.rollout_checkpoint_segments
            ),
            "rollout_checkpoint_implementation": (
                args.rollout_checkpoint_implementation
            ),
            "single_step_rollout_checkpoint": (
                args.single_step_rollout_checkpoint
            ),
            "mesh_split_step": args.mesh_split_step,
            "mesh_checkpoint_granularity": (
                args.mesh_checkpoint_granularity
            ),
            "grid2mesh_stage_device": args.grid2mesh_stage_device,
            "grid2mesh_node_stage_device": (
                args.grid2mesh_node_stage_device
            ),
            "grid2mesh_stage_checkpoint": (
                args.grid2mesh_stage_checkpoint
            ),
            "mesh2grid_stage_device": args.mesh2grid_stage_device,
            "mesh2grid_node_stage_device": (
                args.mesh2grid_node_stage_device
            ),
            "mesh2grid_decoder_stage_device": (
                args.mesh2grid_decoder_stage_device
            ),
            "mesh2grid_stage_checkpoint": (
                args.mesh2grid_stage_checkpoint
            ),
            "mesh_edge_chunk_size_override": args.mesh_edge_chunk_size,
            "edge_chunk_size_override": args.edge_chunk_size,
            "grid_node_chunk_size_override": args.grid_node_chunk_size,
            "loss_grid_chunk_size": args.loss_grid_chunk_size,
            "activation_checkpoint": args.activation_checkpoint,
            "seed": args.seed,
            "progress_unit": (
                "optimizer_update"
                if target_optimizer_updates is not None
                else "epoch"
            ),
            "internal_data_cycle_limit": training_epoch_limit,
            "requested_epochs": (
                None
                if target_optimizer_updates is not None
                else args.epochs
            ),
            "training_start": args.training_start,
            "training_end": args.training_end,
            "lat_start": args.lat_start,
            "lat_count": args.lat_count,
            "lon_start": args.lon_start,
            "lon_count": args.lon_count,
            "mesh_size": args.mesh_size,
            "deterministic": args.deterministic,
            "validation_mode": args.validation_mode,
            "validation_steps": args.validation_steps,
        }
        if context.is_primary:
            write_immutable_json(
                run_dir / "run_config.json",
                {
                    "schema_version": 1,
                    "run_mode": run_mode,
                    "backward_execution": (
                        "serial_musa" if serial_musa_backward else "runtime_default"
                    ),
                    "corpus_index": str(manifest.path),
                    "schedule_recipe": args.schedule_recipe,
                    "schedule_source": schedule_source,
                    "source_priority": [
                        "operator_handoff_graphcast_compat",
                        "pinned_deepmind_weathernext",
                        "flagos_torch_distributed_glue",
                    ],
                    "data_pipeline": {
                        # Measurements change on every reopen; full status is
                        # preserved in the per-rank invocation observations.
                        "split_cache": {"max_gib": args.split_cache_max_gib},
                        "data_prefetch": {
                            "depth": args.data_prefetch_depth,
                            "producer_count": (
                                1 if args.data_prefetch_depth == 1 else 0
                            ),
                            "worker_device": "cpu",
                            "accelerator_transfer_thread": "training_main",
                        },
                    },
                    "resume_contract": resume_contract,
                },
            )
            write_immutable_json(
                run_dir / "schedule.json",
                {
                    **schedule.to_dict(),
                    "schedule_digest": schedule.digest,
                    "source_manifest": schedule_source,
                },
            )
            write_immutable_json(
                run_dir / "environment.json",
                _environment_contract(context),
            )
        context.barrier()
        resume_payload = None
        init_state = None
        initial_lineage: dict[str, Any]
        if args.resume_training_checkpoint:
            resume_payload = load_resume_checkpoint(
                args.resume_training_checkpoint,
                resume_contract=resume_contract,
            )
            cursor = TrainingCursor.from_dict(resume_payload["cursor"])
            if cursor.epoch >= training_epoch_limit:
                raise ValueError(
                    "resume checkpoint already completed all data cycles"
                )
            if (
                target_optimizer_updates is not None
                and cursor.global_optimizer_step >= target_optimizer_updates
            ):
                raise ValueError(
                    "resume checkpoint already completed the update recipe"
                )
            resume_plan = EpochPlan.create(
                dataset.window_indices,
                epoch=cursor.epoch,
                seed=args.seed,
                global_batch_size=args.batch_size,
                shuffle=True,
            )
            resume_plan.validate_cursor(cursor.next_batch_index)
            if cursor.epoch_plan_digest != resume_plan.digest:
                raise ValueError("resume checkpoint EpochPlan disagrees")
            init_state = resume_payload["model_state"]
            metadata = resume_payload["metadata"]
            initial_lineage = dict(metadata.get("initial_lineage", {}))
        else:
            cursor = TrainingCursor(
                epoch=0,
                next_batch_index=0,
                global_optimizer_step=0,
                epoch_plan_digest=initial_plan.digest,
            )
            if args.init_mode == "checkpoint":
                (
                    init_state,
                    init_metadata,
                    init_sha256,
                ) = load_model_only_checkpoint(
                    args.init_checkpoint,
                    formal=args.formal,
                    training_stage=args.training_stage,
                    tensor_core_enabled=tensor_core_enabled,
                )
                initial_lineage = {
                    "mode": "model_only_checkpoint",
                    "checkpoint": str(
                        Path(args.init_checkpoint).expanduser().resolve()
                    ),
                    "checkpoint_sha256": init_sha256,
                    "metadata": init_metadata,
                }
            else:
                initial_lineage = {
                    "mode": "pytorch_seeded_random",
                    "seed": args.seed,
                }

        stage_started = time.perf_counter()
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="model",
            status="start",
            mesh_size=args.mesh_size or "config_default",
            activation_checkpoint=args.activation_checkpoint,
            rollout_checkpoint_segments=args.rollout_checkpoint_segments,
            single_step_rollout_checkpoint=(
                args.single_step_rollout_checkpoint
            ),
        )
        config = GoogleSmallConfig.from_yaml(args.config)
        if args.mesh_size is not None:
            config = replace(config, mesh_size=args.mesh_size)
        if args.mesh_edge_chunk_size:
            config = replace(
                config,
                mesh_edge_chunk_size=args.mesh_edge_chunk_size,
            )
        if args.edge_chunk_size:
            config = replace(
                config,
                mesh2grid_edge_chunk_size=args.edge_chunk_size,
            )
        if args.grid_node_chunk_size:
            config = replace(
                config,
                grid_node_chunk_size=args.grid_node_chunk_size,
            )
        if args.schedule_recipe == OFFICIAL_2_TO_12_RECIPE:
            validate_full_operational_model_config(config)
        stats = load_normalization_stats(args.stats_dir)
        model = GoogleSmallCompatibleModel(config)
        if init_state is not None:
            model.load_state_dict(init_state, strict=True)
        model.init_graph(
            latitudes=dataset.latitudes,
            longitudes=dataset.longitudes,
        )
        configure_model_precision(model, "fp32")
        model.configure_four_stage_pipeline(
            input_device,
            mesh_first_device,
            secondary_device,
            output_device,
            split_step=args.mesh_split_step,
            activation_checkpoint=args.activation_checkpoint,
            checkpoint_mesh_per_step=(
                args.mesh_checkpoint_granularity == "step"
            ),
            grid2mesh_stage_device=args.grid2mesh_stage_device,
            grid2mesh_node_stage_device=(
                args.grid2mesh_node_stage_device
            ),
            checkpoint_grid2mesh_stage=(
                args.grid2mesh_stage_checkpoint == "enabled"
            ),
            mesh2grid_stage_device=args.mesh2grid_stage_device,
            mesh2grid_node_stage_device=(
                args.mesh2grid_node_stage_device
            ),
            mesh2grid_decoder_stage_device=(
                args.mesh2grid_decoder_stage_device
            ),
            checkpoint_mesh2grid_stage=(
                args.mesh2grid_stage_checkpoint == "enabled"
            ),
        )
        model.train()
        unroll = EpochAutoregressiveModule(
            model,
            checkpoint_segments=args.rollout_checkpoint_segments,
            checkpoint_single_step=(
                args.single_step_rollout_checkpoint == "enabled"
            ),
            checkpoint_implementation=(
                args.rollout_checkpoint_implementation
            ),
            loss_grid_chunk_size=args.loss_grid_chunk_size,
        )
        wrapped_unroll = (
            context.wrap_ddp(unroll)
            if args.gradient_sync == "ddp"
            else unroll
        )
        next_schedule_value = schedule.value_for_update(
            cursor.global_optimizer_step
        )
        optimizer = build_graphcast_adamw(
            model,
            lr=next_schedule_value.learning_rate,
        )
        if resume_payload is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            _set_learning_rate(
                optimizer,
                next_schedule_value.learning_rate,
            )
            rng_states = resume_payload["rng_states_by_rank"]
            if len(rng_states) != context.world_size:
                raise ValueError("resume RNG rank count disagrees")
            restore_rng_state(
                rng_states[context.rank],
                cuda_device_indices=context.device_indices,
                accelerator_type=context.accelerator_type,
            )

        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="model",
            status="ready",
            elapsed_s=f"{time.perf_counter() - stage_started:.3f}",
        )
        stage_started = time.perf_counter()
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="replica_initialization",
            status="start",
        )
        replica_count = (
            context.world_size if args.gradient_sync == "ddp"
            else validate_initial_replica_state(context, model)
        )
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="replica_initialization",
            status="ready",
            replicas=replica_count,
            elapsed_s=f"{time.perf_counter() - stage_started:.3f}",
        )
        forward_flops_local = model.estimate_forward_linear_flops(
            batch_size=local_batch_size
        )
        forward_flops_global = int(
            context.global_sum(float(forward_flops_local))
        )
        one_lead_training_flops = forward_flops_global * 3
        graph_contract = {
            **model.graph_topology_contract(),
            "local_batch_size_for_flops": local_batch_size,
            "forward_linear_flops_local": forward_flops_local,
            "data_parallel_replicas": context.world_size,
            "forward_linear_flops_global": forward_flops_global,
            "one_lead_training_flops_global": one_lead_training_flops,
        }
        if args.schedule_recipe == OFFICIAL_2_TO_12_RECIPE:
            validate_canonical_operational_graph_contract(graph_contract)
        graph_contracts = context.gather_objects(graph_contract)
        if any(
            candidate != graph_contracts[0]
            for candidate in graph_contracts[1:]
        ):
            raise RuntimeError(
                "data-parallel ranks initialized different graph contracts: "
                f"{graph_contracts}"
            )
        if context.is_primary:
            write_immutable_json(
                run_dir / "graph_contract.json",
                graph_contract,
            )
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="graph_contract",
            status="ready",
            grid_nodes=graph_contract["grid_nodes"],
            mesh_nodes=graph_contract["mesh_nodes"],
            grid2mesh_edges=graph_contract["grid2mesh_edges"],
            mesh_edges=graph_contract["mesh_edges"],
            mesh2grid_edges=graph_contract["mesh2grid_edges"],
            forward_flops_global=forward_flops_global,
        )
        recompute_counter = RecomputeFlopCounter()
        segment_start_step = cursor.global_optimizer_step + 1
        metrics_path = (
            run_dir
            / "metrics"
            / f"segment_{segment_start_step:08d}.jsonl"
        )
        if context.is_primary and metrics_path.exists():
            raise FileExistsError(
                f"metric segment already exists: {metrics_path}"
            )
        context.barrier()
        # Segment-specific evidence must survive metric-list rewrites at
        # profiling/finalization. Hash real state/data outside update timing.
        from graphcast_compat.training_diagnostics import initial_state, segment_start_record
        states_by_rank = context.gather_objects(initial_state(model, optimizer, context.rank))
        runtime_by_rank = context.gather_objects({"rank": context.rank, **operator_replacement})
        if context.is_primary:
            write_immutable_json(
                metrics_path.with_suffix(".start.json"),
                segment_start_record(
                    start_optimizer_step=cursor.global_optimizer_step,
                    states_by_rank=states_by_rank, runtime_by_rank=runtime_by_rank,
                    resume_contract=resume_contract, resolved_precision=math_mode,
                    manifest=manifest,
                    optimizer=graphcast_optimizer_metadata(lr=next_schedule_value.learning_rate),
                    per_variable_weights=PER_VARIABLE_WEIGHTS,
                    execution_policy={
                        "data_prefetch_depth": args.data_prefetch_depth,
                        "split_cache_max_gib": args.split_cache_max_gib,
                        "backward_execution": "serial_musa" if serial_musa_backward else "runtime_default",
                    },
                ),
            )
        context.barrier()
        profiler_run_dir = (
            run_dir / "processes" / f"from_{segment_start_step:08d}"
        )
        _atomic_json_dump(
            profiler_run_dir / f"rank_{context.rank}" / "runtime.json",
            {
                "schema_version": 1,
                "rank": context.rank,
                "segment_start_optimizer_step": segment_start_step,
                "operator_replacement": operator_replacement,
                "split_cache": dataset.split_cache_status,
            },
        )
        if context.is_primary:
            _atomic_json_dump(
                profiler_run_dir / "process.json",
                {
                    "schema_version": 1,
                    "segment_start_optimizer_step": segment_start_step,
                    "resume_training_checkpoint": (
                        str(
                            Path(args.resume_training_checkpoint)
                            .expanduser()
                            .resolve()
                        )
                        if args.resume_training_checkpoint
                        else None
                    ),
                    "argv": sys.argv,
                    "pid": os.getpid(),
                    "started_at_unix": time.time(),
                },
            )
            _write_heartbeat(
                run_dir,
                state="RUNNING",
                global_optimizer_step=cursor.global_optimizer_step,
                epoch=cursor.epoch,
                next_batch_index=cursor.next_batch_index,
            )
        context.barrier()
        profiler = HotspotProfiler(
            profile_config(
                args,
                run_dir=profiler_run_dir,
                rank=context.rank,
            ),
            input_device,
        )
        data_prefetcher: DepthOneCpuPrefetcher[
            OfficialUpdateDataKey,
            OfficialUpdateDataRequest,
            PreparedOfficialUpdate,
        ] | None = None
        metrics: list[dict[str, Any]] = []
        checkpoint_record: dict[str, Any] | None = None
        stopped = False
        update_recipe_complete = False
        _rank_zero_progress(
            "official_graphcast_init",
            rank=context.rank,
            stage="training_loop",
            status="ready",
            first_update=segment_start_step,
            total_updates=planned_optimizer_updates,
            data_prefetch_depth=args.data_prefetch_depth,
            elapsed_s=f"{time.perf_counter() - startup_started:.3f}",
        )
        profiler.start()
        try:
            if args.data_prefetch_depth == 1:
                data_prefetcher = DepthOneCpuPrefetcher(
                    OfficialUpdateCpuProducer(
                        dataset=dataset,
                        config=config,
                        stats=stats,
                        per_variable_weights=PER_VARIABLE_WEIGHTS,
                    ),
                    thread_name=(
                        f"graphcast-data-prefetch-rank-{context.rank}"
                    ),
                )
            for epoch in range(cursor.epoch, training_epoch_limit):
                plan = EpochPlan.create(
                    dataset.window_indices,
                    epoch=epoch,
                    seed=args.seed,
                    global_batch_size=args.batch_size,
                    shuffle=True,
                )
                batch_start = (
                    cursor.next_batch_index
                    if epoch == cursor.epoch
                    else 0
                )
                if (
                    epoch == cursor.epoch
                    and cursor.epoch_plan_digest != plan.digest
                ):
                    raise ValueError("active EpochPlan digest disagrees")
                for batch_index in range(
                    batch_start,
                    plan.optimizer_steps,
                ):
                    schedule_value = schedule.value_for_update(
                        cursor.global_optimizer_step
                    )
                    active_rollout_steps = schedule_value.rollout_steps
                    if not 1 <= active_rollout_steps <= args.rollout_steps:
                        raise ValueError(
                            "schedule rollout must fit inside the dataset "
                            "rollout window"
                        )
                    global_step = cursor.global_optimizer_step + 1
                    phase_update_total = schedule_phase_updates[
                        schedule_value.phase_index
                    ]
                    phase_update_total_text = (
                        str(phase_update_total)
                        if phase_update_total is not None
                        else "unbounded"
                    )
                    _rank_zero_progress(
                        "official_graphcast_update_start",
                        rank=context.rank,
                        update=(
                            f"{global_step}/{planned_optimizer_updates}"
                        ),
                        progress_unit="optimizer_update",
                        data_cycle=epoch,
                        stage=(
                            f"{schedule_value.phase_index + 1}/"
                            f"{len(schedule_phase_updates)}"
                        ),
                        phase=schedule_value.phase_name,
                        phase_update=(
                            f"{schedule_value.phase_update_index + 1}/"
                            f"{phase_update_total_text}"
                        ),
                        rollout=active_rollout_steps,
                        lr=f"{schedule_value.learning_rate:.8e}",
                        status="data_loading",
                    )
                    _set_learning_rate(
                        optimizer,
                        schedule_value.learning_rate,
                    )
                    if target_optimizer_updates is not None:
                        global_positions = dataset.replacement_indices(
                            update_index=cursor.global_optimizer_step,
                            batch_size=args.batch_size,
                            seed=args.seed,
                            target_rollout_steps=active_rollout_steps,
                        )
                        global_sample_weights = (1.0,) * args.batch_size
                        allow_replacement = True
                    else:
                        planned = plan.batches[batch_index]
                        global_positions = planned.window_indices
                        global_sample_weights = planned.sample_weights
                        allow_replacement = False

                    current_data_request = _official_update_data_request(
                        global_step=global_step,
                        epoch=epoch,
                        batch_index=batch_index,
                        rollout_steps=active_rollout_steps,
                        rank=context.rank,
                        replicas=context.dp_replicas,
                        allow_replacement=allow_replacement,
                        global_positions=global_positions,
                        global_sample_weights=global_sample_weights,
                        microbatch_plan=microbatch_plan,
                    )
                    if (
                        data_prefetcher is not None
                        and not data_prefetcher.has_pending
                    ):
                        # Only the first update reaches this path.  Every later
                        # update is submitted before the preceding update's
                        # accelerator work starts.
                        data_prefetcher.submit(
                            current_data_request.key,
                            current_data_request,
                        )

                    update_started = time.perf_counter()
                    local_data_load_seconds = 0.0
                    local_sample_seconds = 0.0
                    local_batch_prepare_seconds = 0.0
                    local_prefetch_wait_seconds = 0.0
                    local_prefetch_worker_seconds = 0.0
                    local_prefetch_worker_sample_seconds = 0.0
                    local_prefetch_worker_prepare_seconds = 0.0
                    local_prefetch_request_seconds = 0.0
                    local_input_stage_seconds = 0.0
                    optimizer.zero_grad(set_to_none=True)
                    recompute_counter.reset()
                    global_valid = 0
                    global_loss_numerator = 0.0
                    lead_loss_numerator: np.ndarray | None = None
                    variable_loss_numerator: np.ndarray | None = None
                    variable_names: tuple[str, ...] | None = None
                    provenance_chunks: list[list[dict[str, Any]]] = []

                    prepared_update: PreparedOfficialUpdate | None = None
                    if data_prefetcher is not None:
                        prefetch_wait_started = time.perf_counter()
                        prepared_update = data_prefetcher.take(
                            current_data_request.key
                        )
                        local_prefetch_wait_seconds = (
                            time.perf_counter() - prefetch_wait_started
                        )
                        local_data_load_seconds += (
                            local_prefetch_wait_seconds
                        )
                        if (
                            prepared_update.key != current_data_request.key
                            or prepared_update.rollout_steps
                            != active_rollout_steps
                        ):
                            raise RuntimeError(
                                "prefetched data disagrees with the active "
                                "training update"
                            )
                        expected_part_indices = tuple(
                            part.index for part in microbatch_plan.parts
                        )
                        prepared_part_indices = tuple(
                            part.index
                            for part in prepared_update.microbatches
                        )
                        if prepared_part_indices != expected_part_indices:
                            raise RuntimeError(
                                "prefetched microbatch order differs from "
                                "the gradient-accumulation plan"
                            )
                        local_prefetch_worker_seconds = (
                            prepared_update.worker_seconds
                        )
                        local_prefetch_worker_sample_seconds = sum(
                            part.sample_seconds
                            for part in prepared_update.microbatches
                        )
                        local_prefetch_worker_prepare_seconds = sum(
                            part.batch_prepare_seconds
                            for part in prepared_update.microbatches
                        )
                        request_started = time.perf_counter()
                        next_data_request = (
                            _next_official_update_data_request(
                                current_global_step=global_step,
                                current_epoch=epoch,
                                current_batch_index=batch_index,
                                current_plan=plan,
                                training_epoch_limit=training_epoch_limit,
                                planned_optimizer_updates=(
                                    planned_optimizer_updates
                                ),
                                target_optimizer_updates=(
                                    target_optimizer_updates
                                ),
                                stop_after_global_step=(
                                    args.stop_after_global_step
                                ),
                                dataset=dataset,
                                schedule=schedule,
                                seed=args.seed,
                                global_batch_size=args.batch_size,
                                microbatch_plan=microbatch_plan,
                                rank=context.rank,
                                replicas=context.dp_replicas,
                            )
                        )
                        if next_data_request is not None:
                            data_prefetcher.submit(
                                next_data_request.key,
                                next_data_request,
                            )
                        local_prefetch_request_seconds = (
                            time.perf_counter() - request_started
                        )

                    for part_offset, part in enumerate(microbatch_plan.parts):
                        part_positions = global_positions[part.batch_slice]
                        part_weights = global_sample_weights[
                            part.batch_slice
                        ]
                        if prepared_update is None:
                            data_started = time.perf_counter()
                            sample_started = time.perf_counter()
                            sample = dataset.sample_indices(
                                window_indices=part_positions,
                                sample_weights=part_weights,
                                rank=context.rank,
                                replicas=context.dp_replicas,
                                target_rollout_steps=active_rollout_steps,
                                allow_replacement=allow_replacement,
                            )
                            local_sample_seconds += (
                                time.perf_counter() - sample_started
                            )
                            if sample.rollout_steps != active_rollout_steps:
                                raise RuntimeError(
                                    "materialized target horizon disagrees "
                                    "with schedule"
                                )
                            batch_prepare_started = time.perf_counter()
                            batch = prepare_autoregressive_batch(
                                sample.inputs,
                                sample.targets,
                                sample.forcings,
                                config=config,
                                stats=stats,
                                input_device=input_device,
                                per_variable_weights=PER_VARIABLE_WEIGHTS,
                                sample_weights=sample.sample_weights,
                            )
                            local_batch_prepare_seconds += (
                                time.perf_counter() - batch_prepare_started
                            )
                            local_data_load_seconds += (
                                time.perf_counter() - data_started
                            )
                            sample_local_valid_count = (
                                sample.valid_sample_count
                            )
                            sample_global_valid_count = (
                                sample.global_valid_sample_count
                            )
                            microbatch_provenance = sample_provenance(
                                sample,
                                rollout_steps=active_rollout_steps,
                            )
                        else:
                            prepared_part = prepared_update.microbatches[
                                part_offset
                            ]
                            sample_metadata = prepared_part.sample_metadata
                            batch = prepared_part.batch
                            if (
                                sample_metadata.global_window_indices
                                != tuple(part_positions)
                                or sample_metadata.global_sample_weights
                                != tuple(float(value) for value in part_weights)
                            ):
                                raise RuntimeError(
                                    "prefetched sample order or weights differ "
                                    "from the active training plan"
                                )
                            input_stage_started = time.perf_counter()
                            batch = replace(
                                batch,
                                initial_input_features=(
                                    batch.initial_input_features.to(
                                        device=input_device
                                    )
                                ),
                            )
                            input_stage_seconds = (
                                time.perf_counter() - input_stage_started
                            )
                            local_input_stage_seconds += input_stage_seconds
                            local_data_load_seconds += input_stage_seconds
                            sample_local_valid_count = (
                                sample_metadata.valid_sample_count
                            )
                            sample_global_valid_count = (
                                sample_metadata.global_valid_sample_count
                            )
                            microbatch_provenance = (
                                sample_metadata.provenance
                            )
                        current_variable_names = ()
                        if variable_names is None:
                            variable_names = current_variable_names
                        elif variable_names != current_variable_names:
                            raise RuntimeError(
                                "target variable order changed between "
                                "micro-batches"
                            )
                        _rank_zero_progress(
                            "official_graphcast_data_ready",
                            rank=context.rank,
                            update=(
                                f"{global_step}/"
                                f"{planned_optimizer_updates}"
                            ),
                            microbatch=(
                                f"{part.index + 1}/"
                                f"{microbatch_plan.accumulation_steps}"
                            ),
                            source_frames=active_rollout_steps + 2,
                            valid=sample_global_valid_count,
                            status="computing",
                        )
                        context.barrier()
                        synchronize(context.device_indices)
                        with profiler.record(
                            "graphcast/official_autoregressive_update/"
                            f"microbatch_{part.index}"
                        ):
                            synchronization = (
                                wrapped_unroll.no_sync()
                                if args.gradient_sync == "ddp"
                                and part.index + 1 < microbatch_plan.accumulation_steps
                                else nullcontext()
                            )
                            with synchronization, recompute_counter.track():
                                (
                                    local_loss,
                                    lead_tensor,
                                    variable_tensor,
                                ) = wrapped_unroll(batch)
                                replica_scale = (
                                    loss_scale_for_valid_samples(
                                        local_valid_count=(
                                            sample_local_valid_count
                                        ),
                                        global_valid_count=(
                                            sample_global_valid_count
                                        ),
                                        replicas=context.dp_replicas,
                                    )
                                )
                                accumulation_scale = (
                                    sample_global_valid_count
                                    / args.batch_size
                                    if target_optimizer_updates is not None
                                    else 1.0
                                )
                                # Multi-device MUSA autograd worker execution
                                # was unstable in full-model training runs.
                                with (
                                    torch.autograd.set_multithreading_enabled(False)
                                    if serial_musa_backward
                                    else nullcontext()
                                ):
                                    (
                                        local_loss
                                        * replica_scale
                                        * accumulation_scale
                                    ).backward()
                        # Close every full-grid microbatch's accelerator work
                        # before the next official data extraction/materialize
                        # interval starts. Without this fence, asynchronous
                        # pipeline work can be charged to the following
                        # data_load_seconds interval (observed on MUSA), making
                        # the derived compute/system denominator invalid.
                        synchronize(context.device_indices)

                        micro_global_valid = sample_global_valid_count
                        micro_global_loss = (
                            context.global_sum(
                                float(local_loss.detach().item())
                                * sample_local_valid_count
                            )
                            / micro_global_valid
                        )
                        micro_lead_losses = np.asarray(
                            _weighted_distributed_tensor(
                                lead_tensor,
                                local_valid_count=(
                                    sample_local_valid_count
                                ),
                                global_valid_count=micro_global_valid,
                                context=context,
                            ),
                            dtype=np.float64,
                        )
                        micro_variable_losses = np.asarray(
                            _weighted_distributed_tensor(
                                variable_tensor,
                                local_valid_count=(
                                    sample_local_valid_count
                                ),
                                global_valid_count=micro_global_valid,
                                context=context,
                            ),
                            dtype=np.float64,
                        )
                        global_valid += micro_global_valid
                        global_loss_numerator += (
                            micro_global_loss * micro_global_valid
                        )
                        if lead_loss_numerator is None:
                            lead_loss_numerator = np.zeros_like(
                                micro_lead_losses
                            )
                        if variable_loss_numerator is None:
                            variable_loss_numerator = np.zeros_like(
                                micro_variable_losses
                            )
                        lead_loss_numerator += (
                            micro_lead_losses * micro_global_valid
                        )
                        variable_loss_numerator += (
                            micro_variable_losses * micro_global_valid
                        )
                        provenance_chunks.append(
                            context.gather_objects(
                                microbatch_provenance
                            )
                        )
                        del (
                            batch,
                            local_loss,
                            lead_tensor,
                            variable_tensor,
                        )
                        if prepared_update is None:
                            del sample

                    if (
                        global_valid <= 0
                        or lead_loss_numerator is None
                        or variable_loss_numerator is None
                        or variable_names is None
                    ):
                        raise RuntimeError(
                            "optimizer update accumulated no valid samples"
                        )
                    if (
                        target_optimizer_updates is not None
                        and global_valid != args.batch_size
                    ):
                        raise RuntimeError(
                            "official with-replacement update did not "
                            "materialize the complete global batch"
                        )
                    synchronized_gradient_count = (
                        context.synchronize_gradients(model)
                        if args.gradient_sync == "manual"
                        else sum(p.grad is not None for p in model.parameters())
                    )
                    if not math.isfinite(global_loss_numerator):
                        raise FloatingPointError("non-finite GraphCast update loss")
                    # Global-norm clipping already rejects non-finite gradients.
                    # Avoid three device-to-host reductions for every parameter.
                    grad_norm_before_clip = clip_graphcast_gradients(model)
                    grad_ok = True
                    with profiler.record("graphcast/optimizer"):
                        optimizer.step()
                    synchronize(context.device_indices)
                    local_end_to_end_seconds = (
                        time.perf_counter() - update_started
                    )
                    # Reduce each rank's non-data duration directly.  Taking
                    # max(end-to-end) - max(data) mixes two independently
                    # selected ranks and can collapse to zero on asynchronous
                    # vendor runtimes even though every rank completed a full
                    # forward/backward/gradient-sync/optimizer update.
                    local_step_seconds = (
                        local_end_to_end_seconds
                        - local_data_load_seconds
                    )
                    if local_step_seconds <= 0.0:
                        raise RuntimeError(
                            "invalid local training timing partition: "
                            f"rank={context.rank} "
                            f"data_s={local_data_load_seconds:.9f} "
                            f"e2e_s={local_end_to_end_seconds:.9f} "
                            f"compute_s={local_step_seconds:.9f}"
                        )
                    global_data_load_seconds = (
                        context.global_max_seconds(
                            local_data_load_seconds
                        )
                    )
                    end_to_end_seconds = context.global_max_seconds(
                        local_end_to_end_seconds
                    )
                    step_seconds = context.global_max_seconds(
                        local_step_seconds
                    )
                    if data_prefetcher is None:
                        local_data_other_seconds = max(
                            0.0,
                            local_data_load_seconds
                            - local_sample_seconds
                            - local_batch_prepare_seconds,
                        )
                    else:
                        local_data_other_seconds = max(
                            0.0,
                            local_data_load_seconds
                            - local_prefetch_wait_seconds
                            - local_input_stage_seconds,
                        )
                    local_prefetch_overlap_seconds = max(
                        0.0,
                        local_prefetch_worker_seconds
                        - local_prefetch_wait_seconds,
                    )
                    append_jsonl(
                        rank_dir / "timing.jsonl",
                        {
                            "step": global_step,
                            "rank": context.rank,
                            "rollout_steps": active_rollout_steps,
                            "sample_seconds": local_sample_seconds,
                            "batch_prepare_seconds": (
                                local_batch_prepare_seconds
                            ),
                            "data_other_seconds": local_data_other_seconds,
                            "prefetch_enabled": (
                                data_prefetcher is not None
                            ),
                            "prefetch_depth": args.data_prefetch_depth,
                            "prefetch_wait_seconds": (
                                local_prefetch_wait_seconds
                            ),
                            "prefetch_request_seconds": (
                                local_prefetch_request_seconds
                            ),
                            "prefetch_worker_seconds": (
                                local_prefetch_worker_seconds
                            ),
                            "prefetch_worker_sample_seconds": (
                                local_prefetch_worker_sample_seconds
                            ),
                            "prefetch_worker_prepare_seconds": (
                                local_prefetch_worker_prepare_seconds
                            ),
                            "prefetch_overlap_seconds": (
                                local_prefetch_overlap_seconds
                            ),
                            "input_stage_seconds": (
                                local_input_stage_seconds
                            ),
                            "data_total_seconds": local_data_load_seconds,
                            "non_data_system_seconds": local_step_seconds,
                            "end_to_end_step_seconds": (
                                local_end_to_end_seconds
                            ),
                            "timing_definition": (
                                "local_rank_wall_clock_with_device_fences"
                            ),
                        },
                    )
                    _rank_zero_progress(
                        "official_graphcast_timing_partition",
                        rank=context.rank,
                        update=(
                            f"{global_step}/"
                            f"{planned_optimizer_updates}"
                        ),
                        local_data_s=(
                            f"{local_data_load_seconds:.9f}"
                        ),
                        local_compute_s=f"{local_step_seconds:.9f}",
                        local_sample_s=f"{local_sample_seconds:.9f}",
                        local_prepare_s=(
                            f"{local_batch_prepare_seconds:.9f}"
                        ),
                        local_data_other_s=(
                            f"{local_data_other_seconds:.9f}"
                        ),
                        local_prefetch_wait_s=(
                            f"{local_prefetch_wait_seconds:.9f}"
                        ),
                        local_prefetch_worker_s=(
                            f"{local_prefetch_worker_seconds:.9f}"
                        ),
                        local_prefetch_overlap_s=(
                            f"{local_prefetch_overlap_seconds:.9f}"
                        ),
                        local_input_stage_s=(
                            f"{local_input_stage_seconds:.9f}"
                        ),
                        local_e2e_s=(
                            f"{local_end_to_end_seconds:.9f}"
                        ),
                        global_data_s=(
                            f"{global_data_load_seconds:.9f}"
                        ),
                        global_compute_s=f"{step_seconds:.9f}",
                        global_e2e_s=f"{end_to_end_seconds:.9f}",
                    )
                    local_peak_memory = memory_by_device(
                        context.device_indices
                    )
                    peak_allocated_gib = context.global_max_seconds(
                        max(
                            float(value["allocated_gib"])
                            for value in local_peak_memory.values()
                        )
                    )
                    peak_reserved_gib = context.global_max_seconds(
                        max(
                            float(value["reserved_gib"])
                            for value in local_peak_memory.values()
                        )
                    )
                    global_loss = global_loss_numerator / global_valid
                    lead_losses = (
                        lead_loss_numerator / global_valid
                    ).tolist()
                    variable_values = (
                        variable_loss_numerator / global_valid
                    ).tolist()
                    variable_matrix = np.asarray(
                        variable_values,
                        dtype=np.float64,
                    ).reshape(active_rollout_steps, len(variable_names))
                    recompute_flops = int(
                        context.global_sum(
                            float(recompute_counter.recompute_flops)
                        )
                    )
                    origins = merge_microbatch_provenance(
                        provenance_chunks
                    )
                    active_training_flops = (
                        active_rollout_steps
                        * one_lead_training_flops
                        * microbatch_plan.accumulation_steps
                    )
                    useful_training_flops = (
                        active_training_flops
                        * global_valid
                        / args.batch_size
                    )
                    if context.is_primary:
                        metric = {
                            "step": global_step,
                            "optimizer_step": global_step,
                            "progress_unit": "optimizer_update",
                            "data_cycle": epoch,
                            "epoch": epoch,
                            "epoch_batch_index": batch_index,
                            "epoch_plan_digest": plan.digest,
                            "training_sampling": (
                                "trajectory_with_replacement_per_optimizer_update"
                                if target_optimizer_updates is not None
                                else "deterministic_without_replacement_epoch"
                            ),
                            "loss": global_loss,
                            "lead_losses": lead_losses,
                            "per_variable_lead_losses": {
                                name: variable_matrix[:, index].tolist()
                                for index, name in enumerate(variable_names)
                            },
                            "learning_rate": (
                                schedule_value.learning_rate
                            ),
                            "schedule_phase": schedule_value.phase_name,
                            "schedule_phase_index": (
                                schedule_value.phase_index + 1
                            ),
                            "schedule_phase_count": len(
                                schedule_phase_updates
                            ),
                            "schedule_phase_update": (
                                schedule_value.phase_update_index + 1
                            ),
                            "schedule_phase_updates_total": (
                                schedule_phase_updates[
                                    schedule_value.phase_index
                                ]
                            ),
                            "planned_optimizer_updates": (
                                planned_optimizer_updates
                            ),
                            "grad_ok": grad_ok,
                            "gradient_check": "clip_global_norm",
                            "grad_norm_before_clip": (
                                grad_norm_before_clip
                            ),
                            "synchronized_gradient_count": (
                                synchronized_gradient_count
                            ),
                            "gradient_sync_mode": (
                                "ddp"
                                if args.gradient_sync == "ddp"
                                else (
                                    "manual_deterministic_all_parameter_"
                                    "replica_average"
                                )
                            ),
                            "operator_mode": operator_mode,
                            "flagos_replacement": (
                                operator_replacement
                            ),
                            "step_seconds": step_seconds,
                            "data_load_seconds": global_data_load_seconds,
                            "end_to_end_step_seconds": end_to_end_seconds,
                            "rank0_data_breakdown": {
                                "sample_seconds": local_sample_seconds,
                                "batch_prepare_seconds": (
                                    local_batch_prepare_seconds
                                ),
                                "data_other_seconds": (
                                    local_data_other_seconds
                                ),
                                "prefetch_enabled": (
                                    data_prefetcher is not None
                                ),
                                "prefetch_depth": args.data_prefetch_depth,
                                "prefetch_wait_seconds": (
                                    local_prefetch_wait_seconds
                                ),
                                "prefetch_request_seconds": (
                                    local_prefetch_request_seconds
                                ),
                                "prefetch_worker_seconds": (
                                    local_prefetch_worker_seconds
                                ),
                                "prefetch_worker_sample_seconds": (
                                    local_prefetch_worker_sample_seconds
                                ),
                                "prefetch_worker_prepare_seconds": (
                                    local_prefetch_worker_prepare_seconds
                                ),
                                "prefetch_overlap_seconds": (
                                    local_prefetch_overlap_seconds
                                ),
                                "input_stage_seconds": (
                                    local_input_stage_seconds
                                ),
                                "rank": context.rank,
                            },
                            "step_seconds_definition": (
                                "maximum_across_ranks_of_local_e2e_minus_"
                                "local_data; includes forward_backward_"
                                "gradient_sync_optimizer_and_device_fences"
                            ),
                            "data_load_seconds_definition": (
                                (
                                    "maximum_across_ranks_of_foreground_"
                                    "prefetch_wait_and_input_device_staging_"
                                    "wall_time; request_planning_and_"
                                    "worker_sample_prepare_is_reported_"
                                    "separately"
                                )
                                if data_prefetcher is not None
                                else (
                                    "maximum_across_ranks_of_sample_batch_"
                                    "prepare_and_local_bookkeeping_wall_time; "
                                    "not_pure_storage_io"
                                )
                            ),
                            "samples_per_second": (
                                global_valid / end_to_end_seconds
                            ),
                            "updates_per_second": (
                                1.0 / end_to_end_seconds
                            ),
                            "peak_allocated_gib_per_device": (
                                peak_allocated_gib
                            ),
                            "peak_reserved_gib_per_device": (
                                peak_reserved_gib
                            ),
                            "peak_memory_gib_by_local_device": (
                                local_peak_memory
                            ),
                            "training_flops": useful_training_flops,
                            "model_forward_flops_global": (
                                forward_flops_global
                                * active_rollout_steps
                                * microbatch_plan.accumulation_steps
                            ),
                            "recompute_flops": recompute_flops,
                            "inference_only": False,
                            "global_batch_size": args.batch_size,
                            "global_micro_batch_size": (
                                microbatch_plan.micro_batch_size
                            ),
                            "local_micro_batch_size": local_batch_size,
                            "gradient_accumulation_steps": (
                                microbatch_plan.accumulation_steps
                            ),
                            "valid_sample_count": global_valid,
                            "padded_sample_count": (
                                args.batch_size - global_valid
                            ),
                            "data_parallel_replicas": (
                                context.dp_replicas
                            ),
                            "mp_devices_per_replica": (
                                context.mp_devices_per_replica
                            ),
                            "rollout_steps": active_rollout_steps,
                            "eligible_window_count": (
                                available_windows_by_rollout[
                                    active_rollout_steps
                                ]
                            ),
                            "dataset_rollout_steps": args.rollout_steps,
                            "materialized_source_time_steps": (
                                active_rollout_steps + 2
                            ),
                            "target_materialization": (
                                "two_inputs_plus_active_target_lead_times_"
                                "before_load"
                            ),
                            "official_data_utils_reference": (
                                dataset.official_data_utils_reference
                            ),
                            "origins_by_rank": origins,
                        }
                        # System MFU/HFU and achieved TFLOP/s are available
                        # immediately. Kernel-busy MFU is added after the
                        # bounded profiler trace has been finalized.
                        enrich_step_metrics(
                            [metric],
                            profiled_steps={},
                            device_count=(
                                context.participating_device_count
                            ),
                            peak_tflops_per_device=(
                                args.peak_tflops_per_device
                            ),
                        )
                        validate_metric(metric)
                        metric["mfu_self_check"] = True
                        metrics.append(metric)
                        append_jsonl(metrics_path, metric)
                        profile_busy_status = (
                            "deferred"
                            if args.profile_hotspots
                            else "disabled"
                        )
                        mfu_sys_text = (
                            "N/A"
                            if metric["mfu_sys"] is None
                            else f"{metric['mfu_sys']:.6f}"
                        )
                        hfu_sys_text = (
                            "N/A"
                            if metric["hfu_sys"] is None
                            else f"{metric['hfu_sys']:.6f}"
                        )
                        mfu_e2e_text = (
                            "N/A"
                            if metric["mfu_e2e"] is None
                            else f"{metric['mfu_e2e']:.6f}"
                        )
                        print(
                            "official_graphcast_train "
                            f"update={global_step}/"
                            f"{planned_optimizer_updates} "
                            "progress_unit=optimizer_update "
                            f"data_cycle={epoch} "
                            f"stage={schedule_value.phase_index + 1}/"
                            f"{len(schedule_phase_updates)} "
                            f"phase={schedule_value.phase_name} "
                            f"phase_update="
                            f"{schedule_value.phase_update_index + 1}/"
                            f"{phase_update_total_text} "
                            f"rollout={active_rollout_steps} "
                            f"lr={schedule_value.learning_rate:.8e} "
                            f"global_batch={args.batch_size} "
                            f"micro_batch="
                            f"{microbatch_plan.micro_batch_size} "
                            f"accumulation="
                            f"{microbatch_plan.accumulation_steps} "
                            f"operator_mode={operator_mode} "
                            f"valid={global_valid} "
                            f"loss={global_loss:.8e} "
                            f"data_s={global_data_load_seconds:.6f} "
                            f"compute_s={step_seconds:.6f} "
                            f"e2e_s={end_to_end_seconds:.6f} "
                            f"samples_s={metric['samples_per_second']:.6f} "
                            f"tflops={metric['achieved_tflops']:.6f} "
                            f"mfu_sys={mfu_sys_text} "
                            f"hfu_sys={hfu_sys_text} "
                            f"mfu_e2e={mfu_e2e_text} "
                            f"peak_alloc_gib={peak_allocated_gib:.3f} "
                            f"peak_reserved_gib={peak_reserved_gib:.3f} "
                            f"grad_norm={grad_norm_before_clip:.6e} "
                            f"mfu_busy={profile_busy_status}",
                            flush=True,
                        )
                    profiler.step()
                    profile_capture_complete = profiler.stop_if_trace_ready()
                    cursor = TrainingCursor(
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                        global_optimizer_step=global_step,
                        epoch_plan_digest=plan.digest,
                    )
                    if context.is_primary:
                        _write_heartbeat(
                            run_dir,
                            state="RUNNING",
                            global_optimizer_step=global_step,
                            epoch=epoch,
                            next_batch_index=batch_index + 1,
                        )
                    final_by_update_recipe = (
                        target_optimizer_updates is not None
                        and global_step == target_optimizer_updates
                    )
                    final_by_epoch = (
                        target_optimizer_updates is None
                        and epoch == training_epoch_limit - 1
                        and batch_index == plan.optimizer_steps - 1
                    )
                    final = final_by_update_recipe or final_by_epoch
                    should_stop = (
                        args.stop_after_global_step > 0
                        and global_step >= args.stop_after_global_step
                        and not final
                    )
                    should_checkpoint = (
                        final
                        or should_stop
                        or profile_capture_complete
                        or global_step % args.checkpoint_every_steps == 0
                    )
                    if should_checkpoint:
                        checkpoint_record = save_validated_training_checkpoint(
                            args=args,
                            context=context,
                            model=model,
                            optimizer=optimizer,
                            schedule=schedule,
                            applied_schedule_value=schedule_value,
                            cursor=cursor,
                            epoch_plan=plan,
                            run_dir=run_dir,
                            resume_contract=resume_contract,
                            initial_lineage=initial_lineage,
                            final=final,
                        )
                    if profile_capture_complete:
                        profile_postprocess_mode = (
                            "torch_npu_msprof_sqlite"
                            if context.accelerator_type == "npu"
                            else "torch_profiler_trace_json"
                        )
                        profile_postprocess_source = (
                            "torch_npu raw capture -> msprof SQLite"
                            if context.accelerator_type == "npu"
                            else (
                                "torch.profiler "
                                f"{context.accelerator_type} trace JSON"
                            )
                        )
                        _rank_zero_progress(
                            "official_graphcast_profile",
                            rank=context.rank,
                            update=global_step,
                            status="postprocess_start",
                            mode=profile_postprocess_mode,
                        )
                        if context.is_primary:
                            _write_heartbeat(
                                run_dir,
                                state="PROFILE_POSTPROCESSING",
                                global_optimizer_step=global_step,
                                epoch=epoch,
                                next_batch_index=batch_index + 1,
                            )
                        profiler.postprocess_ready_profiles(
                            expected_devices=tuple(
                                str(index) for index in context.device_indices
                            )
                        )
                        boundary_profiles_by_rank = context.gather_objects(
                            profiler.profiled_steps(
                                expected_devices=tuple(
                                    str(index)
                                    for index in context.device_indices
                                )
                            )
                        )
                        if context.is_primary:
                            boundary_profiles = merge_profiled_steps(
                                boundary_profiles_by_rank,
                                expected_devices=tuple(
                                    str(index)
                                    for index in range(
                                        context.participating_device_count
                                    )
                                ),
                            )
                            boundary_profiles_by_global_step = {
                                int(metrics[local_step - 1]["step"]): profile
                                for local_step, profile in (
                                    boundary_profiles.items()
                                )
                                if 0 < local_step <= len(metrics)
                            }
                            enrich_step_metrics(
                                metrics,
                                profiled_steps=(
                                    boundary_profiles_by_global_step
                                ),
                                device_count=(
                                    context.participating_device_count
                                ),
                                peak_tflops_per_device=(
                                    args.peak_tflops_per_device
                                ),
                            )
                            for boundary_metric in metrics:
                                validate_metric(boundary_metric)
                                boundary_metric["mfu_self_check"] = True
                            metrics_path.write_text(
                                "".join(
                                    json.dumps(row, sort_keys=True) + "\n"
                                    for row in metrics
                                ),
                                encoding="utf-8",
                            )
                            _atomic_json_dump(
                                profiler_run_dir / "profile_ready.json",
                                {
                                    "schema_version": 1,
                                    "status": "OK",
                                    "global_updates": sorted(
                                        boundary_profiles_by_global_step
                                    ),
                                    "local_updates": sorted(
                                        boundary_profiles
                                    ),
                                    "device_count": (
                                        context.participating_device_count
                                    ),
                                    "source": (
                                        profile_postprocess_source
                                    ),
                                    "postprocess_mode": (
                                        profile_postprocess_mode
                                    ),
                                },
                            )
                        context.barrier()
                        if context.is_primary:
                            _write_heartbeat(
                                run_dir,
                                state="RUNNING",
                                global_optimizer_step=global_step,
                                epoch=epoch,
                                next_batch_index=batch_index + 1,
                            )
                        _rank_zero_progress(
                            "official_graphcast_profile",
                            rank=context.rank,
                            update=global_step,
                            status="postprocess_complete",
                            mode=profile_postprocess_mode,
                        )
                    if should_stop:
                        stopped = True
                        break
                    if final_by_update_recipe:
                        update_recipe_complete = True
                        break
                if stopped or update_recipe_complete:
                    break
        finally:
            active_exception = sys.exc_info()[0] is not None
            try:
                if data_prefetcher is not None:
                    data_prefetcher.close()
            finally:
                profiler.stop(suppress_errors=active_exception)

        local_profiles = profiler.profiled_steps(
            expected_devices=tuple(
                str(index) for index in context.device_indices
            )
        )
        profiles_by_rank = context.gather_objects(local_profiles)
        merged_local_profiles = (
            merge_profiled_steps(
                profiles_by_rank,
                expected_devices=tuple(
                    str(index)
                    for index in range(context.participating_device_count)
                ),
            )
            if args.profile_hotspots and context.is_primary
            else {}
        )
        if context.is_primary:
            profiled_by_global_step = {
                int(metrics[local_step - 1]["step"]): profile
                for local_step, profile in merged_local_profiles.items()
                if 0 < local_step <= len(metrics)
            }
            enrich_step_metrics(
                metrics,
                profiled_steps=profiled_by_global_step,
                device_count=context.participating_device_count,
                peak_tflops_per_device=args.peak_tflops_per_device,
            )
            for metric in metrics:
                validate_metric(metric)
                metric["mfu_self_check"] = True
                if metric.get("busy_profiled"):
                    mfu_busy_text = (
                        "N/A"
                        if metric["mfu_busy"] is None
                        else f"{metric['mfu_busy']:.6f}"
                    )
                    mfu_sys_text = (
                        "N/A"
                        if metric["mfu_sys"] is None
                        else f"{metric['mfu_sys']:.6f}"
                    )
                    hfu_sys_text = (
                        "N/A"
                        if metric["hfu_sys"] is None
                        else f"{metric['hfu_sys']:.6f}"
                    )
                    mfu_e2e_text = (
                        "N/A"
                        if metric["mfu_e2e"] is None
                        else f"{metric['mfu_e2e']:.6f}"
                    )
                    print(
                        "official_graphcast_profile "
                        f"update={metric['step']}/"
                        f"{planned_optimizer_updates} "
                        f"stage={metric['schedule_phase_index']}/"
                        f"{metric['schedule_phase_count']} "
                        f"phase={metric['schedule_phase']} "
                        f"phase_update="
                        f"{metric['schedule_phase_update']}/"
                        f"{metric['schedule_phase_updates_total']} "
                        f"rollout={metric['rollout_steps']} "
                        f"profile_wall_ms="
                        f"{metric['profile_step_wall_ms']:.3f} "
                        f"device_busy_ms="
                        f"{metric['device_busy_ms']:.3f} "
                        f"total_device_busy_ms="
                        f"{metric['total_device_busy_ms']:.3f} "
                        f"device_sys_ms="
                        f"{metric['device_sys_ms']:.3f} "
                        f"total_device_sys_ms="
                        f"{metric['total_device_sys_ms']:.3f} "
                        f"tflops={metric['achieved_tflops']:.6f} "
                        f"mfu_busy={mfu_busy_text} "
                        f"mfu_sys={mfu_sys_text} "
                        f"hfu_sys={hfu_sys_text} "
                        f"mfu_e2e={mfu_e2e_text}",
                        flush=True,
                    )
            metrics_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in metrics
                ),
                encoding="utf-8",
            )
        context.barrier()

        all_metrics = (
            read_metric_segments(run_dir) if context.is_primary else []
        )
        if stopped:
            if checkpoint_record is None:
                raise RuntimeError("controlled stop has no checkpoint")
            if context.is_primary:
                summary = {
                    "status": "STOPPED",
                    "reason": "controlled_optimizer_boundary_stop",
                    "completed_optimizer_steps": (
                        cursor.global_optimizer_step
                    ),
                    "cursor": cursor.to_dict(),
                    "latest_checkpoint": checkpoint_record,
                    "metrics": all_metrics,
                    "corpus_index": str(manifest.path),
                    "schedule_digest": schedule.digest,
                }
                _atomic_json_dump(run_dir / "summary.json", summary)
                _write_heartbeat(
                    run_dir,
                    state="STOPPED",
                    global_optimizer_step=cursor.global_optimizer_step,
                    epoch=cursor.epoch,
                    next_batch_index=cursor.next_batch_index,
                )
                _write_state_marker(
                    run_dir,
                    "STOPPED",
                    global_optimizer_step=cursor.global_optimizer_step,
                    checkpoint=checkpoint_record,
                )
            context.barrier()
            return 0

        validation = evaluate_validation(
            args=args,
            context=context,
            wrapped_unroll=wrapped_unroll,
            base_model=model,
            config=config,
            stats=stats,
            input_device=input_device,
        )
        context.barrier()
        (
            final_model_hash,
            final_model_hashes,
        ) = validated_replicated_model_hashes(context, model)
        memory_by_rank = context.gather_objects(
            memory_by_device(context.device_indices)
        )
        if context.is_primary:
            all_metrics = read_metric_segments(run_dir)
            if target_optimizer_updates is None:
                coverage = _coverage_summary(
                    all_metrics,
                    epochs=training_epoch_limit,
                    expected_windows=dataset.window_count,
                )
                coverage_accepted = coverage["all_epochs_complete"]
            else:
                coverage = _replacement_sampling_summary(
                    all_metrics,
                    available_windows_by_rollout=(
                        available_windows_by_rollout
                    ),
                    global_batch_size=args.batch_size,
                    target_optimizer_updates=target_optimizer_updates,
                )
                coverage_accepted = coverage["sampling_self_check"]
            validation_complete = (
                not args.validation_corpus_manifest
                or (
                    validation is not None
                    and validation.get("validation_self_check") is True
                )
            )
            profile_complete = (
                not args.profile_hotspots
                or any(
                    bool(row.get("busy_profiled"))
                    for row in all_metrics
                )
            )
            expected_steps = (
                target_optimizer_updates
                if target_optimizer_updates is not None
                else sum(
                    EpochPlan.create(
                        dataset.window_indices,
                        epoch=epoch,
                        seed=args.seed,
                        global_batch_size=args.batch_size,
                        shuffle=True,
                    ).optimizer_steps
                    for epoch in range(training_epoch_limit)
                )
            )
            accepted = (
                len(all_metrics) == expected_steps
                and cursor.global_optimizer_step == expected_steps
                and coverage_accepted
                and all(row["grad_ok"] for row in all_metrics)
                and all(row["mfu_self_check"] for row in all_metrics)
                and all(
                    math.isfinite(float(row["loss"]))
                    for row in all_metrics
                )
                and profile_complete
                and validation_complete
                and checkpoint_record is not None
                and checkpoint_record.get("final") is True
            )
            averages = {}
            for name in (
                "mfu_busy",
                "mfu_sys",
                "hfu_sys",
                "mfu_e2e",
            ):
                values = [
                    float(row[name])
                    for row in all_metrics
                    if row.get(name) is not None
                ]
                averages[f"average_{name}"] = (
                    sum(values) / len(values) if values else None
                )
            timing_totals = {
                "compute_wall_seconds": sum(
                    float(row["step_seconds"]) for row in all_metrics
                ),
                "data_load_wall_seconds": sum(
                    float(row["data_load_seconds"]) for row in all_metrics
                ),
                "end_to_end_wall_seconds": sum(
                    float(row["end_to_end_step_seconds"])
                    for row in all_metrics
                ),
                "compute_device_seconds": sum(
                    float(row["step_seconds"])
                    * context.participating_device_count
                    for row in all_metrics
                ),
                "end_to_end_device_seconds": sum(
                    float(row["end_to_end_step_seconds"])
                    * context.participating_device_count
                    for row in all_metrics
                ),
                "profiled_device_busy_seconds": sum(
                    float(row["total_device_busy_ms"]) / 1.0e3
                    for row in all_metrics
                    if row.get("total_device_busy_ms") is not None
                ),
                "profiled_updates": [
                    int(row["step"])
                    for row in all_metrics
                    if row.get("busy_profiled")
                ],
            }
            summary = {
                "status": "OK" if accepted else "FAIL",
                "run_mode": run_mode,
                "run_dir": str(run_dir),
                "training_stage": args.training_stage,
                "dataset_kind": args.dataset_kind,
                "corpus_index": str(manifest.path),
                "training_window_count": dataset.window_count,
                "training_window_count_by_rollout": {
                    str(key): value
                    for key, value in (
                        available_windows_by_rollout.items()
                    )
                },
                "progress_unit": (
                    "optimizer_update"
                    if target_optimizer_updates is not None
                    else "epoch"
                ),
                "training_sampling": (
                    "trajectory_with_replacement_per_optimizer_update"
                    if target_optimizer_updates is not None
                    else "deterministic_without_replacement_epoch"
                ),
                "internal_data_cycle_limit": training_epoch_limit,
                "epochs": (
                    None
                    if target_optimizer_updates is not None
                    else training_epoch_limit
                ),
                "requested_epochs": (
                    None
                    if target_optimizer_updates is not None
                    else args.epochs
                ),
                "optimizer_steps_per_epoch": (
                    None
                    if target_optimizer_updates is not None
                    else initial_plan.optimizer_steps
                ),
                "target_optimizer_updates": target_optimizer_updates,
                "completed_optimizer_steps": (
                    cursor.global_optimizer_step
                ),
                "global_batch_size": args.batch_size,
                "global_micro_batch_size": (
                    microbatch_plan.micro_batch_size
                ),
                "local_micro_batch_size": local_batch_size,
                "gradient_accumulation_steps": (
                    microbatch_plan.accumulation_steps
                ),
                "data_parallel_replicas": context.dp_replicas,
                "mp_devices_per_replica": (
                    context.mp_devices_per_replica
                ),
                "gradient_sync": args.gradient_sync,
                "operator_replacement": operator_replacement,
                "participating_device_count": (
                    context.participating_device_count
                ),
                "rollout_steps": args.rollout_steps,
                "rollout_steps_semantics": (
                    "maximum_dataset_horizon; active horizon is recorded "
                    "per metric"
                ),
                "rollout_checkpoint_segments": (
                    args.rollout_checkpoint_segments
                ),
                "rollout_checkpoint_implementation": (
                    args.rollout_checkpoint_implementation
                ),
                "single_step_rollout_checkpoint": (
                    args.single_step_rollout_checkpoint
                ),
                "precision": "fp32",
                "tensor_core_enabled": tensor_core_enabled,
                "tensor_core_math_mode": math_mode,
                "peak_tflops_per_device": args.peak_tflops_per_device,
                "peak_tflops_source": args.peak_tflops_source,
                "optimizer": graphcast_optimizer_metadata(
                    lr=first_schedule_value.learning_rate
                ),
                "schedule": schedule.to_dict(),
                "schedule_digest": schedule.digest,
                "source_priority": [
                    "operator_handoff_graphcast_compat",
                    "pinned_deepmind_weathernext",
                    "flagos_torch_distributed_glue",
                ],
                "data_pipeline": {
                    "split_cache": dataset.split_cache_status,
                    "data_prefetch": {
                        "depth": args.data_prefetch_depth,
                        "producer_count": (
                            1 if args.data_prefetch_depth == 1 else 0
                        ),
                        "worker_device": "cpu",
                        "accelerator_transfer_thread": "training_main",
                    },
                    "timing_files": [
                        str(run_dir / f"rank_{rank}/timing.jsonl")
                        for rank in range(context.world_size)
                    ],
                },
                "coverage": coverage,
                "metrics": all_metrics,
                "metrics_segments": [
                    str(path)
                    for path in sorted(
                        (run_dir / "metrics").glob("segment_*.jsonl")
                    )
                ],
                "timing_totals": timing_totals,
                "mfu_self_check": all(
                    row["mfu_self_check"] for row in all_metrics
                ),
                **averages,
                "average_mfu": averages["average_mfu_sys"],
                "average_mfu_model_sys": (
                    averages["average_mfu_sys"]
                ),
                "validation": validation,
                "validation_complete": validation_complete,
                "profile_complete": profile_complete,
                "saved_checkpoint": checkpoint_record["path"],
                "checkpoint_sha256": checkpoint_record["sha256"],
                "initial_lineage": initial_lineage,
                "model_state_sha256": final_model_hash,
                "model_state_sha256_by_rank": final_model_hashes,
                "accelerator_memory_by_rank": memory_by_rank,
            }
            _atomic_json_dump(run_dir / "summary.json", summary)
            _atomic_json_dump(run_dir / "validation.json", validation)
            if validation is not None:
                _atomic_json_dump(
                    run_dir
                    / "validation"
                    / f"epoch_{cursor.epoch:04d}.json",
                    validation,
                )
                _atomic_json_dump(
                    run_dir / "checkpoints/best.json",
                    {
                        "checkpoint": checkpoint_record["path"],
                        "checkpoint_sha256": checkpoint_record["sha256"],
                        "metric": "full_validation_mean_loss",
                        "value": validation["mean_loss"],
                    },
                )
            report = [
                f"# {PLATFORM_LABEL} GraphCast Training",
                "",
                f"- status: `{summary['status']}`",
                f"- stage: `{args.training_stage}`",
                f"- math mode: `{math_mode}`",
                f"- operator mode: `{operator_mode}`",
                (
                    "- progress unit: "
                    f"`{summary['progress_unit']}`"
                ),
                f"- internal data cycles touched: `{cursor.epoch + 1}`",
                f"- optimizer steps: `{cursor.global_optimizer_step}`",
                f"- schedule recipe: `{args.schedule_recipe}`",
                f"- unique windows: `{dataset.window_count}`",
                (
                    "- coverage accepted: "
                    f"`{coverage_accepted}`"
                ),
                f"- checkpoint: `{checkpoint_record['path']}`",
                f"- checkpoint sha256: `{checkpoint_record['sha256']}`",
                "",
            ]
            (run_dir / "report.md").write_text(
                "\n".join(report),
                encoding="utf-8",
            )
            _write_state_marker(
                run_dir,
                "COMPLETED" if accepted else "FAILED",
                global_optimizer_step=cursor.global_optimizer_step,
                checkpoint=checkpoint_record,
            )
            _write_heartbeat(
                run_dir,
                state="COMPLETED" if accepted else "FAILED",
                global_optimizer_step=cursor.global_optimizer_step,
                epoch=cursor.epoch,
                next_batch_index=cursor.next_batch_index,
            )
            if not accepted:
                raise RuntimeError("training completion invariants failed")
        context.barrier()
        return 0
    except Exception as error:
        failure_rank = int(os.environ.get("RANK", "0"))
        failure_payload = {
            "rank": failure_rank,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json_dump(
            run_dir / f"failure_rank_{failure_rank:05d}.json",
            failure_payload,
        )
        print(failure_payload["traceback"], file=sys.stderr, flush=True)
        _write_state_marker(
            run_dir,
            "FAILED",
            rank=failure_rank,
            error_type=type(error).__name__,
            error=str(error),
        )
        if failure_rank == 0:
            _atomic_json_dump(
                run_dir / "failure.json",
                failure_payload,
            )
            _write_heartbeat(
                run_dir,
                state="FAILED",
                global_optimizer_step=0,
            )
        raise
    finally:
        try:
            if dataset is not None:
                dataset.close()
            if context is not None:
                context.close(synchronize=sys.exc_info()[0] is None)
        finally:
            _release_run_lock(run_lock)


# ==================================================
# 6. 程序入口
# ==================================================
def _main() -> int:
    exit_code = int(train())
    bypass_static_destructors = os.environ.get(
        "GRAPHCAST_BYPASS_STATIC_DESTRUCTORS_ON_SUCCESS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if (
        os.environ.get("GRAPHCAST_ACCELERATOR_TYPE", "").lower() == "musa"
        or bypass_static_destructors
    ):
        # torch_musa 2.5 can raise ``std::system_error`` from a C++ static
        # destructor after a successful run has already written its summary
        # and checkpoints.  The explicit opt-in covers NCCL-compatible vendor
        # runtimes with the same post-success teardown defect. Flush Python
        # output, then bypass only that destructor path. Exceptions raised by
        # ``train()`` still follow the normal traceback and non-zero exit path.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
