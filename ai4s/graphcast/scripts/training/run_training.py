#!/usr/bin/env python3
"""Formal FP32-storage training entry. Help/dry-run use only the standard library."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/inference/google_graphcast_operational_25km.yaml"))
    p.add_argument("--corpus")
    p.add_argument("--stats")
    checkpoints = p.add_mutually_exclusive_group()
    checkpoints.add_argument("--pretrained-checkpoint", help="Converted PyTorch model state dict")
    checkpoints.add_argument("--resume-checkpoint", help="Optimizer-boundary checkpoint plus SHA256 sidecar")
    p.add_argument("--output-dir", default="training-output")
    p.add_argument("--device-type", choices=("cuda", "npu", "musa"), default="cuda")
    p.add_argument("--devices", help="All contiguous logical device indices, e.g. 0,1,2,3")
    p.add_argument("--mp", type=int, default=4)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=4)
    p.add_argument("--micro-batch-size", type=int, default=1, help="Maximum samples per DP rank per forward/backward")
    p.add_argument("--precision", choices=("fp32",), default="fp32", help="Parameter/activation/loss storage")
    p.add_argument("--tensor-core", choices=("enabled", "disabled"), default="enabled")
    p.add_argument("--backend", choices=("native", "flaggems"), default="native")
    p.add_argument("--flaggems-config", default=str(ROOT / "configs/flaggems/graphcast_training.yaml"))
    p.add_argument("--flaggems-ops", default=None, help="Optional explicit training allowlist override")
    p.add_argument("--flaggems-variant", choices=("installed", "official", "internal"), default="installed")
    p.add_argument("--schedule", default=str(ROOT / "configs/training/operational_tf32_schedule.json"))
    for phase in ("warmup", "one-step", "curriculum"):
        p.add_argument(f"--{phase}-updates", type=int, help="Override stage budget; zero omits this stage")
    p.add_argument("--curriculum-every-updates", type=int)
    p.add_argument("--max-updates", type=int, help="Stop at this absolute optimizer step; does not change schedule identity")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--checkpoint-segments", type=int, default=0)
    p.add_argument("--loss-grid-chunk-size", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def load_schedule(args):
    # Package __init__ imports the model/backend. Load this standalone, stdlib
    # schedule module directly so dry-run never imports torch or opens a corpus.
    spec = importlib.util.spec_from_file_location("_formal_training_schedule", ROOT / "src/graphcast_compat/training_schedule.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    raw = json.loads(Path(args.schedule).read_text())
    overrides = {"warmup": args.warmup_updates, "one_step": args.one_step_updates,
                 "curriculum": args.curriculum_updates}
    for phase in raw["phases"]:
        if overrides.get(phase["name"]) is not None:
            phase["updates"] = overrides[phase["name"]]
        if phase["name"] == "curriculum" and args.curriculum_every_updates is not None:
            phase["rollout"]["every_updates"] = args.curriculum_every_updates
    raw["phases"] = [phase for phase in raw["phases"] if phase["updates"] != 0]
    return module.TrainingSchedule.from_dict(raw)


def validate_arguments(args):
    if args.mp != 4:
        raise ValueError("formal training requires MP=4")
    if args.dp <= 0 or args.global_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("DP and batch sizes must be positive")
    if args.global_batch_size % args.dp:
        raise ValueError("global batch size must be divisible by DP")
    devices = list(range(args.mp * args.dp))
    if args.devices is not None and [int(i) for i in args.devices.split(",")] != devices:
        raise ValueError("--devices must use contiguous logical indices starting at zero; select physical cards with vendor visibility variables")
    for name in ("seed", "checkpoint_segments", "loss_grid_chunk_size", "warmup_updates", "one_step_updates", "curriculum_updates"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("max_updates", "checkpoint_every", "curriculum_every_updates"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    schedule = load_schedule(args)
    if schedule.total_updates is None:
        raise ValueError("formal training requires finite stage budgets")
    return schedule, {"mp": args.mp, "dp": args.dp,
        "rank_device_groups": [devices[r * args.mp:(r + 1) * args.mp] for r in range(args.dp)],
        "global_batch_size": args.global_batch_size, "local_batch_size": args.global_batch_size // args.dp,
        "micro_batch_size_per_rank": args.micro_batch_size,
        "accumulation_steps": math.ceil(args.global_batch_size / (args.dp * args.micro_batch_size)),
        "storage_dtype": "float32", "requested_precision": "tf32_hf32" if args.tensor_core == "enabled" else "ieee_fp32",
        "resolved_precision": "unverified", "requested_backend": args.backend,
        "resolved_backend": "unverified", "forward_backward_coverage": "unverified", "training_mfu": None,
        "schedule": schedule.to_dict(), "schedule_digest": schedule.digest,
        "stop_at_update": min(args.max_updates or schedule.total_updates, schedule.total_updates),
        "epoch_policy": "fixed_maximum_rollout_universe_v1"}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        schedule, metadata = validate_arguments(args)
        if args.dry_run:
            print(json.dumps(metadata, sort_keys=True))
            return 0
        if not args.corpus or not args.stats:
            p.error("real training requires --corpus and --stats")
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    sys.path.insert(0, str(ROOT / "src"))
    from graphcast_compat.training_runtime import run_formal_training
    run_formal_training(args, schedule=schedule, metadata=metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
