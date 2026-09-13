#!/usr/bin/env python3
"""Formal FP32-storage training entry. Help/dry-run use only the standard library."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_platform import PLATFORMS, platform_policy


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/inference/google_graphcast_operational_25km.yaml"))
    p.add_argument("--corpus")
    p.add_argument("--stats")
    checkpoints = p.add_mutually_exclusive_group()
    checkpoints.add_argument("--pretrained-checkpoint", help="Converted PyTorch model state dict")
    checkpoints.add_argument("--resume-checkpoint", help="Optimizer-boundary checkpoint plus SHA256 sidecar")
    p.add_argument("--output-dir", default="training-output")
    p.add_argument("--device-type", choices=("cuda", "npu", "musa"), default=None)
    p.add_argument("--devices", help="All contiguous logical device indices, e.g. 0,1,2,3")
    p.add_argument("--mp", type=int, default=4)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=16)
    microbatch = p.add_mutually_exclusive_group()
    microbatch.add_argument("--micro-batch-size-per-rank", type=int, default=1,
                           help="Samples per DP rank per forward/backward")
    microbatch.add_argument("--micro-batch-size", dest="micro_batch_size_per_rank", type=int,
                           default=argparse.SUPPRESS,
                           help="Compatibility alias for --micro-batch-size-per-rank")
    p.add_argument("--precision", choices=("fp32",), default="fp32", help="Parameter/activation/loss storage")
    p.add_argument("--tensor-core", choices=("enabled", "disabled"), default="enabled")
    p.add_argument("--backend", choices=("native", "flaggems"), default="native")
    p.add_argument("--flaggems-config", default=str(ROOT / "configs/flaggems/graphcast_training.yaml"))
    p.add_argument("--flaggems-ops", default=None, help="Optional explicit training allowlist override")
    p.add_argument("--flaggems-variant", choices=("installed", "official", "internal"), default="installed")
    p.add_argument("--platform", choices=tuple(PLATFORMS), default=os.environ.get("GRAPHCAST_PLATFORM"))
    p.add_argument("--physical-devices", help="Physical cards to expose, in stage order")
    for phase, count in (("warmup", 1000), ("one-step", 299000), ("curriculum", 11000)):
        p.add_argument(f"--{phase}-updates", type=int, default=count)
    p.add_argument("--max-updates", type=int, help="Stop at this absolute optimizer step; does not change schedule identity")
    p.add_argument("--checkpoint-every", type=int, default=20)
    for name in ("mesh-split-step", "edge-chunk-size", "grid-node-chunk-size", "mesh-edge-chunk-size", "loss-grid-chunk-size"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--split-cache-max-gib", type=float)
    p.add_argument("--data-prefetch-depth", type=int, choices=(0, 1))
    p.add_argument("--gradient-sync", choices=("manual", "ddp"))
    p.add_argument("--mesh-checkpoint-granularity", choices=("auto", "stage", "step"))
    p.add_argument("--mesh-edge-input-projection", choices=("concatenated", "factorized"),
                   help="Opt-in Mesh-only first-layer projection experiment")
    p.add_argument("--rollout-checkpoint-implementation", choices=("non_reentrant", "reentrant"))
    for name in ("grid2mesh-stage-device", "grid2mesh-node-stage-device", "mesh2grid-stage-device", "mesh2grid-node-stage-device", "mesh2grid-decoder-stage-device"):
        choices = ("primary", "mesh_first", "mesh_second", "output")
        if "-node-" in name:
            choices += ("stage",)
        p.add_argument("--" + name, choices=choices)
    for name in ("grid2mesh-stage-checkpoint", "mesh2grid-stage-checkpoint", "single-step-rollout-checkpoint"):
        p.add_argument("--" + name, choices=("enabled", "disabled"))
    p.add_argument("--training-start", default="")
    p.add_argument("--training-end", default="")
    p.add_argument("--peak-tflops-per-device", type=float, default=0.)
    p.add_argument("--peak-tflops-source", default="unavailable")
    p.add_argument("--profile-hotspots", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_contract(args):
    if (args.mp, args.dp) not in {(1, 1), (2, 1), (2, 2), (4, 1), (4, 2)}:
        raise ValueError("formal training supports MP=1/DP=1, or MP=2 or 4 with DP=1 or 2")
    for name in ("global_batch_size", "micro_batch_size_per_rank", "warmup_updates", "one_step_updates", "curriculum_updates", "checkpoint_every"):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if args.seed < 0 or (args.max_updates is not None and args.max_updates <= 0):
        raise ValueError("seed must be non-negative and max-updates positive")
    global_micro_batch_size = args.micro_batch_size_per_rank * args.dp
    if args.global_batch_size % global_micro_batch_size:
        raise ValueError("global batch size must be divisible by DP * local micro-batch size")
    gradient_accumulation_steps = args.global_batch_size // global_micro_batch_size
    if args.curriculum_updates % 11:
        raise ValueError("curriculum-updates must be divisible by 11")
    if not math.isfinite(args.peak_tflops_per_device) or args.peak_tflops_per_device < 0:
        raise ValueError("peak must be finite and non-negative")
    if args.peak_tflops_per_device > 0 and args.peak_tflops_source in ("", "unknown", "unavailable"):
        raise ValueError("a positive peak requires --peak-tflops-source")
    devices = list(range(args.mp * args.dp))
    if args.devices and [int(i) for i in args.devices.split(",")] != devices:
        raise ValueError("--devices must be contiguous logical indices starting at zero")
    execution, env = platform_policy(args)
    for key, value in execution.items():
        if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
            raise ValueError(key + " must be finite and non-negative")
    env.update(GRAPHCAST_FLAGOS_MODE="off" if args.backend == "native" else "flaggems",
        FLAGOS_REPLACE="0" if args.backend == "native" else "1",
        GRAPHCAST_FLAGGEMS_CONFIG=args.flaggems_config,
        GRAPHCAST_FLAGGEMS_OPS=args.flaggems_ops or "",
        GRAPHCAST_FLAGGEMS_VARIANT=args.flaggems_variant, GRAPHCAST_STRICT_FLAGOS="1")
    # Direct standalone import avoids Torch/data through the package initializer.
    spec = importlib.util.spec_from_file_location("_canonical_schedule", ROOT / "src/graphcast_compat/training_schedule.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    raw = json.loads((ROOT / "configs/training/official_2_to_12.json").read_text())
    for phase, count in zip(raw["phases"], (args.warmup_updates, args.one_step_updates, args.curriculum_updates)):
        phase["updates"] = count
    raw["phases"][2]["rollout"]["every_updates"] = args.curriculum_updates // 11
    schedule = module.TrainingSchedule.from_dict(raw)
    values = dict(config=args.config, stats_dir=args.stats or "<stats>", corpus_manifest=args.corpus or "<corpus>",
        dataset_kind="era5", training_stage="era5_pretrain", global_batch_size=args.global_batch_size,
        global_micro_batch_size=global_micro_batch_size, data_parallel_replicas=args.dp, mp_devices_per_replica=args.mp,
        rollout_steps=12, rollout_checkpoint_segments=12, tensor_core=args.tensor_core, precision="fp32",
        init_mode="checkpoint" if args.pretrained_checkpoint else "random", schedule_recipe="official_2_to_12",
        warmup_updates=args.warmup_updates, training_updates=args.one_step_updates, updates=args.curriculum_updates,
        run_dir=args.output_dir, save_checkpoint=str(Path(args.output_dir) / "final_checkpoint.pt"),
        checkpoint_every_steps=args.checkpoint_every, stop_after_global_step=args.max_updates or 0,
        seed=args.seed, peak_tflops_per_device=args.peak_tflops_per_device,
        peak_tflops_source=args.peak_tflops_source, training_start=args.training_start, training_end=args.training_end,
        **execution)
    argv = [item for key, value in values.items() for item in ("--" + key.replace("_", "-"), str(value))]
    argv += ["--formal", "--activation-checkpoint"]
    if args.pretrained_checkpoint:
        argv += ["--init-checkpoint", args.pretrained_checkpoint, "--allow-pretrained-era5-init"]
    if args.resume_checkpoint:
        argv += ["--resume-training-checkpoint", args.resume_checkpoint]
    if args.profile_hotspots:
        argv += ["--profile-hotspots"]
    command = [sys.executable]
    if "RANK" not in os.environ:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(args.dp)]
    command += [str(Path(__file__).with_name("run_official_epoch_training.py"))] + argv
    return dict(mp=args.mp, dp=args.dp, rank_device_groups=[devices[r * args.mp:(r + 1) * args.mp] for r in range(args.dp)],
        global_batch_size=args.global_batch_size, local_batch_size=args.global_batch_size // args.dp,
        micro_batch_size_per_rank=args.micro_batch_size_per_rank,
        global_micro_batch_size=global_micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Preserve existing dry-run consumers; checkpoint field names are unchanged.
        canonical_micro_batch_size=global_micro_batch_size,
        accumulation_steps=gradient_accumulation_steps, storage_dtype="float32",
        requested_precision="tf32_hf32" if args.tensor_core == "enabled" else "ieee_fp32",
        resolved_precision="unverified", requested_backend=args.backend, resolved_backend="unverified",
        forward_backward_coverage="unverified", training_mfu=None, schedule=schedule.to_dict(), schedule_digest=schedule.digest,
        stop_at_update=min(args.max_updates or schedule.total_updates, schedule.total_updates),
        epoch_policy="trajectory_with_replacement_per_optimizer_update", execution=execution, environment=env,
        engine_argv=argv, launch_command=command)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        contract = resolve_contract(args)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(contract, sort_keys=True))
        return 0
    if not args.corpus or not args.stats:
        p.error("real training requires --corpus and --stats")
    os.environ.update(contract["environment"])
    if contract["environment"]["GRAPHCAST_PLATFORM"] == "mthreads":
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard) if hard >= 0 else 65536, hard))
    return subprocess.call(contract["launch_command"])


if __name__ == "__main__":
    raise SystemExit(main())
