"""Training lifecycle; the CLI's dry-run does not import this module."""
from __future__ import annotations

import time
from pathlib import Path
import json
import random
import sys

import numpy as np
import torch

from .epoch_plan import EpochPlan
from .gradient_accumulation import MicrobatchPlan
from .distributed_training import loss_scale_for_valid_samples
from .training_engine import run_training_update
from .training_state import (
    TrainingCursor, atomic_save_checkpoint, build_training_checkpoint,
    capture_rng_state, load_training_checkpoint, restore_rng_state,
    sha256_file, state_object_sha256, validate_resume_contract,
)

# Pinned colleague run_official_parallel_training.py:75-81. Atmospheric
# variables use the batch builder's default multiplier of 1.0.
PER_VARIABLE_WEIGHTS = {"2m_temperature": 1., "10m_u_component_of_wind": .1,
    "10m_v_component_of_wind": .1, "mean_sea_level_pressure": .1,
    "total_precipitation_6hr": .1}


def build_resume_contract(*, data_files, stats_files, config, policy):
    """Content identity in corpus order, independent of deployment paths."""
    return {"data_sha256": [sha256_file(p) for p in data_files],
            "stats_sha256": [sha256_file(p) for p in stats_files],
            "config_sha256": state_object_sha256(config), "policy": policy}


def validate_operational_grid(dataset, pressure_levels):
    expected_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    if tuple(pressure_levels) != expected_levels or tuple(dataset.pressure_levels) != expected_levels:
        raise ValueError("formal training requires the exact 13 operational pressure levels")
    if (len(dataset.latitudes) != 721 or len(dataset.longitudes) != 1440
            or not np.allclose(dataset.latitudes, np.linspace(90, -90, 721), rtol=0, atol=1e-6)
            or not np.allclose(dataset.longitudes, np.arange(1440) / 4, rtol=0, atol=1e-6)):
        raise ValueError("formal training requires the complete 721 x 1440 quarter-degree global grid")


def training_start_record(session, metadata):
    """One small CPU-readable comparison record per invocation, not per update.

    Reuse the existing resume contract; hash only parameters and optimizer state,
    not the large static graph buffers. This runs outside update timing.
    """
    return {"event": "start", **metadata, "loss_comparison": {
        "start_optimizer_step": session.cursor.global_optimizer_step,
        "starting_parameters_sha256": state_object_sha256(dict(session.model.named_parameters())),
        "starting_optimizer_sha256": state_object_sha256(session.optimizer.state_dict()),
        "resume_contract": session.resume_contract,
    }}


class TrainingSession:
    """Fixed maximum-rollout epoch universe and exact next-update cursor.

    ``prepare_batch`` extracts a GLOBAL microbatch and selects the rank slice.
    Only scalar epoch metadata is materialized ahead of backward.
    """
    def __init__(self, model, optimizer, *, context, schedule, window_indices,
                 global_batch_size, micro_batch_size, seed, prepare_batch,
                 resume_contract, shuffle=True, checkpoint_segments=0, loss_grid_chunk_size=0):
        self.model, self.optimizer, self.context = model, optimizer, context
        self.schedule, self.prepare_batch = schedule, prepare_batch
        self.window_indices = tuple(window_indices)
        self.global_batch_size, self.micro_batch_size = global_batch_size, micro_batch_size
        context.local_batch_size(global_batch_size=global_batch_size)
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        self.seed, self.shuffle = seed, shuffle
        self.checkpoint_segments, self.loss_grid_chunk_size = checkpoint_segments, loss_grid_chunk_size
        self.resume_contract = {**resume_contract, "schedule_digest": schedule.digest,
            "epoch_policy": "fixed_maximum_rollout_universe_v1", "seed": seed, "shuffle": shuffle,
            "global_batch_size": global_batch_size, "micro_batch_size_per_rank": micro_batch_size,
            "dp": context.dp_replicas, "mp": context.mp_devices_per_replica,
            "window_indices_sha256": state_object_sha256(self.window_indices),
            "checkpoint_segments": checkpoint_segments, "loss_grid_chunk_size": loss_grid_chunk_size}
        plan = self._epoch_plan = self._plan(0)
        self.cursor = TrainingCursor(0, 0, 0, plan.digest)
        self._safe_to_save = False
        self.model.train()

    def _plan(self, epoch):
        return EpochPlan.create(self.window_indices, epoch=epoch, seed=self.seed,
                                global_batch_size=self.global_batch_size, shuffle=self.shuffle)

    def update(self):
        self._safe_to_save = False
        plan = self._epoch_plan
        if self.cursor.epoch_plan_digest != plan.digest:
            raise ValueError("epoch plan digest disagrees with cursor")
        value = self.schedule.value_for_update(self.cursor.global_optimizer_step)
        epoch_batch = plan.batches[self.cursor.next_batch_index]
        parts = MicrobatchPlan.create(global_batch_size=self.global_batch_size,
                                     micro_batch_size=self.micro_batch_size * self.context.dp_replicas)
        for group in self.optimizer.param_groups:
            group["lr"] = value.learning_rate
        data_seconds = 0.
        def batches():
            nonlocal data_seconds
            for part in parts.parts:
                indices = epoch_batch.window_indices[part.batch_slice]
                weights = epoch_batch.sample_weights[part.batch_slice]
                if not any(weights):
                    continue  # Globally empty; every rank makes this same choice.
                local_size = len(indices) // self.context.dp_replicas
                start = self.context.rank * local_size
                local_valid = int(sum(weights[start:start + local_size]))
                scale = loss_scale_for_valid_samples(local_valid_count=local_valid,
                    global_valid_count=epoch_batch.valid_count, replicas=self.context.dp_replicas)
                before = time.perf_counter()
                prepared = self.prepare_batch(indices, weights, self.context.rank,
                                              self.context.dp_replicas, value.rollout_steps)
                data_seconds += time.perf_counter() - before
                yield prepared, scale
                del prepared
        before = time.perf_counter()
        result = run_training_update(self.model, self.optimizer, batches(), context=self.context,
            checkpoint_segments=self.checkpoint_segments, loss_grid_chunk_size=self.loss_grid_chunk_size)
        if self.context.device_indices and self.context.accelerator_type != "cpu":
            from .training_accelerator import synchronize_devices
            synchronize_devices(self.context.device_indices, self.context.accelerator_type)
        elapsed = time.perf_counter() - before
        next_batch, epoch = self.cursor.next_batch_index + 1, self.cursor.epoch
        if next_batch == plan.optimizer_steps:
            epoch, next_batch = epoch + 1, 0
            plan = self._epoch_plan = self._plan(epoch)
        self.cursor = TrainingCursor(epoch, next_batch, self.cursor.global_optimizer_step + 1, plan.digest)
        self._safe_to_save = True
        return {**result, "optimizer_step": self.cursor.global_optimizer_step,
                "epoch": epoch_batch.epoch, "batch_index": epoch_batch.batch_index,
                "rollout_steps": value.rollout_steps, "phase": value.phase_name,
                "learning_rate": value.learning_rate,
                "data_seconds": self.context.global_max_seconds(data_seconds),
                "compute_seconds": self.context.global_max_seconds(elapsed - data_seconds)}

    def save(self, path):
        if not self._safe_to_save:
            raise RuntimeError("checkpoint requires a successfully completed optimizer update")
        rng = capture_rng_state(cuda_device_indices=self.context.device_indices,
                                accelerator_type=self.context.accelerator_type)
        states = self.context.gather_objects(rng)
        error, record = None, None
        if self.context.is_primary:
            try:
                payload = build_training_checkpoint(model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(), schedule_state=self.schedule.to_dict(),
                    cursor=self.cursor, rng_states_by_rank=states,
                    metadata={"resume_contract": self.resume_contract})
                record = atomic_save_checkpoint(path, payload)
            except Exception as exc:
                error = str(exc)
        errors = self.context.gather_objects(error)
        if any(errors):
            raise RuntimeError(f"checkpoint save failed: {errors}")
        return str(record.path) if record else str(Path(path).resolve())

    def restore(self, path):
        payload = load_training_checkpoint(path)
        validate_resume_contract(payload, self.resume_contract)
        if payload["schedule_state"] != self.schedule.to_dict():
            raise ValueError("checkpoint schedule state mismatch")
        cursor = TrainingCursor.from_dict(payload["cursor"])
        plan = self._plan(cursor.epoch)
        if cursor.epoch_plan_digest != plan.digest or cursor.next_batch_index >= plan.optimizer_steps:
            raise ValueError("checkpoint cursor disagrees with epoch plan")
        if cursor.global_optimizer_step != cursor.epoch * plan.optimizer_steps + cursor.next_batch_index:
            raise ValueError("checkpoint optimizer step disagrees with cursor")
        if len(payload["rng_states_by_rank"]) != self.context.dp_replicas:
            raise ValueError("checkpoint rank RNG count disagrees with topology")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng_state(payload["rng_states_by_rank"][self.context.rank],
            cuda_device_indices=self.context.device_indices, accelerator_type=self.context.accelerator_type)
        self._epoch_plan = plan
        self.cursor, self._safe_to_save = cursor, True


def run_formal_training(args, *, schedule, metadata):
    """Assemble the operational model once and execute the shared update loop.

    Native and FlagGems must each enter in a fresh process. Data/model contract
    validation happens before accelerator discovery or any model allocation.
    """
    import yaml
    from .autoregressive_training import prepare_autoregressive_batch
    from .backend import (configure_flagos_environment, configure_float32_math,
                          resolve_runtime_backend, memory_text, reset_peak_memory, synchronize)
    from .distributed_training import DistributedContext
    from .google_small import GoogleSmallConfig, GoogleSmallCompatibleModel
    from .normalization import load_normalization_stats, REQUIRED_STATS_FILES
    from .official_training_corpus import OfficialTrainingCorpus
    from .training_recipe import (build_graphcast_adamw, graphcast_optimizer_metadata,
                                   validate_full_operational_model_config)

    config = GoogleSmallConfig.from_yaml(args.config)
    validate_full_operational_model_config(config)
    raw_config = yaml.safe_load(Path(args.config).read_text())
    maximum_rollout = max(int(phase.rollout.get("maximum", phase.rollout.get("value", 1)))
                          for phase in schedule.phases)
    with OfficialTrainingCorpus(args.corpus, config_path=args.config, rollout_steps=maximum_rollout) as corpus:
        if (corpus.manifest.dataset_kind, corpus.manifest.training_stage, corpus.manifest.data_role) != (
                "era5", "era5_pretrain", "train"):
            raise ValueError("formal entry requires an ERA5 pretraining train corpus")
        validate_operational_grid(corpus, raw_config["task_config"]["pressure_levels"])
        if args.backend == "native" and any(name == "flag_gems" or name.startswith("flag_gems.") for name in sys.modules):
            raise RuntimeError("Native baseline requires a fresh process before any FlagGems import")
        stats = load_normalization_stats(args.stats)
        try:
            configure_flagos_environment(mode=args.backend, config=args.flaggems_config,
                ops=args.flaggems_ops, strict=True, variant=args.flaggems_variant)
            # Context imports/registers the vendor before constructing vendor
            # torch.device objects and initializes one process per MP replica.
            with DistributedContext.from_environment(dp_replicas=args.dp, mp_devices_per_replica=args.mp,
                    accelerator_type=args.device_type, initialize=args.dp > 1) as ctx:
                runtime = resolve_runtime_backend(str(ctx.primary_device))
                if runtime.device.type != args.device_type or runtime.device.index != ctx.device_indices[0]:
                    raise RuntimeError("resolved device disagrees with requested rank topology")
                if args.backend == "flaggems" and runtime.flagos.status != "enabled_only_enable":
                    raise RuntimeError("FlagGems training registration did not succeed")
                precision = configure_float32_math(args.tensor_core == "enabled", runtime.device.type)
                devices = [torch.device(args.device_type, index) for index in ctx.device_indices]
                policy = {"storage_dtype": "float32", "tensor_core": args.tensor_core,
                    "resolved_precision": precision, "backend": args.backend,
                    "runtime_profile": runtime.profile.name, "accelerator_type": args.device_type,
                    "runtime_versions": runtime.versions,
                    "flaggems_variant": runtime.flagos.variant,
                    "requested_ops": list(runtime.flagos.requested_ops),
                    "per_variable_weights": PER_VARIABLE_WEIGHTS,
                    "optimizer": graphcast_optimizer_metadata(lr=0.),
                    "activation_checkpointing": True, "normalization": "inputs_and_residuals_v1"}
                contract = build_resume_contract(data_files=[shard.path for shard in corpus.manifest.shards],
                    stats_files=[Path(args.stats) / name for name in REQUIRED_STATS_FILES],
                    config={key: raw_config[key] for key in ("model_config", "task_config", "channel_contract")},
                    policy=policy)
                # Identical initialization on each replica; rank-local stochastic
                # execution begins after all model construction has completed.
                random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                model = GoogleSmallCompatibleModel(config, activation_checkpointing=True,
                                                   fuse_post_layer_norm_residual=False).float()
                if args.pretrained_checkpoint:
                    state = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=True)
                model.init_graph(latitudes=corpus.latitudes, longitudes=corpus.longitudes)
                model.set_model_parallel_devices(grid2mesh_device=devices[0],
                    mesh_step_devices=[devices[1]] * 8 + [devices[2]] * 8, mesh2grid_device=devices[3])
                if any(p.dtype != torch.float32 for p in model.parameters()):
                    raise RuntimeError("formal training requires FP32 model storage")
                optimizer = build_graphcast_adamw(model, lr=schedule.value_for_update(0).learning_rate)
                random.seed(args.seed + ctx.rank); np.random.seed(args.seed + ctx.rank)
                torch.manual_seed(args.seed + ctx.rank)

                def prepare(indices, weights, rank, replicas, rollout):
                    sample = corpus.sample_indices(window_indices=indices, sample_weights=weights,
                        rank=rank, replicas=replicas, target_rollout_steps=rollout)
                    prepared = prepare_autoregressive_batch(sample.inputs, sample.targets, sample.forcings,
                        config=config, stats=stats, input_device=devices[0], storage_device=torch.device("cpu"),
                        per_variable_weights=PER_VARIABLE_WEIGHTS, sample_weights=sample.sample_weights)
                    # Include data upload time in data_seconds, not compute.
                    synchronize(devices[0])
                    return prepared

                session = TrainingSession(model, optimizer, context=ctx, schedule=schedule,
                    window_indices=corpus.window_indices, global_batch_size=args.global_batch_size,
                    micro_batch_size=args.micro_batch_size, seed=args.seed, prepare_batch=prepare,
                    resume_contract=contract, checkpoint_segments=args.checkpoint_segments,
                    loss_grid_chunk_size=args.loss_grid_chunk_size)
                if args.resume_checkpoint:
                    session.restore(args.resume_checkpoint)
                output = Path(args.output_dir).resolve()
                if ctx.is_primary:
                    output.mkdir(parents=True, exist_ok=True)
                ctx.barrier()
                resolved = {**metadata, "resolved_precision": precision, "resolved_backend": runtime.to_metadata(),
                    "forward_backward_coverage": "unverified", "training_mfu": None,
                    "memory_acceptance": "unverified", "maximum_rollout_eligibility": maximum_rollout,
                    "resume_checkpoint": args.resume_checkpoint}
                if ctx.is_primary:
                    line = json.dumps(training_start_record(session, resolved), sort_keys=True)
                    print(line, flush=True)
                    with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(line + "\n")
                for device in devices:
                    reset_peak_memory(device)
                while session.cursor.global_optimizer_step < metadata["stop_at_update"]:
                    row = session.update()
                    for device in devices:
                        synchronize(device)
                    row["memory_by_rank"] = ctx.gather_objects({str(device): memory_text(device) for device in devices})
                    step = session.cursor.global_optimizer_step
                    checkpoint = None
                    if step % args.checkpoint_every == 0 or step == metadata["stop_at_update"]:
                        checkpoint = session.save(output / f"update-{step:09d}.pt")
                    row.update({"checkpoint": checkpoint, "checkpoint_status": "saved" if checkpoint else "not_due",
                                "event": "update", "training_mfu": None, "forward_backward_coverage": "unverified"})
                    if ctx.is_primary:
                        line = json.dumps(row, sort_keys=True)
                        print(line, flush=True)
                        with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(line + "\n")
        finally:
            for dataset in (stats.stddev_by_level, stats.mean_by_level, stats.diffs_stddev_by_level):
                dataset.close()
