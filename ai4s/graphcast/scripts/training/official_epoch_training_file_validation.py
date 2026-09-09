"""Checkpoint, replica-state, and metric helpers for the epoch trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from graphcast_compat.training_schedule import TrainingSchedule
from graphcast_compat.training_state import (
    atomic_save_checkpoint,
    build_training_checkpoint,
    capture_rng_state,
    load_training_checkpoint,
    sha256_file,
    state_object_sha256,
    validate_resume_contract,
)
from training_support import (
    differing_component_names,
    state_dict_cpu,
    state_dict_sha256,
    tensor_components_sha256,
)


# ==================================================
# 0. 原子写入
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


def write_immutable_json(path: str | Path, value: Any) -> None:
    """Write a run contract once and reject later content drift."""

    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        existing = json.loads(resolved.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(
                f"immutable run contract changed: {resolved}"
            )
        return
    _atomic_json_dump(resolved, value)


def checkpoint_file_sha256(path: str | Path) -> str:
    """Return the digest stored with a model checkpoint."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {resolved}")
    return sha256_file(resolved)


# ==================================================
# 1. 训练日程
# ==================================================
def load_schedule_manifest(path: str | Path) -> TrainingSchedule:
    """Load a schedule directly from its JSON file."""

    return TrainingSchedule.from_path(
        Path(path).expanduser().resolve()
    )


# ==================================================
# 2. 初始化与恢复 checkpoint 校验
# ==================================================
def load_model_only_checkpoint(
    path: str | Path,
    *,
    formal: bool,
    training_stage: str,
    tensor_core_enabled: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    """Validate a parent checkpoint used only for model initialization."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"initial checkpoint is missing: {resolved}"
        )
    payload = torch.load(
        resolved,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(payload, dict) and payload and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        # Raw historical weights are model-only initialization, never exact resume.
        payload = {"model_state": payload, "metadata": {
            "initialization_format": "raw_state_dict",
        }}
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("initial checkpoint has no model state")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("initial checkpoint metadata is missing")
    model_state = payload["model_state"]
    actual_hash = state_dict_sha256(model_state)
    recorded_hash = metadata.get("model_state_sha256")
    if recorded_hash is not None and recorded_hash != actual_hash:
        raise ValueError("initial checkpoint model-state hash disagrees")
    parent_stage = metadata.get("training_stage")
    parent_kind = metadata.get("dataset_kind")
    if formal and training_stage == "hres_finetune":
        if parent_stage != "era5_pretrain" or parent_kind != "era5":
            raise ValueError(
                "formal HRES adaptation requires an ERA5 pretraining "
                "checkpoint"
            )
        completed = int(
            metadata.get(
                "completed_optimizer_steps",
                payload.get("completed_optimizer_steps", 0),
            )
        )
        if completed < 30:
            raise ValueError(
                "formal ERA5 parent checkpoint contains fewer than 30 updates"
            )
    if metadata.get("precision") not in (None, "fp32"):
        raise ValueError("initial checkpoint is not FP32")
    recorded_tensor_core = metadata.get("tensor_core_enabled")
    if (
        recorded_tensor_core is not None
        and bool(recorded_tensor_core) != tensor_core_enabled
    ):
        raise ValueError("initial checkpoint math mode disagrees")
    return model_state, metadata, checkpoint_file_sha256(resolved)


def load_resume_checkpoint(
    path: str | Path,
    *,
    resume_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete exact-resume checkpoint and its state hashes."""

    resolved = Path(path).expanduser().resolve()
    # load_training_checkpoint already verifies the SHA-256 sidecar before
    # deserialization. Do not read a potentially large checkpoint twice.
    payload = load_training_checkpoint(resolved)
    validate_resume_contract(payload, resume_contract)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("resume checkpoint metadata is missing")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("resume checkpoint model state is missing")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("resume checkpoint optimizer state is missing")
    if (
        metadata.get("model_state_sha256")
        != state_dict_sha256(model_state)
    ):
        raise ValueError("resume model-state hash disagrees")
    if (
        metadata.get("optimizer_state_sha256")
        != state_object_sha256(optimizer_state)
    ):
        raise ValueError("resume optimizer-state hash disagrees")
    return payload


# ==================================================
# 3. 分布式状态一致性与 checkpoint 写入
# ==================================================
def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    return value


def validate_initial_replica_state(context: Any, model: torch.nn.Module) -> int:
    """Require byte-identical model tensors on every data-parallel rank."""

    hashes = context.gather_objects(
        tensor_components_sha256(state_dict_cpu(model))
    )
    differences = differing_component_names(hashes)
    if differences:
        raise RuntimeError(
            "DP ranks started with different model states: "
            + ",".join(differences[:10])
        )
    return len(hashes)


def validated_replicated_model_hashes(
    context: Any,
    model: torch.nn.Module,
) -> tuple[str, list[str]]:
    """Return the final model hash after checking all replicas agree."""

    local_hash = state_dict_sha256(state_dict_cpu(model))
    hashes = context.gather_objects(local_hash)
    if len(set(hashes)) != 1:
        raise RuntimeError("DP ranks ended with different model states")
    return local_hash, hashes


def save_validated_training_checkpoint(
    *,
    args: Any,
    context: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: Any,
    applied_schedule_value: Any,
    cursor: Any,
    epoch_plan: Any,
    run_dir: Path,
    resume_contract: Mapping[str, Any],
    initial_lineage: Mapping[str, Any],
    final: bool,
) -> dict[str, Any]:
    """Atomically save an exact-resume checkpoint after replica validation."""

    context.barrier()
    local_model_state = state_dict_cpu(model)
    local_model_hash = state_dict_sha256(local_model_state)
    model_hashes = context.gather_objects(local_model_hash)
    if len(set(model_hashes)) != 1:
        raise RuntimeError("DP model states differ at checkpoint")
    local_optimizer_state = _cpu_tree(optimizer.state_dict())
    local_optimizer_hash = state_object_sha256(local_optimizer_state)
    optimizer_hashes = context.gather_objects(local_optimizer_hash)
    if len(set(optimizer_hashes)) != 1:
        raise RuntimeError("DP optimizer states differ at checkpoint")
    local_rng = capture_rng_state(
        cuda_device_indices=context.device_indices,
        accelerator_type=context.accelerator_type,
    )
    rng_states = context.gather_objects(local_rng)
    checkpoint_path = (
        Path(args.save_checkpoint).expanduser().resolve()
        if final
        else (
            run_dir
            / "checkpoints"
            / f"step_{cursor.global_optimizer_step:08d}.pt"
        )
    )
    record_value = None
    if context.is_primary:
        metadata = {
            "training_stage": args.training_stage,
            "dataset_kind": args.dataset_kind,
            "precision": "fp32",
            "tensor_core_enabled": args.tensor_core == "enabled",
            "completed_optimizer_steps": cursor.global_optimizer_step,
            "model_state_sha256": local_model_hash,
            "model_state_sha256_by_rank": model_hashes,
            "optimizer_state_sha256": local_optimizer_hash,
            "optimizer_state_sha256_by_rank": optimizer_hashes,
            "resume_contract": dict(resume_contract),
            "initial_lineage": dict(initial_lineage),
            "epoch_plan": epoch_plan.to_metadata(),
            "corpus_index": resume_contract["corpus_index"],
            "schedule_digest": schedule.digest,
            "applied_schedule": {
                "update_index": applied_schedule_value.update_index,
                "phase_index": applied_schedule_value.phase_index,
                "phase_name": applied_schedule_value.phase_name,
                "phase_update_index": (
                    applied_schedule_value.phase_update_index
                ),
                "learning_rate": applied_schedule_value.learning_rate,
                "rollout_steps": applied_schedule_value.rollout_steps,
            },
        }
        payload = build_training_checkpoint(
            model_state=local_model_state,
            optimizer_state=local_optimizer_state,
            schedule_state={
                "name": schedule.name,
                "digest": schedule.digest,
                "next_update_index": cursor.global_optimizer_step,
                "last_applied_update_index": (
                    applied_schedule_value.update_index
                ),
                "last_applied_phase": (
                    applied_schedule_value.phase_name
                ),
                "last_applied_learning_rate": (
                    applied_schedule_value.learning_rate
                ),
                "last_applied_rollout_steps": (
                    applied_schedule_value.rollout_steps
                ),
            },
            cursor=cursor,
            rng_states_by_rank=rng_states,
            metadata=metadata,
        )
        record = atomic_save_checkpoint(
            checkpoint_path,
            payload,
            latest_path=run_dir / "checkpoints/latest.json",
        )
        record_value = {
            "path": str(record.path),
            "sha256": record.sha256,
            "global_optimizer_step": cursor.global_optimizer_step,
            "final": final,
        }
        if not final:
            periodic = sorted(
                (run_dir / "checkpoints").glob("step_*.pt")
            )
            for stale in periodic[: -args.keep_last]:
                stale.unlink()
                stale.with_suffix(stale.suffix + ".sha256").unlink(
                    missing_ok=True
                )
    context.barrier()
    records = context.gather_objects(record_value)
    return records[0]


# ==================================================
# 4. Metrics 分段文件校验
# ==================================================
def read_metric_segments(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read metric JSONL files and reject missing or duplicate steps."""

    root = Path(run_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "metrics").glob("segment_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"metric row must be an object: {path}")
            rows.append(value)
    rows.sort(key=lambda row: int(row["optimizer_step"]))
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != list(range(1, len(rows) + 1)):
        raise RuntimeError("metric segments are missing or duplicate steps")
    return rows
