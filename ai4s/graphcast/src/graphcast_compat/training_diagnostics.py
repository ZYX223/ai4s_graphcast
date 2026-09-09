"""Segment evidence helpers. Readers import no training dependencies."""
import hashlib
from pathlib import Path

SEGMENT_SCHEMA = "graphcast.training_segment_comparison.v1"
SOURCE_CONTRACT_FIELDS = (
    "precision", "tensor_core_enabled", "schedule_digest", "seed", "training_sampling",
    "global_batch_size", "global_micro_batch_size", "local_micro_batch_size",
    "gradient_accumulation_steps", "data_parallel_replicas", "mp_devices_per_replica",
    "gradient_sync", "activation_checkpoint", "rollout_checkpoint_segments",
    "rollout_checkpoint_implementation", "single_step_rollout_checkpoint",
    "mesh_checkpoint_granularity", "loss_grid_chunk_size", "graph_topology_version",
    "mesh_split_step", "grid2mesh_stage_device", "grid2mesh_node_stage_device",
    "grid2mesh_stage_checkpoint", "mesh2grid_stage_device", "mesh2grid_node_stage_device",
    "mesh2grid_decoder_stage_device", "mesh2grid_stage_checkpoint",
    "mesh_edge_chunk_size_override", "edge_chunk_size_override", "grid_node_chunk_size_override",
)
SOURCE_EXECUTION_FIELDS = ("data_prefetch_depth", "split_cache_max_gib", "backward_execution")
# RuntimeBackend._runtime_versions emits all six keys even when a package/runtime
# is absent (not-installed/not-available). FlagGems/FlagTree are the intentional
# replacement-side difference, not the common Torch/vendor runtime identity.
COMMON_RUNTIME_VERSION_FIELDS = (
    "torch", "torch_cuda", "torch_hip", "torch_npu", "torch_musa", "torch_metax",
)
SOURCE_UPDATE_ALIASES = {
    "batch_index": "epoch_batch_index", "phase": "schedule_phase",
    "effective_samples": "valid_sample_count", "accumulation_steps": "gradient_accumulation_steps",
    "finite": "grad_ok", "gradient_norm": "grad_norm_before_clip",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_state(model, optimizer, rank):
    from .training_state import state_object_sha256
    return {"rank": rank,
        "parameters_sha256": state_object_sha256(dict(model.named_parameters())),
        "optimizer_sha256": state_object_sha256(optimizer.state_dict())}


def segment_start_record(*, start_optimizer_step, states_by_rank, runtime_by_rank,
                         resume_contract, resolved_precision, manifest, optimizer,
                         per_variable_weights, execution_policy):
    """Hash consumed bytes once per invocation, outside optimizer update timing."""
    from .normalization import STDDEV_FILE, MEAN_FILE, DIFFS_STDDEV_FILE
    stats_dir = Path(resume_contract["stats_dir"])
    return {"schema": SEGMENT_SCHEMA, "start_optimizer_step": start_optimizer_step,
        "states_by_rank": states_by_rank, "runtime_by_rank": runtime_by_rank,
        "resume_contract": resume_contract, "resolved_precision": resolved_precision,
        "data_sha256": [file_sha256(shard.path) for shard in manifest.shards],
        "data_paths": [str(shard.path) for shard in manifest.shards],
        "corpus_index_sha256": file_sha256(manifest.path),
        "stats_sha256": {name: file_sha256(stats_dir / name)
                         for name in (STDDEV_FILE, MEAN_FILE, DIFFS_STDDEV_FILE)},
        "config_sha256": file_sha256(resume_contract["config"]),
        "optimizer": optimizer, "normalization": "inputs_and_residuals_v1",
        "per_variable_weights": per_variable_weights, "execution_policy": execution_policy}


def source_context(run):
    run = run or {}
    contract = run.get("resume_contract") or {}
    # Content hashes replace path identity; the backend is the intentional paired
    # variable. Keep every other recorded sampler, schedule and memory choice.
    excluded = {"operator_replacement", "corpus_index", "stats_dir", "config",
                "validation_corpus_index", "schedule_source"}
    result = {name: run.get(name) for name in ("start_optimizer_step",
        "data_sha256", "stats_sha256", "config_sha256", "optimizer", "normalization",
        "per_variable_weights", "execution_policy")}
    result["source_recipe"] = {k: v for k, v in contract.items() if k not in excluded}
    result["state_rank_count"] = len(run.get("states_by_rank") or []) or None
    result.update({"source." + name: contract.get(name) for name in SOURCE_CONTRACT_FIELDS})
    execution = run.get("execution_policy") or {}
    result.update({"execution_policy." + name: execution.get(name) for name in SOURCE_EXECUTION_FIELDS})
    for index, state in enumerate(run.get("states_by_rank") or []):
        for name in ("rank", "parameters_sha256", "optimizer_sha256"):
            result[f"state.{index}.{name}"] = state.get(name)
    for index, replacement in enumerate(run.get("runtime_by_rank") or []):
        runtime = replacement.get("runtime") or {}
        for name in ("backend", "device_type", "runtime_family", "chip"):
            result[f"runtime.{index}.{name}"] = runtime.get(name)
        versions = runtime.get("versions") or {}
        for name in COMMON_RUNTIME_VERSION_FIELDS:
            result[f"runtime.{index}.versions.{name}"] = versions.get(name)
    return result


def training_start_record(session, metadata):
    """Read-compatible legacy start event; no legacy execution engine."""
    state = initial_state(session.model, session.optimizer, 0)
    return {"event": "start", **metadata, "loss_comparison": {
        "start_optimizer_step": session.cursor.global_optimizer_step,
        "starting_parameters_sha256": state["parameters_sha256"],
        "starting_optimizer_sha256": state["optimizer_sha256"],
        "resume_contract": session.resume_contract}}
