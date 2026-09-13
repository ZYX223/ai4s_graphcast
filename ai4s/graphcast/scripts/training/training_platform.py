"""Training-only platform policies adapted from b7667b4 train_pipeline.sh."""
import os

PLATFORMS = {
    "h100": ("cuda", "nvidia", "NVIDIA H100", "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF"),
    "metax": ("cuda", "metax", "MetaX C550", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"),
    "hygon": ("cuda", "hygon", "Hygon BW3000", "HIP_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"),
    "thead": ("cuda", "ptg", "T-Head PPU-ZW810E", "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF"),
    "mthreads": ("musa", "mthreads", "MThreads S5000", "MUSA_VISIBLE_DEVICES", "PYTORCH_MUSA_ALLOC_CONF"),
    "ascend910c": ("npu", "ascend", "Ascend 910C", "ASCEND_RT_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF"),
}


def platform_policy(args):
    platform = args.platform or {"cuda": "h100", "npu": "ascend910c", "musa": "mthreads"}[args.device_type or "cuda"]
    device, backend, label, visibility, allocator = PLATFORMS[platform]
    if args.device_type and args.device_type != device:
        raise ValueError("--platform and --device-type disagree")
    env = {"GRAPHCAST_PLATFORM": platform, "GRAPHCAST_PLATFORM_LABEL": label,
        "GRAPHCAST_ACCELERATOR_TYPE": device, "GRAPHCAST_BACKEND": backend,
        allocator: os.environ.get(allocator, "expandable_segments:True")}
    if visibility in os.environ:
        env[visibility] = os.environ[visibility]
    if args.physical_devices:
        cards = args.physical_devices.split(",")
        if len(cards) != args.mp * args.dp or len(set(cards)) != len(cards):
            raise ValueError("--physical-devices must select exactly MP * DP distinct cards")
        env[visibility] = args.physical_devices
    values = dict(gradient_sync="manual" if args.mp in (1, 4) else "ddp",
        mesh_split_step=8, mesh_checkpoint_granularity="auto",
        mesh_edge_input_projection="concatenated",
        edge_chunk_size=131072, grid_node_chunk_size=131072,
        mesh_edge_chunk_size=0, loss_grid_chunk_size=0,
        grid2mesh_stage_device="primary", grid2mesh_node_stage_device="stage",
        grid2mesh_stage_checkpoint="enabled", mesh2grid_stage_device="output",
        mesh2grid_node_stage_device="stage", mesh2grid_decoder_stage_device="output",
        mesh2grid_stage_checkpoint="disabled", single_step_rollout_checkpoint="disabled",
        rollout_checkpoint_implementation="non_reentrant",
        split_cache_max_gib=16.0, data_prefetch_depth=1)
    if platform == "ascend910c":
        env["ASCEND_LAUNCH_BLOCKING"] = os.environ.get("GRAPHCAST_ASCEND_LAUNCH_BLOCKING", "1")
        if env["ASCEND_LAUNCH_BLOCKING"] not in ("0", "1"):
            raise ValueError("GRAPHCAST_ASCEND_LAUNCH_BLOCKING must be 0 or 1")
        values.update(edge_chunk_size=8192, grid_node_chunk_size=4096,
            mesh_edge_chunk_size=8192, loss_grid_chunk_size=65536,
            single_step_rollout_checkpoint="enabled", rollout_checkpoint_implementation="reentrant",
            mesh_split_step=16, mesh_checkpoint_granularity="stage",
            grid2mesh_stage_device="output", grid2mesh_node_stage_device="output",
            mesh2grid_node_stage_device="mesh_second", mesh2grid_decoder_stage_device="mesh_second",
            mesh2grid_stage_checkpoint="enabled")
        if args.mp == 4 and args.dp == 1:
            # Limit mesh replay to one message step instead of keeping all
            # sixteen steps' chunk activations live on mesh_first.
            values.update(edge_chunk_size=131072, grid_node_chunk_size=65536,
                mesh_edge_chunk_size=32768, mesh_checkpoint_granularity="step")
    if platform == "thead":
        env.update(GRAPHCAST_SKIP_PROCESS_GROUP_TEARDOWN="1", GRAPHCAST_BYPASS_STATIC_DESTRUCTORS_ON_SUCCESS="1")
        values.update(single_step_rollout_checkpoint="enabled",
                      mesh2grid_stage_device="mesh_second", mesh2grid_stage_checkpoint="enabled")
    if args.mp in (1, 2):
        values.update(edge_chunk_size=8192, grid_node_chunk_size=4096,
            mesh_edge_chunk_size=8192, loss_grid_chunk_size=65536,
            mesh_split_step=8, mesh_checkpoint_granularity="step",
            rollout_checkpoint_implementation="non_reentrant",
            grid2mesh_stage_device="primary", grid2mesh_node_stage_device="stage",
            mesh2grid_stage_device="output", mesh2grid_node_stage_device="primary",
            mesh2grid_decoder_stage_device="primary", mesh2grid_stage_checkpoint="enabled",
            single_step_rollout_checkpoint="enabled")
        if args.mp == 1:
            # Share the same single-device recipe across all accelerator backends.
            values.update(mesh_split_step=16,
                rollout_checkpoint_implementation="reentrant")
    for key in values:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    if values["gradient_sync"] == "ddp" and values["rollout_checkpoint_implementation"] == "reentrant":
        raise ValueError("DDP requires non_reentrant rollout checkpointing")
    return values, env
