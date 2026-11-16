"""Explicit H100 training entry; importing this module never starts a worker."""
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare_imports():
    """Keep maintained helpers and the original trainer import order explicit."""
    engine = Path(__file__).resolve().parents[1]
    for path in (engine / "training", engine / "original_training", engine.parents[2]):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return engine


def install_remat(candidate, output):
    """Install checkpoint policy before the original model is imported."""
    # Apply only explicitly selected stage barriers; regular3 remains the base.
    if candidate != "regular3":
        import haiku as hk
        from selective_remat import remat_action, function_stats_names

        original_remat = hk.remat
        hook_trace = output / "remat-hook-events.jsonl"
        assert not hook_trace.exists(), "Candidate output must be new"

        def remat_allow_cse(fun, *args, **kwargs):
            action = remat_action(candidate, fun.__name__)
            matched = action == "allow_cse"
            if matched:
                kwargs["prevent_cse"] = False
            names = function_stats_names(candidate, fun.__name__)
            if names:
                import jax

                if "policy" in kwargs:
                    raise ValueError(
                        "Named function already has an unexpected checkpoint policy"
                    )
                kwargs["policy"] = jax.checkpoint_policies.save_only_these_names(*names)
            with hook_trace.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            function_name=fun.__name__,
                            function_module=getattr(fun, "__module__", None),
                            matched=matched,
                            action=action,
                            effective_prevent_cse=None
                            if action == "bypass"
                            else kwargs.get("prevent_cse", True),
                            function_saved_residual_names=list(names),
                        )
                    )
                    + "\n"
                )
            if action == "bypass":
                return fun
            return original_remat(fun, *args, **kwargs)

        hk.remat = remat_allow_cse

    from selective_remat import (
        mesh_group_size,
        shifted_layernorm,
        mesh_gather,
        mesh_vjp,
        saves_layernorm_stats,
        edge_linear,
        analytic_layernorm,
        final_mesh_nodes,
        g2m_mesh_nodes,
    )

    if mesh_group_size(candidate) or candidate in [
        "mesh-message-cse",
        "mesh-message-lnremat",
        "spatial-stage-bypass-lnremat",
    ]:
        from selective_remat import install_message_policy

        install_message_policy(candidate, output)


def load_trainer():
    import streamed_train

    return streamed_train


def install_hooks(candidate, output):
    """Install the existing numerical hooks after importing the trainer."""
    from selective_remat import (
        mesh_group_size,
        shifted_layernorm,
        mesh_gather,
        mesh_vjp,
        saves_layernorm_stats,
        edge_linear,
        analytic_layernorm,
        final_mesh_nodes,
        g2m_mesh_nodes,
    )
    import regular3_hooks

    regular3_hooks.install()
    if saves_layernorm_stats(candidate):
        if analytic_layernorm(candidate):
            import training_layernorm_vjp as layernorm_hook
        else:
            import layernorm_stats as layernorm_hook
        layernorm_hook.install()
    if shifted_layernorm(candidate):
        import shifted_ln_hooks

        shifted_ln_hooks.install()
    if mesh_gather(candidate):
        if mesh_vjp(candidate):
            import training_mesh_vjp as mesh_hook
        else:
            import training_mesh_hooks as mesh_hook
        mesh_hook.install()
    if final_mesh_nodes(candidate):
        import training_final_mesh

        training_final_mesh.install()
    if g2m_mesh_nodes(candidate):
        import training_g2m_mesh

        training_g2m_mesh.install()
    if edge_linear(candidate):
        import training_edge_linear

        training_edge_linear.install()

    def save_all_audits(path):
        regular3_hooks.save_audit(path)
        if saves_layernorm_stats(candidate):
            layernorm_hook.save_audit(path.with_name("layernorm-stats-audit.json"))
        if shifted_layernorm(candidate):
            shifted_ln_hooks.save_audit(path.with_name("shifted-ln-audit.json"))
        if mesh_gather(candidate):
            mesh_hook.save_audit(path.with_name("training-mesh-audit.json"))
        if final_mesh_nodes(candidate):
            training_final_mesh.save_audit(path.with_name("final-mesh-audit.json"))
        if g2m_mesh_nodes(candidate):
            training_g2m_mesh.save_audit(path.with_name("g2m-mesh-audit.json"))
        if edge_linear(candidate):
            training_edge_linear.save_audit(path.with_name("edge-linear-audit.json"))

    return save_all_audits


def write_manifest(context):
    engine = Path(__file__).resolve().parents[1]
    sources = {}
    for folder, prefix, patterns in (
        (engine / "original_training", "/original/experiment/", ("*.py",)),
        (engine / "training", "/candidate/", ("*.py", "*.sh")),
    ):
        for pattern in patterns:
            for path in folder.glob(pattern):
                sources[prefix + path.name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    record = dict(
        candidate=context.config["training"]["candidate"],
        runtime_context=context.to_dict(),
        candidate_xla_flags=os.environ.get("CANDIDATE_XLA_FLAGS", ""),
        precision="FP32 storage; H100 TF32; no explicit RNE",
        environment={
            key: os.environ.get(key)
            for key in (
                "PHYSICAL_DEVICE",
                "EXPECTED_PCI",
                "NVIDIA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "CUDA_VISIBLE_DEVICES",
                "TRAINING_CANDIDATE",
                "MEMORY_COMPILE_ONLY",
                "CANDIDATE_XLA_FLAGS",
                "STREAM_MICROBATCHES",
            )
        },
        source_sha256=sources,
    )
    (context.output / "candidate-manifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def save_failure_memory(output, primary):
    if isinstance(primary, SystemExit) and primary.code in (None, 0):
        return  # argparse help is a successful exit, not a device failure.
    import jax

    payload = dict(
        primary_error=repr(primary),
        devices=[
            dict(device=str(device), memory_stats=device.memory_stats())
            for device in jax.devices()
        ],
        scope="Allocator counters after failure; unavailable counters are not device-wide capacity",
    )
    (output / "failure-memory.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv=None, context=None):
    # A new worker handles each horizon; checkpoint state carries across workers.
    # Install the selected hooks before the trainer traces its forward/backward graph.
    prepare_imports()
    from graphcast_bench.platforms.context import context_arguments
    from graphcast_bench.validation.audit import run_with_audit
    from graphcast_bench.progress import emit

    if context is None:
        context, argv = context_arguments(argv)
    context.require_kind("training")
    if context.platform != "h100":
        raise ValueError("H100 training requires an H100 runtime context")
    candidate, output = context.config["training"]["candidate"], context.output
    install_remat(candidate, output)
    train = load_trainer()
    save_audits = install_hooks(candidate, output)
    write_manifest(context)
    try:
        return run_with_audit(
            lambda: train.main(argv, context=context),
            save_audits,
            output / "regular3-audit.json",
            on_failure=lambda error: save_failure_memory(output, error),
        )
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        emit(
            "Training",
            "Stage failed: {}: {}".format(type(error).__name__, error),
            "ERROR",
        )
        raise


if __name__ == "__main__":
    main()
