"""Install the reviewed runtime policy, then execute one training horizon.

Installation order is explicit: geometry -> remat -> aggregate/LN markers ->
post-AD RNE lowering -> trainer. Each horizon runs in a fresh process.
"""

import json
import os
from pathlib import Path

from graphcast_bench.assets import write_json
from graphcast_bench.validation.audit import run_with_audit
from graphcast_bench.validation.provenance import verify_maintained
from graphcast_bench.platforms.context import context_arguments


def install_remat(output, candidate):
    import haiku as hk
    import jax

    from graphcast_bench.optimizations.training import remat

    remat.require_candidate(candidate)
    original = hk.remat

    def remat_policy(fun, *args, **kwargs):
        action = remat.remat_action(candidate, fun.__name__)
        names = remat.function_stats_names(candidate, fun.__name__)
        if names:
            if "policy" in kwargs:
                raise ValueError("Named function already has a checkpoint policy")
            kwargs["policy"] = jax.checkpoint_policies.save_only_these_names(*names)
        event = dict(
            function_name=fun.__name__,
            function_module=getattr(fun, "__module__", None),
            action=action,
            matched=False,
            function_saved_residual_names=list(names),
            effective_prevent_cse=None if action == "bypass" else kwargs.get("prevent_cse", True),
        )
        with (output / "remat-hook-events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        return fun if action == "bypass" else original(fun, *args, **kwargs)

    hk.remat = remat_policy
    remat.install_message_policy(candidate, output)


def install_rne(output, policy):
    from graphcast_bench.optimizations.rne import lowering
    from graphcast_bench.optimizations.rne.warmup import WarmupGate
    from graphcast_bench.platforms.runtime import Precision

    if policy != "warmup-verify" or os.environ["RNE_PRE_ROUNDED"] != policy:
        raise ValueError("Training requires verified warmup before formal RNE skips")
    lowering.install("rocm", audit_path=output / "lowering.jsonl")
    gate = WarmupGate()
    original = Precision.select
    os.environ["RNE_PRE_ROUNDED"] = "verify"

    def select(self, mode, label):
        os.environ["RNE_PRE_ROUNDED"] = gate.select(label, self.counts())
        if gate.verified_calls and not (output / "warmup-rne-proof.json").exists():
            write_json(
                output / "warmup-rne-proof.json",
                dict(
                    status="passed",
                    warmup_gemm_calls=gate.verified_calls,
                    first_formal_label=label,
                    policy="verify warmup -> skip formal",
                    trainer="Synchronized compiled gradient warmup; same executable for formal microbatches",
                ),
            )
        return original(self, mode, label)

    Precision.select = select
    candidate = dict(
        lowering_enabled=True,
        rne_policy="warmup-verify",
        precision="FP32 storage; post-AD operand RNE in JAX/XLA/FlagTree, XF32 rocBLAS",
        scope="One horizon per process; CPU regression is not GPU revalidation",
    )
    write_json(output / "rne-candidate.json", candidate)
    return candidate


def main(argv=None):
    # Delivery starts one worker per rollout horizon. Target shapes change between
    # horizons; the trainer reuses this shape's executables within the stage.
    context, argv = context_arguments(argv)
    context.require_kind('training')
    from graphcast_bench.platforms.host_memory import configure_training_memory

    # Set host-page policy before accelerator imports or driver allocations.
    configure_training_memory(context.output)
    from graphcast_bench.model import reference_graph
    from graphcast_bench.optimizations.training import (
        final_mesh,
        g2m_mesh,
        layernorm_stats,
        mesh_vjp,
        regular3,
    )
    from graphcast_bench.platforms.runtime import check_runtime

    check_runtime(context.config)
    manifest = verify_maintained()
    output = context.output
    settings = context.config['training']
    reference_graph.install(context.path('training_graph'), settings['reference_graph_sha256'],
                            output / 'reference-graph-audit.jsonl')
    install_remat(output, settings['candidate'])
    regular3.install()
    layernorm_stats.install()
    mesh_vjp.install()
    final_mesh.install()
    g2m_mesh.install()
    candidate = install_rne(output, settings['rne_policy'])
    write_json(
        output / "candidate-manifest.json",
        dict(
            candidate=settings['candidate'],
            runtime_context=context.to_dict(),
            code=manifest,
            environment={
                key: os.environ.get(key)
                for key in (
                    "XLA_MODE",
                    "JAX_DEFAULT_MATMUL_PRECISION",
                    "XLA_FLAGS",
                )
            },
            precision=candidate["precision"],
        ),
    )

    from graphcast_bench.training.runner import main as train

    def save_audits(path):
        regular3.save_audit(path)
        for module, filename in (
            (layernorm_stats, "layernorm-stats-audit.json"),
            (mesh_vjp, "training-mesh-audit.json"),
            (final_mesh, "final-mesh-audit.json"),
            (g2m_mesh, "g2m-mesh-audit.json"),
        ):
            module.save_audit(output / filename)

    def save_failure(primary):
        import jax

        write_json(
            output / "failure-memory.json",
            dict(
                primary_error=repr(primary),
                devices=[dict(device=str(d), memory_stats=d.memory_stats()) for d in jax.devices()],
            ),
        )

    try:
        run_with_audit(lambda: train(argv, context=context), save_audits,
                       output / "regular3-audit.json", on_failure=save_failure)
    finally:
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report["precision_legacy"] = report["precision"]
            report["precision"] = candidate["precision"]
            report["rne_lowering_candidate"] = candidate
            write_json(report_path, report)


if __name__ == "__main__":
    main()
