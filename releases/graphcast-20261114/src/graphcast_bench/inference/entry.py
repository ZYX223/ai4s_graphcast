"""Apply named inference optimizations to the pinned official model."""

from graphcast_bench.validation.audit import run_with_audit
from graphcast_bench.validation.provenance import verify_maintained
from graphcast_bench.platforms.context import context_arguments
from graphcast_bench.progress import phase


def main(argv=None):
    # Delivery supplies paths and options through RuntimeContext. This entry owns
    # hook installation and audit saving; runner owns warmup and measured rollout.
    context, argv = context_arguments(argv)
    context.require_kind('inference')
    with phase('Initialize', 'Import inference dependencies'):
        from graphcast_bench.inference.runner import main as infer
        from graphcast_bench.model import reference_graph
        from graphcast_bench.optimizations.inference import layernorm, mesh, regular3
        from graphcast_bench.platforms.runtime import check_runtime

    with phase('Prepare', 'Verify the runtime and install inference optimizations'):
        check_runtime(context.config)
        verify_maintained()
        reference_graph.install(context.path('inference_graph'),
                                context.config['inference']['reference_graph_sha256'],
                                context.output / 'reference-graph-audit.jsonl')
        regular3.install()
        layernorm.install(include_regular3=False)
        mesh.install()
    output = context.output / "run"

    def save_audits(path):
        layernorm.save_audit(path)
        mesh.save_audit(output / "mesh-gather-audit.json")
        from graphcast_bench.assets import write_json

        write_json(
            output / "integration-verdict.json",
            dict(status="passed", all_optimization_audits_saved=True),
        )

    run_with_audit(lambda: infer(argv, context=context), save_audits,
                   output / "optimization-audit.json")


if __name__ == "__main__":
    main()
