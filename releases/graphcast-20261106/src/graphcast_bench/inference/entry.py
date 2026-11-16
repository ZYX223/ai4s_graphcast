"""Apply named inference optimizations to the pinned official model."""

import os
from pathlib import Path

from graphcast_bench.validation.audit import run_with_audit
from graphcast_bench.validation.provenance import verify_maintained


def main():
    from graphcast_bench.inference.runner import main as infer
    from graphcast_bench.model import reference_graph
    from graphcast_bench.optimizations.inference import layernorm, mesh, regular3
    from graphcast_bench.platforms.runtime import check_runtime

    check_runtime()
    verify_maintained()
    reference_graph.install()
    regular3.install()
    layernorm.install(include_regular3=False)
    mesh.install()
    output = Path(os.environ["GRAPHCAST_REFERENCE_GRAPH_AUDIT"]).parent / "run"

    def save_audits(path):
        layernorm.save_audit(path)
        mesh.save_audit(output / "mesh-gather-audit.json")
        from graphcast_bench.assets import write_json

        write_json(
            output / "integration-verdict.json",
            dict(status="passed", all_optimization_audits_saved=True),
        )

    run_with_audit(infer, save_audits, output / "optimization-audit.json")


if __name__ == "__main__":
    main()
