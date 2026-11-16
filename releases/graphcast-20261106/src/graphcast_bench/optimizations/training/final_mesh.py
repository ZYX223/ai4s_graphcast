"""Mark the original final mesh-node return; install no checkpoint policy.

The time-remat policy may choose to save this FP32 array. Group/message policies,
original mesh calls, parameters, aggregation and arithmetic are unchanged.
"""

import functools
import json
from pathlib import Path

STAT_NAME = "graphcast_final_mesh_nodes"
SHAPE = (40962, 1, 512)
_installed = False
_audit = []
_provenance = {}
_first_scope = None


def validate_result(value):
    if tuple(getattr(value, "shape", ())) != SHAPE or str(getattr(value, "dtype", "")) != "float32":
        raise ValueError("Expected the single original FP32 final mesh array [40962,1,512]")


def residual_bytes(horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Positive integer horizon required")
    return horizon * 40962 * 512 * 4


def wrap_mesh_method(original, marker, scopes, audit):
    """Preserve the already-installed live mesh scope; mark after it completes."""

    @functools.wraps(original)
    def run(owner, *args, **kwargs):
        before = len(scopes)
        value = original(owner, *args, **kwargs)
        if len(scopes) != before + 1:
            raise RuntimeError("Expected exactly one completed live mesh scope per final marker")
        scope = scopes[before]
        if (
            scope.get("status") != "verified"
            or scope.get("actual_replacements") != 16
            or scope.get("original_sender_calls") != 16
        ):
            raise RuntimeError(
                "Final mesh marker requires all16 verified receiver and original sender calls"
            )
        validate_result(value)
        result = marker(value, name=STAT_NAME)
        validate_result(result)
        audit.append(
            dict(
                mesh_scope_id=scope["scope_id"],
                shape=list(SHAPE),
                dtype="float32",
                name=STAT_NAME,
                bytes=residual_bytes(1),
                marker_position="after original _run_mesh_gnn returned and its live mesh scope closed",
                value_representation=type(value).__name__,
                expected_receiver_calls=16,
                actual_receiver_calls=scope["actual_replacements"],
            )
        )
        return result

    return run


def install():
    """Install once, after regular3, original LNstats and analytic mesh VJP."""
    global _installed, _first_scope
    if _installed:
        raise RuntimeError("Final mesh marker already installed")
    import haiku as hk
    import jax
    from graphcast import graphcast
    from jax.ad_checkpoint import checkpoint_name

    import graphcast_bench.optimizations.training.layernorm_stats as layernorm_stats
    import graphcast_bench.optimizations.training.mesh as mesh
    import graphcast_bench.optimizations.training.mesh_vjp as vjp
    import graphcast_bench.optimizations.training.regular3 as regular3_hooks

    if hk.__version__ != "0.0.14" or jax.__version__ != "0.6.0":
        raise RuntimeError("Only the isolated Haiku0.0.14/JAX0.6.0 BW port is supported")
    if not all(m._installed for m in (regular3_hooks, layernorm_stats, mesh, vjp)):
        raise RuntimeError("Install regular3, original LayerNorm statistics and mesh VJP first")
    from graphcast_bench.validation.provenance import verify_module

    for module in (graphcast, regular3_hooks, layernorm_stats, mesh, vjp):
        actual = verify_module(module)
        _provenance[module.__name__] = actual
    original = graphcast.GraphCast._run_mesh_gnn
    if (
        Path(original.__code__.co_filename).resolve() != Path(mesh.__file__).resolve()
        or original.__code__.co_name != "run"
    ):
        raise RuntimeError(
            "Expected the existing live-scope mesh wrapper as the immediate predecessor"
        )
    _first_scope = len(mesh._scopes)
    graphcast.GraphCast._run_mesh_gnn = wrap_mesh_method(
        original, checkpoint_name, mesh._scopes, _audit
    )
    _installed = True


def save_audit(path, require_full_model=True):
    import graphcast_bench.optimizations.training.mesh as mesh

    if not _installed or not _audit:
        raise RuntimeError("Final mesh marker has no actual model trace")
    scopes = mesh._scopes[_first_scope:]
    if len(scopes) != len(_audit) or [s["scope_id"] for s in scopes] != [
        r["mesh_scope_id"] for r in _audit
    ]:
        raise RuntimeError("Exactly one final mesh marker per verified model scope required")
    if require_full_model and any(
        s["status"] != "verified" or s["actual_replacements"] != 16 for s in scopes
    ):
        raise RuntimeError("Incomplete actual mesh model trace")
    Path(path).write_text(
        json.dumps(
            dict(
                status="traced",
                name=STAT_NAME,
                source_sha256=_provenance,
                completed_model_traces=len(_audit),
                entries=_audit,
                policy_installed=False,
                original_method_calls_per_marker=1,
                bytes_per_step=residual_bytes(1),
                bytes_h6=residual_bytes(6),
                bytes_h12=residual_bytes(12),
                scope="Python trace audit; not GPU launches, peak allocation or a claim that a policy saved the marker.",
            ),
            indent=2,
        )
        + "\n"
    )
