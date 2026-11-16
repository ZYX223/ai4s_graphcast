"""Mark only g2m mesh nodes AFTER the frozen outer stage remat returns.

The original stage method, its remat bookkeeping and grid return are preserved.
This module installs no checkpoint policy and saves no grid-domain tensor.
"""

import functools
import json
from pathlib import Path

STAT_NAME = "graphcast_g2m_mesh_nodes"
SHAPE = (40962, 1, 512)
GRID_SHAPE = (1038240, 1, 512)
_installed = False
_audit = []
_provenance = {}


def validate_result(value):
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("Expected the original (mesh_nodes, grid_nodes) tuple")
    for array, shape in zip(value, (SHAPE, GRID_SHAPE)):
        if (
            tuple(getattr(array, "shape", ())) != shape
            or str(getattr(array, "dtype", "")) != "float32"
        ):
            raise ValueError("Expected fixed FP32 production g2m mesh/grid outputs")


def residual_bytes(horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Positive integer horizon required")
    return horizon * 40962 * 512 * 4


def wrap_stage_method(original, marker, audit):
    @functools.wraps(original)
    def run(owner, *args, **kwargs):
        pair = original(owner, *args, **kwargs)
        if "grid2mesh" not in getattr(owner, "remat_stages", ()):
            raise RuntimeError("Original CheckpointedGraphCast stage bookkeeping was not preserved")
        validate_result(pair)
        mesh, grid = pair
        result = (marker(mesh, name=STAT_NAME), grid)
        validate_result(result)
        assert result[1] is grid
        audit.append(
            dict(
                name=STAT_NAME,
                shape=list(SHAPE),
                dtype="float32",
                bytes=residual_bytes(1),
                original_grid_shape=list(GRID_SHAPE),
                original_grid_object_preserved=True,
                grid_marker_added=False,
                original_stage_calls=1,
                remat_stages=sorted(owner.remat_stages),
                marker_position="after CheckpointedGraphCast._run_grid2mesh_gnn including its hk.remat(super) returned",
                mesh_representation=type(mesh).__name__,
            )
        )
        return result

    return run


def install():
    """Install after the frozen final-mesh combination; do not change policy."""
    global _installed
    if _installed:
        raise RuntimeError("g2m mesh output marker already installed")
    import haiku as hk
    import jax
    from graphcast import graphcast
    from jax.ad_checkpoint import checkpoint_name

    import graphcast_bench.model.loss as model
    import graphcast_bench.model.remat as remat_policy
    import graphcast_bench.optimizations.training.final_mesh as final_mesh

    if hk.__version__ != "0.0.14" or jax.__version__ != "0.6.0":
        raise RuntimeError("Only the isolated Haiku0.0.14/JAX0.6.0 BW port is supported")
    if not final_mesh._installed:
        raise RuntimeError("Install the verified final-mesh combination first")
    from graphcast_bench.validation.provenance import verify_module

    for module in (graphcast, remat_policy, model, final_mesh):
        actual = verify_module(module)
        _provenance[module.__name__] = actual
    cls = remat_policy.CheckpointedGraphCast
    if model.CheckpointedGraphCast is not cls:
        raise RuntimeError("The actual loss model no longer uses the frozen stage class alias")
    original = cls._run_grid2mesh_gnn
    if (
        Path(original.__code__.co_filename).resolve() != Path(remat_policy.__file__).resolve()
        or original.__qualname__ != "CheckpointedGraphCast._run_grid2mesh_gnn"
    ):
        raise RuntimeError("Expected the unmodified frozen outer grid2mesh stage method")
    cls._run_grid2mesh_gnn = wrap_stage_method(original, checkpoint_name, _audit)
    _installed = True


def save_audit(path, expected_traces=None):
    if not _installed or not _audit:
        raise RuntimeError("No actual g2m output marker trace")
    if expected_traces is not None and len(_audit) != expected_traces:
        raise RuntimeError("Unexpected number of g2m mesh output traces")
    Path(path).write_text(
        json.dumps(
            dict(
                status="traced",
                name=STAT_NAME,
                source_sha256=_provenance,
                completed_model_traces=len(_audit),
                entries=_audit,
                policy_installed=False,
                bytes_per_step=residual_bytes(1),
                bytes_h6=residual_bytes(6),
                bytes_h12=residual_bytes(12),
                scope="Python trace audit only; selected mesh return only, no grid checkpoint marker. Actual time-policy saving, GPU memory and performance need separate evidence.",
            ),
            indent=2,
        )
        + "\n"
    )
