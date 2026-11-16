"""Optional first-order reverse-AD mesh sum; original forward and guards retained.

Seven independent gathers have seven full-edge scatter transposes by default.
For this verified partition, their sum is exactly cotangent[receiver]. This
custom VJP stores no input activation residual and uses that analytic transpose.
Forward-mode jvp/jacfwd through this function is intentionally unsupported.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import training_mesh_hooks as mesh

_ordinary_sum = mesh.grouped_sum
_installed = False


def receiver_from_groups(groups):
    """Reconstruct and validate the complete forward partition before its VJP."""
    expected_groups = [(0, 12, 35), (12, 42, 36), (42, 162, 30),
                       (162, 642, 24), (642, 2562, 18), (2562, 10242, 12),
                       (10242, 40962, 6)]
    if len(groups) != 7 or [tuple(g[:3]) for g in groups] != expected_groups:
        raise ValueError('Analytic transpose requires the seven fixed ordered groups')
    positions = []
    receivers = np.full(mesh.EDGES, -1, np.int32)
    for start, stop, degree, indices in groups:
        if (not isinstance(indices, np.ndarray) or indices.dtype != np.int32
                or indices.shape != (stop - start, degree)
                or np.any(indices < 0) or np.any(indices >= mesh.EDGES)):
            raise ValueError('Invalid forward gather group for analytic transpose')
        positions.append(indices.ravel())
        receivers[indices] = np.arange(start, stop, dtype=np.int32)[:, None]
    if not np.array_equal(np.sort(np.concatenate(positions)), np.arange(mesh.EDGES)):
        raise ValueError('Analytic transpose requires every edge exactly once')
    mesh.validate_indices(receivers, mesh.NODES)
    receivers.setflags(write=False)
    return receivers


def make_receiver_sum(groups):
    """Build a pure JAX operator for one already guarded immutable partition."""
    import jax
    import jax.numpy as jnp
    receivers = receiver_from_groups(groups)

    def check(data):
        if data.shape != (mesh.EDGES, 1, mesh.CHANNELS) or str(data.dtype) != 'float32':
            raise ValueError('Only the fixed full FP32 mesh message shape is supported')

    @jax.custom_vjp
    def receiver_sum(data):
        check(data)
        return _ordinary_sum(data, groups, jnp)

    def forward(data):
        check(data)
        # No data, intermediate activations, or gathered arrays in residual.
        # Topology is a checked static closure constant, not an activation.
        return _ordinary_sum(data, groups, jnp), ()

    def backward(residual, cotangent):
        if residual != ():
            raise ValueError('Unexpected activation residual')
        if cotangent.shape != (mesh.NODES, 1, mesh.CHANNELS) or str(cotangent.dtype) != 'float32':
            raise ValueError('Unexpected mesh output cotangent')
        return (cotangent[receivers],)

    receiver_sum.defvjp(forward, backward)
    return receiver_sum


def install():
    """Use instead of training_mesh_hooks.install(), after regular3.install()."""
    global _installed
    if _installed or mesh._installed:
        raise RuntimeError('Install the optional analytic VJP once, before the mesh hook')

    def grouped_with_analytic_vjp(data, groups, xp):
        return make_receiver_sum(groups)(data)

    mesh.grouped_sum = grouped_with_analytic_vjp
    try:
        mesh.install()
    except BaseException:
        mesh.grouped_sum = _ordinary_sum
        raise
    _installed = True


def save_audit(path):
    if not _installed:
        raise RuntimeError('Analytic mesh VJP was not installed')
    mesh.save_audit(path)
    path = Path(path)
    report = json.loads(path.read_text())
    report['derivative'] = dict(kind='custom_vjp',
        forward='Exactly the original guarded seven-group gather/reduce function',
        backward='cotangent[verified static receiver]; one gather; no full-edge scatter sum',
        activation_residual='Empty tuple; only static topology closure constants',
        input_shape=[mesh.EDGES, 1, mesh.CHANNELS], dtype='float32',
        supported='This experiment validates first-order reverse-mode AD and remat',
        unsupported='Forward-mode JVP/jacfwd through custom_vjp; other models/shapes unvalidated',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(report, indent=2) + '\n')
