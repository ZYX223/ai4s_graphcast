"""Mark the original final mesh-node return; install no checkpoint policy.

The time-remat policy may choose to save this FP32 array. Group/message policies,
original mesh calls, parameters, aggregation and arithmetic are unchanged.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path

STAT_NAME = 'graphcast_final_mesh_nodes'
SHAPE = (40962, 1, 512)
_installed = False
_audit = []
_provenance = {}
_first_scope = None


def validate_result(value):
    if tuple(getattr(value, 'shape', ())) != SHAPE or str(getattr(value, 'dtype', '')) != 'float32':
        raise ValueError('Expected the single original FP32 final mesh array [40962,1,512]')


def residual_bytes(horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Positive integer horizon required')
    return horizon * 40962 * 512 * 4


def wrap_mesh_method(original, marker, scopes, audit):
    """Preserve the already-installed live mesh scope; mark after it completes."""
    @functools.wraps(original)
    def run(owner, *args, **kwargs):
        before = len(scopes)
        value = original(owner, *args, **kwargs)
        if len(scopes) != before + 1:
            raise RuntimeError('Expected exactly one completed live mesh scope per final marker')
        scope = scopes[before]
        if (scope.get('status') != 'verified' or scope.get('actual_replacements') != 16
                or scope.get('original_sender_calls') != 16):
            raise RuntimeError('Final mesh marker requires all16 verified receiver and original sender calls')
        validate_result(value)
        result = marker(value, name=STAT_NAME)
        validate_result(result)
        audit.append(dict(mesh_scope_id=scope['scope_id'], shape=list(SHAPE),
            dtype='float32', name=STAT_NAME, bytes=residual_bytes(1),
            marker_position='after original _run_mesh_gnn returned and its live mesh scope closed',
            value_representation=type(value).__name__, expected_receiver_calls=16,
            actual_receiver_calls=scope['actual_replacements']))
        return result
    return run


def install():
    """Install once, after regular3, original LNstats and analytic mesh VJP."""
    global _installed, _first_scope
    if _installed:
        raise RuntimeError('Final mesh marker already installed')
    import haiku as hk
    import jax
    from jax.ad_checkpoint import checkpoint_name
    from graphcast import graphcast
    import regular3_hooks
    import layernorm_stats
    import training_mesh_hooks as mesh
    import training_mesh_vjp as vjp
    if hk.__version__ != '0.0.14' or jax.__version__ != '0.4.34':
        raise RuntimeError('Only the frozen Haiku0.0.14/JAX0.4.34 path is supported')
    if not all(m._installed for m in (regular3_hooks, layernorm_stats, mesh, vjp)):
        raise RuntimeError('Install regular3, original LayerNorm statistics and mesh VJP first')
    expected = {
        graphcast: '6aeb9cfa2490b0e2dbfa70f255594914ee1765d5e20c3e66ca912b80bc66e953',
        regular3_hooks: 'e3fa2f332281cdf6ad3743eb97744088cfaf2483ea058faea7b4806d9813e1f5',
        layernorm_stats: 'cf822fe909a94059a3bedd6f939c10bff1dad6df3c28873607b5de6ed55bbdd1',
        mesh: 'e9c075588212e4759600cd2b67b7ebc1e3059b333cf02aa3f1cd7daf895e91e9',
        vjp: '50a30ee63a5e2320c21ad58f7c538f3ca7c4cc9c1cfe0dac1ec7fb77dbd2896e',
    }
    for module, wanted in expected.items():
        path = Path(inspect.getsourcefile(module))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise RuntimeError('Frozen final-mesh dependency changed: '+module.__name__)
        _provenance[module.__name__] = actual
    original = graphcast.GraphCast._run_mesh_gnn
    if (Path(original.__code__.co_filename).resolve() != Path(mesh.__file__).resolve()
            or original.__code__.co_name != 'run'):
        raise RuntimeError('Expected the existing live-scope mesh wrapper as the immediate predecessor')
    _first_scope = len(mesh._scopes)
    graphcast.GraphCast._run_mesh_gnn = wrap_mesh_method(original, checkpoint_name, mesh._scopes, _audit)
    _installed = True


def save_audit(path, require_full_model=True):
    import training_mesh_hooks as mesh
    if not _installed or not _audit:
        raise RuntimeError('Final mesh marker has no actual model trace')
    scopes = mesh._scopes[_first_scope:]
    if len(scopes) != len(_audit) or [s['scope_id'] for s in scopes] != [r['mesh_scope_id'] for r in _audit]:
        raise RuntimeError('Exactly one final mesh marker per verified model scope required')
    if require_full_model and any(s['status'] != 'verified' or s['actual_replacements'] != 16 for s in scopes):
        raise RuntimeError('Incomplete actual mesh model trace')
    Path(path).write_text(json.dumps(dict(status='traced', name=STAT_NAME,
        source_sha256=_provenance, completed_model_traces=len(_audit), entries=_audit,
        policy_installed=False, original_method_calls_per_marker=1,
        bytes_per_step=residual_bytes(1), bytes_h6=residual_bytes(6), bytes_h12=residual_bytes(12),
        scope='Python trace audit; not GPU launches, peak allocation or a claim that a policy saved the marker.'), indent=2)+'\n')
