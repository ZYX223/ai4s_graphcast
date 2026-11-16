"""Optional edge-domain first-Linear K partition; original full parameters.

No node-domain projection, remat policy, dtype conversion, or GPU launch.
Floating-point reduction order changes, although real-arithmetic GEMM work
and ordinary reverse-mode GEMM FLOPs are unchanged. GPU benefit is unproven.
"""
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading

EDGE_ROWS = {'grid2mesh': 1618746, 'mesh2grid': 3114720}
MODULES = {stage: stage+'_gnn/~_networks_builder/processor_edges_0_'+stage+'_mlp/~/linear_0'
           for stage in EDGE_ROWS}
_installed = False
_audit = []
_provenance = {}


def validate_parts(parts):
    if len(parts) != 3:
        raise ValueError('Exactly edge, sender-on-edge, receiver-on-edge inputs required')
    shape = tuple(parts[0].shape)
    if len(shape) != 3 or shape[1:] != (1, 512) or shape[0] <= 0:
        raise ValueError('Expected already gathered edge-domain [edges,1,512] arrays')
    if any(tuple(p.shape) != shape or str(p.dtype) != 'float32' for p in parts):
        raise ValueError('All three edge-domain parts must have identical FP32 shape')
    return int(shape[0])


def validate_empty_globals(value):
    # Frozen GraphCast constructs Context.features=() in all three graphs.
    # The original component fixture used {}; both exact empty containers have
    # no pytree leaves. Do not silently discard arrays, unknown containers or
    # any nonempty globals, even if a leaf happens to contain zero elements.
    if type(value) not in (tuple, dict) or len(value) != 0:
        raise ValueError('Expected known empty tuple/dict globals, got '+type(value).__name__)
    return type(value).__name__


def gemm_flops(rows):
    f = 2 * int(rows) * 1536 * 512
    return dict(forward=f, input_vjp=f, weight_vjp=f, total=3*f)


class EdgeInputs:
    def __init__(self): self.local = threading.local()
    def current(self): return getattr(self.local, 'binding', None)
    @contextmanager
    def bind(self, stage, parts):
        if self.current() is not None: raise ValueError('Nested edge input binding is unsupported')
        binding = dict(stage=stage, parts=tuple(parts), hits=0, seen=[])
        self.local.binding = binding
        try: yield binding
        finally: self.local.binding = None


def partitioned_dot(parts, weight, bias, precision=None):
    """Three edge-domain GEMMs; only the small weight matrix is sliced."""
    import jax.numpy as jnp
    validate_parts(parts)
    if weight.shape != (1536, 512) or bias.shape != (512,):
        raise ValueError('The original complete [1536,512] w and [512] b are required')
    if str(weight.dtype) != 'float32' or str(bias.dtype) != 'float32':
        raise ValueError('Original parameters must remain FP32')
    first = jnp.dot(parts[0], weight[0:512], precision=precision)
    second = jnp.dot(parts[1], weight[512:1024], precision=precision)
    third = jnp.dot(parts[2], weight[1024:1536], precision=precision)
    result = (first + second) + third
    return result + jnp.broadcast_to(bias, result.shape)


def apply_partitioned(module, parts, precision=None):
    """Pinned Haiku Linear parameter creation; complete initializer/fan-in."""
    import haiku as hk
    import numpy as np
    validate_parts(parts)
    if module.output_size != 512 or not module.with_bias:
        raise ValueError('Expected the unchanged affine1536-to512 first Linear')
    module.input_size = 1536
    initializer = module.w_init
    if initializer is None:
        initializer = hk.initializers.TruncatedNormal(stddev=1. / np.sqrt(module.input_size))
    weight = hk.get_parameter('w', [1536, 512], parts[0].dtype, init=initializer)
    bias = hk.get_parameter('b', [512], parts[0].dtype, init=module.b_init)
    return partitioned_dot(parts, weight, bias, precision=precision)


def install():
    global _installed
    if _installed: raise RuntimeError('Edge-domain Linear candidate installed twice')
    import haiku as hk
    import jax
    import jraph
    from haiku._src import basic
    from haiku._src.nets import mlp
    from graphcast import graphcast, deep_typed_graph_net, typed_graph_net
    original = hk.Linear
    if hk.__version__ != '0.0.14' or jax.__version__ != '0.4.34':
        raise RuntimeError('Only the pinned Haiku0.0.14/JAX0.4.34 path is supported')
    if original is not basic.Linear or original is not mlp.hk.Linear or original.__module__ != 'haiku._src.basic':
        raise RuntimeError('Original Haiku Linear was already replaced')
    expected = [(graphcast, '6aeb9cfa2490b0e2dbfa70f255594914ee1765d5e20c3e66ca912b80bc66e953'),
        (deep_typed_graph_net, '7812649e3128a002603b537e61c7eda060b03a7a2d90bd037eeb7791b52cc59a'),
        (typed_graph_net, '58bc7c9b090a65ec83d65f0b3cd48d4a450f57f50b8bba5425f33b25d436e7a4'),
        (original, '4f62c8c96c5b2f0bfda3a983d5f869e34ffddab4620b1eaa7a8c1910d1422c96'),
        (hk.nets.MLP, '140bb612bb5cf9813c4f1c688af63474b179284fa660bb23dbfaf3033da38d87'),
        (jraph.concatenated_args, '9bce036d06ff28c580161df35a353584c8cab09d63348e7dbdc43e307f467bed')]
    # File hashes freeze e,s,r flattening order and original Haiku parameter API.
    for obj, wanted in expected:
        path = Path(inspect.getsourcefile(obj))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted: raise RuntimeError('Frozen edge/Linear dataflow source changed: '+str(path))
        _provenance[str(path)] = actual
    gate = EdgeInputs()
    original_edge = typed_graph_net._edge_update

    class EdgePartitionLinear(original):
        def __call__(self, inputs, *, precision=None):
            live = gate.current()
            if live is not None:
                live['seen'].append(self.module_name)
            if self.module_name not in MODULES.values():
                return super().__call__(inputs, precision=precision)
            binding = gate.current()
            if binding is None or MODULES[binding['stage']] != self.module_name:
                raise ValueError('Target Linear escaped its actual official edge update')
            parts = binding['parts']
            count = validate_parts(parts)
            if count != EDGE_ROWS[binding['stage']] or tuple(inputs.shape) != (count, 1, 1536) or str(inputs.dtype) != 'float32':
                raise ValueError('Target edge/concat shape differs from fixed production model')
            binding['hits'] += 1
            if binding['hits'] != 1: raise ValueError('Expected exactly one target first Linear per edge update')
            result = apply_partitioned(self, parts, precision=precision)
            _audit.append(dict(stage=binding['stage'], module_name=self.module_name,
                edge_rows=count, original_concat_shape=list(inputs.shape),
                actual_part_shapes=[list(v.shape) for v in parts], dtype=str(inputs.dtype),
                full_weight_shape=[1536,512], full_bias_shape=[512], bias_additions=1,
                accumulation_order='(edge@W0 + gathered_sender@W1) + gathered_receiver@W2 + bias',
                useful_gemm_flops=gemm_flops(count), edge_domain_preserved=True,
                globals_container=binding['globals_container'], global_feature_leaf_count=0,
                explicit_precision=str(precision), input_representation=type(parts[0]).__name__))
            return result

    @functools.wraps(original_edge)
    def edge_update(graph, edge_fn, edge_set_key):
        stage = edge_set_key.name
        if stage not in EDGE_ROWS:
            return original_edge(graph, edge_fn, edge_set_key)
        expected_nodes = ('grid_nodes','mesh_nodes') if stage == 'grid2mesh' else ('mesh_nodes','grid_nodes')
        if tuple(edge_set_key.node_sets) != expected_nodes:
            raise ValueError('Target edge direction changed')
        def bound_edge(edge, sender, receiver, globals_):
            # Original _edge_update already gathered the two node inputs.
            # Preserve that work and the actual edge ordering without caching.
            globals_container = validate_empty_globals(globals_)
            parts = (edge, sender, receiver)
            if validate_parts(parts) != EDGE_ROWS[stage]:
                raise ValueError('Unexpected production edge count')
            with gate.bind(stage, parts) as binding:
                binding['globals_container'] = globals_container
                result = edge_fn(edge, sender, receiver, globals_)
                if binding['hits'] != 1:
                    raise ValueError('Target first Linear was not replaced; actual modules: '+repr(binding['seen']))
                return result
        return original_edge(graph, bound_edge, edge_set_key)

    # Pinned MLP has a private class named hk that captured basic.Linear at
    # import time. Update that exact guarded alias as well as both public/core
    # aliases, preserving normal Haiku method interceptors for all new layers.
    basic.Linear = EdgePartitionLinear
    hk.Linear = EdgePartitionLinear
    mlp.hk.Linear = EdgePartitionLinear
    typed_graph_net._edge_update = edge_update
    _installed = True


def save_audit(path):
    if not _installed or not _audit: raise RuntimeError('No actual target edge Linear trace')
    if {v['module_name'] for v in _audit} != set(MODULES.values()):
        raise RuntimeError('Both grid2mesh and mesh2grid first Linear traces are required')
    Path(path).write_text(json.dumps(dict(status='traced', source_sha256=_provenance,
        entries=_audit, replaced_modules=MODULES,
        policy_changed=False, parameters_renamed_or_split=False,
        warning='Different floating-point reduction order. Graph evidence alone does not show CUDA concat elimination, beta1 reuse, peak memory, speed or full training loss.'), indent=2)+'\n')
