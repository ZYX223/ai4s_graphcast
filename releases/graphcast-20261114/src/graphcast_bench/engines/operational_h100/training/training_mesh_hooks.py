"""Fixed mesh6 receiver gather/reduce with remat-safe live-topology binding.

Only the frozen GraphCast mesh processor is changed. The pure JAX sum has its
ordinary autodiff transpose; no custom derivative or precision conversion.
Each successful _run_mesh_gnn trace must replace all sixteen receiver sums.
"""
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading

import numpy as np

EXPECTED = 'f91c5061f868a234cb0f0e8cc3651f856c4cfe37967b9e50ac92a2a211819b5a'
NODES, EDGES, CHANNELS = 40962, 327660, 512
_audit, _scopes, _provenance = [], [], {}
_installed = False


def validate_indices(ids, num_segments):
    if (not isinstance(ids, np.ndarray) or ids.dtype != np.int32
            or ids.shape != (EDGES,) or num_segments != NODES):
        raise ValueError('Expected fixed mesh int32 receiver topology')
    if np.any(ids < 0) or np.any(ids >= NODES):
        raise ValueError('Receiver index outside fixed mesh')
    digest = hashlib.sha256(ids.tobytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError('Actual current mesh receiver topology changed')
    return digest


def build_plan(ids, num_segments):
    validate_indices(ids, num_segments)
    degree = np.bincount(ids, minlength=num_segments)
    cuts = np.r_[0, np.flatnonzero(degree[1:] != degree[:-1]) + 1, num_segments]
    # mergesort is stable on both the local old NumPy and the pinned container.
    order = np.argsort(ids, kind='mergesort').astype(np.int32)
    starts = np.r_[0, np.cumsum(degree)]
    groups = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        count = int(degree[start])
        indices = order[starts[start]:starts[stop]].reshape(stop - start, count)
        if not np.all(ids[indices] == np.arange(start, stop)[:, None]):
            raise ValueError('Receiver group covers incorrect targets')
        indices.setflags(write=False)
        groups.append((int(start), int(stop), count, indices))
    expected_groups = [(0, 12, 35), (12, 42, 36), (42, 162, 30),
                       (162, 642, 24), (642, 2562, 18), (2562, 10242, 12),
                       (10242, 40962, 6)]
    if [g[:3] for g in groups] != expected_groups:
        raise ValueError('Unexpected fixed mesh degree groups')
    positions = np.concatenate([g[3].ravel() for g in groups])
    if not np.array_equal(np.sort(positions), np.arange(EDGES)):
        raise ValueError('Each edge must contribute exactly once')
    return tuple(groups)


def grouped_sum(data, groups, xp):
    """Pure sum/concatenate. The install hook separately enforces full shape."""
    if data.shape[0] != EDGES:
        raise ValueError('Wrong number of mesh messages')
    return xp.concatenate([xp.sum(data[g[3]], axis=1) for g in groups], axis=0)


class ScopedMeshTopology:
    """Authority exists only within one current model and node invocation."""
    def __init__(self):
        self.local = threading.local()

    def current(self):
        return getattr(self.local, 'binding', None)

    def check_current(self):
        b = self.current()
        if b is None:
            raise ValueError('No live verified mesh model call')
        if (b['owner']._mesh_graph_structure is not b['graph']
                or b['graph'].edges.get(b['key']) is not b['edge']
                or b['edge'].indices.receivers is not b['ids']):
            raise ValueError('Current model topology changed inside verified scope')
        return b

    @contextmanager
    def model(self, owner):
        if self.current() is not None:
            raise ValueError('Nested mesh model scope is unsupported')
        graph = owner._mesh_graph_structure
        keys = list(graph.edges)
        if (len(keys) != 1 or keys[0].name != 'mesh'
                or tuple(keys[0].node_sets) != ('mesh_nodes', 'mesh_nodes')
                or list(graph.nodes) != ['mesh_nodes']
                or not np.array_equal(graph.nodes['mesh_nodes'].n_node, np.array([NODES]))):
            raise ValueError('Only the current fixed mesh receiver graph is supported')
        key = keys[0]
        edge = graph.edges[key]
        ids = edge.indices.receivers
        groups = build_plan(ids, NODES)
        self.local.binding = dict(owner=owner, graph=graph, key=key, edge=edge,
            ids=ids, groups=groups, indices_sha256=EXPECTED, receiver=None,
            sender=None, message=None, selected_calls=0, sender_calls=0)
        try:
            yield self.local.binding
        finally:
            self.local.binding = None

    @contextmanager
    def receiver(self, receiver, sender, message):
        b = self.check_current()
        if b['receiver'] is not None or receiver is sender:
            raise ValueError('Nested or ambiguous sender/receiver scope')
        # Never let an earlier validated graph authorize in-place mutation.
        validate_indices(b['ids'], NODES)
        b['receiver'], b['sender'], b['message'] = receiver, sender, message
        try:
            yield b
        finally:
            b['receiver'], b['sender'], b['message'] = None, None, None

    def authorize(self, receiver, message):
        b = self.check_current()
        if b['receiver'] is None or receiver is not b['receiver'] or message is not b['message']:
            raise ValueError('Not the exact current official receiver/message invocation')
        return b

    def node_is_target(self, graph, node_set_key):
        b = self.current()
        if b is None or node_set_key != 'mesh_nodes':
            return False
        self.check_current()
        if list(graph.edges) != [b['key']]:
            raise ValueError('Unexpected graph in mesh node update')
        return True


def install():
    global _installed
    if _installed:
        raise RuntimeError('Training mesh hook already installed')
    import jax
    import jax.numpy as jnp
    import jraph
    from graphcast import graphcast, deep_typed_graph_net, typed_graph_net
    expected = {
        graphcast: '6aeb9cfa2490b0e2dbfa70f255594914ee1765d5e20c3e66ca912b80bc66e953',
        deep_typed_graph_net: '7812649e3128a002603b537e61c7eda060b03a7a2d90bd037eeb7791b52cc59a',
        typed_graph_net: '58bc7c9b090a65ec83d65f0b3cd48d4a450f57f50b8bba5425f33b25d436e7a4',
    }
    for module, digest in expected.items():
        actual = hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError('Frozen official mesh dataflow changed: ' + module.__name__)
        _provenance[module.__name__] = actual
    original_run = graphcast.GraphCast._run_mesh_gnn
    original_node = typed_graph_net._node_update
    gate = ScopedMeshTopology()

    @functools.wraps(original_run)
    def run(owner, *args, **kwargs):
        network = owner._mesh_gnn
        if (network._aggregate_normalization is not None or network._f32_aggregation
                or network._aggregate_edges_for_nodes_fn is not jraph.segment_sum
                or network._num_message_passing_steps != 16
                or network._num_processor_repetitions != 1):
            raise ValueError('Mesh aggregation/processor configuration changed')
        with gate.model(owner) as b:
            scope = dict(scope_id=len(_scopes) + 1, status='tracing',
                         expected_replacements=16, indices_sha256=b['indices_sha256'])
            _scopes.append(scope)
            b['scope_id'] = scope['scope_id']
            try:
                result = original_run(owner, *args, **kwargs)
                if b['selected_calls'] != 16:
                    raise RuntimeError('Expected16 mesh receiver replacements in current model scope, got ' + str(b['selected_calls']))
            except BaseException as error:
                scope.update(status='failed', error=repr(error))
                raise
            else:
                scope['status'] = 'verified'
                return result
            finally:
                scope.update(actual_replacements=b['selected_calls'],
                             original_sender_calls=b['sender_calls'])

    @functools.wraps(original_node)
    def node(graph, node_fn, node_set_key, aggregation_fn):
        if not gate.node_is_target(graph, node_set_key):
            return original_node(graph, node_fn, node_set_key, aggregation_fn)
        binding = gate.check_current()
        edge = graph.edges[binding['key']]
        receiver, sender = edge.indices.receivers, edge.indices.senders
        if isinstance(receiver, np.ndarray):
            if receiver is not binding['ids']:
                raise ValueError('Unrelated static receiver array in current mesh call')
        elif not isinstance(receiver, jax.core.Tracer):
            raise ValueError('Receiver must be original static array or official remat tracer')
        if receiver.shape != (EDGES,) or np.dtype(receiver.dtype) != np.dtype(np.int32):
            raise ValueError('Unexpected mesh receiver representation')
        with gate.receiver(receiver, sender, edge.features):
            def aggregate(data, segment_ids, num_segments, *a, **kw):
                b = gate.check_current()
                # Official mesh node_update also computes sender aggregation;
                # preserve it (including any later dead-code elimination).
                if segment_ids is b['sender'] and data is b['message']:
                    b['sender_calls'] += 1
                    return aggregation_fn(data, segment_ids, num_segments, *a, **kw)
                b = gate.authorize(segment_ids, data)
                if (a or kw or int(num_segments) != NODES
                        or data.shape != (EDGES, 1, CHANNELS) or str(data.dtype) != 'float32'):
                    raise ValueError('Unexpected fixed mesh message shape/dtype/arguments')
                result = grouped_sum(data, b['groups'], jnp)
                b['selected_calls'] += 1
                _audit.append(dict(scope_id=b['scope_id'], receiver_call=b['selected_calls'],
                    selected=True, input_shape=list(data.shape), output_shape=list(result.shape),
                    indices_sha256=b['indices_sha256'], all_edges_exactly_once=True,
                    receiver_representation=type(receiver).__name__,
                    receiver_identity_bound=True, live_model_id=id(b['owner']),
                    live_graph_id=id(b['graph']),
                    groups=[dict(start=g[0], stop=g[1], degree=g[2]) for g in b['groups']]))
                return result
            return original_node(graph, node_fn, node_set_key, aggregate)

    graphcast.GraphCast._run_mesh_gnn = run
    typed_graph_net._node_update = node
    _installed = True


def save_audit(path):
    if not _installed or not _scopes or any(s['status'] != 'verified' or s['actual_replacements'] != 16 for s in _scopes):
        raise RuntimeError('Missing or incomplete16-receiver mesh scope; no fallback accepted')
    for scope in _scopes:
        rows = [a for a in _audit if a['scope_id'] == scope['scope_id']]
        if [a['receiver_call'] for a in rows] != list(range(1, 17)):
            raise RuntimeError('Mesh receiver audit is incomplete or duplicated')
    Path(path).write_text(json.dumps(dict(status='verified',
        expected_replacements_per_model_trace=16, completed_model_traces=len(_scopes),
        actual_replacements=len(_audit), official_dataflow_sha256=_provenance,
        scopes=_scopes, entries=_audit,
        scope='Live _mesh_graph_structure validated before remat; exact receiver/message binding; sender path preserved; each successful trace replaces16 receivers. Counts are Python trace events, not GPU launches.',
        derivative='Ordinary JAX autodiff of gather/sum; no custom VJP or stop_gradient; FP32 reduction order changes.'), indent=2) + '\n')
