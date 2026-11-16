"""Training candidate: static mesh2grid sum, with automatic differentiation.

Independent snapshot of h100-inference-mfu-20261102/candidate_hooks.py.
Official files and the inference candidate remain unchanged.
"""
import hashlib
import json
import numpy as np
import functools
import inspect
import threading
from contextlib import contextmanager

_audit=[]
_installed=False
_provenance={}

class ScopedTopology:
    """A live model call and its immediate receiver invocation; never a cache."""
    def __init__(self,num_segments):
        self.num_segments=num_segments
        self.local=threading.local()

    def current(self):
        return getattr(self.local,'binding',None)

    def check_current(self):
        b=self.current()
        if b is None:raise ValueError('No verified current mesh2grid model call')
        if (b['owner']._mesh2grid_graph_structure is not b['graph'] or
            b['graph'].edges.get(b['key']) is not b['edge'] or
            b['edge'].indices.receivers is not b['ids']):
            raise ValueError('Current model topology changed inside verified call')
        return b

    @contextmanager
    def model(self,owner):
        if self.current() is not None:raise ValueError('Nested mesh2grid scope is unsupported')
        graph=owner._mesh2grid_graph_structure
        keys=list(graph.edges)
        if len(keys)!=1 or keys[0].name!='mesh2grid' or tuple(keys[0].node_sets)!=('mesh_nodes','grid_nodes'):
            raise ValueError('Only the current mesh2grid receiver graph is authorized')
        key=keys[0]; edge=graph.edges[key]; ids=edge.indices.receivers
        if not np.array_equal(graph.nodes['grid_nodes'].n_node,np.array([self.num_segments])):
            raise ValueError('Wrong grid node count')
        validate_regular3(ids,self.num_segments)
        self.local.binding=dict(owner=owner,graph=graph,key=key,edge=edge,ids=ids,
            indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),receiver=None,message=None)
        try:yield self.local.binding
        finally:self.local.binding=None

    @contextmanager
    def receiver(self,ids,message):
        b=self.check_current()
        if b['receiver'] is not None:raise ValueError('Nested receiver scope is unsupported')
        # Revalidate the actual static array, including in-place mutations.
        validate_regular3(b['ids'],self.num_segments)
        b['receiver']=ids; b['message']=message
        try:yield b
        finally:b['receiver']=None; b['message']=None

    def authorize(self,ids,message):
        b=self.check_current()
        if b['receiver'] is None or ids is not b['receiver'] or message is not b['message']:
            raise ValueError('Receiver/message is not the exact current official node-update invocation')
        return b

    def node_is_target(self,graph,node_set_key):
        b=self.current()
        if b is None or node_set_key!='grid_nodes':return False
        self.check_current()
        if list(graph.edges)!=[b['key']]:raise ValueError('Unexpected graph in mesh2grid node update')
        return True

def validate_regular3(segment_ids,num_segments):
    if not isinstance(segment_ids,np.ndarray) or segment_ids.ndim!=1:
        raise ValueError('Aggregation requires a static one-dimensional numpy index array')
    if segment_ids.dtype.kind not in 'iu' or len(segment_ids)!=3*num_segments:
        raise ValueError('Index dtype/count does not prove three complete incoming edges')
    if not np.all(segment_ids.reshape(num_segments,3)==np.arange(num_segments)[:,None]):
        raise ValueError('Indices must be exactly 0,0,0,1,1,1,... through the final segment')

def regular3_sum(data,num_segments,xp):
    if data.shape[0]!=3*num_segments:
        raise ValueError('Message count does not match segment count')
    return xp.sum(xp.reshape(data,(num_segments,3)+data.shape[1:]),axis=1)

def install():
    global _installed
    if _installed:
        raise RuntimeError('Candidate hook already installed')
    import jax
    import jax.numpy as jnp
    import jraph
    from pathlib import Path
    from graphcast import graphcast,deep_typed_graph_net,typed_graph_net
    # The provenance proof is specific to these frozen official implementations:
    # GraphCast replaces features only; the processor preserves indices; node_update
    # forwards exactly edge.indices.receivers to aggregation_fn. Fail on drift.
    expected={graphcast:'6aeb9cfa2490b0e2dbfa70f255594914ee1765d5e20c3e66ca912b80bc66e953',
        deep_typed_graph_net:'7812649e3128a002603b537e61c7eda060b03a7a2d90bd037eeb7791b52cc59a',
        typed_graph_net:'58bc7c9b090a65ec83d65f0b3cd48d4a450f57f50b8bba5425f33b25d436e7a4'}
    for module,digest in expected.items():
        actual=hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
        if actual!=digest:raise RuntimeError('Frozen official graph dataflow changed: '+module.__name__)
        _provenance[module.__name__]=actual
    original_run=graphcast.GraphCast._run_mesh2grid_gnn
    original_node=typed_graph_net._node_update
    gate=ScopedTopology(1038240)

    @functools.wraps(original_run)
    def run(owner,*args,**kwargs):
        network=owner._mesh2grid_gnn
        if (network._aggregate_normalization is not None or network._f32_aggregation or
            network._aggregate_edges_for_nodes_fn is not jraph.segment_sum):
            raise ValueError('Mesh2grid aggregation configuration differs from the frozen candidate')
        with gate.model(owner):return original_run(owner,*args,**kwargs)

    @functools.wraps(original_node)
    def node(graph,node_fn,node_set_key,aggregation_fn):
        if not gate.node_is_target(graph,node_set_key):
            return original_node(graph,node_fn,node_set_key,aggregation_fn)
        binding=gate.check_current()
        edge=graph.edges[binding['key']]
        ids=edge.indices.receivers
        if isinstance(ids,np.ndarray):
            if ids is not binding['ids']:raise ValueError('Unrelated static receiver array')
        elif not isinstance(ids,jax.core.Tracer):
            raise ValueError('Expected remat tracer from the frozen official graph dataflow')
        if ids.shape!=(3114720,) or np.dtype(ids.dtype).kind not in 'iu':
            raise ValueError('Unexpected receiver representation')
        # Only this receiver and this message leaf can enter the replacement.
        # The original frozen node_update forwards them directly; no global
        # aggregation factory is replaced and no certificate survives the call.
        with gate.receiver(ids,edge.features):
            def aggregate(data,segment_ids,num_segments,*args,**kwargs):
                b=gate.authorize(segment_ids,data)
                if args or kwargs or int(num_segments)!=1038240 or data.shape!=(3114720,1,512) or str(data.dtype)!='float32':
                    raise ValueError('Unexpected batch/channel/dtype/arguments for fixed mesh2grid case')
                result=regular3_sum(data,int(num_segments),jnp)
                _audit.append(dict(selected=True,input_shape=list(data.shape),output_shape=list(result.shape),
                    num_segments=int(num_segments),batch_axis_preserved=True,complete_ordered_coverage=True,
                    indices_sha256=b['indices_sha256'],dtype=str(data.dtype),
                    receiver_representation=type(ids).__name__,live_model_id=id(b['owner']),
                    live_graph_id=id(b['graph']),receiver_identity_bound=True))
                return result
            return original_node(graph,node_fn,node_set_key,aggregate)
    graphcast.GraphCast._run_mesh2grid_gnn=run
    typed_graph_net._node_update=node
    _installed=True

def save_audit(path):
    selected=[row for row in _audit if row['selected']]
    if not _installed or not selected:
        raise RuntimeError('Required fixed mesh2grid substitution did not execute during tracing')
    path.write_text(json.dumps(dict(status='verified',selected_trace_calls=len(selected),
        scope='Live current model static graph validated before remat; exact receiver/message invocation bound inside frozen official node_update; scopes cleared on exit.',
        official_dataflow_sha256=_provenance,
        entries=_audit),indent=2)+'\n')
