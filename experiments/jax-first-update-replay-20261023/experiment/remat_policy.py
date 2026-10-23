"""Training-only stage and message-step recomputation for official GraphCast.

Retains parameter paths, math, dtype and rocBLAS. No inference RNE hook is used.
hk.remat preserves Haiku parameter/state handling, unlike raw jax.checkpoint.
"""
from contextlib import contextmanager
import haiku as hk
from graphcast import graphcast, deep_typed_graph_net

class CheckpointedGraphCast(graphcast.GraphCast):
    """GraphCast is not an hk.Module: stage wrappers must be explicit."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.remat_stages=set()

    def _run_grid2mesh_gnn(self,grid_node_features):
        self.remat_stages.add('grid2mesh')
        return hk.remat(super()._run_grid2mesh_gnn)(grid_node_features)

    def _run_mesh_gnn(self,latent_mesh_nodes):
        self.remat_stages.add('mesh')
        return hk.remat(super()._run_mesh_gnn)(latent_mesh_nodes)

    def _run_mesh2grid_gnn(self,updated_latent_mesh_nodes,latent_grid_nodes):
        self.remat_stages.add('mesh2grid')
        return hk.remat(super()._run_mesh2grid_gnn)(updated_latent_mesh_nodes,latent_grid_nodes)

@contextmanager
def rematerialize():
    coverage={'steps':set()}
    def _intercept(next_f,args,kwargs,context):
        if isinstance(context.module,deep_typed_graph_net.DeepTypedGraphNet) and context.method_name=='_process_step':
            # The callable is static; only the graph argument is a JAX tree.
            processor,graph=args
            assert not kwargs
            coverage['steps'].add((context.module.module_name,id(processor)))
            return hk.remat(lambda g:next_f(processor,g))(graph)
        return next_f(*args,**kwargs)
    with hk.intercept_methods(_intercept):
        yield coverage
