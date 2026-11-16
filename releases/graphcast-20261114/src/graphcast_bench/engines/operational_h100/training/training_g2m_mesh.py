"""Mark only g2m mesh nodes AFTER the frozen outer stage remat returns.

The original stage method, its remat bookkeeping and grid return are preserved.
This module installs no checkpoint policy and saves no grid-domain tensor.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path

STAT_NAME='graphcast_g2m_mesh_nodes'
SHAPE=(40962,1,512)
GRID_SHAPE=(1038240,1,512)
_installed=False
_audit=[]
_provenance={}


def validate_result(value):
    if type(value) is not tuple or len(value)!=2:
        raise ValueError('Expected the original (mesh_nodes, grid_nodes) tuple')
    for array,shape in zip(value,(SHAPE,GRID_SHAPE)):
        if tuple(getattr(array,'shape',()))!=shape or str(getattr(array,'dtype',''))!='float32':
            raise ValueError('Expected fixed FP32 production g2m mesh/grid outputs')


def residual_bytes(horizon):
    if type(horizon) is not int or horizon<1:raise ValueError('Positive integer horizon required')
    return horizon*40962*512*4


def wrap_stage_method(original,marker,audit):
    @functools.wraps(original)
    def run(owner,*args,**kwargs):
        pair=original(owner,*args,**kwargs)
        if 'grid2mesh' not in getattr(owner,'remat_stages',()):
            raise RuntimeError('Original CheckpointedGraphCast stage bookkeeping was not preserved')
        validate_result(pair)
        mesh,grid=pair
        result=(marker(mesh,name=STAT_NAME),grid)
        validate_result(result)
        assert result[1] is grid
        audit.append(dict(name=STAT_NAME,shape=list(SHAPE),dtype='float32',
            bytes=residual_bytes(1),original_grid_shape=list(GRID_SHAPE),
            original_grid_object_preserved=True,grid_marker_added=False,
            original_stage_calls=1,remat_stages=sorted(owner.remat_stages),
            marker_position='after CheckpointedGraphCast._run_grid2mesh_gnn including its hk.remat(super) returned',
            mesh_representation=type(mesh).__name__))
        return result
    return run


def install():
    """Install after the frozen final-mesh combination; do not change policy."""
    global _installed
    if _installed:raise RuntimeError('g2m mesh output marker already installed')
    import haiku as hk
    import jax
    from jax.ad_checkpoint import checkpoint_name
    from graphcast import graphcast
    import model
    import remat_policy
    import training_final_mesh as final_mesh
    if hk.__version__!='0.0.14' or jax.__version__!='0.4.34':
        raise RuntimeError('Only the frozen Haiku0.0.14/JAX0.4.34 path is supported')
    if not final_mesh._installed:
        raise RuntimeError('Install the verified final-mesh combination first')
    expected={
        graphcast:{'6aeb9cfa2490b0e2dbfa70f255594914ee1765d5e20c3e66ca912b80bc66e953'},
        remat_policy:{'c0edb0448b0b0a0e0d3137fa8416d8542624fedf1775c88e065a83a1ba65819d'},
        model:{'0bbfe6e8b8f5496ec86688c30b0aacc75847ca98c14dfe8345e45e92fd109c1b',
               'e1ac6210cbb78c413d125abd00321f7345e62f013ef7c18e3ad674875750b272'},
        final_mesh:{'d70450293ffc0f1d23517dc8b518c45bb2a1345dbb48314736c6ff6824ebce39'},
    }
    for module,wanted in expected.items():
        path=Path(inspect.getsourcefile(module));actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual not in wanted:raise RuntimeError('Frozen g2m source changed: '+module.__name__)
        _provenance[module.__name__]=dict(path=str(path),sha256=actual)
    cls=remat_policy.CheckpointedGraphCast
    if model.CheckpointedGraphCast is not cls:
        raise RuntimeError('The actual loss model no longer uses the frozen stage class alias')
    original=cls._run_grid2mesh_gnn
    if (Path(original.__code__.co_filename).resolve()!=Path(remat_policy.__file__).resolve()
            or original.__qualname__!='CheckpointedGraphCast._run_grid2mesh_gnn'):
        raise RuntimeError('Expected the unmodified frozen outer grid2mesh stage method')
    cls._run_grid2mesh_gnn=wrap_stage_method(original,checkpoint_name,_audit)
    _installed=True


def save_audit(path,expected_traces=None):
    if not _installed or not _audit:raise RuntimeError('No actual g2m output marker trace')
    if expected_traces is not None and len(_audit)!=expected_traces:
        raise RuntimeError('Unexpected number of g2m mesh output traces')
    Path(path).write_text(json.dumps(dict(status='traced',name=STAT_NAME,source_sha256=_provenance,
        completed_model_traces=len(_audit),entries=_audit,policy_installed=False,
        bytes_per_step=residual_bytes(1),bytes_h6=residual_bytes(6),bytes_h12=residual_bytes(12),
        scope='Python trace audit only; selected mesh return only, no grid checkpoint marker. Actual time-policy saving, GPU memory and performance need separate evidence.'),indent=2)+'\n')
