"""Trace useful differentiated graphs on CPU, without executing full training."""
import dataclasses,hashlib,json,os,time
from pathlib import Path
import haiku as hk,jax,jax.numpy as jnp,numpy as np,xarray as xr
from graphcast import checkpoint,graphcast,data_utils
from model import make_loss
from flops import summarize_jaxpr
out=Path(os.environ['FLOPS_OUTPUT']);out.mkdir(exist_ok=True,parents=True)
weekly=Path(os.environ['TRAINING_WEEKLY'])/'assets'
with (weekly/'jax_weights.npz').open('rb') as f:w=checkpoint.load(f,graphcast.CheckPoint)
raw=xr.load_dataset(Path(os.environ['TRAINING_DATA'])/'hres-20220101-025-13-12.nc',decode_timedelta=True)
stats={n:xr.load_dataset(weekly/'stats'/(n+'.nc')).astype(np.float32) for n in ['stddev_by_level','mean_by_level','diffs_stddev_by_level']}
from graphcast_bench.logical_inventory import export_jaxpr
horizon=1
loss=make_loss(w.model_config,w.task_config,stats,remat=False,temporal_remat=False)
batch=data_utils.extract_inputs_targets_forcings(raw.isel(time=slice(0,3)),target_lead_times=slice('6h','6h'),**dataclasses.asdict(w.task_config))
batch=tuple(b.astype(np.float32) for b in batch)
def f(params,b):return loss.apply(params,{},jax.random.PRNGKey(27),*b)[0]
closed=jax.make_jaxpr(jax.value_and_grad(f))(w.params,batch)
r=export_jaxpr(closed,out/'logical')
print('LOGICAL_CAPTURE',len(r['equations']),len(r['groups']),flush=True)
