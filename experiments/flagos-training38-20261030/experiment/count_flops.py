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
records=[]
for horizon in range(1,13):
 start=time.monotonic()
 loss=make_loss(w.model_config,w.task_config,stats,remat=False,temporal_remat=False)
 batch=data_utils.extract_inputs_targets_forcings(raw.isel(time=slice(0,horizon+2)),target_lead_times=slice('6h',str(6*horizon)+'h'),**dataclasses.asdict(w.task_config))
 batch=tuple(b.astype(np.float32) for b in batch)
 def f(params,b):return loss.apply(params,{},jax.random.PRNGKey(27),*b)[0]
 forward=summarize_jaxpr(jax.make_jaxpr(f)(w.params,batch))
 training=summarize_jaxpr(jax.make_jaxpr(jax.value_and_grad(f))(w.params,batch))
 record=dict(horizon=horizon,forward=forward,training=training,microbatches_per_update=2,
   useful_flops_per_update=2*training['gemm_flops'],method='DCE differentiated JAXPR without spatial/temporal remat; dense GEMM only; FP32 arrays',seconds=time.monotonic()-start)
 (out/('horizon-%02d.json'%horizon)).write_text(json.dumps(record,indent=2)+'\n')
 records.append({k:record[k] for k in ['horizon','useful_flops_per_update','seconds']})
 (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
 print('FLOPS',horizon,forward['gemm_calls'],forward['gemm_flops'],training['gemm_calls'],training['gemm_flops'],flush=True)
