"""Actual official R2 path: remat gradients, feedback and checkpoint resume."""
import dataclasses,pickle,unittest
import jax,jax.numpy as jnp,numpy as np,xarray as xr
from graphcast import checkpoint,data_utils,graphcast
from model import make_loss
from recipe import optimizer
from data_transfer import to_device

class OfficialModelTests(unittest.TestCase):
 def test_two_lead_gradients_and_resumed_update(self):
  with open('/weekly/assets/jax_weights.npz','rb') as f:w=checkpoint.load(f,graphcast.CheckPoint)
  d=xr.load_dataset('/previous-data/hres-20220101-025-13-1.nc',decode_timedelta=True).isel(lat=slice(None,None,120),lon=slice(None,None,120))
  # Synthetic fourth frame only for a small unit test, never capacity/accuracy.
  extra=d.isel(time=[-1]).copy();extra=extra.assign_coords(time=extra.time+np.timedelta64(6,'h'),datetime=extra.datetime+np.timedelta64(6,'h'))
  d=xr.concat([d,extra],dim='time',data_vars='minimal',coords='minimal',compat='override')
  b=tuple(x.astype(np.float32) for x in data_utils.extract_inputs_targets_forcings(d,target_lead_times=slice('6h','12h'),**dataclasses.asdict(w.task_config)))
  cfg=dataclasses.replace(w.model_config,resolution=30,mesh_size=1,latent_size=8,gnn_msg_steps=2)
  stats={n:xr.load_dataset('/weekly/assets/stats/'+n+'.nc').astype(np.float32) for n in ['stddev_by_level','mean_by_level','diffs_stddev_by_level']}
  original=make_loss(cfg,w.task_config,stats,remat=False)
  candidate=make_loss(cfg,w.task_config,stats,remat=True)
  params,_=original.init(jax.random.PRNGKey(27),*b)
  def vg(model):
   def loss(params,b):return model.apply(params,{},jax.random.PRNGKey(27),*b)[0]
   return jax.jit(jax.value_and_grad(loss))
  ref=vg(original)(params,b);grad=vg(candidate);actual=grad(params,to_device(b))
  for x,y in zip(jax.tree.leaves(ref),jax.tree.leaves(actual)):np.testing.assert_allclose(x,y,rtol=3e-4,atol=3e-5)
  self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(actual)))
  tx=optimizer(params);state=tx.init(params)
  def step(params,state):
   loss,g=grad(params,b);u,state=tx.update(g,state,params)
   return jax.tree.map(lambda p,u:p+.0001*u,params,u),state,loss
  p1,s1,_=step(params,state)
  cp=pickle.loads(pickle.dumps(jax.device_get((p1,s1))))
  full=step(p1,s1);resumed=step(*cp)
  for x,y in zip(jax.tree.leaves(full),jax.tree.leaves(resumed)):np.testing.assert_array_equal(x,y)

if __name__=='__main__':unittest.main()
