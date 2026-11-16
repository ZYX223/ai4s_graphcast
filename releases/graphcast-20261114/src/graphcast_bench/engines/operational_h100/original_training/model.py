"""Shared actual training loss: official autoregression, optional spatial remat."""
from contextlib import nullcontext
import haiku as hk,jax
from graphcast import graphcast,normalization,autoregressive,xarray_jax
from remat_policy import rematerialize,CheckpointedGraphCast

def make_loss(model_config,task_config,stats,remat=True):
 @hk.transform_with_state
 def loss_fn(inputs,targets,forcings):
  with (rematerialize() if remat else nullcontext()) as coverage:
   cls=CheckpointedGraphCast if remat else graphcast.GraphCast
   raw_model=cls(model_config,task_config)
   model=normalization.InputsAndResiduals(raw_model,**stats)
   model=autoregressive.Predictor(model,gradient_checkpointing=True)
   loss,_=model.loss(inputs,targets,forcings)
   if remat:
    assert raw_model.remat_stages=={'grid2mesh','mesh','mesh2grid'}
    assert len(coverage['steps'])==model_config.gnn_msg_steps+2,len(coverage['steps'])
   return xarray_jax.unwrap_data(loss).mean()
 return loss_fn
