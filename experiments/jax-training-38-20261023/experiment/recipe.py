"""The approved short curriculum, deterministic windows and FP32 optimizer."""
import math
import numpy as np
import jax
import optax

def schedule():
 rows=[]
 for i in range(38):
  if i<6:r,lr=1,.001*i/5
  elif i<16:r,lr=1,.001*.5*(1+math.cos(math.pi*(i-6)/9))
  else:r,lr=2+(i-16)//2,3e-7
  rows.append(dict(update=i+1,rollout=r,lr=lr))
 return rows

def sample_plan(frame_count,seed):
 rng=np.random.default_rng(seed);rows=schedule()
 for row in rows:
  choices=frame_count-2-row['rollout']+1
  if choices<1:raise ValueError('Insufficient real future targets')
  row['starts']=rng.integers(0,choices,size=2).tolist()
 return rows

def optimizer(params):
 return optax.chain(optax.clip_by_global_norm(32.),optax.adamw(
  learning_rate=1.,b1=.9,b2=.95,eps=1e-8,weight_decay=.1,
  mask=jax.tree.map(lambda x:x.ndim>1,params)))

def accumulate(value_grad,params,batches):
 loss_sum=None;grad_sum=None
 for batch in batches:
  loss,grad=value_grad(params,batch)
  loss_sum=loss if loss_sum is None else loss_sum+loss
  grad_sum=grad if grad_sum is None else jax.tree.map(lambda a,b:a+b,grad_sum,grad)
 count=len(batches)
 return loss_sum/count,jax.tree.map(lambda x:x/count,grad_sum)
