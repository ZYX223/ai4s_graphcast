"""Full-model short training, one precision and rollout horizon per process.

Each process resumes its own trajectory. Recompilation and cache lifetime cannot
make later horizons retain earlier executables. Native SGEMM precision is selected
at runtime; JAX differentiates the original floating-point model.
"""
import argparse,ctypes as C,dataclasses,gc,hashlib,importlib.metadata as md,json,os,pickle,subprocess,threading,time
from pathlib import Path
import haiku as hk,jax,jax.numpy as jnp,numpy as np,optax,xarray as xr
from graphcast import autoregressive,checkpoint,data_utils,normalization,xarray_jax
from remat_policy import rematerialize,CheckpointedGraphCast
from precision import Precision
from data_transfer import to_device
from model import make_loss
import recipe

OUT=Path('/output')
def save(name,obj):
 p=OUT/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)
def tree_numpy(tree):return jax.tree.map(lambda x:np.asarray(x),jax.device_get(tree))
def flatten(tree):return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])
def comparison(actual,reference):
 # Blockwise float64 to avoid several extra full-size double buffers.
 sq=0.;ref_sq=0.;l1=0.;ref_l1=0.;mx=0.
 for i in range(0,actual.size,1_000_000):
  a=actual[i:i+1_000_000].astype(np.float64);b=reference[i:i+1_000_000].astype(np.float64);d=a-b
  sq+=float(d@d);ref_sq+=float(b@b);l1+=float(np.abs(d).sum());ref_l1+=float(np.abs(b).sum());mx=max(mx,float(np.max(np.abs(d))))
 return dict(relative_l2=float(np.sqrt(sq/ref_sq)) if ref_sq else None,relative_l1=l1/ref_l1 if ref_l1 else None,rmse=float(np.sqrt(sq/actual.size)),max_abs=mx)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--rollout',type=int,required=True)
 p.add_argument('--resume',default='');p.add_argument('--limit',type=int,default=38);p.add_argument('--tiny',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 dev=jax.devices()[0];assert len(jax.devices())==1
 hip=C.CDLL('libamdhip64.so');bus=C.create_string_buffer(64)
 assert hip.hipDeviceGetPCIBusId(bus,64,0)==0 and bus.value.decode().lower()==os.environ['EXPECTED_PCI']
 jax.config.update('jax_compilation_cache_dir','/cache')
 jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
 with open('/weekly/assets/jax_weights.npz','rb') as f:w=checkpoint.load(f,__import__('graphcast.graphcast',fromlist=['CheckPoint']).CheckPoint)
 data_root=Path('/previous-data' if a.preflight else '/training-data')
 source=json.loads((data_root/'source.json').read_text())
 raw=xr.load_dataset(data_root/('hres-20220101-025-13-1.nc' if a.preflight else 'hres-20220101-025-13-12.nc'),decode_timedelta=True)
 if a.tiny:raw=raw.isel(lat=slice(None,None,120),lon=slice(None,None,120));cfg=dataclasses.replace(w.model_config,resolution=30,mesh_size=1,latent_size=8,gnn_msg_steps=2)
 else:cfg=w.model_config
 assert a.tiny or (raw.sizes['lat']==721 and raw.sizes['lon']==1440 and raw.sizes['level']==13)
 assert (raw.sizes['time']>=14 or a.preflight) and raw.sizes['batch']==1
 if a.preflight:
  assert a.rollout==1
  plan=[dict(r,starts=[0,0]) for r in recipe.schedule()]
 else:plan=recipe.sample_plan(raw.sizes['time'],27)
 save('sample-plan.json',plan)
 stats={n:xr.load_dataset('/weekly/assets/stats/'+n+'.nc').astype(np.float32) for n in ['stddev_by_level','mean_by_level','diffs_stddev_by_level']}
 def get_batch(start):
  selected=raw.isel(time=slice(start,start+a.rollout+2))
  b=data_utils.extract_inputs_targets_forcings(selected,target_lead_times=slice('6h',str(6*a.rollout)+'h'),**dataclasses.asdict(w.task_config))
  b=tuple(x.astype(np.float32) for x in b)
  assert b[0].sizes['time']==2 and b[1].sizes['time']==a.rollout
  assert all(np.isfinite(v.values).all() for ds in b for v in ds.data_vars.values())
  assert all(np.any(v.values!=0) for v in b[1].data_vars.values())
  return b
 loss_fn=make_loss(cfg,w.task_config,stats)
 def loss_apply(params,batch):
  loss,state=loss_fn.apply(params,{},jax.random.PRNGKey(27),*batch);assert not state;return loss
 control=Precision();control.select(a.mode,'initialization')
 first=next(r for r in plan if r['rollout']==a.rollout)
 example=get_batch(first['starts'][0])
 params=w.params
 if a.tiny:params,_=loss_fn.init(jax.random.PRNGKey(27),*example)
 optimizer=recipe.optimizer(params);state=optimizer.init(params);completed=0
 if a.resume:
  with open(a.resume,'rb') as f:saved=pickle.load(f)
  assert saved['mode']==a.mode and saved['data_sha256']==source['sha256']
  assert saved['plan']==plan and saved['tiny']==a.tiny
  params,state,completed=saved['params'],saved['opt_state'],saved['updates']
 selected=[r for r in plan if r['rollout']==a.rollout and completed<r['update']<=a.limit]
 assert selected and selected[0]['update']==completed+1,(completed,selected)
 params,state=jax.device_put((params,state));example=to_device(example)
 report=dict(schema='graphcast.training38.v1',status='compiling',mode=a.mode,rollout=a.rollout,tiny=a.tiny,preflight=a.preflight,
  device=dict(index=int(os.environ['PHYSICAL_DEVICE']),pci=bus.value.decode(),driver_name=dev.device_kind,user_name='BW1100'),
  model=dataclasses.asdict(cfg),task=dataclasses.asdict(w.task_config),versions={n:md.version(n) for n in ['jax','jaxlib','dm-haiku','optax']},
  data_sha256=source['sha256'],scope='Short mechanics/numerics test on a small real HRES example; not convergence or weather-skill validation',
  precision='FP32 arrays/states; native SGEMM FP32 or both operands RNE + XF32 in forward/backward/recomputation',
  optimizer=dict(name='AdamW',betas=[.9,.95],eps=1e-8,matrix_decay=.1,clip_global_norm=32,accumulation=2),rows=[])
 save('report.json',report)
 stop=threading.Event();samples=[]
 def monitor():
  with (OUT/'device-memory.jsonl').open('w') as f:
   while not stop.is_set():
    try:
     d=json.loads(subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',os.environ['PHYSICAL_DEVICE'],'--showuse','--showmeminfo','vram','--json'],text=True))
     row=dict(time_ns=time.time_ns(),device=d);samples.append(row);f.write(json.dumps(row)+'\n');f.flush()
    except Exception as e:f.write(json.dumps(dict(error=repr(e)))+'\n')
    stop.wait(.2)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  t=time.perf_counter();print('COMPILING',a.mode,'R',a.rollout,flush=True)
  grad=jax.jit(jax.value_and_grad(loss_apply)).lower(params,example).compile()
  report['compile_seconds']=time.perf_counter()-t
  report['compiled_memory_analysis']=str(grad.memory_analysis())
  (OUT/'optimized-hlo.txt').write_text(grad.as_text())
  print('COMPILED',report['compile_seconds'],report['compiled_memory_analysis'],flush=True)
  control.select(a.mode,'warmup');loss,grads=grad(params,example);jax.block_until_ready((loss,grads))
  assert np.isfinite(float(loss))
  avg=jax.jit(lambda x,y:jax.tree.map(lambda a,b:(a+b)*.5,x,y),donate_argnums=(0,))
  def apply(params,state,grads,lr):
   updates,state=optimizer.update(grads,state,params)
   params=jax.tree.map(lambda p,u:p+lr*u,params,updates)
   return params,state
  update=jax.jit(apply,donate_argnums=(0,1)).lower(params,state,grads,jnp.float32(0)).compile()
  avg.lower(grads,grads).compile()
  warm_grads=avg(jax.device_put(tree_numpy(grads)),grads)
  warm_params,warm_state=jax.device_put(tree_numpy((params,state)))
  warm_result=update(warm_params,warm_state,warm_grads,jnp.float32(0))
  jax.block_until_ready(warm_result)
  del warm_grads,warm_result,warm_params,warm_state
  del grads,loss,example;gc.collect()
  report['status']='running';save('report.json',report)
  previous_flat=flatten(tree_numpy(params))
  for row0 in selected:
   row=dict(row0);u=row['update'];lr=row['lr']
   stage_t=time.perf_counter();print('STAGING',a.mode,u,flush=True)
   batches=[to_device(get_batch(s)) for s in row['starts']];jax.block_until_ready(batches)
   row['input_preparation_and_h2d_seconds']=time.perf_counter()-stage_t
   print('EXECUTING',a.mode,u,'staging_s',row['input_preparation_and_h2d_seconds'],flush=True)
   before=control.counts();control.select(a.mode,'update-%02d'%u)
   begin=time.time_ns();assert hip.hipProfilerStart()==0
   t=time.perf_counter()
   l0,g0=grad(params,batches[0]);l1,g1=grad(params,batches[1]);grads=avg(g0,g1)
   # Keep raw averaged gradients for diagnostics; optimizer owns clipping.
   params,state=update(params,state,grads,jnp.float32(lr))
   jax.block_until_ready((params,state,grads,l0,l1))
   elapsed=time.perf_counter()-t;assert hip.hipProfilerStop()==0;end=time.time_ns()
   row.update(loss=(float(l0)+float(l1))*.5,micro_losses=[float(l0),float(l1)],synchronized_update_seconds=elapsed,trace_begin_ns=begin,trace_end_ns=end,gemm=control.delta(before),jax_memory=dev.memory_stats())
   assert row['gemm']['calls']>0 and row['gemm']['rounded']==(row['gemm']['calls'] if a.mode=='rne_xf32' else 0),row['gemm']
   grad_host=tree_numpy(grads);flat=flatten(grad_host);assert np.isfinite(flat).all() and np.isfinite(row['loss'])
   row['gradient_norm']=float(np.sqrt(sum(float(np.square(x.astype(np.float64)).sum()) for x in jax.tree.leaves(grad_host))))
   assert row['gradient_norm']>0
   row['gradient_groups']={k:float(np.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in group.values()))) for k,group in grad_host.items()}
   current_flat=flatten(tree_numpy(params));assert np.isfinite(current_flat).all()
   row['parameter_update']=comparison(current_flat,previous_flat)
   if lr==0:assert row['parameter_update']['max_abs']==0
   else:assert row['parameter_update']['max_abs']>0
   previous_flat=current_flat
   row['gradient_sha256']=hashlib.sha256(flat.tobytes()).hexdigest()
   if a.mode=='fp32':np.save(OUT/('gradient-%02d.npy'%u),flat,allow_pickle=False)
   else:
    ref=Path('/experiment-root')/('fp32-r%02d'%a.rollout)/('gradient-%02d.npy'%u)
    if ref.exists():row['gradient_vs_fp32']=comparison(flat,np.load(ref,mmap_mode='r'))
   report['rows'].append(row);save('report.json',report)
   print('UPDATE',a.mode,u,'R',a.rollout,'loss',row['loss'],'norm',row['gradient_norm'],'seconds',elapsed,flush=True)
   del batches,l0,l1,grads,grad_host,flat;gc.collect()
  params_host=tree_numpy(params);state_host=tree_numpy(state)
  assert all(np.isfinite(v).all() for v in jax.tree.leaves((params_host,state_host)))
  payload=dict(params=params_host,opt_state=state_host,updates=selected[-1]['update'],mode=a.mode,data_sha256=source['sha256'],plan=plan,tiny=a.tiny)
  with (OUT/'checkpoint.pkl').open('wb') as f:pickle.dump(payload,f,protocol=4)
  report['checkpoint_sha256']=hashlib.sha256((OUT/'checkpoint.pkl').read_bytes()).hexdigest()
  report['final_params_sha256']=hashlib.sha256(flatten(params_host).tobytes()).hexdigest()
  if a.mode=='rne_xf32':
   ref=Path('/experiment-root')/('fp32-r%02d'%a.rollout)/'checkpoint.pkl'
   if ref.exists():
    with ref.open('rb') as f:ref=pickle.load(f)
    report['parameters_vs_fp32']=comparison(flatten(params_host),flatten(ref['params']))
  report['status']='completed'
 except Exception as e:report.update(status='failed',error=repr(e));raise
 finally:
  stop.set();thread.join(5)
  report['observed_device_peak_mib']=max((float(s['device']['card'+os.environ['PHYSICAL_DEVICE']]['vram Total Used Memory (MiB)']) for s in samples),default=None)
  save('report.json',report)

if __name__=='__main__':main()
