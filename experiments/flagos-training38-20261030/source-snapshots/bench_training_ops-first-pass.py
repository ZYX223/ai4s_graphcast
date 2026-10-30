"""Single-horizon training logical operators, native vs fl_only on one BW1100.

Synthetic positive floating activations (also saved backward tensors), original
index constants, true scalar variance masks. Times are synchronized independent
operator wall time, not whole-model time. Two microbatches scale call weights.
"""
import gc,hashlib,json,os,statistics,time,traceback
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from graphcast_bench.logical_bench import Inventory,sample_values
from operator_replay import replay_group,training_cases,pending_cases
from precision import Precision
base=Path('/tmp/zyx-training38-operators');mode=os.environ['XLA_MODE'];out=base/('bench-'+mode);out.mkdir(exist_ok=True)
inv=Inventory(base/'logical');cases,excluded=training_cases(inv);control=Precision()
report=dict(backend=mode,status='running',rows=[],cases=len(cases),excluded=excluded,scope='horizon1 useful forward/backward including loss; excludes remat and optimizer; model profile includes both',
 precision='FP32 arrays; every observed rocBLAS GEMM uses RNE+XF32',
 timing='3 rounds, each 5 warmups and20 synchronized calls; median of medians, weighted calls=logical count x2 microbatches',
 inputs='seed41030+argument_index, positive uniform[.25,1.25] synthetic floats; real constant indices; runtime variance masks=true; backward saved values synthetic',
 reference='same cases and generated inputs, FlagOS-container native XLA/RNE+XF32; sampled outputs, not independent weather truth')
if (out/'report.json').exists():
 old=json.loads((out/'report.json').read_text());assert old['backend']==mode and old['cases']==len(cases)
 report['rows']=old['rows']
pending=pending_cases(cases,report['rows'])
def save():
 p=out/'report.json';tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
save()
for index,c in enumerate(pending):
 row=dict(id=c['id'],operator=c['operator'],phase=c['phase'],calls_per_microbatch=c['calls'],calls_per_update=2*c['calls'],status='running',signature=c['signature'])
 report['active_case']=row['id'];save();print('CASE',index+1,len(cases),row['operator'],row['phase'],flush=True)
 args=[];fn=None;ex=None;ys=None;x=None
 try:
  g=c['group'];literals={};positions=[]
  for i,key in enumerate(g['inputs']):
   d=inv.variables[key];constant=inv.constant(key);shape=tuple(d['shape']);dtype=np.dtype(d['dtype'])
   if shape==() and constant is not None:literals[key]=np.asarray(constant,dtype=dtype);continue
   positions.append(key)
   if constant is not None:host=np.asarray(constant,dtype=dtype)
   elif dtype.kind=='b':host=np.ones(shape,dtype)
   elif dtype.kind=='f':host=np.random.RandomState(41030+i).uniform(.25,1.25,shape).astype(dtype)
   else:raise ValueError('Missing real index input '+key)
   x=jax.device_put(host);args.append(x);del host
  jax.block_until_ready(args)
  fn=replay_group(g['inputs'],g['outputs'],[inv.equations[i] for i in g['equation_ids']],literals)
  ex=fn.lower(*args).compile();hlo=ex.as_text();(out/(c['id']+'-hlo.txt')).write_text(hlo)
  row['flagtree_hlo']='xla.flagtree' in hlo
  control.select('rne_xf32','op-'+c['id']);before=control.counts();ys=fn(*args);jax.block_until_ready(ys)
  delta=control.delta(before);row['gemm']=delta
  if c['operator']=='linear':assert delta['calls']>0 and delta['calls']==delta['rounded'],delta
  elif delta['calls']:assert delta['calls']==delta['rounded'],delta
  samples={};finite=True
  for oi,y in enumerate(ys):
   size=y.size;flat=np.unique(np.r_[0,max(0,size-1),np.random.RandomState(503+oi).randint(0,max(1,size),min(size,256))])
   coords=np.array(np.unravel_index(flat,y.shape)).T if y.shape else np.zeros((len(flat),0),np.int64)
   samples['output%d'%oi]=sample_values(y,coords)
   finite=finite and bool(jnp.all(jnp.isfinite(y)))
  np.savez(out/(c['id']+'-samples.npz'),**samples);assert finite
  row['full_output_finite']=True
  reference_path=base/'bench-native'/(c['id']+'-samples.npz')
  native_ok=mode=='native' or any(r['id']==c['id'] and r['status']=='passed' for r in json.loads((base/'bench-native'/'report.json').read_text())['rows'])
  row['reference_status']='native_reference' if mode=='native' else ('available' if native_ok and reference_path.exists() else 'unavailable_native_case_failed')
  if mode=='fl_only' and native_ok and reference_path.exists():
   reference=np.load(base/'bench-native'/(c['id']+'-samples.npz'))
   a=np.concatenate(list(samples.values()));b=np.concatenate([reference[k] for k in samples]);diff=a-b
   row.update(max_abs_error=float(np.abs(diff).max()),sample_relative_l2_percent=float(np.linalg.norm(diff)/max(np.linalg.norm(b),1e-30)*100))
   np.testing.assert_allclose(a,b,atol=3e-3,rtol=3e-3)
  rounds=[]
  for _ in range(3):
   for _ in range(5):jax.block_until_ready(fn(*args))
   times=[]
   for _ in range(20):
    t=time.perf_counter();ys=fn(*args);jax.block_until_ready(ys);times.append((time.perf_counter()-t)*1000)
   rounds.append(times)
  row.update(status='measured_reference_unavailable' if row['reference_status']=='unavailable_native_case_failed' else 'passed',round_samples_ms=rounds,median_ms=statistics.median(statistics.median(r) for r in rounds))
  print('PASS',c['id'],row['median_ms'],flush=True)
 except Exception as e:
  row.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('FAIL',row['id'],repr(e),flush=True)
 report['rows'].append(row);save()
 del args,fn,ex,ys,x;gc.collect();jax.clear_caches()
report.pop('active_case',None);report['status']='passed' if all(r['status']=='passed' for r in report['rows']) else 'completed_with_failures';save()
print('DONE',mode,report['status'],flush=True)
