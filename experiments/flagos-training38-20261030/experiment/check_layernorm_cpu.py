"""Independent CPU sampled correctness for native-GPU compiler failures.

This does not manufacture missing native GPU timing or speedups. Full shapes,
identical generated inputs and all saved forward outputs are preserved.
"""
import gc,json,time
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from graphcast_bench.logical_bench import Inventory,sample_values
from operator_replay import replay_group,training_cases
base=Path('/tmp/zyx-training38-operators');out=base/'cpu-layernorm';out.mkdir(exist_ok=True)
assert len(jax.devices())==1 and jax.devices()[0].platform=='cpu'
native=json.loads((base/'bench-native/report.json').read_text())
ids={r['id'] for r in native['rows'] if r['status']!='passed' and r['operator']=='layer_norm'}
inv=Inventory(base/'logical');cases,_=training_cases(inv);rows=[]
for c in cases:
 if c['id'] not in ids:continue
 t=time.monotonic();g=c['group'];args=[];literals={}
 eqs=[inv.equations[i] for i in g['equation_ids']]
 assert not any(e['primitive']=='dot_general' for e in eqs)
 for i,key in enumerate(g['inputs']):
  d=inv.variables[key];constant=inv.constant(key);shape=tuple(d['shape']);dtype=np.dtype(d['dtype'])
  if shape==() and constant is not None:literals[key]=np.asarray(constant,dtype=dtype);continue
  if constant is not None:host=np.asarray(constant,dtype=dtype)
  elif dtype.kind=='b':host=np.ones(shape,dtype)
  elif dtype.kind=='f':host=np.random.RandomState(41030+i).uniform(.25,1.25,shape).astype(dtype)
  else:raise ValueError('Missing index')
  args.append(jax.device_put(host));del host
 fn=replay_group(g['inputs'],g['outputs'],eqs,literals);ys=fn(*args);jax.block_until_ready(ys)
 samples={}
 for oi,y in enumerate(ys):
  size=y.size;flat=np.unique(np.r_[0,max(0,size-1),np.random.RandomState(503+oi).randint(0,max(1,size),min(size,256))])
  coords=np.array(np.unravel_index(flat,y.shape)).T if y.shape else np.zeros((len(flat),0),np.int64)
  samples['output%d'%oi]=sample_values(y,coords)
  assert bool(jnp.all(jnp.isfinite(y)))
 np.savez(out/(c['id']+'-samples.npz'),**samples)
 rows.append(dict(id=c['id'],status='cpu_reference_computed',outputs=len(ys),seconds=time.monotonic()-t))
 (out/'report.json').write_text(json.dumps(dict(reference='CPU JAX FP32; no GEMM in these LayerNorm groups; full shapes, same inputs, sampled comparison only',rows=rows),indent=2)+'\n')
 print('CPU_REFERENCE',c['id'],len(ys),time.monotonic()-t,flush=True)
 del args,ys,fn,y;gc.collect();jax.clear_caches()
assert len(rows)==len(ids)==5
comparisons=[]
for row in rows:
 k=row['id'];actual=np.load(base/'bench-fl_only'/(k+'-samples.npz'));reference=np.load(out/(k+'-samples.npz'))
 assert actual.files==reference.files
 a=np.concatenate([actual[n] for n in actual.files]);b=np.concatenate([reference[n] for n in reference.files])
 np.testing.assert_allclose(a,b,atol=3e-3,rtol=3e-3)
 comparisons.append(dict(id=k,status='passed',reference='JAX CPU FP32',sample_values=a.size,max_abs_error=float(np.max(np.abs(a-b))),relative_l2_percent=float(np.linalg.norm(a-b)/np.linalg.norm(b)*100)))
(out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
