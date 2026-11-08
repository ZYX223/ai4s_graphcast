"""Read-only state comparison; does not erase failed lifecycle/summary status."""
import hashlib,json,math,sys
from pathlib import Path
import torch
import numpy as np
torch.set_num_threads(2)
pa,pb=map(Path,sys.argv[1:3]);a=torch.load(pa,map_location='cpu',weights_only=False);b=torch.load(pb,map_location='cpu',weights_only=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def eq(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
def compare(x,y):
 rows=[]
 def walk(x,y,path):
  if isinstance(x,torch.Tensor):
   assert isinstance(y,torch.Tensor) and x.shape==y.shape and x.dtype==y.dtype,path
   xf=x.double();yf=y.double();delta=xf-yf
   rows.append({'path':path,'equal':torch.equal(x,y),'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all()),'delta_sum_sq':float(delta.square().sum()),'reference_sum_sq':float(xf.square().sum()),'max_abs':float(delta.abs().max()) if delta.numel() else 0.})
  elif isinstance(x,dict):
   assert x.keys()==y.keys(),path
   for k in x:walk(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   assert len(x)==len(y),path
   for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,path+'/'+str(i))
  else:assert x==y,(path,x,y)
 walk(x,y,'')
 return {'tensor_count':len(rows),'bitwise_equal':all(z['equal'] for z in rows),'differing_tensors':sum(not z['equal'] for z in rows),'all_finite':all(z['finite'] for z in rows),'relative_l2':math.sqrt(sum(z['delta_sum_sq'] for z in rows)/max(sum(z['reference_sum_sq'] for z in rows),1e-300)),'max_abs':max((z['max_abs'] for z in rows),default=0.)}
d={'continuous_checkpoint':str(pa),'resumed_checkpoint':str(pb),'continuous_sha256':sha(pa),'resumed_sha256':sha(pb),'model':compare(a['model_state'],b['model_state']),'optimizer':compare(a['optimizer_state'],b['optimizer_state']),'cursor_equal':eq(a['cursor'],b['cursor']),'cursor':b['cursor'],'schedule_equal':eq(a['schedule_state'],b['schedule_state']),'rng_equal':eq(a['rng_states_by_rank'],b['rng_states_by_rank']),'resume_contract_equal':eq(a['metadata']['resume_contract'],b['metadata']['resume_contract']),'initial_lineage_equal':eq(a['metadata']['initial_lineage'],b['metadata']['initial_lineage']),'lifecycle_success':False,'formal_acceptance':False}
d['hashes_match_recorded']=all(sha(p)==p.with_suffix(p.suffix+'.sha256').read_text().split()[0] for p in [pa,pb])
assert not torch.cuda.is_initialized()
d['cuda_initialized']=False
print(json.dumps(d))
