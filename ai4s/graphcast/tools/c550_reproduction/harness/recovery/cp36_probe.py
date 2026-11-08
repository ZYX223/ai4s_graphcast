import sys,json,torch
"""Read-only CPU audit; does not claim an actual resume or training acceptance."""
import hashlib,json,math,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
p,initial=map(Path,sys.argv[1:3])
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(p)=='211b19a84964215a65694d4ad1a7d88299c19c60cffe9636d142f406ef13935f'
assert sha(initial)=='2bd1786e197b432d0f471ce4118f34b231c73c322dbaba0684e5f71097e010b4'
a=torch.load(p,map_location='cpu',weights_only=False)
b=torch.load(initial,map_location='cpu',weights_only=False)
if 'model_state' in b:b=b['model_state']
elif 'state_dict' in b:b=b['state_dict']
model=a['model_state'];changed=[];unchanged=[];mismatch=[];norm2=0.;maxabs=0.
for k,t in model.items():
 if k not in b or t.shape!=b[k].shape:mismatch.append(k);continue
 if torch.equal(t,b[k]):unchanged.append(k)
 else:
  changed.append(k);d=(t.to(torch.float64)-b[k].to(torch.float64));norm2+=float((d*d).sum());maxabs=max(maxabs,float(d.abs().max()))
def tensors(x):
 if isinstance(x,torch.Tensor):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from tensors(v)
 elif isinstance(x,(tuple,list)):
  for v in x:yield from tensors(v)
opt=a['optimizer_state'];states=opt.get('state',{});steps=[float(v['step']) for v in states.values() if 'step' in v]
r={'scope':'CPU checkpoint structural and parameter-delta audit; actual resume NOT tested','checkpoint':str(p),'checkpoint_sha256':sha(p),'recorded_sha256':p.with_suffix(p.suffix+'.sha256').read_text().split()[0], 'initial_checkpoint_sha256':sha(initial),'schema_version':a.get('schema_version'),'parameter_tensor_count':len(model),'parameter_elements':sum(x.numel() for x in model.values()),'changed_tensor_count':len(changed),'unchanged_tensor_count':len(unchanged),'unchanged_names':unchanged,'shape_or_key_mismatches':mismatch,'initial_extra_keys':sorted(set(b)-set(model)),'all_model_finite':all(bool(torch.isfinite(t).all()) for t in model.values()),'parameter_delta_l2':math.sqrt(norm2),'parameter_delta_abs_max':maxabs,'optimizer_states':len(states),'optimizer_steps':sorted(set(steps)),'optimizer_all_finite':all(bool(torch.isfinite(t).all()) for t in tensors(opt)),'optimizer_param_groups':[{k:v for k,v in g.items() if k!='params'}|{'parameter_count':len(g['params'])} for g in opt.get('param_groups',[])],'schedule_state':a.get('schedule_state'),'cursor':a.get('cursor'),'rng_rank_count':len(a.get('rng_states_by_rank',[])),'metadata_keys':list(a.get('metadata',{}))}
r['hash_matches']=r['checkpoint_sha256']==r['recorded_sha256']

audit=r
contract=a['metadata']['resume_contract']
runtime=contract['operator_replacement']['runtime']
audit['required_resume_paths']={
 'config':contract['config'], 'corpus_index':contract['corpus_index'],
 'stats_dir':contract['stats_dir'], 'flaggems_source_root':runtime['flaggems_source_root'],
 'flaggems_ops_config':runtime['flaggems_ops_config'],
 'flaggems_module_origin':runtime['flaggems_module_origin']}

sys.path.insert(0, str(Path(sys.argv[3]) / 'src'))
from graphcast_compat.training_state import state_object_sha256
obj=torch.load(p,map_location='cpu',weights_only=False)
assert audit['checkpoint_sha256']=='211b19a84964215a65694d4ad1a7d88299c19c60cffe9636d142f406ef13935f'
for group in obj['optimizer_state']['param_groups']:group['lr']=3e-07
guard=dict(expected_resume_lr=3e-07,expected_resume_parameter_sha=state_object_sha256(obj['model_state']),expected_resume_optimizer_sha=state_object_sha256(obj['optimizer_state']),expected_resume_rng_sha=state_object_sha256(obj['rng_states_by_rank'][0]))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(audit=audit,guard=guard,cuda_initialized=False)))
