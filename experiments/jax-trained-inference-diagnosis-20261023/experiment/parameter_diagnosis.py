"""Compare parameter gaps after update 16 and update 38 on CPU."""
import csv,gc,hashlib,pickle
from pathlib import Path
import jax
import numpy as np
from graphcast_bench.assets import read_json,sha256,write_json

trees={};sources={}
for mode in ['fp32','rne_xf32']:
 for horizon,updates in [(1,16),(12,38)]:
  folder=Path('/training')/('%s-r%02d'%(mode,horizon))
  report=read_json(folder/'report.json')
  assert sha256(folder/'checkpoint.pkl')==report['checkpoint_sha256']
  with (folder/'checkpoint.pkl').open('rb') as f: saved=pickle.load(f)
  assert saved['updates']==updates and saved['mode']==mode
  key=mode+str(updates);trees[key]=saved['params']
  h=hashlib.sha256()
  for leaf in jax.tree.leaves(trees[key]):h.update(leaf.tobytes())
  assert h.hexdigest()==report['final_params_sha256']
  sources[key]=dict(checkpoint_sha256=report['checkpoint_sha256'],params_sha256=report['final_params_sha256'])
  del saved;gc.collect()
keys=list(trees['fp3238'])
assert all(set(t)==set(keys) for t in trees.values())
rows=[];totals=dict(fp32_sq16=0.,fp32_sq38=0.,gap_sq16=0.,gap_sq38=0.,gap_change_sq=0.,gap_dot=0.)
for module in sorted(keys):
 s={k:0. for k in totals}
 for name in trees['fp3238'][module]:
  a=trees['fp3216'][module][name].astype(np.float64).ravel()
  b=trees['rne_xf3216'][module][name].astype(np.float64).ravel()
  c=trees['fp3238'][module][name].astype(np.float64).ravel()
  d=trees['rne_xf3238'][module][name].astype(np.float64).ravel()
  g16=b-a;g38=d-c;change=g38-g16
  values=dict(fp32_sq16=float(a@a),fp32_sq38=float(c@c),gap_sq16=float(g16@g16),gap_sq38=float(g38@g38),gap_change_sq=float(change@change),gap_dot=float(g16@g38))
  for k,v in values.items():s[k]+=v;totals[k]+=v
 rows.append(dict(module=module,**s))
for row in rows:row['fraction_of_final_gap_squared']=row['gap_sq38']/totals['gap_sq38']
rows.sort(key=lambda r:r['gap_sq38'],reverse=True)
with open('/output/by-module.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(status='completed',sources=sources,
 relative_parameter_l2_gap_update16=float(np.sqrt(totals['gap_sq16']/totals['fp32_sq16'])),
 relative_parameter_l2_gap_update38=float(np.sqrt(totals['gap_sq38']/totals['fp32_sq38'])),
 change_in_gap_vector_from_update16_to38_relative_to_final_gap=float(np.sqrt(totals['gap_change_sq']/totals['gap_sq38'])),
 cosine_between_gap_vectors_update16_update38=totals['gap_dot']/float(np.sqrt(totals['gap_sq16']*totals['gap_sq38'])),
 totals=totals,caveat='Parameter-gap location is measured; per-module parameter norms do not establish causal importance for output error.')
write_json('/output/summary.json',summary)
print({k:v for k,v in summary.items() if k not in ['sources','totals']})
