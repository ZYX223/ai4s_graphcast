import hashlib,json
from pathlib import Path
import numpy as np
a=Path('/tmp/zyx-training38-operators-retry/bench-fl_only')
b=Path('/tmp/zyx-training38-operators-native-retry/bench-native')
fa=json.loads((a/'report.json').read_text());nb=json.loads((b/'report.json').read_text())
assert nb['status']=='passed' and len(fa['rows'])==len(nb['rows'])==1
f=fa['rows'][0];n=nb['rows'][0]
assert n['status']=='passed' and f['status']=='measured_reference_unavailable'
assert f['id']==n['id'] and f['signature']==n['signature'] and f['calls_per_update']==n['calls_per_update']
assert f['full_output_finite'] and n['full_output_finite']
for r in [f,n]:
 assert len(r['round_samples_ms'])==3 and all(len(t)==20 for t in r['round_samples_ms'])
 assert r['gemm']['calls']==r['gemm']['rounded']>0
name=f['id']+'-samples.npz';x=np.load(a/name);y=np.load(b/name);assert x.files==y.files
actual=np.concatenate([x[k] for k in x.files]);reference=np.concatenate([y[k] for k in y.files])
np.testing.assert_allclose(actual,reference,atol=3e-3,rtol=3e-3)
result=dict(id=f['id'],status='passed',reference='isolated native GPU, same80% pool',
 max_abs_error=float(np.max(np.abs(actual-reference))),sample_relative_l2_percent=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)*100),
 native_ms=n['median_ms'],flagos_ms=f['median_ms'],speedup=n['median_ms']/f['median_ms'],
 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a/'report.json',b/'report.json',a/name,b/name]})
(a.parent/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
