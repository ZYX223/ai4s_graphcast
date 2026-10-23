"""Device-kernel accounting with native call policy verification."""
import argparse,json,collections,csv
from pathlib import Path
from graphcast_bench.profile_summary import summarize_events
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--fused',action='store_true');a=p.parse_args()
report=json.loads((a.run/'run/report.json').read_text());assert report['status']=='completed' and report['measured_steps']==1
trace=json.loads((a.run/'trace.json').read_text());events=trace['traceEvents'] if isinstance(trace,dict) else trace
summary=summarize_events(events,(a.run/'run/optimized-hlo.txt').read_text())
rows=[json.loads(l) for l in (a.run/'gemm.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['label']=='round1-rne_xf32']
assert len(rows)==84 and len(set(r['layer'] for r in rows))==84
weights={r['module']:r for r in json.loads((a.run.parent/'baseline-source/kernels/xf32_rne/weights.json').read_text())['layers']}
assert {r['layer'] for r in rows}==set(weights)
assert all(r['weight_sha256'] in weights[r['layer']]['sha256'].values() and r['bit']==weights[r['layer']]['bit'] for r in rows)
policy=json.loads((a.run.parent/'experiment-r1/policy.json').read_text())['layers']
assert {r['layer'] for r in rows if r.get('activation_rne_skipped',False)}==({r['module'] for r in policy} if a.fused else set())
assert all(r['status']==0 and r['mode']==1 and r['rne_inputs'] for r in rows)
assert sum(r.get('activation_rne_skipped',False) for r in rows)==(79 if a.fused else 0)
assert summary['categories']['gemm']['calls']==84
assert summary['categories']['rne']['calls']==(89 if a.fused else 168)
summary['native_gemm_audit']=dict(calls=84,xf32_mode_verified=True,activation_rne_skipped=79 if a.fused else 0)
(a.run/'profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.run/'kernel-details.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(summary['kernels'][0]));writer.writeheader();writer.writerows(summary['kernels'])
print(json.dumps({k:v for k,v in summary.items() if k not in ['kernels','host_apis']},indent=2))
