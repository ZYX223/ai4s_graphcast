"""Two-by-two control: training precision versus inference precision."""
import csv,json
from pathlib import Path
import numpy as np
import xarray as xr
from graphcast_bench.assets import load_case,read_json,sha256,write_json
from graphcast_bench.metrics import compare,VARIABLES
case,_=load_case('/data')
folders={'A':Path('/original-results/fp32'),'B':Path('/original-results/rne_xf32'),
 'C':Path('/results/train-fp32-infer-rne_xf32'),'D':Path('/results/train-rne_xf32-infer-fp32')}
reports={};outputs={}
for label,folder in folders.items():
 r=read_json(folder/'report.json')
 assert r['status']=='completed' and r['training_updates']==38
 assert int((folder/'exit-code.txt').read_text())==0
 assert r['rows'][0]['steps']==40 and len(r['rows'][0]['audit'])==40
 assert all(x['calls']==84 and x['rounded']==(84 if r['mode']=='rne_xf32' else 0) for x in r['rows'][0]['audit'])
 assert r['rows'][0]['output_sha256']==sha256(folder/'prediction.nc')
 assert r['checkpoint_sha256']==sha256(r['checkpoint']) and r['case']==case
 reports[label]=r;outputs[label]=xr.load_dataset(folder/'prediction.nc',decode_timedelta=True)
assert reports['A']['checkpoint_sha256']==reports['C']['checkpoint_sha256']
assert reports['B']['checkpoint_sha256']==reports['D']['checkpoint_sha256']
assert reports['A']['mode']==reports['D']['mode']=='fp32'
assert reports['B']['mode']==reports['C']['mode']=='rne_xf32'
for field in ['model','task','code_sha256']:
 assert len({json.dumps(r[field],sort_keys=True) for r in reports.values()})==1
pairs=[('B','A','combined_independent_training_and_inference'),
 ('D','A','different_checkpoints_both_fp32_inference'),
 ('C','A','same_fp32_checkpoint_change_inference_precision'),
 ('B','D','same_rne_checkpoint_change_inference_precision'),
 ('B','C','different_checkpoints_both_rne_inference')]
rows=[]
for candidate,reference,purpose in pairs:
 metrics=compare(outputs[candidate],outputs[reference],case=case)
 metrics.update(reference=reference,candidate=candidate,purpose=purpose)
 metrics['per_variable_mean_percent']={v:float(np.mean([r['relative_error_percent'] for r in metrics['layers'] if r['variable']==v])) for v in VARIABLES}
 write_json('/output/'+purpose+'.json',metrics)
 rows.append({k:metrics[k] for k in ['purpose','candidate','reference','overall_mean_percent','z500_percent','z500_mae_m','z500_rmse_m','metrics_within_limits']})
 print(rows[-1],flush=True)
with open('/output/comparisons.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_json('/output/summary.json',dict(status='completed',rows=rows,
 configurations={k:dict(training_mode=r.get('training_mode',r['mode']),inference_mode=r['mode'],checkpoint_sha256=r['checkpoint_sha256'],output_sha256=r['rows'][0]['output_sha256']) for k,r in reports.items()},
 caveat='Controlled contrasts, not additive causal percentages. Same fixed 40-step case, one execution per configuration. Training-randomness and FP32 repeat floor not separately measured.'))
