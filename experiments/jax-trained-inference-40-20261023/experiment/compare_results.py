"""Same 78 equal-weight relative MAEs as weekly inference, new FP32 reference."""
import csv
from pathlib import Path
import numpy as np
import xarray as xr
from graphcast_bench.assets import load_case, read_json, sha256, write_json
from graphcast_bench.metrics import compare, VARIABLES

root=Path('/results')
case,_=load_case('/data')
reports={mode:read_json(root/mode/'report.json') for mode in ('fp32','rne_xf32')}
outputs={}
for mode,r in reports.items():
    assert r['status']=='completed' and r['mode']==mode and r['training_updates']==38
    row=r['rows'][0]
    assert row['steps']==40 and len(row['audit'])==40
    assert all(x['calls']==84 and x['rounded']==(84 if mode=='rne_xf32' else 0) for x in row['audit'])
    assert sha256(root/mode/'prediction.nc')==row['output_sha256']
    assert r['checkpoint_sha256']==sha256(Path(r['checkpoint']))
    outputs[mode]=xr.load_dataset(root/mode/'prediction.nc',decode_timedelta=True)
assert reports['fp32']['case']==reports['rne_xf32']['case']
assert reports['fp32']['model']==reports['rne_xf32']['model']
assert reports['fp32']['task']==reports['rne_xf32']['task']
assert reports['fp32']['code_sha256']==reports['rne_xf32']['code_sha256']
metrics=compare(outputs['rne_xf32'],outputs['fp32'],case=case)
# The historical helper has a fixed reference label; explicitly relabel it.
metrics['reference']='update38_fp32_checkpoint_with_fp32_inference'
metrics['candidate']='update38_rne_xf32_checkpoint_with_rne_xf32_inference'
assert len(metrics['layers'])==78
independent=[]
for row in metrics['layers']:
    a=outputs['rne_xf32'][row['variable']].sel(level=row['level_hpa']).values.astype(np.float64)
    b=outputs['fp32'][row['variable']].sel(level=row['level_hpa']).values.astype(np.float64)
    ratio=100*np.sum(np.abs(a-b))/np.sum(np.abs(b))
    assert np.isclose(ratio,row['relative_error_percent'],rtol=1e-12,atol=1e-12)
    independent.append(ratio)
assert np.isclose(np.mean(independent),metrics['overall_mean_percent'],rtol=1e-12)
summary=dict(schema='graphcast.trained_inference40_comparison.v1',status='completed',
    initialization=case['initialization_time'],steps=40,lead_hours=240,
    metrics=metrics,per_variable_mean_percent={v:float(np.mean([r['relative_error_percent'] for r in metrics['layers'] if r['variable']==v])) for v in VARIABLES},
    requirements=dict(overall_mean_percent_strictly_less_than=5,z500_percent_strictly_less_than=5),
    reference=dict(mode='fp32',checkpoint_sha256=reports['fp32']['checkpoint_sha256'],output_sha256=reports['fp32']['rows'][0]['output_sha256']),
    candidate=dict(mode='rne_xf32',checkpoint_sha256=reports['rne_xf32']['checkpoint_sha256'],output_sha256=reports['rne_xf32']['rows'][0]['output_sha256']),
    process_exit_codes={m:int((root/m/'exit-code.txt').read_text()) for m in reports},
    method='For each of 6 atmospheric variables and 13 levels: mean(abs(candidate-reference))/mean(abs(reference))*100. Equal mean of 78 percentages; unweighted grid. Separate Z500. Final +240h only; independent autoregression.',
    scope='Combined independent-training divergence and inference precision difference. Consistency against trained FP32; not truth-based weather skill or a comparison to original pretrained weights.',
    independent_recalculation_passed=True,metric_source_sha256=sha256('/repo/src/graphcast_bench/metrics.py'))
write_json('/output/summary.json',summary)
with open('/output/by-variable-level.csv','w') as f:
    writer=csv.DictWriter(f,fieldnames=list(metrics['layers'][0]));writer.writeheader();writer.writerows(metrics['layers'])
with open('/output/by-variable.csv','w') as f:
    writer=csv.writer(f);writer.writerow(['variable','mean_relative_error_percent']);writer.writerows(summary['per_variable_mean_percent'].items())
print({k:v for k,v in metrics.items() if k!='layers'})
