"""Compare all 3x3 final frames, retaining the three matching rounds as primary."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from graphcast_bench.metrics import compare, final_frame

p=argparse.ArgumentParser()
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--reference-report',type=Path,help='Manifest if the reference archive separates top-level metadata')
p.add_argument('--case',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False)
case=json.loads(a.case.read_text())
loaded={};files=[];reports={}
for platform,root,mode in [('candidate',a.candidate,'rne_xf32'),('h100',a.reference,'tf32')]:
    manifest=a.reference_report if platform=='h100' and a.reference_report else root/'report.json'
    reports[platform]=json.loads(manifest.read_text())
    assert reports[platform]['status']=='completed'
    assert reports[platform]['measured_steps']==40 and reports[platform]['repeats']==3
    for r in (1,2,3):
        folder=root/('round{}-{}'.format(r,mode));path=folder/'prediction.nc'
        report=json.loads((folder/'report.json').read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==report['output_sha256']
        raw=xr.load_dataset(path,decode_cf=False)
        assert raw.time.attrs.get('units')=='hours' and np.array_equal(raw.time.values,[240])
        raw.time.attrs.pop('dtype',None)
        loaded[platform,r]=xr.decode_cf(raw,decode_timedelta=True)
        final_frame(loaded[platform,r])
        files.append(dict(platform=platform,round=r,path=str(path),sha256=digest))
assert reports['candidate']['source']['commit']==reports['h100']['source']['commit']
assert reports['candidate']['data']==reports['h100']['data']
rows=[];details=[]
for cr in (1,2,3):
    for hr in (1,2,3):
        result=compare(loaded['candidate',cr],loaded['h100',hr],case=case)
        result['reference']='gpu41_H100_JAX_TF32_no_explicit_RNE_20261029'
        result['internal_under_5_percent']=result.pop('metrics_within_limits')
        result['stage_at_most_10_percent']=result['overall_mean_percent']<=10 and result['z500_percent']<=10
        details.extend(dict(candidate_round=cr,h100_round=hr,**x) for x in result.pop('layers'))
        row=dict(candidate_round=cr,h100_round=hr,primary_pair=cr==hr,**result)
        rows.append(row);print(json.dumps(row),flush=True)
summary=dict(status='verified',candidate='BW1100 FlagOS JAX fl_only fused RNE+XF32 GEMM_EX candidate, 20261030',
    reference='gpu41 H100 JAX TF32, no explicit RNE, 20261029',
    scope='Same original pretrained weights/input/stats; independent autoregression; +240h final frame only, not weather truth.',
    metric='mean(abs(candidate-reference))/mean(abs(reference))*100 per field, H100 denominator; equal mean of 6 upper-air variables x 13 levels; Z500 separate.',
    pairing='Same-index 3 pairs are primary; all 3x3 are reported for repeat variability.',
    input_case_match=True,inputs=files,rows=rows,
    all_nine_internal_pass=all(r['internal_under_5_percent'] for r in rows),
    all_nine_stage_pass=all(r['stage_at_most_10_percent'] for r in rows))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
for name,data in [('pairs.csv',rows),('variable-level-details.csv',details)]:
    with (a.output/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
