"""Recompute +240h consistency against the immutable Oct29 H100 outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from graphcast_bench.metrics import compare, final_frame

def load(path):
    raw=xr.load_dataset(path,decode_cf=False)
    assert raw.time.attrs.get('units')=='hours' and np.array_equal(raw.time.values,[240])
    raw.time.attrs.pop('dtype',None)
    result=xr.decode_cf(raw,decode_timedelta=True)
    final_frame(result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    case=json.loads(a.case.read_text())
    reports={name:json.loads((root/'report.json').read_text()) for name,root in [('candidate',a.candidate),('reference',a.reference)]}
    for name,r in reports.items():
        assert r['status']=='completed' and r['measured_steps']==40 and r['warmup_steps']==5
        assert r['mode']=='tf32' and r['storage']=='float32' and r['router_build'] is None
    assert reports['reference']['repeats']==3
    assert reports['candidate']['source']['commit']==reports['reference']['source']['commit']
    assert reports['candidate']['data']==reports['reference']['data']
    files=[]
    refs={}
    def check_file(root,row,kind):
        folder=root/row['run']
        path=folder/'prediction.nc'
        record=json.loads((folder/'report.json').read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==record['output_sha256']
        files.append(dict(kind=kind,round=row['round'],path=str(path),sha256=digest))
        return path
    for row in reports['reference']['rows']:
        refs[row['round']]=load(check_file(a.reference,row,'frozen_reference'))
    rows=[];details=[]
    for row in reports['candidate']['rows']:
        candidate=load(check_file(a.candidate,row,'candidate'))
        for rr,reference in sorted(refs.items()):
            result=compare(candidate,reference,case=case)
            result['reference']='gpu41_H100_JAX_TF32_no_explicit_RNE_20261029'
            result['internal_under_5_percent']=result.pop('metrics_within_limits')
            details.extend(dict(candidate_round=row['round'],reference_round=rr,**x) for x in result.pop('layers'))
            result.update(candidate_round=row['round'],reference_round=rr,primary_pair=row['round']==rr)
            rows.append(result)
        del candidate
    summary=dict(status='verified',candidate_variant=reports['candidate'].get('experiment_variant','unknown'),
        reference='gpu41 H100 JAX TF32, no explicit RNE, 20261029; frozen predictions unchanged',
        metric='mean(abs(candidate-reference))/mean(abs(reference))*100 per field; H100 denominator; equal mean of 6 upper-air variables x 13 levels; Z500 separately.',
        scope='Independent 40-step autoregressions from identical pretrained weights/input/stats; +240h only; not weather truth.',
        pairing='Matching rounds primary; all candidate rounds versus all 3 frozen reference rounds also checked.',
        input_case_match=True,files=files,rows=rows,
        all_pairs_under_5_percent=all(r['internal_under_5_percent'] for r in rows),
        primary_pairs_under_5_percent=all(r['internal_under_5_percent'] for r in rows if r['primary_pair']))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,data in [('pairs.csv',rows),('variable-level-details.csv',details)]:
        with (a.output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('files','rows')},indent=2),flush=True)
    if not summary['all_pairs_under_5_percent']:
        raise SystemExit('Candidate failed final-frame 5% consistency check')

if __name__=='__main__':
    main()
