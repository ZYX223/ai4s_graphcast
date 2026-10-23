"""Compare complete outputs and the fixed-case weekly metrics, outside timing."""
import argparse,json,hashlib,statistics
from pathlib import Path
import numpy as np
import xarray as xr
from graphcast_bench.assets import load_case,write_json,sha256
from graphcast_bench.metrics import compare
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,required=True);a=p.parse_args()
case,_=load_case(Path('/data'));case=dict(case,lead_hours=a.steps*6)
result=dict(steps=a.steps,lead_hours=a.steps*6,baseline=str(a.baseline),candidate=str(a.candidate),pairs=[])
reports=[json.loads((root/'run/report.json').read_text()) for root in [a.baseline,a.candidate]]
assert all(r['status']=='completed' and r['measured_steps']==a.steps for r in reports)
assert reports[0]['pci']==reports[1]['pci']
assert reports[0]['data']==reports[1]['data'] and reports[0]['source']==reports[1]['source']
assert len(reports[0]['rows'])==len(reports[1]['rows'])
for i in range(len(reports[0]['rows'])):
 files=[root/'run'/('round%d-rne_xf32'%(i+1))/'prediction.nc' for root in [a.baseline,a.candidate]]
 ref,cand=[xr.load_dataset(f,decode_timedelta=True) for f in files]
 assert set(ref.data_vars)==set(cand.data_vars)
 details=[]
 for name in ref.data_vars:
  x,y=xr.align(ref[name],cand[name],join='exact')
  y=y.transpose(*x.dims)
  x,y=x.values,y.values
  assert x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all()
  unequal=int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32)))
  # Chunk along level/time dimensions to bound temporary memory.
  flatx,flaty=x.reshape(-1),y.reshape(-1)
  max_abs=0.;sum_abs=0.;sum_square=0.
  for start in range(0,flatx.size,1000000):
   diff=flaty[start:start+1000000].astype(np.float64)-flatx[start:start+1000000].astype(np.float64)
   max_abs=max(max_abs,float(np.max(np.abs(diff))));sum_abs+=float(np.sum(np.abs(diff)));sum_square+=float(np.dot(diff,diff))
  details.append(dict(variable=name,elements=x.size,unequal_bits=unequal,mae=sum_abs/x.size,rmse=float(np.sqrt(sum_square/x.size)),max_abs=max_abs))
 metrics=compare(cand,ref,case=case);metrics['reference']='same_day_unfused_rne_xf32'
 result['pairs'].append(dict(round=i+1,baseline_sha256=sha256(files[0]),candidate_sha256=sha256(files[1]),all_11_variables_bitwise_equal=all(d['unequal_bits']==0 for d in details),variables=details,weekly_metrics=metrics))
 del ref,cand
result['performance']={}
for variant,report in zip(['baseline','fused'],reports):
 wall=[row['rollout_wall_seconds'] for row in report['rows']];apply=[row['apply_sum_seconds'] for row in report['rows']]
 result['performance'][variant]=dict(rollout_seconds=wall,rollout_median_seconds=statistics.median(wall),apply_sum_seconds=apply,apply_sum_median_seconds=statistics.median(apply))
b=result['performance']['baseline']['rollout_median_seconds'];c=result['performance']['fused']['rollout_median_seconds']
result['performance'].update(rollout_reduction_percent=(b-c)/b*100,rollout_speedup=b/c)
write_json(a.output,result)
print(json.dumps(dict(performance=result['performance'],bitwise_equal=[r['all_11_variables_bitwise_equal'] for r in result['pairs']],overall_percent=[r['weekly_metrics']['overall_mean_percent'] for r in result['pairs']]),indent=2))
