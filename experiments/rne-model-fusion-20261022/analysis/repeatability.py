"""Measure within-variant variation separately from cross-variant differences."""
import json
from pathlib import Path
import xarray as xr
from graphcast_bench.assets import load_case,write_json
from graphcast_bench.metrics import compare
case,_=load_case(Path('/data'));result={}
for variant in ['baseline','fused']:
 root=Path('/results/bench-'+variant+'-r1/run')
 report=json.loads((root/'report.json').read_text());assert report['status']=='completed' and report['repeats']==3
 ref=xr.load_dataset(root/'round1-rne_xf32/prediction.nc',decode_timedelta=True)
 rows=[]
 for n in [2,3]:
  other=xr.load_dataset(root/('round%d-rne_xf32'%n)/'prediction.nc',decode_timedelta=True)
  metrics=compare(other,ref,case=case);metrics['reference']=variant+'_round1'
  rows.append(dict(round=n,accuracy=metrics));del other
 del ref
 result[variant]=rows
write_json(Path('/results/repeatability.json'),result)
print(json.dumps({k:[dict(round=r['round'],overall_percent=r['accuracy']['overall_mean_percent'],z500_percent=r['accuracy']['z500_percent']) for r in rows] for k,rows in result.items()},indent=2))
