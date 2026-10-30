"""Audit one measured model step and retain FlagTree kernel/HLO attribution."""
import argparse
import collections
import csv
import json
from pathlib import Path
from graphcast_bench.profile_summary import summarize_events, hlo_kernel_tags, category

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
report=json.loads((a.run/'run/report.json').read_text())
assert report['status']=='completed' and report['measured_steps']==1
hlo=(a.run/'run/optimized-hlo.txt').read_text()
events=json.loads((a.run/'trace.json').read_text())['traceEvents']
summary=summarize_events(events,hlo)
# FlagTree's emitted name has an _impl suffix; use the original name in the
# attachment but strip only that suffix when looking up the HLO fusion body.
tags=hlo_kernel_tags(hlo)
groups=collections.defaultdict(lambda:dict(calls=0,total_ms=0.0))
for row in summary['kernels']:
    name=row['name'];key=name[:-5] if name.endswith('_impl') else name
    row['hlo_tags']=tags.get(key,[])
    row['category']=category(name,row['hlo_tags'])
    g=groups[row['category']];g['calls']+=row['calls'];g['total_ms']+=row['total_ms']
for g in groups.values():g['percent_of_kernel_sum']=100*g['total_ms']/summary['gpu_kernel_sum_ms']
summary['categories']=dict(sorted(groups.items(),key=lambda x:-x[1]['total_ms']))
gemms=[json.loads(l) for l in (a.run/'gemm.jsonl').read_text().splitlines()]
gemms=[g for g in gemms if g['label']=='round1-rne_xf32']
assert len(gemms)==len({g['layer'] for g in gemms})==84
assert all(g['mode']==1 and g['rne_inputs'] and g['status']==0 for g in gemms)
assert sum(g['activation_rne_skipped'] for g in gemms)==79
assert summary['categories']['gemm']['calls']==84
assert summary['categories']['rne']['calls']==89
summary.update(api_counts=dict(collections.Counter(g['api'] for g in gemms)),
    fused_activation_calls=79,
    xf32_mx_gemm_kernels=sum(r['calls'] for r in summary['kernels'] if r['category']=='gemm' and '_MX_' in r['name']),
    native_fp32_fallback_gemm_kernels=sum(r['calls'] for r in summary['kernels'] if r['category']=='gemm' and '_MX_' not in r['name']),
    useful_gemm_flops=sum(2*g['m']*g['n']*g['k'] for g in gemms))
assert summary['useful_gemm_flops']==28784948924416
summary['mfu_percent']=100*summary['useful_gemm_flops']/(summary['gpu_kernel_busy_union_ms']/1000*170.3936e12)
summary['mfu_scope']='Single-step active GPU kernel union, GEMM FLOPs approximation; not measured 40-step GPU time'
summary['telemetry_note']='Container hy-smi telemetry could not access /dev/mkfd/service. HIPPROF captured kernel intervals; separate host telemetry retained.'
(a.run/'profile-summary.json').write_text(json.dumps(summary,indent=2))
with (a.run/'kernel-details.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary['kernels'][0]));w.writeheader();w.writerows(summary['kernels'])
print(json.dumps({k:v for k,v in summary.items() if k not in ['kernels','host_apis']},indent=2))
