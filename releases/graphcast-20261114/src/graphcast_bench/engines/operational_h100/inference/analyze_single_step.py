"""Analyze two isolated one-step captures without scaling their time to 40 steps."""
import argparse
import json
import os
from pathlib import Path
from analyze import read_trace,workload_signature,FLOPS40,PEAK

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-dir',type=Path,required=True)
    a=p.parse_args();root=a.profile_dir
    report=json.loads((root/'run/report.json').read_text())
    assert report['status']=='completed' and report['measured_steps']==1 and report['repeats']==2
    assert report['warmup_steps']==5 and len(json.loads((root/'run/step-audit.json').read_text()))==12
    traces=sorted(root.glob('model-trace*.sqlite'));assert len(traces)==2
    rows=[]
    for trace,record,label in zip(traces,report['rows'],['prior_best_mesh2grid3','mesh2grid3_'+os.environ['GRAPHCAST_VARIANT']]):
        assert len(record['apply_seconds'])==1 and record['jit_cache_before']==record['jit_cache_after']
        row=read_trace(trace,useful_flops=FLOPS40/40)
        row['workload_signature']=workload_signature(row,1)
        row.update(variant=label,captured_steps=1,profiled_rollout_wall_seconds=record['rollout_wall_seconds'],synchronized_apply_seconds=record['apply_sum_seconds'])
        assert row['tf32_named_kernel_count']>0 and row['rne_named_kernel_count']==0
        rows.append(row)
    result=dict(status='measured_and_verified',scope='Two separate cudaProfilerApi captures, one original step and one candidate step, each after 5 warmups and input reset. Not 40-step measurement.',
        useful_flops_per_capture=FLOPS40/40,peak_flops_per_second=PEAK,
        formula='F_step / (union of all kernels in that one-step capture * dense TF32 peak)',rows=rows,
        limitation='One sample per variant; full-rollout repeated performance evidence remains the separate three-round 40-step experiment.')
    (root/'single-step-timing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
