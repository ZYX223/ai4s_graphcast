"""Read all traced GPU kernels; keep MFU, synchronization and wall scopes separate."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics

FLOPS40 = 1151397956976640
PEAK = 494.7e12

def union_ns(intervals):
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('Negative event duration')
        total += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = stop if end is None else max(end, stop)
    return total

def category(name):
    lower = name.lower()
    if any(token in lower for token in ('gemm', 'cutlass', 'matmul')):
        return 'gemm_name_matched'
    for token in ('scatter', 'reduce', 'gather', 'concatenate'):
        if token in lower:
            return token + '_named_fusion'
    return 'other_fusion_auxiliary'

def trace_integrity(con):
    """Check both directions of every successful non-graph GPU launch.

    Nsight encodes globalTid as globalPid plus a low 24-bit thread ID.
    CUDA graph capture is disabled for this experiment; a graph-launch API is
    rejected rather than assuming one graph launch corresponds to one kernel.
    """
    tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernel_keys=collections.Counter(con.execute('SELECT globalPid,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL'))
    launch_keys=collections.Counter();apis=collections.Counter()
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
        if table not in tables:continue
        rows=con.execute('SELECT r.globalTid,r.correlationId,r.returnValue,s.value FROM '+table+' r JOIN StringIds s ON r.nameId=s.id')
        for tid,correlation,code,name in rows:
            if 'Launch' not in name or 'LaunchHostFunc' in name:continue
            if not re.fullmatch(r'(cuLaunchKernel(?:Ex)?|cudaLaunchKernel(?:ExC)?)(?:_v[0-9]+)?',name):
                raise ValueError('Unsupported GPU launch API in non-graph measurement: '+name)
            if code!=0 or tid is None or correlation is None:
                raise ValueError('Failed or uncorrelated GPU launch: '+name)
            launch_keys[(tid & ~0xffffff,correlation)]+=1;apis[name]+=1
    if not launch_keys or launch_keys!=kernel_keys:
        missing=list((launch_keys-kernel_keys).items())[:5]
        orphan=list((kernel_keys-launch_keys).items())[:5]
        raise ValueError('Kernel/launch correlation mismatch; missing kernels={}, orphan kernels={}'.format(missing,orphan))
    if any(v!=1 for v in kernel_keys.values()) or len({pid for pid,cid in kernel_keys})!=1:
        raise ValueError('Expected exactly one kernel per launch in a single process')
    diagnostics=[]
    if 'DIAGNOSTIC_EVENT' in tables:
        diagnostics=[dict(timestamp=t,severity=s,text=message) for t,s,message in con.execute('SELECT timestamp,severity,text FROM DIAGNOSTIC_EVENT ORDER BY timestamp')]
    return dict(status='kernel_launch_correlations_complete',matched_kernel_launches=sum(kernel_keys.values()),
        launch_api_counts=dict(apis),unmatched_launches=0,orphan_kernels=0,diagnostics=diagnostics,
        interpretation='All captured successful GPU launches and kernels match one-to-one. Nsight general CUDA warnings are retained, not dismissed. cuBLAS/NVTX hook warnings prevent library-hook attribution, but actual GPU kernel names/HLO still identify the backend. Correlation matching alone cannot prove events absent from both tables; repeated-workload signature and benchmark step audit are checked separately.')

def workload_signature(result,steps):
    gemms=[r for r in result['kernels'] if category(r['name'])=='gemm_name_matched']
    if not gemms or any(r['count']%steps for r in gemms):
        raise ValueError('GEMM launch signature does not contain the expected number of complete steps')
    signature={r['name']:r['count']//steps for r in gemms}
    return dict(expected_steps=steps,complete_repeated_gemm_signature=True,
        gemm_kernel_launches_per_step=sum(signature.values()),kernel_name_counts_per_step=signature,
        limitation='A kernel-launch signature supports workload completeness alongside the benchmark audit; it is not a replacement for explicit step-boundary tracing.')

def read_trace(path, useful_flops=FLOPS40):
    con = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
    kernels = con.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
    if not kernels or len(con.execute('SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()) != 1:
        raise ValueError('Expected nonempty trace from exactly one GPU')
    copies = con.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
    integrity=trace_integrity(con)
    con.close()
    active = union_ns([(a,b) for a,b,n in kernels]) / 1e9
    groups = collections.defaultdict(lambda: [0, 0])
    cats = collections.defaultdict(lambda: [0, 0])
    for a,b,name in kernels:
        for group,key in [(groups,name),(cats,category(name))]:
            group[key][0] += 1
            group[key][1] += b-a
    total = sum(b-a for a,b,n in kernels)
    def table(data):
        return [dict(name=k, count=v[0], seconds=v[1]/1e9, fraction_kernel_sum_percent=100*v[1]/total)
                for k,v in sorted(data.items(), key=lambda pair: -pair[1][1])]
    return dict(trace=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kernel_count=len(kernels), kernel_active_seconds=active,
        kernel_sum_seconds=total/1e9, kernel_span_seconds=(max(b for a,b,n in kernels)-min(a for a,b,n in kernels))/1e9,
        memcpy_sum_seconds=sum(b-a for a,b in copies)/1e9,
        kernel_and_copy_union_seconds=union_ns([(a,b) for a,b,n in kernels]+copies)/1e9,
        tf32_named_kernel_count=sum('tf32' in n.lower() for a,b,n in kernels),
        rne_named_kernel_count=sum('round_rne' in n.lower() for a,b,n in kernels),
        mfu_percent=useful_flops/(active*PEAK)*100, categories=table(cats), kernels=table(groups),trace_integrity=integrity)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads((a.profile_dir/'run/report.json').read_text())
    assert report['status']=='completed' and report['measured_steps']==40 and report['warmup_steps']==5
    assert report['mode']=='tf32' and report['storage']=='float32' and report['router_build'] is None
    assert report['jax_matmul_precision']=='tensorfloat32'
    traces=sorted(a.profile_dir.glob('model-trace*.sqlite'))
    assert len(traces)==report['repeats']==len(report['rows'])
    capture=json.loads((a.profile_dir/'run/capture-ranges.json').read_text())
    assert len(capture)==len(traces) and all(r['measured_steps']==40 for r in capture)
    rows=[]
    for trace,row,bounds in zip(traces,report['rows'],capture):
        assert len(row['apply_seconds'])==40 and row['jit_cache_before']==row['jit_cache_after']
        assert bounds['label']==row['run']
        r=read_trace(trace)
        r['workload_signature']=workload_signature(r,40)
        assert r['rne_named_kernel_count']==0
        r.update(round=row['round'],label=row['run'],rollout_wall_seconds=row['rollout_wall_seconds'],
            synchronized_apply_sum_seconds=row['apply_sum_seconds'])
        rows.append(r)
    hlo=(a.profile_dir/'run/optimized-hlo.txt').read_text()
    result=dict(status='measured_and_verified', variant=report.get('experiment_variant','historical_baseline'),
        useful_flops_per_40_steps=FLOPS40, peak_flops_per_second=PEAK,
        formula='F_40 / (union of all captured GPU kernel intervals * dense TF32 peak)',
        scope='Full 40 steps after 5 warmups/reset per round; GPU-active GEMM-FLOPs approximation, not end-to-end MFU.',
        device=report['device'],pci=report['pci'],versions=report['versions'],xla_flags=report['xla_flags'],
        rounds=rows,median_gpu_active_seconds=statistics.median(r['kernel_active_seconds'] for r in rows),
        median_mfu_percent=statistics.median(r['mfu_percent'] for r in rows),
        median_profiled_rollout_wall_seconds=statistics.median(r['rollout_wall_seconds'] for r in rows),
        hlo_custom_call_targets=dict(collections.Counter(re.findall('custom_call_target="([^"]+)"',hlo))),
        hlo_bf16_occurrences=len(re.findall(r'\bbf16\[',hlo)),
        limitations=['Kernel-name categories overlap fused semantic work; not instruction-level attribution.',
            'Profiler overhead and shared-host activity can affect timings; no clock lock.',
            'Legacy benchmark accuracy remains official JAX FP32. Current H100 consistency is in separate comparison output.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rounds'},indent=2))

if __name__=='__main__':
    main()
