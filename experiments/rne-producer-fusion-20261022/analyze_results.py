"""Verify and summarize the independent-chain experiment, never extrapolate latency."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import shutil
import statistics

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

bench=read(ROOT/'bench-r2/report.json')
profile_path=ROOT/'profile-r3/report.json'
profile=read(profile_path)
assert profile['iterations']==1 and bench['iterations']==20
assert all(len(v)==1 for row in profile['rows'] for rounds in row['samples_ms'].values() for v in rounds)
assert all(r['all_runtime_weight_hashes_match'] for r in profile['rows'])
for report in [bench,profile]:
    assert report['status']=='completed' and report['complete_catalog'] and report['source_unchanged']
    assert len(report['rows'])==12 and sum(r['calls'] for r in report['rows'])==79
    assert report['all_producer_bits_equal'] and report['all_chain_bits_equal']
    assert report['native_rne_probe']['jax_equals_hip'] and report['native_rne_probe']['jax_equals_independent_oracle']
assert (ROOT/'bench-r2/exit-code.txt').read_text().strip()=='0'
assert (ROOT/'profile-r3/exit-code.txt').read_text().strip()=='0'
assert sha(ROOT/'experiment-final-v2/probe.py')==sha(ROOT/'experiment-profile-v2/probe.py')
assert sha(ROOT/'bench-r2/native/xf32_rne_router.cc')==sha(ROOT/'profile-r3/native/xf32_rne_router.cc')
events=read(ROOT/'profile-r3/trace.json')['traceEvents']
kernels=sorted([e for e in events if e.get('cat')=='HIPOPS' and e.get('ph')=='X'],key=lambda e:e['ts'])
assert len(kernels)==324
logs=[json.loads(s) for s in (ROOT/'profile-r3/gemm.jsonl').read_text().splitlines()]
logs=[r for r in logs if '-round' in r['label'] and not r['label'].endswith('-warmup')]
assert len(logs)==72 and all(r['mode']==1 and r['status']==0 and r['rne_inputs'] for r in logs)
index=0;li=0;trace_rows=[];details=[]
for row in profile['rows']:
    times=collections.defaultdict(list);names=collections.defaultdict(set)
    for ri in range(3):
        order=['baseline','candidate'] if ri%2==0 else ['candidate','baseline']
        for mode in order:
            n=5 if mode=='baseline' else 4
            chunk=kernels[index:index+n];index+=n
            label=row['id']+'-round'+str(ri+1)+'-'+mode
            log=logs[li];li+=1
            assert log['label']==label and log['weight_sha256']==row['weight_sha256']
            assert log['m']==row['weight_shape'][1] and log['k']==row['weight_shape'][0] and log['n']==row['input_shapes'][0][0]
            assert chunk[-1]['name']=='loop_add_fusion' and chunk[-2]['name'].startswith('Cijk_')
            assert all(e['name'].startswith('round_rne(') for e in chunk[1:-2])
            expect={('concatenate','baseline'):'wrapped_concatenate',('concatenate','candidate'):'loop_concatenate_fusion',
                    ('silu','baseline'):'loop_multiply_fusion',('silu','candidate'):'loop_bitcast_convert_fusion'}
            assert chunk[0]['name']==expect[row['kind'],mode]
            # Original weight is operand A in all observed calls (W[N,K] as
            # column-major); first HIP RNE is weight, second is activation.
            components=dict(producer=chunk[0],weight_rne=chunk[1],gemm=chunk[-2],bias=chunk[-1])
            if mode=='baseline':components['activation_rne']=chunk[2]
            for key,e in components.items():
                ms=int(e['args']['DurationNs'])/1e6
                assert ms>=0
                times[mode+'_'+key+'_ms'].append(ms)
                details.append(dict(case=row['id'],round=ri+1,variant=mode,component=key,name=e['name'],ms=ms))
            names[mode].add(chunk[-2]['name'])
    assert names['baseline']==names['candidate'] and len(names['baseline'])==1
    trace_rows.append(dict(id=row['id'],kind=row['kind'],baseline_kernels_per_call=5,candidate_kernels_per_call=4,
                           same_gemm_kernel=True,medians_ms={k:statistics.median(v) for k,v in times.items()}))
assert index==len(kernels) and li==len(logs)

for row in bench['rows']:
    assert row['producer_validation']['bitwise_equal'] and row['chain_validation']['bitwise_equal']
    assert row['producer_validation']['max_abs_error']==row['chain_validation']['max_abs_error']==0
    for variant,rounds in row['samples_ms'].items():
        assert len(rounds)==3 and all(len(r)==20 for r in rounds)
        med=[statistics.median(r) for r in rounds]
        assert med==row['round_medians_ms'][variant] and statistics.median(med)==row['median_ms'][variant]
    assert row['hlo']['producer_rne']['entry_fusions']==1 and row['hlo']['candidate']['entry_gemm_calls']==1
    hlo_path=ROOT/'bench-r2'/row['hlo']['producer_rne']['path']
    assert sha(hlo_path)==row['hlo']['producer_rne']['sha256']
    hlo=hlo_path.read_text()
    assert 'shift-right-logical(' in hlo and 'bitcast-convert(' in hlo and 'select(' in hlo
    assert ('concatenate(' if row['kind']=='concatenate' else 'exponential(') in hlo

original=Path('/home/zhangyuxin/zyx/graphcast-sugon')
manifest=read(ROOT/'baseline-manifest.json')
assert all(sha(original/p)==h and sha(ROOT/'baseline-source'/p)==h for p,h in manifest.items())
totals={}
for kind in ['concatenate','silu','all']:
    rows=[r for r in bench['rows'] if kind=='all' or r['kind']==kind]
    b=sum(r['calls']*r['median_ms']['baseline'] for r in rows)
    c=sum(r['calls']*r['median_ms']['candidate'] for r in rows)
    totals[kind]=dict(logical_calls=sum(r['calls'] for r in rows),independent_weighted_baseline_ms=b,
                      independent_weighted_candidate_ms=c,speedup=b/c,reduction_pct=100*(1-c/b))
summary=dict(status='verified',scope='Independent real-shape producer+Linear chains with synthetic activations; NOT whole-model performance/accuracy.',
             baseline_files_unchanged=len(manifest),cases=12,logical_calls_represented=79,all_full_outputs_bitwise_equal=True,
             native_rne_pattern_count=bench['native_rne_probe']['elements'],trace_kernel_count=324,trace_gemm_count=72,
             trace_weight_rne_count=72,trace_activation_rne_count=36,all_gemm_kernel_names_unchanged=True,
             profile_effective_calls_per_round=1,bench_calls_per_round=20,weighted_independent_chain_totals=totals,
             profile_cases=trace_rows,
             sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'bench-r2/report.json',profile_path,ROOT/'profile-r3/trace.json',ROOT/'profile-r3/gemm.jsonl',ROOT/'baseline-manifest.json']})
write(ROOT/'summary.json',summary)
with (ROOT/'kernel-breakdown.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['case','round','variant','component','name','ms']);w.writeheader();w.writerows(details)
with (ROOT/'timings.csv').open('w',newline='') as f:
    fields=['id','kind','input_shapes','weight_shape','logical_calls','baseline_ms','candidate_ms','speedup','reduction_pct','full_output_bitwise_equal']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in bench['rows']:
        w.writerow(dict(id=r['id'],kind=r['kind'],input_shapes=json.dumps(r['input_shapes']),weight_shape=json.dumps(r['weight_shape']),logical_calls=r['calls'],
                        baseline_ms=r['median_ms']['baseline'],candidate_ms=r['median_ms']['candidate'],speedup=r['speedup'],reduction_pct=r['reduction_pct'],full_output_bitwise_equal=True))
print(json.dumps({k:v for k,v in summary.items() if k not in ('profile_cases','sources')},indent=2))
