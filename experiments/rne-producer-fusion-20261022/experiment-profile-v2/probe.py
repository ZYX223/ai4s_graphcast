"""Throwaway feasibility probe: producer RNE fusion, unchanged rocBLAS GEMM.

Times are independent producer+Linear chains, NOT model inference latency.
Every representative uses actual graph shape and frozen real weights/bias.
"""
import argparse
import collections
import ctypes as C
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from importlib import metadata

import jax
import jax.numpy as jnp
import numpy as np
from graphcast_bench.assets import sha256, write_json, verify_source
from graphcast_bench.graph_operator_bench import device_identity
from graphcast_bench.logical_bench import Inventory
from rne_ops import rne
from test_rne_ops import oracle

OUT=Path('/output')
lib=C.CDLL(os.environ['GRAPHCAST_ROUTER_LIBRARY'])
hip=C.CDLL('libamdhip64.so')
lib.graphcast_stage_select.argtypes=[C.c_uint64,C.c_char_p]
lib.graphcast_round_select.argtypes=[C.c_uint64]
lib.experiment_activation_pre_rounded.argtypes=[C.c_int]
for name in ['graphcast_stage_calls','graphcast_round_calls','graphcast_stage_unknown','experiment_skipped_activation_calls']:
    getattr(lib,name).restype=C.c_uint64


def snapshot():
    return [getattr(lib,n)() for n in ['graphcast_stage_calls','graphcast_round_calls','graphcast_stage_unknown','experiment_skipped_activation_calls']]


def select(candidate,label):
    lib.graphcast_stage_select(511,label.encode())
    lib.graphcast_round_select(511)
    lib.experiment_activation_pre_rounded(int(candidate))


def verify_counts(before,candidate,expected):
    delta=[b-a for a,b in zip(before,snapshot())]
    assert delta==[expected,expected,0,expected if candidate else 0],delta
    return delta


def native_rne_probe():
    bits=np.r_[np.array([0,0x80000000,1,0x80000001,0x7fffff,0x800000,0x7f7fffff,0xff7fffff,
                        0x7f800000,0xff800000,0x7fc01234,0xffc05678,0x7f801234,
                        0x3f801000,0x3f803000,0xbf801000,0xbf803000],np.uint32),
               np.random.RandomState(22).randint(0,2**32,100000,dtype=np.uint32)]
    x=jax.device_put(bits.view(np.float32)); y=jnp.zeros(x.shape,jnp.float32)
    jax.block_until_ready((x,y))
    lib.graphcast_round_into.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p]
    assert lib.graphcast_round_into(x.unsafe_buffer_pointer(),y.unsafe_buffer_pointer(),len(bits),1,len(bits),None)==0
    assert hip.hipDeviceSynchronize()==0
    hip_bits=np.asarray(y).view(np.uint32)
    jax_bits=np.asarray(jax.jit(rne)(x)).view(np.uint32)
    np.testing.assert_array_equal(hip_bits,oracle(bits));np.testing.assert_array_equal(jax_bits,hip_bits)
    np.savez(OUT/'rne-bits.npz',input_bits=bits,hip_bits=hip_bits,jax_bits=jax_bits)
    return dict(elements=len(bits),jax_equals_hip=True,jax_equals_independent_oracle=True)


def pair_cases(inv):
    catalog=json.loads(Path('/catalog/catalog.json').read_text())
    scopes={s for c in catalog['cases'] if c['operator']=='linear' for s in c['scopes']}
    producers={v:g for g in inv.data['groups'] for v in g['outputs']}
    cases=collections.OrderedDict()
    for linear in inv.data['groups']:
        if linear['operator']!='linear' or linear['scope'] not in scopes:continue
        producer=producers.get(linear['inputs'][0])
        if producer is None or producer['operator'] not in ('concatenate','silu'):continue
        describe=lambda v:{k:inv.variables[v][k] for k in ('shape','dtype')}
        signature=dict(kind=producer['operator'],inputs=[describe(v) for v in producer['inputs']],
                       linear_inputs=[describe(v) for v in linear['inputs']],
                       producer_equations=[(inv.equations[i]['primitive'],inv.equations[i]['params']) for i in producer['equation_ids']])
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
        if key not in cases:
            cases[key]=dict(id=producer['operator']+'-'+key,signature=signature,producer=producer,linear=linear,calls=0,scopes=[])
        cases[key]['calls']+=1;cases[key]['scopes'].append(linear['scope'])
    cases=list(cases.values())
    assert len(cases)==12 and sum(c['calls'] for c in cases)==79
    # First smoke probes cover the two Mesh edge producer classes.
    cases.sort(key=lambda c:(c['signature']['inputs'][0]['shape'][0]!=327660,c['signature']['kind'],c['id']))
    return cases


@jax.jit
def compare_full(a,b):
    same=jnp.all(jax.lax.bitcast_convert_type(a,jnp.uint32)==jax.lax.bitcast_convert_type(b,jnp.uint32))
    return same,jnp.max(jnp.abs(a-b)),jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b))


def compare_values(a,b):
    exact,error,finite=jax.device_get(compare_full(a,b))
    return dict(bitwise_equal=bool(exact),max_abs_error=float(error),all_finite=bool(finite))


def save_hlo(fn,args,path):
    hlo=fn.lower(*args).compile().as_text();path.write_text(hlo)
    entry=hlo[hlo.rfind('ENTRY '):]
    return dict(path=path.name,sha256=sha256(path),entry_fusions=entry.count(' fusion('),
                entry_custom_calls=entry.count(' custom-call('),entry_gemm_calls=entry.count('custom_call_target="__cublas$gemm"'))


def run():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');opts=ap.parse_args()
    identity=device_identity();verify_source()
    source_paths=[p for folder in [Path('/experiment'),Path('/repo')] for p in folder.rglob('*') if p.is_file()]
    source_paths += [Path(p) for p in ['/inventory/logical/inventory.json','/inventory/logical/constants.npz','/inventory/optimized-hlo.txt','/inventory/report.json','/catalog/catalog.json']]
    source={str(p):sha256(p) for p in source_paths}
    inv=Inventory('/inventory/logical');cases=pair_cases(inv)
    write_json(OUT/'cases.json',cases)
    report=dict(status='running',device=identity,versions={k:metadata.version(k) for k in ['jax','jaxlib','numpy']},
                mode='RNE+XF32',profile=opts.profile,input_policy='Original graph shapes, frozen actual weights/bias; seeded uniform [-4,4] synthetic producer inputs, not captured activations.',
                timing=('Trace only: 3 interleaved rounds, 5 warmups +1 captured call per variant per round; use the unprofiled benchmark for performance.' if opts.profile else 'Independent producer+Linear chain, includes bias, weight RNE, native rocBLAS, JAX dispatch/sync; 3 interleaved rounds, 5 warmups +20 measurements per variant per round. Compile/input generation/correctness excluded. NOT model latency.'),
                rounds=3,iterations=1 if opts.profile else 20,source_sha256=source,rows=[],native_rne_probe=native_rne_probe())
    write_json(OUT/'report.json',report)
    if os.environ.get('EXPERIMENT_SMOKE')=='1':cases=cases[:2]
    # The unchanged baseline router fingerprints (pointer, size) once, assuming
    # model parameters stay alive. Preserve that same lifetime across cases.
    frozen_parameters={case['id']:tuple(jax.device_put(np.asarray(inv.constant(k),np.float32))
                       for k in case['linear']['inputs'][1:]) for case in cases}
    jax.block_until_ready(frozen_parameters)
    pointers=[pair[0].unsafe_buffer_pointer() for pair in frozen_parameters.values()]
    assert len(set(pointers))==len(pointers)
    report['weight_lifetime']='All representative weights remain resident until process exit; no pointer reuse across cases.'
    for ci,case in enumerate(cases):
        print('CASE',ci+1,len(cases),case['id'],case['signature']['inputs'],flush=True)
        producer=inv.function(dict(group=case['producer']))
        linear=inv.function(dict(group=case['linear']))
        producer_rne=jax.jit(lambda *x:rne(producer(*x)))
        standalone_rne=jax.jit(rne)
        baseline=jax.jit(lambda *x:linear(producer(*x[:-2]),x[-2],x[-1]))
        candidate=jax.jit(lambda *x:linear(rne(producer(*x[:-2])),x[-2],x[-1]))
        xs=[jax.random.uniform(jax.random.PRNGKey(20261022+i),tuple(d['shape']),dtype=jnp.float32,minval=-4,maxval=4) for i,d in enumerate(case['signature']['inputs'])]
        w,b=frozen_parameters[case['id']]
        jax.block_until_ready((xs,w,b))
        args=tuple(xs)+(w,b)
        row=dict(id=case['id'],kind=case['signature']['kind'],calls=case['calls'],input_shapes=[list(x.shape) for x in xs],
                 output_shape=inv.variables[case['linear']['outputs'][0]]['shape'],weight_shape=list(w.shape),
                 weight_sha256=hashlib.sha256(np.asarray(w).tobytes()).hexdigest())
        row['hlo']={name:save_hlo(fn,vals,OUT/(case['id']+'-'+name+'.hlo.txt'))
                    for name,fn,vals in [('producer',producer,tuple(xs)),('producer_rne',producer_rne,tuple(xs)),('baseline',baseline,args),('candidate',candidate,args)]}
        # Materialize the unfused producer before applying a separately compiled
        # RNE, then compare all bits against the fused producer (not samples).
        raw=producer(*xs);jax.block_until_ready(raw)
        expected=standalone_rne(raw);jax.block_until_ready(expected)
        actual=producer_rne(*xs);jax.block_until_ready(actual)
        row['producer_validation']=compare_values(expected,actual)
        del raw,expected,actual
        select(False,case['id']+'-validation-baseline');before=snapshot()
        y0=baseline(*args);jax.block_until_ready(y0);verify_counts(before,False,1)
        select(True,case['id']+'-validation-candidate');before=snapshot()
        y1=candidate(*args);jax.block_until_ready(y1);verify_counts(before,True,1)
        row['chain_validation']=compare_values(y0,y1)
        assert row['producer_validation']['all_finite'] and row['chain_validation']['all_finite']
        # Keep compact output samples in addition to full-device comparisons.
        indices=np.linspace(0,y0.shape[0]-1,32,dtype=np.int32)
        np.savez(OUT/(case['id']+'-samples.npz'),rows=indices,baseline=np.asarray(y0[indices]),candidate=np.asarray(y1[indices]))
        del y0,y1
        caches=[f._cache_size() for f in [baseline,candidate]]
        times={'baseline':[],'candidate':[]};audit=[]
        for ri in range(3):
            order=[False,True] if ri%2==0 else [True,False]
            for is_candidate in order:
                name='candidate' if is_candidate else 'baseline';fn=candidate if is_candidate else baseline
                label=case['id']+'-round'+str(ri+1)+'-'+name
                select(is_candidate,label+'-warmup')
                before=snapshot()
                for _ in range(5):jax.block_until_ready(fn(*args))
                verify_counts(before,is_candidate,5)
                select(is_candidate,label)
                before=snapshot();samples=[]
                if opts.profile:assert hip.hipProfilerStart()==0
                for _ in range(1 if opts.profile else 20):
                    start=time.perf_counter();y=fn(*args);jax.block_until_ready(y)
                    samples.append((time.perf_counter()-start)*1000)
                if opts.profile:assert hip.hipProfilerStop()==0
                delta=verify_counts(before,is_candidate,len(samples))
                times[name].append(samples);audit.append(dict(label=label,counters=delta))
                del y
        assert caches==[f._cache_size() for f in [baseline,candidate]]
        row['samples_ms']=times
        row['round_medians_ms']={k:[statistics.median(r) for r in v] for k,v in times.items()}
        row['median_ms']={k:statistics.median(v) for k,v in row['round_medians_ms'].items()}
        row['speedup']=row['median_ms']['baseline']/row['median_ms']['candidate']
        row['reduction_pct']=100*(1-1/row['speedup']);row['audit']=audit
        row['jit_cache_unchanged']=True
        if opts.profile:
            logged=[json.loads(line) for line in (OUT/'gemm.jsonl').read_text().splitlines()]
            logged=[r for r in logged if r['label'].startswith(case['id']+'-')]
            assert len(logged)==38
            assert all(r['weight_sha256']==row['weight_sha256'] and r['mode']==1 and r['status']==0 for r in logged)
            row['all_runtime_weight_hashes_match']=True
        report['rows'].append(row);write_json(OUT/'report.json',report)
        print(json.dumps({k:row[k] for k in ['id','median_ms','speedup','producer_validation','chain_validation']},ensure_ascii=False),flush=True)
        del xs,args,w,b,producer,producer_rne,standalone_rne,linear,baseline,candidate
        gc.collect();jax.clear_caches()
    assert source=={str(p):sha256(p) for p in source_paths}
    report['all_producer_bits_equal']=all(r['producer_validation']['bitwise_equal'] for r in report['rows'])
    report['all_chain_bits_equal']=all(r['chain_validation']['bitwise_equal'] for r in report['rows'])
    if not (report['all_producer_bits_equal'] and report['all_chain_bits_equal']):
        report['status']='validation_failed';write_json(OUT/'report.json',report)
        raise RuntimeError('Producer/chain outputs differ from baseline')
    report.update(status='completed',complete_catalog=len(cases)==12,source_unchanged=True)
    write_json(OUT/'report.json',report)
    print('COMPLETED',len(cases),'cases, exact chain outputs:',report['all_chain_bits_equal'],flush=True)


if __name__=='__main__':run()
