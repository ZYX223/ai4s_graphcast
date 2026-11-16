"""Link Nsight kernels to their CUDA-launch NVTX HLO scope, not CPU duration."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import re
import sqlite3


def containing_scope(scopes, start, end):
    index = bisect.bisect_right(scopes, (start, float('inf'), '\uffff')) - 1
    for i in range(index, -1, -1):
        a, b, label = scopes[i]
        if a <= start and end <= b:
            return label
    return None


def phase(name):
    if name.endswith('/transpose'):
        return 'backward_gemm'
    if name.endswith('/dot_general'):
        return 'recomputed_forward_gemm' if 'rematted_computation' in name else 'forward_gemm'
    return 'unclassified_gemm'


def attribute(folder):
    con = sqlite3.connect('file:' + str(folder/'train-trace.sqlite') + '?mode=ro', uri=True)
    scopes = collections.defaultdict(list)
    query = """select e.start,e.end,e.globalTid,coalesce(e.text,s.value)
        from NVTX_EVENTS e left join StringIds s on e.textId=s.id
        where e.end is not null and coalesce(e.text,s.value) like 'Thunk:%'"""
    for start, end, tid, label in con.execute(query):
        scopes[tid].append((start, end, label))
    for intervals in scopes.values():
        intervals.sort()
    launches = {}
    for start, end, correlation, tid in con.execute('select start,end,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME'):
        launches[correlation] = (start, end, tid)
    groups = collections.defaultdict(lambda: dict(calls=0, summed_gpu_ns=0))
    gemm_groups = collections.defaultdict(lambda: dict(calls=0, summed_gpu_ns=0))
    mapping = []
    query = 'select k.start,k.end,k.correlationId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id'
    for start, end, correlation, kernel in con.execute(query):
        launch = launches.get(correlation)
        scope = containing_scope(scopes[launch[2]], *launch[:2]) if launch else None
        matched = re.search(r'name=(.*),hlo_op=', scope) if scope else None
        name = matched.group(1) if matched else ''
        stage = next((s for s in ['grid2mesh_gnn', 'mesh2grid_gnn', 'mesh_gnn'] if s in name), 'other_or_unattributed')
        groups[stage]['calls'] += 1
        groups[stage]['summed_gpu_ns'] += end - start
        if any(s in kernel.lower() for s in ['gemm', 'cublas', 'cutlass']):
            key = stage + '/' + phase(name)
            gemm_groups[key]['calls'] += 1
            gemm_groups[key]['summed_gpu_ns'] += end - start
        if scope is None:
            mapping.append(dict(kernel=kernel, correlation=correlation))
    con.close()
    return dict(stages=dict(groups), gemm_scopes=dict(gemm_groups),
                unmatched_kernels=len(mapping), unmatched_examples=mapping[:10],
                timing='GPU kernel duration sum; scopes from correlated CPU launch, never NVTX CPU duration',
                classification='Forward/recompute/backward GEMM labels inferred from HLO scope suffix and rematted_computation; not an exhaustive model FLOPs ledger')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = attribute(a.folder)
    value = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.write_text(value)
    else:
        print(value)
