"""Reject incomplete kernel capture using successful launch correlation multisets."""
import argparse
import collections
import json
from pathlib import Path
import re
import sqlite3


def inspect_connection(con):
    launch_pattern = re.compile(r'^(?:cuLaunchKernel(?:Ex)?|cudaLaunchKernel(?:ExC)?)(?:_v\d+)?$')
    launches = collections.Counter()
    apis = collections.Counter()
    for tid, correlation, name, status in con.execute('''select r.globalTid,r.correlationId,s.value,r.returnValue
            from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id'''):
        if status == 0 and launch_pattern.fullmatch(name):
            # Nsight encodes globalPid in the high bits, with the thread id in 24 low bits.
            launches[(tid & ~0xffffff, correlation)] += 1
            apis[name] += 1
    kernels = collections.Counter(con.execute('select globalPid,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'))
    missing = launches - kernels
    extra = kernels - launches
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    diagnostics = []
    if 'DIAGNOSTIC_EVENT' in tables:
        diagnostics = [dict(severity=s, text=t, occurrences=n) for s,t,n in con.execute(
            'select severity,text,count(*) from DIAGNOSTIC_EVENT group by severity,text')]
    return dict(passed=bool(kernels) and launches == kernels,
                successful_kernel_launches=sum(launches.values()), kernel_events=sum(kernels.values()),
                missing_kernel_events=sum(missing.values()), unmatched_kernel_events=sum(extra.values()),
                missing_examples=[dict(global_pid=p, correlation=c, count=n) for (p,c),n in list(missing.items())[:10]],
                unmatched_examples=[dict(global_pid=p, correlation=c, count=n) for (p,c),n in list(extra.items())[:10]],
                launch_api_counts=dict(apis), diagnostics=diagnostics,
                method='Successful CUDA kernel-launch API and GPU kernel (globalPid,correlationId) multisets must match; host functions excluded. Diagnostic warnings retained separately.')


def inspect_trace(path):
    con = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    try:
        if con.execute('pragma integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Corrupt SQLite trace')
        return inspect_connection(con)
    finally:
        con.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect_trace(args.trace)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['diagnostics','method']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
