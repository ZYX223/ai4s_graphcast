"""Read-only historical38 replay: loss alignment, device union, useful-GEMM MFU.

No GPU experiment is rerun. Original reports/checkpoints/traces remain frozen.
The native numerator is reconstructed from the equivalent no-remat model graph;
it is not a claim that native hardware executed only the useful FLOPs.
"""
import csv
import hashlib
import json
import math
import pathlib
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parent
OLD = ROOT.parent / 'jax-training-38-20261023'
OUT = ROOT / 'native-comparison'
OUT.mkdir(exist_ok=True)
sources = {}

def read(path):
    content = path.read_bytes()
    sources[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)

def union(intervals):
    intervals = sorted(intervals)
    assert intervals
    lo, hi = intervals[0]
    total = 0
    for a, b in intervals[1:]:
        assert b >= a
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + hi - lo

current = read(ROOT / 'summary.json')
rows, checks = [], []
native_total_ns, useful = 0, 0
native_wall, whole_peak, jax_peak = 0., 0., 0.
for horizon in range(1, 13):
    old_dir = OLD / ('rne_xf32-r%02d' % horizon)
    native = read(old_dir / 'report.json')
    nv = read(ROOT / ('h100-r%02d' % horizon) / 'report.json')
    fl = read(ROOT / ('flagos-r%02d' % horizon) / 'report.json')
    counts = read(ROOT / 'flops' / ('horizon-%02d.json' % horizon))
    timing = {r['update']: r for r in read(old_dir / 'device-timing.json')['rows']}
    plan = read(old_dir / 'sample-plan.json')
    assert plan == read(ROOT / ('flagos-r%02d' % horizon) / 'sample-plan.json')
    assert native['status'] == 'completed'
    assert (old_dir / 'exit-code.txt').read_text().strip() == '0'
    for key in ['model', 'task', 'optimizer', 'data_sha256']:
        assert native[key] == nv[key] == fl[key], (horizon, key)
    assert len(native['rows']) == len(nv['rows']) == len(fl['rows'])
    db = old_dir / 'trace.db'
    con = sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True)
    tables = [x[0] for x in con.execute("select name from sqlite_master where type='table'") if x[0].startswith('HIPOPS_')]
    assert tables
    for a, b, c in zip(native['rows'], nv['rows'], fl['rows']):
        for key in ['update', 'rollout', 'lr', 'starts']:
            assert a[key] == b[key] == c[key], (horizon, key)
        assert all(math.isfinite(x['loss']) for x in (a, b, c)) and b['loss'] != 0
        intervals = []
        for table in tables:
            intervals.extend(con.execute('select BeginNs,EndNs from "' + table + '" where BeginNs>=? and EndNs<=?', (a['trace_begin_ns'], a['trace_end_ns'])).fetchall())
        ns = union(intervals)
        previous = timing[a['update']]
        assert len(intervals) == previous['gpu_kernel_count']
        assert abs(ns / 1e6 - previous['gpu_kernel_busy_union_ms']) < 1e-6
        assert a['gemm']['calls'] == a['gemm']['rounded']
        native_total_ns += ns
        useful += counts['useful_flops_per_update']
        native_wall += a['synchronized_update_seconds']
        jax_peak = max(jax_peak, a['jax_memory']['peak_bytes_in_use'] / 2**30)
        rows.append(dict(update=a['update'], horizon=horizon, lr=a['lr'], starts=a['starts'],
                         h100_loss=b['loss'], native_loss=a['loss'], flagos_loss=c['loss'],
                         native_abs_difference=abs(a['loss']-b['loss']),
                         native_relative_percent=100*abs(a['loss']-b['loss'])/abs(b['loss']),
                         flagos_abs_difference=abs(c['loss']-b['loss']),
                         flagos_relative_percent=100*abs(c['loss']-b['loss'])/abs(b['loss']),
                         native_kernel_seconds=ns/1e9,
                         native_useful_gemm_flops=counts['useful_flops_per_update']))
    con.close()
    whole_peak = max(whole_peak, native['observed_device_peak_mib']/1024)
    checks.append(dict(horizon=horizon, updates=len(native['rows']), device_sqlite=str(db),
                       model_task_optimizer_data_plan_equal=True, all_device_unions_match=True))

assert [r['update'] for r in rows] == list(range(1,39))
assert useful == current['backends']['flagos']['useful_gemm_flops']
seconds = native_total_ns / 1e9
peak = 170.3936
mfu = 100 * useful / (seconds * peak * 1e12)
native_summary = dict(updates=38, gpu_kernel_seconds=seconds, synchronized_update_seconds=native_wall,
    useful_gemm_flops=useful, peak_tflops=peak, mfu_percent=mfu,
    mfu_over_h100_percent=100*mfu/current['backends']['h100']['mfu_percent'],
    whole_card_peak_gib=whole_peak, jax_peak_gib=jax_peak,
    last_loss=rows[-1]['native_loss'], max_loss_relative_percent=max(r['native_relative_percent'] for r in rows),
    mean_loss_relative_percent=sum(r['native_relative_percent'] for r in rows)/38,
    last_loss_relative_percent=rows[-1]['native_relative_percent'],
    last_loss_abs_difference=rows[-1]['native_abs_difference'],
    maximum_error_row=max(rows, key=lambda r:r['native_relative_percent']))
result = dict(native=native_summary, current=current,
    native_speedup_over_flagos=current['backends']['flagos']['gpu_kernel_seconds']/seconds,
    native_time_reduction_vs_flagos_percent=100*(1-seconds/current['backends']['flagos']['gpu_kernel_seconds']),
    native_to_h100_internal5_pass=all(r['native_relative_percent']<5 for r in rows),
    stage_checks=checks, limitations=[
        'Historical 2026-10-23 native measurements recomputed 2026-10-30; no native rerun.',
        'Native useful FLOPs reconstructed from matched model/task/loss source, horizon/microbatch count and graph dimensions, using current BW no-remat differentiated GEMM counts.',
        'Native vendor JAX0.6.0+das.opt1.dtk2604 differs from FlagOS JAX0.6.0; native uses original remat, FlagOS additional LayerNorm remat; different physical cards of same BW1100 model. Not isolated backend A/B.',
        'Original native reports lack runtime initial-parameter/optimizer/batch hashes; archived loader, recipe and current archived weight/stat files match, but historical runtime fingerprint is not retroactively proven.',
        'Native historical HIPPROF signal11 cleanup warnings retained. Outer exits0 and all38 measured HIPOPS kernel intervals independently match saved results; no claim that historical teardown was clean.',
        'Loss is pre-update mean of2 microbatches; independent trajectories, not convergence or trained-checkpoint inference comparison.',
        'MFU uses GPU compute-kernel interval union and useful-GEMM approximation. Relative MFU is candidate peak normalization, not formal phase efficiency acceptance.'
    ])
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/'source-hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
with (OUT/'loss-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(native_summary,ensure_ascii=False,indent=2))
print('Native speedup over FlagOS:',result['native_speedup_over_flagos'])
