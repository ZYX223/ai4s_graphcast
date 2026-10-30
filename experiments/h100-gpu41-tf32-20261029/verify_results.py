"""Independently verify saved predictions against the frozen FP32 reference."""
import json
import statistics
from pathlib import Path
import xarray as xr
from graphcast_bench.assets import load_case, sha256, verify_source, write_json
from graphcast_bench.metrics import compare, load_reference

root = Path('/result')
case, prepared = load_case(Path('/data'))
report = json.loads((root/'benchmark-r1/run/report.json').read_text())
assert report['status'] == 'completed'
assert report['source'] == verify_source() and report['data'] == prepared
assert report['mode'] == 'tf32' and report['jax_matmul_precision'] == 'tensorfloat32'
assert report['storage'] == 'float32' and report['router_build'] is None
assert report['warmup_steps'] == 5 and report['measured_steps'] == 40 and report['repeats'] == 3
assert len(report['rows']) == len(report['warmups']) == 3
probe = json.loads((root/'probe-r1/probe/report.json').read_text())
assert probe['status'] == 'passed' and probe['actual_tf32_numerical_effect']
audit = json.loads((root/'benchmark-r1/run/step-audit.json').read_text())
assert len(audit) == 135 and all('native_calls' not in row for row in audit)
reference = load_reference(Path('/data/reference.npz'), case)
rows = []
for row in report['rows']:
    folder = root/'benchmark-r1/run'/row['run']
    full = json.loads((folder/'report.json').read_text())
    assert len(row['apply_seconds']) == 40 and row['jit_cache_before'] == row['jit_cache_after']
    assert sha256(folder/'prediction.nc') == full['output_sha256']
    with xr.open_dataset(folder/'prediction.nc', decode_timedelta=True) as opened:
        prediction = opened.load()
    assert len(prediction.data_vars) == 11
    recomputed = compare(prediction, reference, case=case)
    assert recomputed == full['accuracy'], 'Saved metrics must match complete recomputation'
    rows.append(dict(round=row['round'], rollout_wall_seconds=row['rollout_wall_seconds'],
                     apply_sum_seconds=row['apply_sum_seconds'],
                     overall_mean_percent=recomputed['overall_mean_percent'],
                     z500_percent=recomputed['z500_percent'], z500_rmse_m=recomputed['z500_rmse_m'],
                     passed=recomputed['metrics_within_limits'], output_sha256=full['output_sha256']))
    del prediction
summary = dict(status='verified', model_source=report['source']['commit'],
               device=report['device'], pci=report['pci'], versions=report['versions'],
               precision='TF32 matmul; FP32 arrays; no explicit RNE',
               steps=40, warmup_steps=5, repeats=3, rows=rows,
               rollout_wall_median_seconds=statistics.median(r['rollout_wall_seconds'] for r in rows),
               apply_sum_median_seconds=statistics.median(r['apply_sum_seconds'] for r in rows),
               all_rounds_below_5_percent=all(r['passed'] for r in rows),
               metric='Final +240h: equal mean of 78 variable-level relative MAEs; Z500 separately; existing official JAX FP32 reference',
               scope='One initialization case; three independent autoregressive forecasts, not observations or a multi-case skill evaluation',
               cpu_suite='33/35 passed. Two Hygon logical-operator replay tests require JAX 0.6 batching-dimension API; these tools are unused by this H100 inference.')
write_json(root/'summary.json', summary)
print(json.dumps(summary, indent=2), flush=True)
