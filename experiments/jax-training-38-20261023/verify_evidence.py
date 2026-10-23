"""Read-only cross-check of completed training stages and their GPU traces."""
import collections
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
baseline = json.loads((root / 'fp32-r01/launch.json').read_text())
stages = []
for horizon in range(1, 13):
    for mode in ['fp32', 'rne_xf32']:
        run = root / ('%s-r%02d' % (mode, horizon))
        if not (run / 'device-timing.json').exists():
            continue
        report = json.loads((run / 'report.json').read_text())
        assert report['status'] == 'completed'
        launch = json.loads((run / 'launch.json').read_text())
        assert launch['source_sha256'] == baseline['source_sha256']
        for name, expected in launch['source_sha256'].items():
            assert hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() == expected
        calls = collections.Counter()
        for line in (run / 'gemm.jsonl').read_text().splitlines():
            event = json.loads(line)
            if event['label'].startswith('update-'):
                assert event['mode'] == int(mode == 'rne_xf32')
                assert event['rounded'] == (mode == 'rne_xf32')
                calls[event['label']] += 1
        targets = collections.Counter(re.findall(r'custom_call_target="([^"]+)"', (run / 'optimized-hlo.txt').read_text()))
        # R1 is the official direct-loss special case; R>=2 uses scan and
        # checkpointed autoregressive forward/reverse bodies.
        expected_static = 402 if horizon == 1 else 489
        assert dict(targets) == {'__cublas$gemm': expected_static}, dict(targets)
        expected_runtime = expected_static * 2 * horizon
        rounding_ns = 0
        for row in report['rows']:
            assert calls['update-%02d' % row['update']] == row['gemm']['calls'] == expected_runtime
            kernels = json.loads((run / ('kernels-update-%02d.json' % row['update'])).read_text())
            rounding = [v for name, v in kernels.items() if name.startswith('round_rne')]
            assert sum(v['calls'] for v in rounding) == (expected_runtime * 2 if mode == 'rne_xf32' else 0)
            rounding_ns += sum(v['duration_ns'] for v in rounding)
        stages.append(dict(mode=mode, rollout=horizon, updates=len(report['rows']),
                           static_gemm_calls=expected_static,
                           runtime_gemm_calls_per_update=expected_runtime,
                           mean_rounding_gpu_seconds=rounding_ns / len(report['rows']) / 1e9))
result = dict(status='completed' if len(stages) == 24 else 'partial', stages=stages,
              checks=['Frozen source manifests match across stages and files',
                      'All HLO custom calls are GEMM, with expected static/runtime counts',
                      'Runtime precision modes and operand rounding flags match',
                      'Exactly two traced RNE kernels per rounded GEMM'])
(root / 'evidence-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], verified_stages=len(stages))))
