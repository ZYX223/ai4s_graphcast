"""Run a bounded, source-verified 38-update full-course arm; preserve checkpoints for recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('arm', choices=['baseline', 'candidate'])
p.add_argument('--prepare-only', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent
c = json.loads((root / 'config.json').read_text())
arm = c['arms'][a.arm]
ref = json.loads((root / 'reference.json').read_text())
template = json.loads((root / (a.arm + '-template.json')).read_text())
out = root / a.arm
source = Path(arm['source'])
model = Path(c['model'])

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def verify():
    expected = dict(ref['flaggems_source_sha256'], **arm['overlay'])
    assert json.loads((source / 'retest-source-manifest.json').read_text()) == expected
    for rel, digest in expected.items():
        assert sha(source / rel) == digest, ('source drift', rel)
    for category in ['frozen_source_sha256', 'integration_source_sha256']:
        for rel, digest in ref[category].items():
            assert sha(model / rel) == digest, ('model drift', rel)
    for path, digest in template['input_file_sha256'].items():
        assert sha(path) == digest, ('input drift', path)
    return dict(candidate_manifest=sha(source / 'retest-source-manifest.json'),
                verified_source_files=len(expected), model_verified=True, inputs_verified=True)

before = verify()
assert Path(arm['cache_seed']).is_dir(), ('missing cache seed before preparation', arm['cache_seed'])
oldout = template['output_dir']
cmd = [s.replace(oldout, str(out)) for s in template['command']]
assert cmd[cmd.index('--stop-after-global-step') + 1] == '3'
cmd[cmd.index('--stop-after-global-step') + 1] = '38'
assert '--profile-hotspots' in cmd
cmd.remove('--profile-hotspots')
assert cmd[cmd.index('--checkpoint-every-steps') + 1] == '3'
assert cmd[cmd.index('--warmup-updates') + 1] == '6'
assert not any('trace_training_release.py' in s for s in cmd)
env = dict(template['environment'])
env.update(c['visibility'])
env['GRAPHCAST_FLAGGEMS_SOURCE_ROOT'] = str(source)
env['PYTHONPATH'] = str(source / 'src') + ':' + str(model / 'src') + ':' + c['dependency_pythonpath']
for key, subdir in [('FLAGGEMS_CACHE_DIR', 'flaggems'), ('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg')]:
    env[key] = str(out / 'cache' / subdir)
env.update(PET_LOCAL_ADDR='127.0.0.1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
if c['key'] == 's5000':
    env.pop('DISABLE_LLVM_OPT', None)
plan = dict(model=str(model), new_source=str(source), backend='flaggems', command=cmd,
    environment=env, output_dir=str(out), expected_ln_sha=c['ln_sha'], ln_commit=c['ln_commit'],
    matrix_operator=c['operator'], expected_matrix_sha=arm['operator_sha256'],
    source_verification=before, input_file_sha256=template['input_file_sha256'],
    requested_updates=38, warmup_steps=[], learning_rate_warmup_included=list(range(1,7)), timed_steps=list(range(1,39)), profile_enabled=False,
    observer='none', formal_mfu_eligible=False, cache_seed_copy=arm['cache_seed'],
    physical_gpu=c['physical_gpu'], expected_pci_bus=c['expected_pci_bus'],
    coverage_extension_sha256='d9b97aedbcbdfa836edfdd0d4c46c4ccbf4d31e0bf4f086ad338e46f09f097b8',
    expected_clip_source_sha256='56e8e0d4edc7c0532262d069a48557523ba69176fdab70afe1aeae87baa7cf8d',
    comparison_scope='complete38 independent course, all update E2E included; pending source/numerical/MFU review')
if a.prepare_only:
    (root / (a.arm + '-prepared-plan.json')).write_text(json.dumps(plan, indent=2))
    print('PREPARED', a.arm, json.dumps(before), flush=True)
    raise SystemExit(0)
assert Path(arm['cache_seed']).is_dir()
out.mkdir(exist_ok=False)
shutil.copytree(arm['cache_seed'], out / 'cache')
(out / 'launch.json').write_text(json.dumps(plan, indent=2))
from reproduce import training_command
prefix, worker = training_command(plan, root)
started = time.time()
with (out / 'training.log').open('x') as stream:
    z = subprocess.run(['timeout', '--foreground', '--signal=TERM', '--kill-after=30s', '7200s', *prefix, *worker],
                       cwd=model, env=env, stdout=stream, stderr=subprocess.STDOUT)
record = dict(returncode=z.returncode, before=before, after=verify(), started_unix=started,
              ended_unix=time.time(), formal_mfu_eligible=False, actual_command=prefix + worker)
(out / 'exit.json').write_text(json.dumps(record, indent=2))
if z.returncode == 0:
    summary = json.loads((out / 'run/summary.json').read_text())
    metrics = summary['metrics']
    assert summary['completed_optimizer_steps'] == 38
    assert [m['step'] for m in metrics] == list(range(1,39))
    assert all(m['grad_ok'] for m in metrics)
    expected_r=[1]*16+[v for r in range(2,13) for v in [r,r]]
    assert [m['rollout_steps'] for m in metrics]==expected_r
    assert abs(sum(m['training_flops'] for m in metrics)/1e12-29434.23636529152)<1e-8
    assert not list(out.rglob('trace.json'))
    (out / 'timing-window.json').write_text(json.dumps({
        'warmup_steps': [], 'learning_rate_warmup_included': list(range(1,7)), 'timed_steps': list(range(1,39)), 'all_metrics': metrics,
        'timed_metrics': metrics, 'profile_enabled': False, 'observer': 'none',
        'full_course_mfu': None}, indent=2))
print('EXIT', a.arm, z.returncode, flush=True)
raise SystemExit(z.returncode)
