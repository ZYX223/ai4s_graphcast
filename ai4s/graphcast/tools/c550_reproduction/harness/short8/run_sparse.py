"""Verify and run 8 or 9 representative updates in one process."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('arm',choices=['candidate']);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent;c=json.loads((root/'config.json').read_text());plan=json.loads((root/'plan-template.json').read_text())
ref=json.loads((root/'reference.json').read_text());source=Path(plan['new_source']);model=Path(plan['model']);out=Path(plan['output_dir'])
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
    expected=dict(ref['flaggems_source_sha256'],**c['arms']['candidate']['overlay'])
    assert json.loads((source/'retest-source-manifest.json').read_text())==expected
    for rel,digest in expected.items():assert sha(source/rel)==digest,rel
    for category in ('frozen_source_sha256','integration_source_sha256'):
        for rel,digest in ref[category].items():assert sha(model/rel)==digest,rel
    for path,digest in plan['input_file_sha256'].items():assert sha(path)==digest,path
    cp=Path(plan['checkpoint']);assert cp.is_file()
    assert sha(cp)==plan['checkpoint_sha256']
    return dict(source_files=len(expected),source_manifest_sha256=sha(source/'retest-source-manifest.json'),model_inputs_verified=True,
                initial_checkpoint_sha256=plan['checkpoint_sha256'])
before=verify();assert Path(plan['cache_seed_copy']).is_dir()
if args.prepare_only:
    (root/'prepared-source-receipt.json').write_text(json.dumps(before,indent=2));print('PREPARED',json.dumps(before));raise SystemExit(0)
started=time.time();out.mkdir(exist_ok=False)
cache=Path(plan['cache_seed_copy'])
cache_manifest={str(f.relative_to(cache)):sha(f) for f in cache.rglob('*') if f.is_file()}
shutil.copytree(cache,out/'cache')
assert {str(f.relative_to(out/'cache')):sha(f) for f in (out/'cache').rglob('*') if f.is_file()}==cache_manifest
(out/'cache-seed-manifest.json').write_text(json.dumps(cache_manifest,indent=2))
plan['source_verification']=before;(out/'launch.json').write_text(json.dumps(plan,indent=2))
from reproduce import training_command
prefix, worker = training_command(plan, root)
with (out/'training.log').open('x') as stream:
    run=subprocess.run(['timeout','--foreground','--signal=TERM','--kill-after=30s','1500s',*prefix,*worker],cwd=model,
        env=plan['environment'],stdout=stream,stderr=subprocess.STDOUT)
ended=time.time()
record=dict(returncode=run.returncode,started_unix=started,ended_unix=ended,
            sampling_wall_seconds_including_cache_copy=ended-started,before=before,after=verify(),formal_mfu_eligible=False)
(out/'exit.json').write_text(json.dumps(record,indent=2))
if run.returncode==0:assert (out/'sparse-window.json').is_file() and (out/'final-state.json').is_file()
print('EXIT',run.returncode,flush=True);raise SystemExit(run.returncode)
