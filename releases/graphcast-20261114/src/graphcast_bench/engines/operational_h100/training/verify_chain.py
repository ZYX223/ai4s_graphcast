"""Validate an independent 38-update chain without comparing checkpoint outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit_reports(reports,reference_initial):
    if [r['rollout'] for r in reports]!=list(range(1,13)):
        raise ValueError('Require all twelve horizons in order')
    for key in ['initial_params_sha256','initial_optimizer_sha256']:
        if reports[0][key]!=reference_initial[key]:
            raise ValueError('First stage is not anchored to frozen reference: '+key)
    updates=[]; previous=None
    for report in reports:
        horizon=report['rollout']
        expected=list(range(1,17)) if horizon==1 else [13+2*horizon,14+2*horizon]
        if report['status']!='completed' or [r['update'] for r in report['rows']]!=expected:
            raise ValueError('Missing, duplicated or incomplete updates')
        reload=report['checkpoint_reload']
        if reload['status']!='passed':
            raise ValueError('Checkpoint reload validation failed')
        if reload['optimizer_counters']!=[expected[-1]]:
            raise ValueError('Optimizer counter does not match curriculum')
        if previous is not None:
            # Initial/reload use tree_digest (paths, shapes and bytes); final_params
            # uses flattened bytes only. Compare the same tree hash convention.
            if report['initial_params_sha256']!=previous['checkpoint_reload']['params_sha256']:
                raise ValueError('Parameter trajectory is not continuous')
            if report['initial_optimizer_sha256']!=previous['checkpoint_reload']['optimizer_sha256']:
                raise ValueError('Optimizer trajectory is not continuous')
        for row in report['rows']:
            if row['rollout']!=horizon or not math.isfinite(row['loss']):
                raise ValueError('Invalid rollout or loss')
            change=row['parameter_update']['max_abs']
            if not math.isfinite(change) or change<0 or (row['lr']==0)!=(change==0):
                raise ValueError('Actual parameter change disagrees with zero/nonzero learning rate')
            updates.append(row)
        previous=report
    zero=sum(r['lr']==0 for r in updates)
    zero_positions=[r['update'] for r in updates if r['lr']==0]
    changed=sum(r['parameter_update']['max_abs']>0 for r in updates)
    if len(updates)!=38 or zero_positions!=[1,16] or changed!=36:
        raise ValueError('Expected 38 updates, two zero-LR and 36 actual parameter changes')
    return dict(status='passed',updates=38,zero_lr_updates=zero,zero_lr_update_positions=zero_positions,actual_parameter_changes=changed,
        independent_parameter_and_optimizer_chain=True,
        initial_params_sha256=reports[0]['initial_params_sha256'],
        final_params_tree_sha256=reports[-1]['checkpoint_reload']['params_sha256'],
        final_params_flat_sha256=reports[-1]['final_params_sha256'])


def validate_launch(launch,horizon,candidate,run_tag):
    if launch.get('full_chain') is not True or launch.get('candidate')!=candidate:
        raise ValueError('Launch is not the expected complete-chain candidate')
    command=launch['command']
    if '--limit' in command:raise ValueError('Full chain must not limit updates')
    if command.count('--rollout')!=1 or command[command.index('--rollout')+1]!=str(horizon):
        raise ValueError('Wrong launched horizon')
    if horizon==1:
        if '--resume' in command:raise ValueError('First horizon must use frozen pretrained state')
    elif (command.count('--resume')!=1 or
          command[command.index('--resume')+1]!='/run-chain/r%02d/checkpoint.pkl'%(horizon-1)):
        raise ValueError('Resume does not point to own preceding horizon')
    mounts=[command[i+1] for i,x in enumerate(command[:-1]) if x=='-v' and command[i+1].endswith(':/run-chain:ro')]
    if len(mounts)!=1:raise ValueError('Expected one readonly chain mount')
    source=mounts[0][:-len(':/run-chain:ro')]
    if Path(source).name!=run_tag:raise ValueError('Chain mount has a different run tag')
    return source


def runtime_signature(manifest, launch, horizon, previous=None):
    source = {k:v for k,v in manifest['source_sha256'].items()
              if k.startswith('/original/') or k.startswith('/candidate/')}
    for required in ['/original/experiment/train.py', '/candidate/entry.py',
                     '/candidate/regular3_hooks.py', '/candidate/selective_remat.py',
                     '/candidate/run-h100.sh']:
        if not source.get(required):
            raise ValueError('Missing runtime source digest: ' + required)
    environment = dict(manifest['environment'])
    if environment.get('STREAM_MICROBATCHES') == '1':
        for required in ['/candidate/streamed_train.py','/candidate/streamed_microbatches.py']:
            if not source.get(required):raise ValueError('Missing streamed trainer source: '+required)
        if launch.get('stream_microbatches') is not True:
            raise ValueError('Streaming environment does not match launch')
    schedule = launch['memory_fraction_schedule']
    if set(schedule) != set(map(str, range(1,13))):
        raise ValueError('Incomplete memory fraction schedule')
    fraction = environment.pop('XLA_PYTHON_CLIENT_MEM_FRACTION')
    if fraction != schedule[str(horizon)]:
        raise ValueError('Runtime memory fraction differs from fixed horizon schedule')
    command = launch['command']
    launch_env = {}
    for i, token in enumerate(command[:-1]):
        if token == '-e':
            key, value = command[i+1].split('=',1)
            if key in launch_env:raise ValueError('Duplicate launch environment key')
            launch_env[key] = value
    for key in ['NVIDIA_VISIBLE_DEVICES', 'XLA_PYTHON_CLIENT_MEM_FRACTION'] + (['STREAM_MICROBATCHES'] if 'STREAM_MICROBATCHES' in environment else []):
        if launch_env.get(key) != manifest['environment'].get(key):
            raise ValueError('Runtime environment differs from launch: ' + key)
    signature = dict(source_sha256=source, environment=environment,
                     memory_fraction_schedule=schedule, candidate=manifest['candidate'],
                     candidate_xla_flags=manifest['candidate_xla_flags'])
    if previous is not None and signature != previous:
        raise ValueError('Runtime source or environment changed within chain')
    return signature


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True,help='Frozen original H100 report directory')
    parser.add_argument('--expected-candidate',default='regular3')
    parser.add_argument('--require-checkpoints',action='store_true')
    args=parser.parse_args(); reports=[]; hashes={}; comparisons=[]; chain_source=None; runtime_hashes=None
    from compare import compare_reports
    for h in range(1,13):
        folder=args.run_dir/('r%02d'%h)
        if (folder/'exit-code.txt').read_text().strip()!='0':raise ValueError('Failed process')
        launch=json.loads((folder/'launch.json').read_text())
        source=validate_launch(launch,h,args.expected_candidate,args.run_dir.name)
        if chain_source is not None and source!=chain_source:raise ValueError('Chain mount changed between horizons')
        chain_source=source
        manifest=json.loads((folder/'candidate-manifest.json').read_text())
        if manifest['candidate']!=args.expected_candidate:raise ValueError('Runtime candidate differs from launch')
        runtime_hashes=runtime_signature(manifest,launch,h,previous=runtime_hashes)
        report=json.loads((folder/'report.json').read_text())
        reference=json.loads((args.reference/('h100-r%02d'%h)/'report.json').read_text())
        comparison=compare_reports(report,reference,same_initial_state=h==1)
        comparisons.append(comparison)
        if args.require_checkpoints:
            digest=hashlib.sha256((folder/'checkpoint.pkl').read_bytes()).hexdigest()
            if digest!=report['checkpoint_sha256']:raise ValueError('Checkpoint file hash mismatch')
            hashes[str(h)]=digest
        reports.append(report)
    result=audit_reports(reports,json.loads((args.reference/'h100-r01/report.json').read_text()))
    result['checkpoint_file_hashes']=hashes
    from graphcast_bench.validation.loss_metrics import aggregate_comparisons
    result['loss_validation']=aggregate_comparisons(comparisons,expected_updates=range(1,39))
    result['accuracy_status']='passed' if result['loss_validation']['metric_passed'] else 'failed'
    result['first_stage_frozen_reference_initial_state_verified']=True
    result['corresponding_update_loss_recomputed_from_reports']=True
    result['full_chain_launch_and_resume_paths_verified']=True
    result['chain_mount_source']=chain_source
    result['runtime_training_sha256']=runtime_hashes['source_sha256']
    result['runtime_environment_and_schedule']= {k:v for k,v in runtime_hashes.items() if k!='source_sha256'}
    result['reference_report_sha256']={str(h):hashlib.sha256((args.reference/('h100-r%02d'%h)/'report.json').read_bytes()).hexdigest() for h in range(1,13)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['accuracy_status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
