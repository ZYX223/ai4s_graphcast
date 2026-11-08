"""Stage real pre-resume metrics in a NEW isolated run; never repair failed runs.

The caller must supply a fresh CPU/hash checkpoint audit and the intended
resume context. This stages history only, not checkpoint loading or acceptance.
"""
import hashlib
import json
from pathlib import Path

CONTEXT_KEYS = ('schema', 'config_sha256', 'corpus_index_sha256', 'data_sha256',
                'stats_sha256', 'resume_contract', 'execution_policy',
                'normalization', 'per_variable_weights', 'resolved_precision')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stage_prefix(parent_run, target_run, checkpoint_audit, expected_context):
    parent, target = Path(parent_run), Path(target_run)
    if target.exists():
        raise FileExistsError('target must be a new isolated run')
    audit = checkpoint_audit
    digest = audit['checkpoint_sha256']
    if (not audit.get('hash_matches') or len(digest) != 64
            or audit.get('recorded_sha256') != digest
            or not audit.get('all_model_finite') or not audit.get('optimizer_all_finite')):
        raise ValueError('checkpoint audit is not valid')
    step = audit['cursor']['global_optimizer_step']
    if not isinstance(step, int) or step < 1 or expected_context['start_optimizer_step'] != step:
        raise ValueError('checkpoint/resume step mismatch')
    schedule = audit['schedule_state']
    if (schedule['next_update_index'] != step
            or schedule['digest'] != expected_context['resume_contract']['schedule_digest']):
        raise ValueError('checkpoint schedule mismatch')

    rows, files, evidence = [], {}, []
    for path in sorted((parent / 'metrics').glob('segment_*.jsonl')):
        original = path.read_bytes()
        values = [json.loads(line) for line in original.decode().splitlines() if line.strip()]
        selected = [(line, value) for line, value in zip(
            [x for x in original.decode().splitlines() if x.strip()], values)
            if 1 <= int(value['optimizer_step']) <= step]
        if not selected:
            continue
        start_path = path.with_suffix('.start.json')
        start_bytes = start_path.read_bytes()
        start = json.loads(start_bytes)
        for key in CONTEXT_KEYS:
            if key not in start or key not in expected_context or start[key] != expected_context[key]:
                raise ValueError('parent context mismatch: ' + key)
        old_optimizer = {k: v for k, v in start['optimizer'].items() if k != 'lr'}
        new_optimizer = {k: v for k, v in expected_context['optimizer'].items() if k != 'lr'}
        if old_optimizer != new_optimizer:
            raise ValueError('parent optimizer recipe mismatch')
        if int(start['start_optimizer_step']) + 1 != int(selected[0][1]['optimizer_step']):
            raise ValueError('segment start does not match metric prefix')
        content = ('\n'.join(line for line, _ in selected) + '\n').encode()
        files[path.name] = content
        files[start_path.name] = start_bytes
        rows.extend(value for _, value in selected)
        evidence.append({'source': str(path.resolve()), 'source_sha256': sha(original),
                         'source_start_sha256': sha(start_bytes), 'staged_name': path.name,
                         'staged_sha256': sha(content),
                         'selected_steps': [int(v['optimizer_step']) for _, v in selected]})
    rows.sort(key=lambda row: int(row['optimizer_step']))
    if [int(row['optimizer_step']) for row in rows] != list(range(1, step + 1)):
        raise ValueError('prefix metrics have missing or duplicate steps')
    last = rows[-1]
    if last['epoch_plan_digest'] != audit['cursor']['epoch_plan_digest']:
        raise ValueError('checkpoint epoch_plan mismatch')
    if (last['epoch'] != audit['cursor']['epoch']
            or last['epoch_batch_index'] + 1 != audit['cursor']['next_batch_index']):
        raise ValueError('checkpoint cursor mismatch')
    if last['learning_rate'] != schedule['last_applied_learning_rate']:
        raise ValueError('checkpoint applied learning rate mismatch')

    # No destination write occurs until the full prefix and lineage validate.
    target.mkdir(parents=True, exist_ok=False)
    (target / 'metrics').mkdir()
    for name, content in files.items():
        (target / 'metrics' / name).write_bytes(content)
    receipt = {'status': 'metrics_prefix_staged', 'training_updates_replayed': 0,
               'parent_run': str(parent.resolve()), 'checkpoint': audit['checkpoint'],
               'checkpoint_sha256': digest, 'resume_step': step, 'sources': evidence,
               'expected_resume_context_sha256': sha(json.dumps(expected_context, sort_keys=True).encode()),
               'lifecycle_success': False, 'formal_acceptance': False}
    (target / 'metrics-prefix-receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt
