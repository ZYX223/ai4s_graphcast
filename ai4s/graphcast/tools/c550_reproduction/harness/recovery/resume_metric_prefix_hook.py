"""Attach validated real history before the resumed segment starts writing."""
import hashlib
import json
from pathlib import Path

from resume_metric_prefix import stage_prefix


def stage_into_existing_run(parent_run, run_dir, audit, actual_context):
    run = Path(run_dir)
    metrics = run / 'metrics'
    if not run.is_dir():
        raise FileNotFoundError('training must create its own run directory')
    if metrics.exists() and any(metrics.iterdir()):
        raise FileExistsError('metrics must be empty before history inheritance')
    staging = run.parent / 'verified-metric-prefix'
    receipt = stage_prefix(parent_run, staging, audit, actual_context)
    metrics.mkdir(exist_ok=True)
    files = sorted((staging / 'metrics').iterdir())
    if any((metrics / path.name).exists() for path in files):
        raise FileExistsError('history destination changed before publication')
    published = []
    for path in files:
        content = path.read_bytes()
        with (metrics / path.name).open('xb') as stream:
            stream.write(content)
        published.append({'name': path.name, 'sha256': hashlib.sha256(content).hexdigest()})
    receipt.update(actual_context_checked=True, published=published,
                   integration='before resumed segment_start_record is written')
    with (run / 'metrics-prefix-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt
