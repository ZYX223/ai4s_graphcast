"""Join HIPPROF JSON shards without changing timestamps or original evidence."""
import hashlib
import json
import re
from pathlib import Path

def normalize(run):
    run = Path(run)
    target = run / 'trace.json'
    if target.exists():
        return
    shards = sorted((p for p in run.glob('trace_*.json')
                     if re.fullmatch(r'trace_[0-9]+\.json', p.name)),
                    key=lambda p: int(p.stem.split('_')[-1]))
    numbers = [int(p.stem.split('_')[-1]) for p in shards]
    if not shards or numbers != list(range(1, len(shards)+1)):
        raise ValueError('Missing or non-contiguous HIPPROF trace shards: %s' % numbers)
    events = []
    sources = []
    device_keys = set()
    for path in shards:
        content = path.read_bytes()
        trace = json.loads(content)
        part = trace['traceEvents'] if isinstance(trace, dict) else trace
        sources.append(dict(file=path.name, sha256=hashlib.sha256(content).hexdigest(), events=len(part)))
        for event in part:
            if event.get('cat') == 'HIPOPS' and event.get('ph') == 'X':
                args = event['args']
                key = (event.get('pid'), event.get('tid'), event['name'],
                       int(args['BeginNs']), int(args['EndNs']))
                if key in device_keys:
                    raise ValueError('Duplicate GPU event across trace shards: %r' % (key,))
                device_keys.add(key)
        events.extend(part)
    temporary = run / 'trace-combined.tmp'
    with temporary.open('w') as f:
        json.dump(dict(traceEvents=events), f, separators=(',', ':'))
    temporary.replace(target)
    (run/'trace-shards.json').write_text(json.dumps(dict(
        sources=sources, events=len(events), device_events=len(device_keys),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        method='Concatenate numbered HIPPROF trace shards, preserving absolute BeginNs/EndNs; reject duplicate GPU events. Original shards retained.'), indent=2)+'\n')

if __name__ == '__main__':
    import sys
    normalize(Path(sys.argv[1]))
