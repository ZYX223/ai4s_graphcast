"""Read optional host samples; unavailable monitoring is never zero utilization."""
import json
import math
from pathlib import Path
import time


def read_host_sample(stage, card, now_ns=None, platform='bw1100'):
    """Return legacy JSON-string values and explicit sample provenance/status."""
    metadata = dict(source='host', path='../host-memory.jsonl', status='unavailable')
    try:
        path = Path(stage) / 'host-memory.jsonl'
        with path.open('rb') as stream:
            stream.seek(0, 2)
            start = max(0, stream.tell() - 65536)
            stream.seek(start)
            # Ignore an unfinished append and a prefix cut by the bounded read.
            lines = stream.read().split(b'\n')[:-1]
            if start:
                lines = lines[1:]
        complete = [line for line in lines if line.strip()]
        if not complete:
            raise ValueError('No complete host sample yet')
        sample = json.loads(complete[-1])
        if sample.get('error'):
            raise ValueError('Host monitor failed: ' + str(sample['error']))
        stamp = sample['time_ns']
        if type(stamp) is not int or stamp <= 0:
            raise ValueError('Invalid host sample timestamp')
        observed = int(time.time() * 1e9) if now_ns is None else now_ns
        age = (observed - stamp) / 1e9
        metadata.update(time_ns=stamp, age_seconds=age)
        if not 0 <= age <= 120:
            raise ValueError('Host sample is stale or from the future')
        values = sample['devices'][str(card)]
        usage = float(values['utilization'] if platform == 'h100' else values['HCU use (%)'])
        memory = float(values['memory_mib'] if platform == 'h100' else values['vram Total Used Memory (MiB)'])
        if not (math.isfinite(usage) and 0 <= usage <= 100 and
                math.isfinite(memory) and memory >= 0):
            raise ValueError('Invalid host utilization or memory sample')
        metadata['status'] = 'available'
        if sample.get('warning'):
            metadata['warning'] = sample['warning']
        key = str(card) if platform == 'h100' else 'card' + str(card)
        return json.dumps({key: values}), metadata
    except (OSError, ValueError, OverflowError, KeyError, TypeError, AttributeError) as exc:
        metadata['reason'] = str(exc)
        return json.dumps(dict(status='unavailable', source='host')), metadata
