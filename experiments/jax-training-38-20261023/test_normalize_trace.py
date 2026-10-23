import json
import tempfile
import unittest
from pathlib import Path
from normalize_trace import normalize

class Shards(unittest.TestCase):
    def test_two_shards_preserve_device_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = [dict(cat='HIPOPS', ph='X', name='round_rne', pid=1, tid=2,
                           args=dict(BeginNs=i, EndNs=i+2)) for i in [10, 20]]
            for i, event in enumerate(events, 1):
                (root / ('trace_%d.json' % i)).write_text(json.dumps({'traceEvents':[event]}))
            normalize(root)
            self.assertEqual(json.loads((root/'trace.json').read_text())['traceEvents'], events)
            self.assertEqual(json.loads((root/'trace-shards.json').read_text())['events'], 2)
            normalize(root)  # Existing canonical file is preserved.
            self.assertTrue((root/'trace_1.json').exists())

    def test_missing_or_duplicate_device_events_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = dict(cat='HIPOPS', ph='X', name='gemm', pid=1, tid=2,
                         args=dict(BeginNs=10, EndNs=20))
            for i in [1, 3]:
                (root/('trace_%d.json'%i)).write_text(json.dumps({'traceEvents':[event]}))
            with self.assertRaises(ValueError): normalize(root)
            (root/'trace_3.json').rename(root/'trace_2.json')
            with self.assertRaises(ValueError): normalize(root)

if __name__ == '__main__': unittest.main()
