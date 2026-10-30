"""Replay all 15 Linear shapes with the repaired router; keep old evidence frozen."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

old = Path('/tmp/zyx-flagos-logical-operators-20261029')
out = Path(os.environ['GRAPHCAST_REPLAY_OUTPUT'])
library = Path(os.environ['LD_PRELOAD'])
assert hashlib.sha256(library.read_bytes()).hexdigest() == '27a3966aba19349480bdd30c2f820a230c12494f1a4634b7c83311b352834f5e'
hip = C.CDLL('libamdhip64.so')
bus = C.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
assert bus.value.decode().lower() == '0000:88:00.0', bus.value
source = (old / 'bench.py').read_text()
def replace_once(a, b):
    global source
    assert source.count(a) == 1, a
    source = source.replace(a, b)
replace_once("OUT = ROOT / ('bench-' + MODE + os.environ.get('BENCH_SUFFIX',''))", "OUT = Path(os.environ['GRAPHCAST_REPLAY_OUTPUT'])")
replace_once('assert len(cases) == 108', "assert len(cases) == 108\ncases = [c for c in cases if c['operator'] == 'linear']\nassert len(cases) == 15")
replace_once('expected_cases=108', 'expected_cases=15')
replace_once('successful Linear rows require one RNE+XF32 audited rocBLAS SGEMM per invocation.', 'successful Linear rows require one audited rocBLAS call with RNE and XF32 mode requested; K=4 may use native FP32 fallback.')
replace_once('expected [1 SGEMM,1 RNE,0 unknown]', 'expected [1 GEMM,1 RNE,0 unknown]')
(out / 'replay-source.py').write_text(source)
manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), physical_gpu=9, pci=bus.value.decode(),
    source=str(old / 'bench.py'), source_sha256=hashlib.sha256((old / 'bench.py').read_bytes()).hexdigest(),
    executed_sha256=hashlib.sha256(source.encode()).hexdigest(), library=str(library),
    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), expected_cases=15,
    baseline_modified=False, scope='All Linear shapes, repaired GEMM_EX candidate, original standalone timing method')
(out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
namespace = {'__name__': '__main__', '__file__': str(old / 'bench.py')}
exec(compile(source, str(out / 'replay-source.py'), 'exec'), namespace)
manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), status=namespace['report']['status'])
(out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
if namespace['report']['status'] != 'passed':
    raise SystemExit(1)
