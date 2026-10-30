"""Extend only the frozen profiler capture to three full 40-step rollouts."""
import ctypes
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

platform = os.environ['GRAPHCAST_PLATFORM']
source_path = Path(os.environ['GRAPHCAST_FROZEN_BENCHMARK'])
original = source_path.read_text()
source = original
def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)

replace_once('"measured_steps": 1 if args.profile or args.logical_catalog else 40,',
             '"measured_steps": 1 if args.logical_catalog else 40,')
replace_once('"repeats": 1 if args.profile or args.logical_catalog else args.repeats,',
             '"repeats": 1 if args.logical_catalog else args.repeats,')
replace_once('rows, warmups, audit = [], [], []', 'rows, warmups, audit = [], [], []\n    capture_ranges = []')
replace_once('assert start_profile() == 0',
             'capture_wall_start_ns = time.time_ns()\n                    capture_mono_start_ns = time.monotonic_ns()\n                    assert start_profile() == 0')
replace_once('assert stop_profile() == 0',
             '''assert stop_profile() == 0
                        capture_ranges.append(dict(label=name, wall_start_ns=capture_wall_start_ns,
                            wall_end_ns=time.time_ns(), monotonic_start_ns=capture_mono_start_ns,
                            monotonic_end_ns=time.monotonic_ns(), measured_steps=40))
                        write_json(args.output / "capture-ranges.json", capture_ranges)''')
if platform == 'h100':
    # Save final predictions outside the profiler range, as the standard benchmark does.
    replace_once('if not (args.profile or args.logical_catalog):', 'if not args.logical_catalog:')
else:
    libc=ctypes.CDLL(None,use_errno=True)
    assert libc.prctl(41,1,0,0,0)==0 and libc.prctl(42,0,0,0,0)==1

evidence = Path(os.environ['GRAPHCAST_PROFILE_EVIDENCE'])
evidence.mkdir(exist_ok=True)
(evidence/'executed-benchmark.py').write_text(source)
(evidence/'instrumentation.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=str(source_path),tofile='profile40-instrumented')))
manifest = dict(platform=platform,source=str(source_path),source_sha256=hashlib.sha256(original.encode()).hexdigest(),
    executed_sha256=hashlib.sha256(source.encode()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    changes='Capture 40 steps in each of 3 rounds; save capture bounds and H100 final outputs outside timing. Model, precision, inputs and fusion policy unchanged.',
    baseline_modified=False)
(evidence/'instrumentation.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert '--profile' in sys.argv
ns={'__name__':'__main__','__package__':'graphcast_bench' if platform=='h100' else None,'__file__':str(source_path)}
exec(compile(source,str(evidence/'executed-benchmark.py'),'exec'),ns)
