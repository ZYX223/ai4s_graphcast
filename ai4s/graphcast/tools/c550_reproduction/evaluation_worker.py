"""Bind the frozen evaluator only after source and actual device checks."""
import os
from pathlib import Path
import sys

from reproduce import dump, read
from runtime_gate import check_runtime
import evaluate_remaining_cp38 as entry

job = Path(__file__).resolve().parent
spec = read(job / 'spec.json')
entry.preflight(spec)
out = Path(spec['output_dir'])
out.mkdir(exist_ok=False)
os.environ.update(spec['environment'])
check_runtime(spec)
dump(out / 'gpu-launch-gate.json', read(job / 'resource-gate.json'))
spec.update(launch_ready=True, prepared_only=False)
bound = job / 'launch-bound-spec.json'
dump(bound, spec)
sys.argv = [str(job / 'evaluate_remaining_cp38.py'), '--spec', str(bound)]
entry.main()
