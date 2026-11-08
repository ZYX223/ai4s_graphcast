"""CPU-only contract tests; no Torch import, device call, or real-model claim."""
import ast
import argparse
import builtins
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
class Array:
    def __init__(self,x): self.x=x
    def reshape(self,r,c): self.r=r; self.c=len(self.x)//r; return self
    def __getitem__(self,k): return type('Column',(),{'tolist':lambda s:self.x[k[1]::self.c]})()
np=NS(asarray=lambda x,dtype:Array(x),float64=float)


ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parents[2]
runner = MODEL / 'scripts/training/run_official_epoch_training.py'
spec = importlib.util.spec_from_file_location('cp38_entry', ROOT.parent / 'harness/evaluation/evaluate_remaining_cp38.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
tree = ast.parse(runner.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_validation')
compiled = compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])), str(runner), 'exec')

grad = {'enabled': True, 'forwards': 0}


class NoGrad:
    def __enter__(self):
        grad['enabled'] = False

    def __exit__(self, *exc):
        grad['enabled'] = True


class Loss:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def item(self):
        return self.value


class Dataset:
    path = '/cpu-fixture-only'
    window_indices = list(range(1, 112))
    window_count = 111
    closed = False

    def sample_indices(self, *, window_indices, sample_weights, rank, replicas):
        assert rank == 0 and replicas == 1 and sample_weights == [1.0]
        return NS(inputs=window_indices[0], targets=None, forcings=None,
                  sample_weights=sample_weights, valid_sample_count=1,
                  global_valid_sample_count=1, global_padded_sample_count=0,
                  window_indices=window_indices)

    def close(self):
        self.closed = True


class Plan:
    @staticmethod
    def create(indices, *, epoch, seed, global_batch_size, shuffle):
        assert indices == list(range(1, 112)) and global_batch_size == 1 and shuffle is False
        return NS(optimizer_steps=111, batches=[NS(window_indices=[i], sample_weights=[1.0]) for i in indices])


class Model:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def forward(batch, *, collect_variable_diagnostics):
    assert not grad['enabled'] and not model.training and collect_variable_diagnostics
    grad['forwards'] += 1
    value = batch.index / 10.0
    return Loss(value), [value] * 12, [value, value * 2] * 12


def zip_compatible(*iterables, strict=False):
    # Python3.9 host-only shim for the original 3.12 runner's zip(strict=True).
    if strict:
        assert len({len(x) for x in iterables}) == 1
    return builtins.zip(*iterables)


dataset = Dataset()
model = Model()
context = NS(rank=0, dp_replicas=1, device_indices=(0,), is_primary=True,
             global_max_seconds=lambda x: x, global_sum=lambda x: x,
             gather_objects=lambda x: [x])
namespace = dict(torch=NS(no_grad=NoGrad), np=np, math=math,
                 time=NS(perf_counter=lambda: 0.0), zip=zip_compatible,
                 _validation_dataset=lambda args: dataset,
                 validate_formal_dataset=lambda data, formal: None,
                 EpochPlan=Plan, PER_VARIABLE_WEIGHTS={},
                 synchronize=lambda devices: None,
                 prepare_autoregressive_batch=lambda inputs, *pos, **kw: NS(index=inputs, target_channel_slices={'a': None, 'b': None}),
                 _weighted_distributed_tensor=lambda x, **kw: x,
                 sample_provenance=lambda sample: dict(local_window_indices=sample.window_indices, sample_weights=sample.sample_weights))
exec(compiled, namespace)
