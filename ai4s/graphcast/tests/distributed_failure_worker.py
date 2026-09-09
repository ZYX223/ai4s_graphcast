"""CPU Gloo collective failure fixture; no GraphCast model or training data.

Run on two CPU ranks with torchrun. Rank 1 fails while rank 0 waits
in an all-reduce. The launcher must report the injected failure and exit
without waiting for a distributed timeout. An outer test sets a time limit.
"""
import importlib.util
from pathlib import Path
import sys

import torch
import torch.distributed as dist

path = Path(__file__).resolve().parents[1] / 'src/graphcast_compat/distributed_training.py'
spec = importlib.util.spec_from_file_location('distributed_failure', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
context = module.DistributedContext.from_environment(
    dp_replicas=2,
    mp_devices_per_replica=1,
    available_device_count=2,
    backend="gloo",
)
try:
    value = torch.ones(1)
    context.barrier()
    if context.rank == 1:
        raise RuntimeError('injected worker failure before collective')
    dist.all_reduce(value)
finally:
    context.close(synchronize=sys.exc_info()[0] is None)
