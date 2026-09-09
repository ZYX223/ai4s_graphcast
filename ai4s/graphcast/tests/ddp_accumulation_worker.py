"""Distributed plumbing test, not a reduced GraphCast model acceptance run.

Run with torchrun --standalone --nproc-per-node=2. Compare DDP gradients and
AdamW updates with a serial global batch across changing checkpointed horizons.
"""
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint

path = Path(__file__).resolve().parents[1] / "src/graphcast_compat/distributed_training.py"
spec = importlib.util.spec_from_file_location("distributed_plumbing", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Recurrence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 3)
        self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, horizon):
        losses = []
        for _ in range(horizon):
            x = checkpoint(lambda v: self.layer(v).tanh(), x, use_reentrant=False)
            losses.append(x.square().mean())
        return torch.stack(losses).mean()


context = module.DistributedContext.from_environment(
    dp_replicas=2, mp_devices_per_replica=1,
    available_device_count=2, backend="gloo",
)
try:
    torch.manual_seed(27)
    model = Recurrence()
    reference = copy.deepcopy(model)
    ddp = context.wrap_ddp(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    serial_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    inputs = torch.randn(4, 3)
    for horizon in (1, 2, 12):
        optimizer.zero_grad(set_to_none=True)
        serial_optimizer.zero_grad(set_to_none=True)
        reference(inputs, horizon).backward()
        for micro in range(2):
            with ddp.no_sync() if micro == 0 else nullcontext():
                index = micro * 2 + context.rank
                (ddp(inputs[index:index + 1], horizon) / 2).backward()
        for p, q in zip(model.parameters(), reference.parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-7)
        optimizer.step()
        serial_optimizer.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)
    if context.is_primary:
        print("PASS: DDP accumulation, unused parameters, checkpointed horizons 1/2/12, gradients and AdamW match serial global batch")
finally:
    context.close()
