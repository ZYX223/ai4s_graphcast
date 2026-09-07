from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from graphcast_compat.distributed_training import (
    DistributedContext,
    loss_scale_for_valid_samples,
)


def _context(monkeypatch, rank: int) -> DistributedContext:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    return DistributedContext.from_environment(
        dp_replicas=2,
        mp_devices_per_replica=4,
        available_device_count=8,
        initialize=False,
    )


def test_dp_replicas_own_disjoint_model_parallel_device_groups(monkeypatch):
    rank_zero = _context(monkeypatch, 0)
    rank_one = _context(monkeypatch, 1)

    assert rank_zero.device_indices == (0, 1, 2, 3)
    assert rank_one.device_indices == (4, 5, 6, 7)
    assert set(rank_zero.device_indices).isdisjoint(rank_one.device_indices)
    assert rank_zero.participating_device_count == 8
    assert rank_zero.local_batch_size(global_batch_size=4) == 2


def test_valid_sample_scale_turns_replica_average_into_global_mean():
    assert loss_scale_for_valid_samples(
        local_valid_count=2,
        global_valid_count=3,
        replicas=2,
    ) == pytest.approx(4 / 3)
    assert loss_scale_for_valid_samples(
        local_valid_count=1,
        global_valid_count=3,
        replicas=2,
    ) == pytest.approx(2 / 3)
    assert loss_scale_for_valid_samples(
        local_valid_count=0,
        global_valid_count=1,
        replicas=2,
    ) == 0.0
    with pytest.raises(ValueError, match="global_valid_count"):
        loss_scale_for_valid_samples(local_valid_count=0, global_valid_count=0, replicas=2)


@pytest.mark.skipif(shutil.which("torchrun") is None, reason="torchrun is unavailable")
def test_two_gloo_ranks_apply_exact_valid_sample_scaling_and_one_manual_sync(tmp_path):
    output = tmp_path / "result.json"
    script = tmp_path / "manual_dp.py"
    script.write_text(
        """
import json
import os
from pathlib import Path

import torch

from graphcast_compat.distributed_training import DistributedContext, loss_scale_for_valid_samples

context = DistributedContext.from_environment(
    dp_replicas=2,
    mp_devices_per_replica=1,
    available_device_count=2,
    backend="gloo",
)
try:
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = (
        torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        if context.rank == 0
        else torch.tensor([[3.0]], dtype=torch.float64)
    )
    local_valid = 2 if context.rank == 0 else 1
    scale = loss_scale_for_valid_samples(
        local_valid_count=local_valid,
        global_valid_count=3,
        replicas=2,
    )
    optimizer.zero_grad(set_to_none=True)
    local_mean = model(values).square().mean()
    (local_mean * scale).backward()
    synchronized = context.synchronize_gradients(model)
    optimizer.step()

    zero_rank_model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        zero_rank_model.weight.fill_(1.0)
    zero_rank_optimizer = torch.optim.SGD(zero_rank_model.parameters(), lr=0.1)
    zero_rank_optimizer.zero_grad(set_to_none=True)
    local_valid = 1 if context.rank == 0 else 0
    scale = loss_scale_for_valid_samples(
        local_valid_count=local_valid,
        global_valid_count=1,
        replicas=2,
    )
    dummy = torch.tensor([[1.0 if context.rank == 0 else 9.0]], dtype=torch.float64)
    (zero_rank_model(dummy).square().mean() * scale).backward()
    zero_rank_synchronized = context.synchronize_gradients(zero_rank_model)
    zero_rank_optimizer.step()

    rows = context.gather_objects({
        "rank": context.rank,
        "weight": float(model.weight.detach()),
        "synchronized": synchronized,
        "zero_rank_weight": float(zero_rank_model.weight.detach()),
        "zero_rank_synchronized": zero_rank_synchronized,
    })
    if context.is_primary:
        Path(os.environ["RESULT_PATH"]).write_text(json.dumps(rows), encoding="utf-8")
finally:
    context.close()
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["RESULT_PATH"] = str(output)

    result = subprocess.run(
        [shutil.which("torchrun") or "torchrun", "--standalone", "--nproc-per-node=2", str(script)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert {row["rank"] for row in rows} == {0, 1}
    assert {row["synchronized"] for row in rows} == {1}
    # Full-batch gradient is 2 * mean([1**2, 2**2, 3**2]) = 28 / 3.
    expected_weight = 1.0 - 0.1 * 28 / 3
    assert all(row["weight"] == pytest.approx(expected_weight) for row in rows)
    assert {row["zero_rank_synchronized"] for row in rows} == {1}
    assert all(row["zero_rank_weight"] == pytest.approx(0.8) for row in rows)
