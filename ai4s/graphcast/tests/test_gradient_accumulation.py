from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat.gradient_accumulation import MicrobatchPlan


def test_unequal_microbatches_cover_batch_with_sample_weights():
    plan = MicrobatchPlan.create(global_batch_size=5, micro_batch_size=2)

    assert plan.accumulation_steps == 3
    assert [(part.batch_slice.start, part.batch_slice.stop) for part in plan.parts] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [part.sample_count for part in plan.parts] == [2, 2, 1]
    assert [part.sample_weight for part in plan.parts] == pytest.approx([0.4, 0.4, 0.2])


def test_unequal_accumulated_update_matches_full_batch_mean():
    features = torch.tensor(
        [[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0.25], [-1.5], [2.0]], dtype=torch.float64)
    full_batch = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)
    accumulated = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        full_batch.weight.copy_(torch.tensor([[0.4, -0.3]], dtype=torch.float64))
        full_batch.bias.copy_(torch.tensor([0.2], dtype=torch.float64))
        accumulated.load_state_dict(full_batch.state_dict())

    full_optimizer = torch.optim.SGD(full_batch.parameters(), lr=0.05)
    full_optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(full_batch(features), targets).backward()
    full_optimizer.step()

    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    accumulated_optimizer.zero_grad(set_to_none=True)
    for part in MicrobatchPlan.create(global_batch_size=3, micro_batch_size=2).parts:
        loss = torch.nn.functional.mse_loss(
            accumulated(features[part.batch_slice]),
            targets[part.batch_slice],
        )
        (loss * part.sample_weight).backward()
    accumulated_optimizer.step()

    for expected, actual in zip(full_batch.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("global_size,micro_size", [(0, 1), (2, 0), (-1, 1)])
def test_nonpositive_batch_sizes_are_rejected(global_size, micro_size):
    with pytest.raises(ValueError, match="positive"):
        MicrobatchPlan.create(global_batch_size=global_size, micro_batch_size=micro_size)
