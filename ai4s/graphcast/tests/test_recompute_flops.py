from __future__ import annotations

import torch

from graphcast_compat.precision import PrecisionLinear
from graphcast_compat.recompute_flops import RecomputeFlopCounter, tracked_checkpoint


def test_tracked_checkpoint_counts_only_backward_linear_replay() -> None:
    layer = PrecisionLinear(3, 2)
    inputs = torch.randn(4, 3, requires_grad=True)
    counter = RecomputeFlopCounter()

    with counter.track():
        output = tracked_checkpoint(layer, inputs)
        assert counter.recompute_flops == 0
        assert counter.recompute_linear_calls == 0
        output.sum().backward()

    assert counter.recompute_flops == 2 * 4 * 3 * 2
    assert counter.recompute_linear_calls == 1
