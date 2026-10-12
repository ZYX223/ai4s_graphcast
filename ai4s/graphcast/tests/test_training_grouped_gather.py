"""Grouped edge replay must reduce source-wide gradients without changing math."""
from copy import deepcopy

import pytest
import torch

from test_training_chunk_checkpoint import graphnet, inputs, run
from test_training_sequential_chunks import (
    _assert_forward_and_all_gradients_equal, _autograd_node_counts,
)
from graphcast_compat.recompute_flops import tracked_reentrant_checkpoint


@pytest.mark.parametrize("group,expected", [(1, 6), (2, 4), (4, 2)])
def test_grouping_reduces_full_node_index_backward_count(group, expected):
    network = graphnet()
    network.edge_gather_group_size = group
    result = run(network, inputs(), chunks=False)
    joined = sum(value.sum() for value in result)
    counts = _autograd_node_counts(joined)
    assert sum(n for key, n in counts.items() if key.startswith("IndexBackward")) == expected


@pytest.mark.parametrize("group", [2, 4])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("chunks,stage,outer", [
    (False, False, False), (True, False, False), (True, True, True),
])
def test_grouped_replay_keeps_tail_duplicate_indices_and_all_gradients(group, steps, chunks, stage, outer):
    torch.manual_seed(101)
    reference = graphnet(steps)
    candidate = deepcopy(reference)
    candidate.edge_gather_group_size = group
    source = inputs()
    copied = tuple(x.detach().clone().requires_grad_() for x in source)
    expected = run(reference, source, chunks=chunks, stage=stage, per_step=stage)
    def forward(*values):
        return run(candidate, values, chunks=chunks, stage=stage, per_step=stage)
    actual = tracked_reentrant_checkpoint(forward, *copied) if outer else forward(*copied)
    _assert_forward_and_all_gradients_equal(reference, candidate, source, copied, expected, actual)


def test_group_checkpoint_does_not_save_concatenated_inputs():
    network = graphnet(3)
    network.edge_gather_group_size = 2
    shapes = []
    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        run(network, inputs(), chunks=True)
    assert not any(len(shape) == 3 and shape[-1] == 12 for shape in shapes)
