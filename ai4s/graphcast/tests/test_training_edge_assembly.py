"""Shared training gathers and sequential edges avoid source-wide slice replay."""

from copy import deepcopy
import weakref

import pytest
import torch
from test_training_chunk_checkpoint import graphnet, run
from test_training_sequential_chunks import (
    _assert_forward_and_all_gradients_equal,
    _autograd_node_counts,
    _full_input_backward_count,
)

from graphcast_compat.recompute_flops import tracked_reentrant_checkpoint
from graphcast_compat.training_model import _gather_pipeline_nodes


@pytest.mark.parametrize("mode", [None, "advanced_index"])
def test_training_advanced_index_has_no_source_wide_slice_backward(monkeypatch, mode):
    # Reintroducing features[:, indices, :] adds two full-source slice gradients.
    if mode is None:
        monkeypatch.delenv("GRAPHCAST_GATHER_MODE", raising=False)
    else:
        monkeypatch.setenv("GRAPHCAST_GATHER_MODE", mode)
    source = torch.randn(1, 13, 4, dtype=torch.float64, requires_grad=True)
    result = _gather_pipeline_nodes(source, torch.tensor([2, 0, 2, 10]))
    counts = _autograd_node_counts(result)
    assert sum(n for name, n in counts.items() if name.startswith("SliceBackward")) == 0
    assert sum(n for name, n in counts.items() if name.startswith("IndexBackward")) == 1


def test_training_gather_honors_explicit_index_select_override(monkeypatch):
    monkeypatch.setenv("GRAPHCAST_GATHER_MODE", "index_select")
    source = torch.arange(12.0).reshape(1, 3, 4).requires_grad_()
    result = _gather_pipeline_nodes(source, torch.tensor([2, 0, 2]))
    counts = _autograd_node_counts(result)
    assert sum(n for name, n in counts.items() if name.startswith("IndexSelectBackward")) == 1
    torch.testing.assert_close(result.detach(), torch.tensor([[[8., 9., 10., 11.],
                                                             [0., 1., 2., 3.],
                                                             [8., 9., 10., 11.]]]))
    result.sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[[1., 1., 1., 1.],
                                                         [0., 0., 0., 0.],
                                                         [2., 2., 2., 2.]]]))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("index_values", [[], [2, 0, 2, 6], [-1, 0, -1, 2]])
def test_training_gather_preserves_values_and_repeated_index_gradients(
    monkeypatch, batch, noncontiguous, index_values,
):
    monkeypatch.setenv("GRAPHCAST_GATHER_MODE", "advanced_index")
    torch.manual_seed(53)
    source = torch.randn(batch, 7, 4, dtype=torch.float64)
    if noncontiguous:
        source = source.transpose(1, 2).contiguous().transpose(1, 2)
    source.requires_grad_()
    copied = source.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    # Also cover a non-contiguous index vector, independent of feature strides.
    indices = torch.tensor(index_values, dtype=torch.long).repeat_interleave(2)[::2]
    expected = source[:, indices, :]
    actual = _gather_pipeline_nodes(copied, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(copied.grad, source.grad, rtol=0, atol=0)


@pytest.mark.parametrize("index", [7, -8])
def test_training_gather_retains_index_bounds_errors(monkeypatch, index):
    monkeypatch.setenv("GRAPHCAST_GATHER_MODE", "advanced_index")
    with pytest.raises(IndexError):
        _gather_pipeline_nodes(torch.zeros(1, 7, 4), torch.tensor([index]))


def test_edge_state_backward_assembly_does_not_grow_with_chunk_count():
    # Undoing split/cat restores per-chunk SliceBackward and CopySlices chains.
    observed = []
    for chunk_size in (4, 2, 1):
        network = graphnet(steps=3)
        network.edge_update_chunk_sizes["edge"] = chunk_size
        edges = torch.randn(1, 8, 4, dtype=torch.float64, requires_grad=True)
        _, output = network._message_step(
            1,
            {
                "sender": torch.randn(1, 3, 4, dtype=torch.float64),
                "receiver": torch.randn(1, 4, 4, dtype=torch.float64),
            },
            {"edge": edges},
            {"edge": (torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
                      torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))},
        )
        observed.append(_full_input_backward_count(output["edge"]))
        output["edge"].square().sum().backward()
        assert edges.grad is not None
    assert observed == [1, 1, 1]


@pytest.mark.parametrize("chunks", [False, True])
def test_residual_chunk_outputs_are_released_before_node_updates(monkeypatch, chunks):
    # Keeping the list or final loop result after cat retains a second edge
    # state during node processing, even though CatBackward needs only sizes.
    network = graphnet(steps=3)
    references = []
    observed = []
    compute = network._edge_chunk

    def observe_chunks(*args, **kwargs):
        result = compute(*args, **kwargs)
        if result[1] is not None:
            references.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(network, "_edge_chunk", observe_chunks)
    hook = network.processor_nodes_0_receiver_mlp.register_forward_pre_hook(
        lambda module, args: observed.append([ref() is not None for ref in references])
    )
    try:
        _, edges = network._message_step(
            0,
            {
                "sender": torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True),
                "receiver": torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True),
            },
            {"edge": torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)},
            {"edge": (torch.tensor([0, 1, 2, 0, 1]), torch.tensor([0, 1, 2, 3, 0]))},
            checkpoint_chunks=chunks,
        )
        assert observed == [[False, False, False]]
        edges["edge"].square().sum().backward()
    finally:
        hook.remove()


@pytest.mark.parametrize("steps", [1, 3], ids=["adapter", "mesh"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("stage,per_step,outer", [
    (False, False, False), (True, True, False), (True, True, True),
])
def test_tail_edge_chunks_match_unchunked_values_and_all_gradients(
    monkeypatch, steps, batch, stage, per_step, outer,
):
    # Wrong residual order, dropped tails or chunk state retained across replay
    # must change outputs or input/parameter gradients against the unchunked GNN.
    monkeypatch.setenv("GRAPHCAST_GATHER_MODE", "advanced_index")
    torch.manual_seed(59)
    reference = graphnet(steps=steps)
    reference.edge_update_chunk_sizes["edge"] = 0
    candidate = deepcopy(reference)
    candidate.edge_update_chunk_sizes["edge"] = 2
    source = tuple(
        torch.randn(batch, width, rows, dtype=torch.float64).transpose(1, 2).requires_grad_()
        for rows, width in ((3, 3), (4, 3), (5, 2))
    )
    copied = tuple(
        value.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        for value in source
    )
    assert all(not value.is_contiguous() for value in (*source, *copied))
    expected = run(reference, source, chunks=False)

    def forward(*values):
        return run(candidate, values, chunks=True, stage=stage, per_step=per_step)

    actual = tracked_reentrant_checkpoint(forward, *copied) if outer else forward(*copied)
    # One-step adapters intentionally omit their unused residual edge output.
    if steps == 1:
        expected = expected[:2]
        assert len(actual) == 2
    _assert_forward_and_all_gradients_equal(
        reference, candidate, source, copied, expected, actual,
    )
