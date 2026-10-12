"""Sequential training chunks must have constant backward assembly cost."""

from collections import Counter
from copy import deepcopy

import pytest
import torch

from graphcast_compat.recompute_flops import tracked_reentrant_checkpoint
from graphcast_compat.training_model import GoogleSmallConfig, GoogleSmallGraphNet


def _node_only_graphnet(*, chunk_size: int, decoder: bool) -> GoogleSmallGraphNet:
    return GoogleSmallGraphNet(
        edge_types=(),
        edge_node_sets={},
        node_types=("node",),
        node_input_dims=None,
        edge_input_dims={},
        node_processor_input_dims={"node": 4},
        edge_processor_input_dims={},
        node_output_dims={"node": 3} if decoder else None,
        extra_final_node_output_dims=None,
        edge_update_chunk_sizes={},
        node_update_chunk_sizes={"node": chunk_size},
        decoder_node_chunk_sizes=None,
        message_passing_steps=0 if decoder else 1,
        config=GoogleSmallConfig(latent_size=4, hidden_layers=1),
    ).double()


def _received_feature_graphnet(*, chunk_size: int) -> GoogleSmallGraphNet:
    return GoogleSmallGraphNet(
        edge_types=("edge",),
        edge_node_sets={"edge": ("sender", "receiver")},
        node_types=("sender", "receiver"),
        node_input_dims=None,
        edge_input_dims={"edge": 3},
        node_processor_input_dims={"sender": 4, "receiver": 8},
        edge_processor_input_dims={"edge": 12},
        node_output_dims={"receiver": 3},
        extra_final_node_output_dims=None,
        edge_update_chunk_sizes={},
        node_update_chunk_sizes={"sender": chunk_size, "receiver": chunk_size},
        decoder_node_chunk_sizes=None,
        message_passing_steps=1,
        config=GoogleSmallConfig(latent_size=4, hidden_layers=1),
    ).double()


def _noncontiguous_leaf(node_count: int, width: int) -> torch.Tensor:
    return torch.randn(1, width, node_count, dtype=torch.float64).transpose(1, 2).requires_grad_()


def _autograd_node_counts(output: torch.Tensor) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen = set()
    pending = [output.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(parent for parent, _ in node.next_functions)
    return counts


def _full_input_backward_count(output: torch.Tensor) -> int:
    """Count backward nodes whose contract can process a source-wide buffer.

    CopySlices assembles a full destination gradient, SliceBackward scatters a
    view gradient into a source-sized buffer, and SplitBackward jointly maps
    all split gradients back to one source. One split therefore costs one by
    this allocation contract, while repeated slice/write costs per chunk.
    """

    counts = _autograd_node_counts(output)
    return sum(
        count
        for name, count in counts.items()
        if name.startswith(("CopySlices", "SliceBackward", "Split"))
    )


@pytest.mark.parametrize("decoder", [False, True], ids=["node-update", "decoder"])
def test_full_input_backward_assembly_does_not_grow_with_chunk_count(decoder):
    observed = []
    for chunk_size in (4, 2, 1):
        network = _node_only_graphnet(chunk_size=chunk_size, decoder=decoder)
        source = torch.randn(1, 8, 4, dtype=torch.float64, requires_grad=True)
        output = network({"node": source}, {}, {})[0]["node"]
        observed.append(_full_input_backward_count(output))
        output.square().sum().backward()

    assert observed == [1, 1, 1]


def _flat_forward(
    network: GoogleSmallGraphNet,
    tensors: tuple[torch.Tensor, ...],
    *,
    checkpoint_chunks: bool,
) -> tuple[torch.Tensor, ...]:
    nodes, edges = network(
        {"sender": tensors[0], "receiver": tensors[1]},
        {"edge": tensors[2]},
        {
            "edge": (
                torch.tensor([0, 1, 2, 0, 2, 1]),
                torch.tensor([0, 1, 1, 3, 3, 0]),
            )
        },
        checkpoint_chunks=checkpoint_chunks,
    )
    return (*nodes.values(), *(value for value in edges.values() if value is not None))


def _assert_forward_and_all_gradients_equal(
    reference: GoogleSmallGraphNet,
    candidate: GoogleSmallGraphNet,
    source: tuple[torch.Tensor, ...],
    copied: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
) -> None:
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    for left, right in zip(source, copied, strict=True):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-10)
    for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize(
    "checkpoint_chunks,outer_checkpoint",
    [(False, False), (True, False), (True, True)],
)
def test_tail_chunks_received_features_decoder_and_checkpoints_match_unchunked(
    checkpoint_chunks,
    outer_checkpoint,
):
    torch.manual_seed(41)
    reference = _received_feature_graphnet(chunk_size=0)
    candidate = deepcopy(reference)
    candidate.node_update_chunk_sizes = {"sender": 2, "receiver": 2}
    source = (
        _noncontiguous_leaf(3, 4),
        _noncontiguous_leaf(5, 4),
        _noncontiguous_leaf(6, 3),
    )
    copied = tuple(
        value.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        for value in source
    )
    assert all(not value.is_contiguous() for value in (*source, *copied))
    expected = _flat_forward(reference, source, checkpoint_chunks=False)

    def forward(*values):
        return _flat_forward(candidate, values, checkpoint_chunks=checkpoint_chunks)

    actual = (
        tracked_reentrant_checkpoint(forward, *copied)
        if outer_checkpoint
        else forward(*copied)
    )
    _assert_forward_and_all_gradients_equal(
        reference,
        candidate,
        source,
        copied,
        expected,
        actual,
    )


@pytest.mark.parametrize("decoder", [False, True], ids=["node-update", "decoder"])
@pytest.mark.parametrize(
    "node_count,chunk_size",
    [(0, 2), (5, 0), (5, 8)],
    ids=["empty", "disabled", "larger-than-input"],
)
def test_empty_and_unchunked_fallback_match_copied_graphnet(
    decoder,
    node_count,
    chunk_size,
):
    torch.manual_seed(43)
    reference = _node_only_graphnet(chunk_size=0, decoder=decoder)
    candidate = deepcopy(reference)
    candidate.node_update_chunk_sizes = {"node": chunk_size}
    source = (_noncontiguous_leaf(node_count, 4),)
    copied = (source[0].detach().clone(memory_format=torch.preserve_format).requires_grad_(),)

    def run(network, values):
        return (network({"node": values[0]}, {}, {})[0]["node"],)

    _assert_forward_and_all_gradients_equal(
        reference,
        candidate,
        source,
        copied,
        run(reference, source),
        run(candidate, copied),
    )
