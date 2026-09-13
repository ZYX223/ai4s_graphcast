"""Checkpoint boundaries must release edge temporaries and preserve gradients."""
from copy import deepcopy

import pytest
import torch

from graphcast_compat.recompute_flops import tracked_reentrant_checkpoint
from graphcast_compat.training_model import (
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    GoogleSmallGraphNet,
)


def graphnet(steps=1):
    return GoogleSmallGraphNet(
        edge_types=("edge",),
        edge_node_sets={"edge": ("sender", "receiver")},
        node_types=("sender", "receiver"),
        node_input_dims={"sender": 3, "receiver": 3},
        edge_input_dims={"edge": 2},
        node_processor_input_dims={"sender": 4, "receiver": 8},
        edge_processor_input_dims={"edge": 12},
        node_output_dims=None,
        extra_final_node_output_dims=None,
        edge_update_chunk_sizes={"edge": 2},
        message_passing_steps=steps,
        config=GoogleSmallConfig(latent_size=4, hidden_layers=1),
    ).double()


def inputs():
    return (
        torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True),
    )


def run(network, tensors, *, chunks, stage=False, per_step=False):
    nodes, edges = GoogleSmallCompatibleModel._run_graphnet_stage(
        network,
        {"sender": tensors[0], "receiver": tensors[1]},
        {"edge": tensors[2]},
        {"edge": (torch.tensor([0, 1, 2, 0, 1]), torch.tensor([0, 1, 2, 3, 0]))},
        checkpoint_stage=stage,
        checkpoint_chunks=chunks,
        checkpoint_per_step=per_step,
    )
    # One-step adapters deliberately discard the unused edge state.
    return (*nodes.values(), *(x for x in edges.values() if x is not None))


def test_chunk_checkpoint_recomputes_raw_edge_encoder():
    network = graphnet()
    calls = []
    handle = network.encoder_edges_edge_mlp.register_forward_pre_hook(
        lambda module, args: calls.append(tuple(args[0].shape))
    )
    try:
        outputs = run(network, inputs(), chunks=True)
        forward_calls = len(calls)
        sum(x.square().sum() for x in outputs).backward()
    finally:
        handle.remove()
    assert forward_calls == 3  # Two full chunks and a nonempty tail.
    assert len(calls) == 6  # The raw encoder is inside, not before, checkpoint.


def test_chunk_checkpoint_does_not_save_concatenated_edge_features():
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        run(graphnet(), inputs(), chunks=True)
    # Latent width 4: edge + sender + receiver would retain width 12.
    assert not any(len(shape) == 3 and shape[-1] == 12 for shape in saved_shapes)


@pytest.mark.parametrize("stage", [False, True])
def test_chunk_checkpoint_trains_encoder_when_inputs_do_not_require_grad(stage):
    torch.manual_seed(23)
    eager = graphnet()
    checkpointed = deepcopy(eager)
    values = tuple(x.detach() for x in inputs())
    for network, chunks in ((eager, False), (checkpointed, True)):
        outputs = run(network, values, chunks=chunks, stage=stage)
        sum(x.square().sum() for x in outputs).backward()
    for a, b in zip(eager.parameters(), checkpointed.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("stage,per_step,outer", [
    (False, False, False), (True, False, False),
    (True, True, False), (True, True, True),
])
def test_chunk_checkpoint_matches_eager_outputs_and_all_gradients(steps, stage, per_step, outer):
    torch.manual_seed(19)
    eager = graphnet(steps)
    checkpointed = deepcopy(eager)
    source = inputs()
    cloned = tuple(x.detach().clone().requires_grad_() for x in source)
    expected = run(eager, source, chunks=False)

    def forward(*values):
        return run(checkpointed, values, chunks=True, stage=stage, per_step=per_step)

    actual = tracked_reentrant_checkpoint(forward, *cloned) if outer else forward(*cloned)
    for values in (expected, actual):
        sum(x.square().sum() for x in values).backward()
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    for a, b in zip((*source, *eager.parameters()), (*cloned, *checkpointed.parameters()), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-9)
