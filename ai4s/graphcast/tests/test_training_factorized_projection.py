"""Mesh projection must remove edge cat without changing training gradients."""
from copy import deepcopy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from graphcast_compat.recompute_flops import (
    RecomputeFlopCounter, tracked_reentrant_checkpoint,
)
from graphcast_compat.training_model import GoogleSmallCompatibleModel, GoogleSmallConfig
from test_training_chunk_checkpoint import graphnet, inputs, run


class EdgeCatTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        # Split-weight backward can legitimately concatenate a tiny 2D weight
        # gradient. We prohibit edge-sized activations, not parameter gradients.
        if func is torch.ops.aten.cat.default and result.ndim == 3 and result.shape[-1] == 12:
            self.shapes.append(tuple(result.shape))
        return result


def test_edge_cat_trace_detects_the_unoptimized_training_path():
    with EdgeCatTrace() as trace:
        run(graphnet(steps=3), inputs(), chunks=True)
    assert trace.shapes == [(1, 2, 12), (1, 2, 12), (1, 1, 12)] * 3


@pytest.mark.parametrize("chunk_size", [0, 2, 8])
def test_factorized_training_does_not_materialize_edge_cat_even_during_replay(chunk_size):
    network = graphnet(steps=3)
    network.edge_input_projection_mode = "factorized"
    network.edge_update_chunk_sizes["edge"] = chunk_size
    with EdgeCatTrace() as trace:
        values = run(network, inputs(), chunks=True, stage=True, per_step=True)
        sum(x.square().sum() for x in values).backward()
    assert trace.shapes == []


@pytest.mark.parametrize("chunk_size", [0, 2, 8])
@pytest.mark.parametrize("stage,per_step,outer", [
    (False, False, False), (True, False, False),
    (True, True, False), (True, True, True),
])
def test_factorized_training_matches_two_updates_and_all_gradients(chunk_size, stage, per_step, outer):
    torch.manual_seed(31)
    reference = graphnet(steps=3)
    reference.edge_update_chunk_sizes["edge"] = chunk_size
    candidate = deepcopy(reference)
    candidate.edge_input_projection_mode = "factorized"
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    optimizers = [torch.optim.SGD(net.parameters(), lr=0.001) for net in (reference, candidate)]
    for _ in range(2):
        source = inputs()
        cloned = tuple(x.detach().clone().requires_grad_() for x in source)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        expected = run(reference, source, chunks=True, stage=stage, per_step=per_step)

        def forward(*values):
            return run(candidate, values, chunks=True, stage=stage, per_step=per_step)

        actual = tracked_reentrant_checkpoint(forward, *cloned) if outer else forward(*cloned)
        for values in (expected, actual):
            sum(x.square().sum() for x in values).backward()
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        for a, b in zip((*source, *reference.parameters()), (*cloned, *candidate.parameters()), strict=True):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-8)
        for opt in optimizers:
            opt.step()


def test_factorized_training_recompute_counts_split_linears():
    torch.manual_seed(37)
    network = graphnet(steps=3)
    network.edge_input_projection_mode = "factorized"
    counter = RecomputeFlopCounter()
    # No inner checkpoints: the outer replay executes the complete forward once.
    with counter.track():
        values = tracked_reentrant_checkpoint(
            lambda *xs: run(network, xs, chunks=False), *inputs()
        )
        assert counter.recompute_flops == 0
        sum(x.square().sum() for x in values).backward()
    # Encoder: edge 5*(2*4+4*4), nodes 7*(3*4+4*4).
    # Each message: edges 5*(4*4+4*4), node projections 7*4*4,
    # node updates sender 3*(4*4+4*4), receiver 4*(8*4+4*4).
    assert counter.recompute_flops == 2 * (120 + 196 + 3 * (160 + 112 + 96 + 192))


def test_factorized_forward_flop_estimate_removes_repeated_mesh_node_projection():
    model = GoogleSmallCompatibleModel(GoogleSmallConfig(latent_size=4, gnn_msg_steps=3))
    # Only tensor dimensions are consumed by the estimator; no device graph is built.
    model.grid_node_geo_features = torch.empty(7, 1)
    model.mesh_node_geo_features = torch.empty(5, 1)
    model.grid2mesh_edge_features = torch.empty(11, 1)
    model.mesh_edge_features = torch.empty(17, 1)
    model.mesh2grid_edge_features = torch.empty(13, 1)
    model._graph_initialized = True
    before = model.estimate_forward_linear_flops(batch_size=2)
    model.mesh_gnn.edge_input_projection_mode = "factorized"
    after = model.estimate_forward_linear_flops(batch_size=2)
    assert before - after == 4608  # 2 * B=2 * steps=3 * (2*17-2*5) * 4*4.
