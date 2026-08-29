from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat.google_small import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    GoogleSmallGraphNet,
)


def _single_node_graph_net(*, fuse_post_layer_norm_residual: bool) -> GoogleSmallGraphNet:
    config = GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=1,
        node_input_dim=4,
        edge_input_dim=1,
        output_dim=1,
    )
    return GoogleSmallGraphNet(
        edge_types=(),
        edge_node_sets={},
        node_types=("node",),
        node_input_dims={"node": 4},
        edge_input_dims={},
        node_processor_input_dims={"node": 4},
        edge_processor_input_dims={},
        node_output_dims=None,
        extra_final_node_output_dims=None,
        edge_update_chunk_sizes=None,
        node_update_chunk_sizes=None,
        decoder_node_chunk_sizes=None,
        message_passing_steps=1,
        config=config,
        fuse_post_layer_norm_residual=fuse_post_layer_norm_residual,
    )


def test_node_update_uses_post_layer_norm_residual_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    graph_net = _single_node_graph_net(fuse_post_layer_norm_residual=False)
    nodes = {"node": torch.randn(1, 3, 4)}

    expected, _ = graph_net(nodes, {}, {})
    calls = []

    def post_layer_norm_residual(
        input: torch.Tensor,
        residual: torch.Tensor,
        normalized_shape: tuple[int, ...],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        calls.append((input.shape, residual.shape, normalized_shape, eps))
        return torch.layer_norm(input, normalized_shape, weight, bias, eps) + residual

    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        types.SimpleNamespace(post_layer_norm_residual=post_layer_norm_residual),
    )
    graph_net.fuse_post_layer_norm_residual = True

    actual, _ = graph_net(nodes, {}, {})

    torch.testing.assert_close(actual["node"], expected["node"])
    assert calls == [(torch.Size([1, 3, 4]), torch.Size([1, 3, 4]), (4,), 1e-5)]


def test_requested_fusion_requires_public_flaggems_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_net = _single_node_graph_net(fuse_post_layer_norm_residual=True)
    monkeypatch.setitem(sys.modules, "flag_gems", types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="post_layer_norm_residual"):
        graph_net({"node": torch.randn(1, 3, 4)}, {}, {})


def test_model_reads_post_layer_norm_residual_fusion_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL", "1")

    model = GoogleSmallCompatibleModel(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        )
    )

    assert model.fuse_post_layer_norm_residual is True
    assert model.grid2mesh_gnn.fuse_post_layer_norm_residual is True
    assert model.mesh_gnn.fuse_post_layer_norm_residual is True
    assert model.mesh2grid_gnn.fuse_post_layer_norm_residual is True
