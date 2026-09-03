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


def test_operational_config_chunks_grid2mesh_edges() -> None:
    config = GoogleSmallConfig.from_yaml(
        GRAPHCAST_ROOT / "configs/inference/google_graphcast_operational_25km.yaml"
    )

    assert config.grid2mesh_edge_chunk_size == 131072


def test_model_applies_grid2mesh_edge_chunk_contract() -> None:
    model = GoogleSmallCompatibleModel(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            grid2mesh_edge_chunk_size=128,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        )
    )

    assert model.grid2mesh_gnn.edge_update_chunk_sizes == {"grid2mesh": 128}


def _single_edge_graph_net(
    *,
    edge_chunk_size: int | None,
    edge_input_projection_mode: str = "concatenated",
    receiver_node_chunk_size: int | None = None,
    index_add_execution: str = "inline",
) -> GoogleSmallGraphNet:
    config = GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=1,
        node_input_dim=3,
        edge_input_dim=2,
        output_dim=1,
    )
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
        edge_update_chunk_sizes=(
            {"edge": edge_chunk_size} if edge_chunk_size is not None else None
        ),
        node_update_chunk_sizes=(
            {"receiver": receiver_node_chunk_size} if receiver_node_chunk_size is not None else None
        ),
        decoder_node_chunk_sizes=None,
        message_passing_steps=1,
        config=config,
        edge_input_projection_mode=edge_input_projection_mode,
        index_add_execution=index_add_execution,
    )


def _edge_graph_inputs(
    *,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    nodes = {
        "sender": torch.randn(1, 4, 3, dtype=dtype, requires_grad=requires_grad),
        "receiver": torch.randn(1, 3, 3, dtype=dtype, requires_grad=requires_grad),
    }
    edges = {
        "edge": torch.randn(1, 6, 2, dtype=dtype, requires_grad=requires_grad),
    }
    edge_indices = {
        "edge": (
            torch.tensor([0, 1, 2, 3, 0, 2]),
            torch.tensor([0, 1, 2, 0, 2, 1]),
        )
    }
    return nodes, edges, edge_indices


def _clone_edge_graph_inputs(
    nodes: dict[str, torch.Tensor],
    edges: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {
            name: value.detach().clone().requires_grad_(value.requires_grad)
            for name, value in nodes.items()
        },
        {
            name: value.detach().clone().requires_grad_(value.requires_grad)
            for name, value in edges.items()
        },
    )


def test_factorized_edge_projection_preserves_forward_and_checkpoint_contract() -> None:
    torch.manual_seed(11)
    concatenated = _single_edge_graph_net(edge_chunk_size=None)
    factorized = _single_edge_graph_net(
        edge_chunk_size=None,
        edge_input_projection_mode="factorized",
    )
    factorized.load_state_dict(concatenated.state_dict(), strict=True)
    nodes, edges, edge_indices = _edge_graph_inputs()

    expected_nodes, expected_edges = concatenated(nodes, edges, edge_indices)
    actual_nodes, actual_edges = factorized(nodes, edges, edge_indices)

    assert tuple(factorized.state_dict()) == tuple(concatenated.state_dict())
    for name in expected_nodes:
        torch.testing.assert_close(actual_nodes[name], expected_nodes[name])
    for name in expected_edges:
        torch.testing.assert_close(actual_edges[name], expected_edges[name])


def test_factorized_edge_projection_preserves_input_and_parameter_gradients() -> None:
    torch.manual_seed(13)
    concatenated = _single_edge_graph_net(edge_chunk_size=None).double()
    factorized = _single_edge_graph_net(
        edge_chunk_size=None,
        edge_input_projection_mode="factorized",
    ).double()
    factorized.load_state_dict(concatenated.state_dict(), strict=True)
    expected_nodes_in, expected_edges_in, edge_indices = _edge_graph_inputs(
        requires_grad=True,
        dtype=torch.float64,
    )
    actual_nodes_in, actual_edges_in = _clone_edge_graph_inputs(
        expected_nodes_in,
        expected_edges_in,
    )

    expected_nodes, expected_edges = concatenated(
        expected_nodes_in,
        expected_edges_in,
        edge_indices,
    )
    actual_nodes, actual_edges = factorized(
        actual_nodes_in,
        actual_edges_in,
        edge_indices,
    )
    sum(
        value.square().sum() for value in (*expected_nodes.values(), *expected_edges.values())
    ).backward()
    sum(
        value.square().sum() for value in (*actual_nodes.values(), *actual_edges.values())
    ).backward()

    for name in expected_nodes_in:
        torch.testing.assert_close(actual_nodes_in[name].grad, expected_nodes_in[name].grad)
    for name in expected_edges_in:
        torch.testing.assert_close(actual_edges_in[name].grad, expected_edges_in[name].grad)
    for (expected_name, expected), (actual_name, actual) in zip(
        concatenated.named_parameters(),
        factorized.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(actual.grad, expected.grad)


def test_factorized_edge_projection_preserves_chunked_node_result() -> None:
    torch.manual_seed(17)
    unchunked = _single_edge_graph_net(
        edge_chunk_size=None,
        edge_input_projection_mode="factorized",
    )
    chunked = _single_edge_graph_net(
        edge_chunk_size=2,
        edge_input_projection_mode="factorized",
    )
    chunked.load_state_dict(unchunked.state_dict(), strict=True)
    nodes, edges, edge_indices = _edge_graph_inputs()

    expected_nodes, _ = unchunked(nodes, edges, edge_indices)
    actual_nodes, _ = chunked(nodes, edges, edge_indices)

    for name in expected_nodes:
        torch.testing.assert_close(actual_nodes[name], expected_nodes[name])


def test_factorized_edge_projection_removes_edge_input_cat() -> None:
    torch.manual_seed(19)
    concatenated = _single_edge_graph_net(edge_chunk_size=None)
    factorized = _single_edge_graph_net(
        edge_chunk_size=None,
        edge_input_projection_mode="factorized",
    )
    factorized.load_state_dict(concatenated.state_dict(), strict=True)
    nodes, edges, edge_indices = _edge_graph_inputs()

    def cat_calls(model: GoogleSmallGraphNet) -> int:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            model(nodes, edges, edge_indices)
        return sum(event.count for event in profile.key_averages() if event.key == "aten::cat")

    assert cat_calls(factorized) == cat_calls(concatenated) - 1


def test_factorized_edge_projection_keeps_node_projections_on_addmm() -> None:
    torch.manual_seed(23)
    factorized = _single_edge_graph_net(
        edge_chunk_size=None,
        edge_input_projection_mode="factorized",
    )
    nodes, edges, edge_indices = _edge_graph_inputs()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        factorized(nodes, edges, edge_indices)
    counts = {event.key: event.count for event in profile.key_averages()}

    assert counts.get("aten::mm", 0) == 0
    assert counts.get("aten::addmm", 0) >= 2


def test_edge_projection_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="edge input projection mode"):
        _single_edge_graph_net(
            edge_chunk_size=None,
            edge_input_projection_mode="unknown",
        )


def test_chunked_edge_update_preserves_node_result() -> None:
    torch.manual_seed(7)
    unchunked = _single_edge_graph_net(edge_chunk_size=None)
    chunked = _single_edge_graph_net(edge_chunk_size=2)
    chunked.load_state_dict(unchunked.state_dict())
    nodes = {
        "sender": torch.randn(1, 4, 3),
        "receiver": torch.randn(1, 3, 3),
    }
    edges = {"edge": torch.randn(1, 6, 2)}
    edge_indices = {
        "edge": (
            torch.tensor([0, 1, 2, 3, 0, 2]),
            torch.tensor([0, 1, 2, 0, 2, 1]),
        )
    }

    expected_nodes, _ = unchunked(nodes, edges, edge_indices)
    actual_nodes, _ = chunked(nodes, edges, edge_indices)

    torch.testing.assert_close(actual_nodes["sender"], expected_nodes["sender"])
    torch.testing.assert_close(actual_nodes["receiver"], expected_nodes["receiver"])


def _model_with_mesh2grid_partition_buffers(
    *,
    full_grid_fast_path: bool,
) -> GoogleSmallCompatibleModel:
    model = GoogleSmallCompatibleModel(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        ),
        mesh2grid_full_grid_fast_path=full_grid_fast_path,
    )
    model._set_nonpersistent_buffer(
        "mesh2grid_edge_features",
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    model._set_nonpersistent_buffer(
        "mesh2grid_senders",
        torch.tensor([3, 2, 1, 0]),
    )
    model._set_nonpersistent_buffer(
        "mesh2grid_receivers",
        torch.tensor([0, 1, 2, 2]),
    )
    model._num_grid_nodes = 3
    return model


def test_full_grid_mesh2grid_fast_path_reuses_static_graph_buffers() -> None:
    model = _model_with_mesh2grid_partition_buffers(full_grid_fast_path=True)

    edge_features, senders, receivers = model._select_mesh2grid_partition_inputs(
        grid_start=0,
        grid_end=3,
    )

    assert edge_features.data_ptr() == model.mesh2grid_edge_features.data_ptr()
    assert senders.data_ptr() == model.mesh2grid_senders.data_ptr()
    assert receivers.data_ptr() == model.mesh2grid_receivers.data_ptr()


def test_mesh2grid_partial_partition_keeps_filtered_local_receiver_contract() -> None:
    model = _model_with_mesh2grid_partition_buffers(full_grid_fast_path=True)

    edge_features, senders, receivers = model._select_mesh2grid_partition_inputs(
        grid_start=1,
        grid_end=3,
    )

    torch.testing.assert_close(edge_features, torch.tensor([[2.0], [3.0], [4.0]]))
    torch.testing.assert_close(senders, torch.tensor([2, 1, 0]))
    torch.testing.assert_close(receivers, torch.tensor([0, 1, 1]))
    assert edge_features.data_ptr() != model.mesh2grid_edge_features.data_ptr()


def test_disabled_mesh2grid_fast_path_preserves_legacy_full_grid_selection() -> None:
    model = _model_with_mesh2grid_partition_buffers(full_grid_fast_path=False)

    edge_features, senders, receivers = model._select_mesh2grid_partition_inputs(
        grid_start=0,
        grid_end=3,
    )

    torch.testing.assert_close(edge_features, model.mesh2grid_edge_features)
    torch.testing.assert_close(senders, model.mesh2grid_senders)
    torch.testing.assert_close(receivers, model.mesh2grid_receivers)
    assert edge_features.data_ptr() != model.mesh2grid_edge_features.data_ptr()
