import sys
import types

import torch
import torch.nn.functional as F

import graphcast_compat.google_small as google_small
from graphcast_compat.google_small import (
    GoogleSmallConfig,
    GoogleSmallGraphNet,
    HaikuStyleMLP,
)


def test_mlp_uses_addmm_silu_when_opted_in(monkeypatch):
    calls = []

    def fake_addmm_silu(bias, mat1, mat2, *, beta=1, alpha=1):
        calls.append((bias, mat1, mat2, beta, alpha))
        return F.silu(torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha))

    flag_gems = types.ModuleType("flag_gems")
    flag_gems.addmm_silu = fake_addmm_silu
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    mlp = HaikuStyleMLP(
        7,
        5,
        hidden_features=11,
        hidden_layers=1,
        fuse_addmm_silu=True,
    )
    mlp.eval()
    x = torch.randn(2, 32768, 7)

    with torch.no_grad():
        actual = mlp(x)
        expected = mlp.linear_1(F.silu(mlp.linear_0(x)))

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    assert calls[0][0] is mlp.linear_0.bias
    assert calls[0][1].shape == (65536, 7)
    assert calls[0][1].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert calls[0][2].stride() == (1, mlp.linear_0.weight.stride(0))


def test_mlp_preserves_standalone_silu_for_small_row_workloads(monkeypatch):
    calls = []
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.addmm_silu = lambda *args, **kwargs: calls.append(args)
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    mlp = HaikuStyleMLP(
        7,
        5,
        hidden_features=11,
        hidden_layers=1,
        fuse_addmm_silu=True,
    )
    x = torch.randn(2, 13, 7)

    with torch.no_grad():
        actual = mlp(x)
        expected = mlp.linear_1(F.silu(mlp.linear_0(x)))

    torch.testing.assert_close(actual, expected)
    assert calls == []


def test_mlp_uses_packed_silu_addmm_when_opted_in(monkeypatch):
    calls = []

    def fake_silu_addmm(hidden, bias, mat2, *, beta=1, alpha=1):
        calls.append((hidden, bias, mat2))
        return torch.addmm(bias, F.silu(hidden), mat2, beta=beta, alpha=alpha)

    monkeypatch.setattr(google_small, "PACKED_MLP_MIN_ROWS", 1)
    monkeypatch.setenv("GRAPHCAST_PACKED_MLP_SCOPE", "standard")
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        types.SimpleNamespace(silu_addmm=fake_silu_addmm),
    )
    mlp = HaikuStyleMLP(
        7,
        5,
        hidden_features=11,
        hidden_layers=1,
        fuse_packed_mlp=True,
    )
    x = torch.randn(2, 13, 7)

    with torch.no_grad():
        actual = mlp(x)
        expected = mlp.linear_1(F.silu(mlp.linear_0(x)))

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1


def test_packed_mlp_scope_can_exclude_standard_path(monkeypatch):
    monkeypatch.setenv("GRAPHCAST_PACKED_MLP_SCOPE", "factorized")
    monkeypatch.setattr(google_small, "PACKED_MLP_MIN_ROWS", 1)
    mlp = HaikuStyleMLP(
        7,
        5,
        hidden_features=11,
        hidden_layers=1,
        fuse_packed_mlp=True,
    )
    x = torch.randn(2, 13, 7)

    with torch.no_grad():
        actual = mlp(x)
        expected = mlp.linear_1(F.silu(mlp.linear_0(x)))

    torch.testing.assert_close(actual, expected)


def test_packed_mlp_defaults_to_factorized_scope(monkeypatch):
    monkeypatch.delenv("GRAPHCAST_PACKED_MLP_SCOPE", raising=False)

    assert google_small._packed_mlp_scope_enabled("factorized")
    assert not google_small._packed_mlp_scope_enabled("standard")


def test_packed_mlp_row_threshold_includes_graphcast_grid_nodes():
    assert not google_small._packed_mlp_rows_eligible(40961)
    assert google_small._packed_mlp_rows_eligible(40962)


def test_mlp_keeps_standard_path_by_default(monkeypatch):
    monkeypatch.delenv("GRAPHCAST_FUSE_ADDMM_SILU", raising=False)
    mlp = HaikuStyleMLP(
        7,
        5,
        hidden_features=11,
        hidden_layers=1,
    )

    assert mlp.fuse_addmm_silu is False


def _factorized_graph_net(
    *,
    fuse_factorized_add_add_silu: bool,
    fuse_packed_mlp: bool = False,
):
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
        edge_update_chunk_sizes=None,
        node_update_chunk_sizes=None,
        decoder_node_chunk_sizes=None,
        message_passing_steps=1,
        config=config,
        edge_input_projection_mode="factorized",
        fuse_factorized_add_add_silu=fuse_factorized_add_add_silu,
        fuse_packed_mlp=fuse_packed_mlp,
    )


def test_factorized_edge_path_uses_add_add_silu_when_opted_in(monkeypatch):
    calls = []

    def fake_add_add_silu(addend, residual_a, residual_b):
        calls.append((addend, residual_a, residual_b))
        return F.silu((addend + residual_a) + residual_b)

    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        types.SimpleNamespace(add_add_silu=fake_add_add_silu),
    )
    graph_net = _factorized_graph_net(fuse_factorized_add_add_silu=True)
    mlp = graph_net.processor_edges_0_edge_mlp
    edge_features = torch.randn(1, 5, 4)
    senders = torch.tensor([0, 1, 2, 1, 0])
    receivers = torch.tensor([1, 0, 1, 0, 1])
    sender_projection = torch.randn(1, 3, 4)
    receiver_projection = torch.randn(1, 2, 4)
    edge_weight = mlp.linear_0.weight[:, :4]

    with torch.no_grad():
        actual = graph_net._apply_factorized_edge_projection(
            "processor_edges_0_edge",
            edge_features,
            senders,
            receivers,
            (edge_weight, sender_projection, receiver_projection),
        )
        hidden = F.linear(edge_features, edge_weight, mlp.linear_0.bias)
        hidden = hidden + sender_projection[:, senders]
        hidden = hidden + receiver_projection[:, receivers]
        expected = graph_net.processor_edges_0_edge_layer_norm(
            mlp.linear_1(F.silu(hidden))
        )

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    assert calls[0][0].shape == calls[0][1].shape == calls[0][2].shape


def test_factorized_edge_path_fuses_gathers_for_large_inference(monkeypatch):
    gather_calls = []

    def fake_gather_gather_add_silu(
        hidden,
        sender_projection,
        senders,
        receiver_projection,
        receivers,
    ):
        gather_calls.append(
            (
                hidden,
                sender_projection,
                senders,
                receiver_projection,
                receivers,
            )
        )
        return F.silu(
            (hidden + sender_projection[:, senders])
            + receiver_projection[:, receivers]
        )

    monkeypatch.setattr(google_small, "FACTORIZED_GATHER_FUSION_MIN_ROWS", 1)
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        types.SimpleNamespace(
            gather_gather_add_silu=fake_gather_gather_add_silu,
            add_add_silu=lambda addend, residual_a, residual_b: F.silu(
                (addend + residual_a) + residual_b
            ),
        ),
    )
    graph_net = _factorized_graph_net(fuse_factorized_add_add_silu=True)
    mlp = graph_net.processor_edges_0_edge_mlp
    edge_features = torch.randn(1, 5, 4)
    senders = torch.tensor([0, 1, 2, 1, 0])
    receivers = torch.tensor([1, 0, 1, 0, 1])
    sender_projection = torch.randn(1, 3, 4)
    receiver_projection = torch.randn(1, 2, 4)
    edge_weight = mlp.linear_0.weight[:, :4]

    with torch.no_grad():
        actual = graph_net._apply_factorized_edge_projection(
            "processor_edges_0_edge",
            edge_features,
            senders,
            receivers,
            (edge_weight, sender_projection, receiver_projection),
        )
        hidden = F.linear(edge_features, edge_weight, mlp.linear_0.bias)
        hidden = hidden + sender_projection[:, senders]
        hidden = hidden + receiver_projection[:, receivers]
        expected = graph_net.processor_edges_0_edge_layer_norm(
            mlp.linear_1(F.silu(hidden))
        )

    torch.testing.assert_close(actual, expected)
    assert len(gather_calls) == 1
    assert gather_calls[0][0].shape == hidden.shape


def test_factorized_edge_path_uses_packed_residual_mlp(monkeypatch):
    calls = []

    def fake_packed(addend, residual_a, residual_b, bias, mat2):
        calls.append((addend, residual_a, residual_b, bias, mat2))
        activated = F.silu((addend + residual_a) + residual_b)
        return torch.addmm(bias, activated, mat2)

    monkeypatch.setattr(google_small, "PACKED_MLP_MIN_ROWS", 1)
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        types.SimpleNamespace(add_add_silu_addmm=fake_packed),
    )
    graph_net = _factorized_graph_net(
        fuse_factorized_add_add_silu=False,
        fuse_packed_mlp=True,
    )
    mlp = graph_net.processor_edges_0_edge_mlp
    edge_features = torch.randn(1, 5, 4)
    senders = torch.tensor([0, 1, 2, 1, 0])
    receivers = torch.tensor([1, 0, 1, 0, 1])
    sender_projection = torch.randn(1, 3, 4)
    receiver_projection = torch.randn(1, 2, 4)
    edge_weight = mlp.linear_0.weight[:, :4]

    with torch.no_grad():
        actual = graph_net._apply_factorized_edge_projection(
            "processor_edges_0_edge",
            edge_features,
            senders,
            receivers,
            (edge_weight, sender_projection, receiver_projection),
        )
        hidden = F.linear(edge_features, edge_weight, mlp.linear_0.bias)
        hidden = hidden + sender_projection[:, senders]
        hidden = hidden + receiver_projection[:, receivers]
        expected = graph_net.processor_edges_0_edge_layer_norm(
            mlp.linear_1(F.silu(hidden))
        )

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1


def test_packed_mlp_scope_can_exclude_factorized_path(monkeypatch):
    monkeypatch.setenv("GRAPHCAST_PACKED_MLP_SCOPE", "standard")
    graph_net = _factorized_graph_net(
        fuse_factorized_add_add_silu=False,
        fuse_packed_mlp=True,
    )
    mlp = graph_net.processor_edges_0_edge_mlp
    edge_features = torch.randn(1, 5, 4)
    senders = torch.tensor([0, 1, 2, 1, 0])
    receivers = torch.tensor([1, 0, 1, 0, 1])
    sender_projection = torch.randn(1, 3, 4)
    receiver_projection = torch.randn(1, 2, 4)
    edge_weight = mlp.linear_0.weight[:, :4]

    with torch.no_grad():
        actual = graph_net._apply_factorized_edge_projection(
            "processor_edges_0_edge",
            edge_features,
            senders,
            receivers,
            (edge_weight, sender_projection, receiver_projection),
        )
        hidden = F.linear(edge_features, edge_weight, mlp.linear_0.bias)
        hidden = hidden + sender_projection[:, senders]
        hidden = hidden + receiver_projection[:, receivers]
        expected = graph_net.processor_edges_0_edge_layer_norm(
            mlp.linear_1(F.silu(hidden))
        )

    torch.testing.assert_close(actual, expected)


def test_grad_enabled_mlp_preserves_training_factory_and_gradients(monkeypatch):
    from graphcast_compat.precision import PrecisionLinear

    monkeypatch.setenv("GRAPHCAST_PACKED_MLP_SCOPE", "all")
    monkeypatch.setattr(google_small, "ADDMM_SILU_MIN_ROWS", 1)
    monkeypatch.setattr(google_small, "PACKED_MLP_MIN_ROWS", 1)
    # No public fusion symbol is supplied: grad-enabled execution must not use it.
    monkeypatch.setitem(sys.modules, "flag_gems", types.SimpleNamespace())
    model = HaikuStyleMLP(
        7, 5, hidden_features=11, hidden_layers=1,
        linear_type=PrecisionLinear, fuse_addmm_silu=True, fuse_packed_mlp=True,
    )
    reference = HaikuStyleMLP(
        7, 5, hidden_features=11, hidden_layers=1,
        linear_type=PrecisionLinear, fuse_addmm_silu=False, fuse_packed_mlp=False,
    )
    reference.load_state_dict(model.state_dict())
    x = torch.randn(2, 13, 7, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = reference(ref_x)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert isinstance(model.linear_0, PrecisionLinear)
    assert isinstance(model.linear_1, PrecisionLinear)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for actual_p, expected_p in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_p.grad, expected_p.grad)
