from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
for path in (GRAPHCAST_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphcast_compat.backend import (  # noqa: E402
    use_prevalidated_flaggems_index_add,
)
from graphcast_compat.google_small import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    aggregate_edges_to_nodes,
)


def _model_with_static_receivers() -> GoogleSmallCompatibleModel:
    model = GoogleSmallCompatibleModel(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        ),
        mesh2grid_full_grid_fast_path=True,
        index_add_execution="prevalidated-static",
    ).eval()
    model._set_nonpersistent_buffer("grid2mesh_receivers", torch.tensor([0, 3]))
    model._set_nonpersistent_buffer("mesh_receivers", torch.tensor([1, 1, 3]))
    model._set_nonpersistent_buffer("mesh2grid_receivers", torch.tensor([0, 4]))
    model._graph_initialized = True
    model._num_mesh_nodes = 4
    model._num_grid_nodes = 5
    return model


def test_prevalidated_index_add_preserves_aggregation_semantics() -> None:
    edge_features = torch.tensor([[[1.0], [2.0], [3.0]]])
    receivers = torch.tensor([0, 2, 2])

    result = aggregate_edges_to_nodes(
        edge_features,
        receivers,
        4,
        index_add_execution="prevalidated-static",
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[[1.0], [0.0], [5.0], [0.0]]]),
    )


def test_index_add_execution_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="index_add execution"):
        aggregate_edges_to_nodes(
            torch.ones((1, 1, 1)),
            torch.tensor([0]),
            1,
            index_add_execution="skip-check",
        )


def test_static_receiver_contract_records_bounds_and_detects_mutation() -> None:
    model = _model_with_static_receivers()

    contract = model.prevalidate_static_index_add()

    metadata = contract.to_metadata()
    assert metadata["status"] == "verified"
    assert [item["name"] for item in metadata["buffers"]] == [
        "grid2mesh_receivers",
        "mesh_receivers",
        "mesh2grid_receivers",
    ]
    assert [item["upper_bound"] for item in metadata["buffers"]] == [4, 4, 5]
    model.verify_static_index_add(contract)

    model.mesh_receivers[0] = 2
    with pytest.raises(RuntimeError, match="mesh_receivers.*changed"):
        model.verify_static_index_add(contract)


def test_static_receiver_contract_rejects_invalid_or_partial_graphs() -> None:
    invalid = _model_with_static_receivers()
    invalid.grid2mesh_receivers[0] = 4
    with pytest.raises(ValueError, match="grid2mesh_receivers.*out of bounds"):
        invalid.prevalidate_static_index_add()

    partitioned = _model_with_static_receivers()
    partitioned.mesh2grid_full_grid_fast_path = False
    with pytest.raises(ValueError, match="full-grid"):
        partitioned.prevalidate_static_index_add()


def test_runtime_prevalidated_index_add_enters_public_trusted_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_flag_gems = types.ModuleType("flag_gems")
    exec(
        "from contextlib import contextmanager\n"
        "trusted_scope_active = False\n"
        "def _assert_index_in_bounds(index, upper_bound):\n"
        "    return ('checked', index, upper_bound)\n"
        "@contextmanager\n"
        "def use_trusted_index_add_inference():\n"
        "    global trusted_scope_active\n"
        "    trusted_scope_active = True\n"
        "    try:\n"
        "        yield\n"
        "    finally:\n"
        "        trusted_scope_active = False\n"
        "def index_add_(out, dim, index, src, alpha=1):\n"
        "    return (trusted_scope_active, _assert_index_in_bounds(index, out.shape[dim]))\n",
        fake_flag_gems.__dict__,
    )
    original_check = fake_flag_gems._assert_index_in_bounds
    fake_flag_gems.FULL_CONFIG_BY_FUNC = {"index_add_": [("index_add_", fake_flag_gems.index_add_)]}
    monkeypatch.setitem(sys.modules, "flag_gems", fake_flag_gems)
    runtime = SimpleNamespace(
        profile=SimpleNamespace(name="metax"),
        flagos=SimpleNamespace(mode="flaggems", strict=True),
    )

    with use_prevalidated_flaggems_index_add(runtime) as evidence:
        assert fake_flag_gems.index_add_(torch.zeros(1), 0, "index", None) == (
            True,
            ("checked", "index", 1),
        )
        assert evidence["platform_profile"] == "metax"
        assert evidence["scope_api"] == "use_trusted_index_add_inference"
        assert fake_flag_gems._assert_index_in_bounds is original_check

    assert fake_flag_gems._assert_index_in_bounds is original_check
    assert fake_flag_gems.index_add_(torch.zeros(1), 0, "index", None) == (
        False,
        ("checked", "index", 1),
    )
    assert evidence["closed"] is True


def test_runtime_prevalidated_index_add_rejects_missing_public_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_flag_gems = types.ModuleType("flag_gems")
    exec(
        "def index_add_(out, dim, index, src, alpha=1):\n"
        "    return out\n",
        fake_flag_gems.__dict__,
    )
    fake_flag_gems.FULL_CONFIG_BY_FUNC = {
        "index_add_": [("index_add_", fake_flag_gems.index_add_)]
    }
    monkeypatch.setitem(sys.modules, "flag_gems", fake_flag_gems)
    runtime = SimpleNamespace(
        profile=SimpleNamespace(name="mthreads"),
        flagos=SimpleNamespace(mode="flaggems", strict=True),
    )

    with pytest.raises(RuntimeError, match="trusted inference scope"):
        with use_prevalidated_flaggems_index_add(runtime):
            pass
