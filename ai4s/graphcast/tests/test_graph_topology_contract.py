from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat.graph import (
    canonical_mesh_hierarchy,
    radius_query_indices,
    validate_canonical_operational_graph_contract,
)


class CanonicalGraphTopologyTest(unittest.TestCase):
    def test_mesh_hierarchy_uses_h100_baseline_vertices(self) -> None:
        expected_digests = (
            "7181d71d4e7a777569b33369b68c0a69a0353c5728061bb488a8cc85a5fe6e1e",
            "2f1ef9d93d1d8b685a8b4cd5e78cd07554cc1a090f57b821bcc1d764cc8c1962",
            "1d2479fe3e3aa9940b8e711e610f6f85136acb64080ce1708cf75495ae0c1b22",
            "e743a1184b5ca72eca30a5d10d84503a953d19c0fe2a957d516ac08862ded19f",
            "5bd5b5a9939d29b0fdeab81f2b02c41e46a0462a0f73b90a035690dea44cf225",
            "4909d8aef7a35b4331cedcd249dabf06632e5ad098ff72a71782726330e98325",
            "0c6a10735f0f73a4478d4c4665b22f7f556be2cb59b0201d644e757e69fdcb5d",
        )
        meshes = canonical_mesh_hierarchy(6)
        self.assertEqual(len(meshes), len(expected_digests))
        self.assertEqual(
            [mesh.vertices.dtype for mesh in meshes],
            [np.dtype(np.float32)] * len(meshes),
        )
        self.assertEqual(
            [
                hashlib.sha256(mesh.vertices.tobytes()).hexdigest()
                for mesh in meshes
            ],
            list(expected_digests),
        )

    def test_radius_query_orders_neighbors_by_mesh_index(self) -> None:
        mesh = SimpleNamespace(
            vertices=np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        grid_indices, mesh_indices = radius_query_indices(
            grid_latitude=np.asarray([0.0], dtype=np.float32),
            grid_longitude=np.asarray([0.0], dtype=np.float32),
            mesh=mesh,
            radius=3.0,
        )
        np.testing.assert_array_equal(grid_indices, np.zeros(4, dtype=np.int64))
        np.testing.assert_array_equal(mesh_indices, np.arange(4, dtype=np.int64))

    def test_formal_graph_contract_fails_closed_on_edge_drift(self) -> None:
        contract = {
            "resolution_degrees": 0.25,
            "mesh_size": 6,
            "mesh_message_passing_steps": 16,
            "grid_nodes": 1038240,
            "mesh_nodes": 40962,
            "grid2mesh_edges": 1618745,
            "mesh_edges": 327660,
            "mesh2grid_edges": 3114720,
        }
        validate_canonical_operational_graph_contract(contract)
        contract["grid2mesh_edges"] += 2
        with self.assertRaisesRegex(ValueError, "grid2mesh_edges"):
            validate_canonical_operational_graph_contract(contract)


if __name__ == "__main__":
    unittest.main()
