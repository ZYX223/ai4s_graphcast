import unittest
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

from graphcast_bench.optimizations.training.regular3 import ScopedTopology

Key = namedtuple("Key", "name node_sets")
KEY = Key("mesh2grid", ("mesh_nodes", "grid_nodes"))


def fixture():
    ids = np.repeat(np.arange(7, dtype=np.int32), 3)
    edge = SimpleNamespace(indices=SimpleNamespace(receivers=ids))
    graph = SimpleNamespace(
        edges={KEY: edge}, nodes={"grid_nodes": SimpleNamespace(n_node=np.array([7]))}
    )
    owner = SimpleNamespace(_mesh2grid_graph_structure=graph)
    return owner, graph, edge, ids


class ScopeChecks(unittest.TestCase):
    def test_no_previous_verification_authorizes_tracers(self):
        gate = ScopedTopology(7)
        owner, graph, edge, ids = fixture()
        token = object()
        with self.assertRaises(ValueError):
            gate.authorize(token, object())
        with gate.model(owner):
            with self.assertRaises(ValueError):
                gate.authorize(token, object())
        with self.assertRaises(ValueError):
            gate.authorize(token, object())

    def test_exact_receiver_and_message_binding_and_cleanup(self):
        gate = ScopedTopology(7)
        owner, graph, edge, ids = fixture()
        token = object()
        data = object()
        with gate.model(owner):
            with gate.receiver(token, data):
                gate.authorize(token, data)
                with self.assertRaises(ValueError):
                    gate.authorize(object(), data)
                with self.assertRaises(ValueError):
                    gate.authorize(token, object())
            with self.assertRaises(ValueError):
                gate.authorize(token, data)
        self.assertIsNone(gate.current())

    def test_invalid_graph_and_mutation_rejected(self):
        gate = ScopedTopology(7)
        for problem in ["order", "key", "owner", "edge", "indices"]:
            owner, graph, edge, ids = fixture()
            if problem == "order":
                ids[0] = 1
            if problem == "key":
                graph.edges = {Key("other", KEY.node_sets): edge}
            if problem in ["order", "key"]:
                with self.assertRaises(ValueError):
                    with gate.model(owner):
                        pass
                continue
            with gate.model(owner):
                if problem == "owner":
                    owner._mesh2grid_graph_structure = object()
                if problem == "edge":
                    graph.edges[KEY] = object()
                if problem == "indices":
                    edge.indices.receivers = ids.copy()
                with self.assertRaises(ValueError):
                    with gate.receiver(object(), object()):
                        pass

    def test_exception_cleans_both_scopes(self):
        gate = ScopedTopology(7)
        owner, _, _, _ = fixture()
        with self.assertRaises(RuntimeError):
            with gate.model(owner):
                with gate.receiver(object(), object()):
                    raise RuntimeError("expected")
        self.assertIsNone(gate.current())

    def test_non_target_calls_keep_original_path(self):
        gate = ScopedTopology(7)
        owner, graph, _, _ = fixture()
        self.assertFalse(gate.node_is_target(graph, "grid_nodes"))
        with gate.model(owner):
            self.assertFalse(gate.node_is_target(graph, "mesh_nodes"))
            self.assertTrue(gate.node_is_target(graph, "grid_nodes"))
            bad = SimpleNamespace(edges={Key("other", KEY.node_sets): object()})
            with self.assertRaises(ValueError):
                gate.node_is_target(bad, "grid_nodes")
        self.assertFalse(gate.node_is_target(graph, "grid_nodes"))


if __name__ == "__main__":
    unittest.main()
