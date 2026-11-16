"""Use frozen official H100 static geometry; neural math and weights unchanged."""

import dataclasses
import functools
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

_installed = False


def validate_coordinates(expected, actual, name):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if (
        expected.dtype != actual.dtype
        or expected.shape != actual.shape
        or not np.array_equal(expected, actual)
    ):
        raise ValueError("Fixed reference coordinate changed: " + name)


def validate_config(expected, actual):
    if json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True):
        raise ValueError("Model/task configuration differs from frozen official graph")


def install(root, expected_sha256, audit_path):
    global _installed
    if _installed:
        raise RuntimeError("Reference graph installed twice")
    from graphcast import graphcast

    root, audit_path = Path(root), Path(audit_path)
    meta = json.loads((root / "reference-graph.json").read_text())
    path = root / "reference-graph.pkl.gz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta["file_sha256"] or digest != expected_sha256:
        raise ValueError("Reference graph payload SHA mismatch")
    source = hashlib.sha256(Path(graphcast.__file__).read_bytes()).hexdigest()
    if source != meta["source_sha256"]:
        raise ValueError("Official source changed")
    with gzip.open(path, "rb") as f:
        state = pickle.load(f)
    if sorted(state) != meta["state_keys"] or state["_initialized"] is not True:
        raise ValueError("Incomplete official static graph state")
    for graph_name in [
        "_grid2mesh_graph_structure",
        "_mesh_graph_structure",
        "_mesh2grid_graph_structure",
    ]:
        for key, edge in state[graph_name].edges.items():
            for field, value in [
                ("senders", edge.indices.senders),
                ("receivers", edge.indices.receivers),
                ("features", edge.features),
            ]:
                array = np.asarray(value)
                expected = meta["arrays"][graph_name + "/" + key.name + "/" + field]
                if (
                    list(array.shape) != expected["shape"]
                    or str(array.dtype) != expected["dtype"]
                    or hashlib.sha256(array.tobytes()).hexdigest() != expected["sha256"]
                ):
                    raise ValueError("Decoded graph array differs from reference")
    init = graphcast.GraphCast.__init__
    maybe = graphcast.GraphCast._maybe_init

    @functools.wraps(init)
    def initialize(self, model_config, task_config):
        validate_config(meta["model_config"], dataclasses.asdict(model_config))
        validate_config(meta["task_config"], dataclasses.asdict(task_config))
        init(self, model_config, task_config)
        self._graphcast_reference_config_verified = True

    @functools.wraps(maybe)
    def restore(self, sample_inputs):
        if not getattr(self, "_graphcast_reference_config_verified", False):
            raise ValueError("Missing model configuration proof")
        validate_coordinates(state["_grid_lat"], sample_inputs.lat.values, "lat")
        validate_coordinates(state["_grid_lon"], sample_inputs.lon.values, "lon")
        if not self._initialized:
            for name, value in state.items():
                setattr(self, name, value)
            record = dict(
                reference_id=meta["reference_id"],
                payload_sha256=digest,
                official_source_sha256=source,
                all_static_graph_arrays_verified=True,
                grid2mesh_edges=len(
                    state["_grid2mesh_graph_structure"]
                    .edges[next(iter(state["_grid2mesh_graph_structure"].edges))]
                    .indices.senders
                ),
                scope="Static graph from official H100 CPU construction; model weights/parameters and neural arithmetic unchanged",
            )
            with audit_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    graphcast.GraphCast.__init__ = initialize
    graphcast.GraphCast._maybe_init = restore
    _installed = True
