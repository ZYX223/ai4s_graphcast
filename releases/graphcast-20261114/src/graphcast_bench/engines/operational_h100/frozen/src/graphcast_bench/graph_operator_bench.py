"""Independent graph operations with official graph topology and generated features."""

import ctypes as C
import gc
import hashlib
import os
import statistics
import time
from pathlib import Path

import numpy as np

from .assets import ROOT, read_json, write_json


def make_operation(kind, node_count=None):
    import haiku as hk
    import jax

    if kind == "gather":
        return jax.jit(lambda x, indices: x[indices])
    if kind == "scatter":
        return jax.jit(lambda x, indices: jax.ops.segment_sum(x, indices, num_segments=node_count))
    if kind == "layer_norm":

        @hk.transform
        def norm(x, scale, offset):
            return hk.LayerNorm(axis=-1, create_scale=False, create_offset=False, eps=1e-5)(
                x, scale=scale, offset=offset
            )

        return jax.jit(lambda x, scale, offset: norm.apply({}, None, x, scale, offset))
    raise ValueError("Unknown operation: " + kind)


def device_identity():
    import jax

    hip = C.CDLL("libamdhip64.so")
    bus = C.create_string_buffer(64)
    assert hip.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
    assert bus.value.decode().lower() == os.environ["GRAPHCAST_EXPECTED_PCI"].lower()
    devices = jax.devices("gpu")
    expected = read_json(ROOT / "configs/hygon.json")["expected_device"]
    if len(devices) != 1 or expected not in devices[0].device_kind:
        raise RuntimeError("Expected exactly one configured accelerator")
    return {"device_name": devices[0].device_kind, "pci": bus.value.decode()}


def run_graph_operators(data, output, repeats, iterations):
    import jax
    import jax.numpy as jnp
    import xarray as xr
    from graphcast import checkpoint, graphcast, grid_mesh_connectivity, icosahedral_mesh

    output = Path(output)
    output.mkdir()
    identity = device_identity()
    with (Path(data) / "assets/jax_weights.npz").open("rb") as f:
        weights = checkpoint.load(f, graphcast.CheckPoint)
    cfg = weights.model_config
    del weights
    with xr.open_dataset(Path(data) / "assets/inputs.nc") as ds:
        lat, lon = np.asarray(ds.lat, np.float32), np.asarray(ds.lon, np.float32)
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=cfg.mesh_size)
    finest = meshes[-1]
    ngrid, nmesh = len(lat) * len(lon), len(finest.vertices)
    radius = graphcast._get_max_edge_distance(finest) * cfg.radius_query_fraction_edge_length
    print("Building official graph indices", flush=True)
    gsend, grecv = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=lat, grid_longitude=lon, mesh=finest, radius=radius
    )
    msend, mrecv = icosahedral_mesh.faces_to_edges(icosahedral_mesh.merge_meshes(meshes).faces)
    drecv, dsend = grid_mesh_connectivity.in_mesh_triangle_indices(
        grid_latitude=lat, grid_longitude=lon, mesh=finest
    )
    graphs = [
        ("grid2mesh", gsend, grecv, ngrid, nmesh, 1618745),
        ("mesh", msend, mrecv, nmesh, nmesh, 327660),
        ("mesh2grid", dsend, drecv, nmesh, ngrid, 3114720),
    ]
    topology = []
    for name, senders, receivers, ns, nr, expected_edges in graphs:
        assert len(senders) == len(receivers) == expected_edges
        assert (
            min(senders) >= 0 and max(senders) < ns and min(receivers) >= 0 and max(receivers) < nr
        )
        topology.append(
            {
                "name": name,
                "senders": ns,
                "receivers": nr,
                "edges": len(senders),
                "sender_sha256": hashlib.sha256(senders.astype(np.int32).tobytes()).hexdigest(),
                "receiver_sha256": hashlib.sha256(receivers.astype(np.int32).tobytes()).hexdigest(),
            }
        )
    np.savez(
        output / "graph-indices.npz",
        **{
            name + "_" + key: value.astype(np.int32)
            for name, s, r, *_ in graphs
            for key, value in (("senders", s), ("receivers", r))
        },
    )
    write_json(output / "topology.json", topology)
    rows = []

    def measure(name, kind, x, arguments, indices=None, node_count=None):
        fn = make_operation(kind, node_count)
        values = (x,) + tuple(arguments)
        y = fn(*values)
        jax.block_until_ready(y)
        sample = np.linspace(0, y.shape[0] - 1, 32, dtype=np.int32)
        actual = np.asarray(y[sample])
        if kind == "gather":
            expected = np.asarray(x[np.asarray(indices)[sample]])
            np.testing.assert_array_equal(actual, expected)
        elif kind == "scatter":
            expected = np.stack(
                [
                    np.asarray(x[np.flatnonzero(indices == i)], dtype=np.float64).sum(axis=0)
                    for i in sample
                ]
            )
            np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=5e-4)
        else:
            host = np.asarray(x[sample], dtype=np.float64)
            scale, offset = [np.asarray(v, dtype=np.float64) for v in arguments]
            expected = (host - host.mean(-1, keepdims=True)) / np.sqrt(
                host.var(-1, keepdims=True) + 1e-5
            )
            expected = expected * scale + offset
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        np.savez(output / (name + "-samples.npz"), rows=sample, actual=actual, expected=expected)
        (output / (name + "-optimized-hlo.txt")).write_text(fn.lower(*values).compile().as_text())
        cache = fn._cache_size()
        times = []
        for _ in range(repeats):
            for _ in range(5):
                jax.block_until_ready(fn(*values))
            samples = []
            for _ in range(iterations):
                start = time.perf_counter()
                y = fn(*values)
                jax.block_until_ready(y)
                samples.append((time.perf_counter() - start) * 1000)
            times.append(samples)
        assert fn._cache_size() == cache
        result = {
            "name": name,
            "operation": kind,
            "input_shape": list(x.shape),
            "output_shape": list(y.shape),
            "dtype": str(x.dtype),
            "round_samples_ms": times,
            "round_median_ms": [statistics.median(s) for s in times],
            "median_ms": statistics.median(v for s in times for v in s),
            "sample_rows_checked": 32,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "status": "passed",
            "jit_cache_unchanged": True,
        }
        rows.append(result)
        write_json(output / "report.json", {"status": "running", "rows": rows})
        print(name, result["median_ms"], "ms", flush=True)

    # Synthesized features; topology and dimensions are taken from the official model.
    for index, (name, senders, receivers, ns, nr, _) in enumerate(graphs):
        for side, ids, nodes in (("senders", senders, ns), ("receivers", receivers, nr)):
            x = jax.random.normal(
                jax.random.PRNGKey(20261022 + index), (nodes, 512), dtype=jnp.float32
            )
            placed = jax.device_put(ids.astype(np.int32))
            jax.block_until_ready((x, placed))
            measure(name + "-gather-" + side, "gather", x, (placed,), indices=ids)
            del x, placed
            gc.collect()
        x = jax.random.normal(
            jax.random.PRNGKey(20261122 + index), (len(receivers), 512), dtype=jnp.float32
        )
        placed = jax.device_put(receivers.astype(np.int32))
        jax.block_until_ready((x, placed))
        measure(name + "-scatter", "scatter", x, (placed,), indices=receivers, node_count=nr)
        del x, placed
        gc.collect()
    for nodes in sorted({ngrid, nmesh, len(gsend), len(msend), len(dsend)}):
        x = jax.random.normal(jax.random.PRNGKey(20261022), (nodes, 512), dtype=jnp.float32)
        scale, offset = jnp.linspace(0.8, 1.2, 512), jnp.linspace(-0.1, 0.1, 512)
        jax.block_until_ready((x, scale, offset))
        measure("layer-norm-" + str(nodes), "layer_norm", x, (scale, offset))
        del x, scale, offset
        gc.collect()
    report = {
        "status": "passed",
        "rows": rows,
        "topology": topology,
        **identity,
        "input_policy": "Official graph indices; seeded JAX normal FP32 features, affine LayerNorm parameters synthesized. Batch=1 is flattened.",
        "timing_scope": "Synchronized JAX wall time; compile, input generation/transfer and validation excluded. Independent operations do not reproduce the model's fused kernels.",
    }
    write_json(output / "report.json", report)
    return report
