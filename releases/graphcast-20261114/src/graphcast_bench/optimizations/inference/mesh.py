"""Fixed mesh receiver sum via verified complete gather groups, inference only."""

import hashlib
import json

import numpy as np

EXPECTED = "f91c5061f868a234cb0f0e8cc3651f856c4cfe37967b9e50ac92a2a211819b5a"
_audit = []
_cache = {}


def build_plan(ids, n):
    if (
        not isinstance(ids, np.ndarray)
        or ids.dtype != np.int32
        or ids.shape != (327660,)
        or n != 40962
    ):
        raise ValueError("Unexpected fixed mesh index shape/dtype")
    sha = hashlib.sha256(ids.tobytes()).hexdigest()
    if sha != EXPECTED:
        raise ValueError("Actual mesh receiver indices differ from inspected topology")
    if sha in _cache:
        return _cache[sha]
    degree = np.bincount(ids, minlength=n)
    cuts = np.r_[0, np.flatnonzero(degree[1:] != degree[:-1]) + 1, n]
    order = np.argsort(ids, kind="stable").astype(np.int32)
    starts = np.r_[0, np.cumsum(degree)]
    groups = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = int(degree[a])
        idx = order[starts[a] : starts[b]].reshape(b - a, k)
        if not np.all(ids[idx] == np.arange(a, b)[:, None]):
            raise ValueError("Receiver group coverage failed")
        groups.append((int(a), int(b), k, idx))
    if not np.array_equal(
        np.sort(np.concatenate([x[3].ravel() for x in groups])), np.arange(len(ids))
    ):
        raise ValueError("Each incoming edge must appear exactly once")
    _cache[sha] = groups
    return groups


def grouped_sum(data, groups):
    import jax.numpy as jnp

    return jnp.concatenate([jnp.sum(data[g[3]], axis=1) for g in groups], axis=0)


def install():
    from graphcast import deep_typed_graph_net

    original_get = deep_typed_graph_net._get_aggregate_edges_for_nodes_fn

    def get(name):
        original = original_get(name)
        if name != "segment_sum":
            return original

        def aggregate(data, segment_ids, num_segments, *args, **kwargs):
            eligible = (
                int(num_segments) == 40962
                and data.shape == (327660, 1, 512)
                and isinstance(segment_ids, np.ndarray)
            )
            sha = hashlib.sha256(segment_ids.tobytes()).hexdigest() if eligible else None
            if sha == EXPECTED:
                if args or kwargs or str(data.dtype) != "float32":
                    raise ValueError("Unexpected fixed mesh aggregation arguments")
                groups = build_plan(segment_ids, int(num_segments))
                out = grouped_sum(data, groups)
                assert out.shape == (40962, 1, 512)
                _audit.append(
                    dict(
                        selected=True,
                        indices_sha256=sha,
                        input_shape=list(data.shape),
                        output_shape=list(out.shape),
                        all_edges_exactly_once=True,
                        groups=[dict(start=a, stop=b, degree=k) for a, b, k, idx in groups],
                    )
                )
                return out
            return original(data, segment_ids, num_segments, *args, **kwargs)

        return aggregate

    deep_typed_graph_net._get_aggregate_edges_for_nodes_fn = get


def save_audit(path):
    if len(_audit) != 16:
        raise RuntimeError(
            "Expected exactly 16 fixed mesh receiver replacements, got " + str(len(_audit))
        )
    path.write_text(
        json.dumps(
            dict(
                status="traced",
                expected_replacements=16,
                actual_replacements=len(_audit),
                entries=_audit,
                scope="Only exact fixed mesh receiver topology; no dropped/repeated edges; FP32 reduction order changes",
            ),
            indent=2,
        )
        + "\n"
    )
