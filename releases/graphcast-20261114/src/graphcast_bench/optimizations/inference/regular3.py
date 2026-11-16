"""One strict inference-only substitution; official files are never modified."""

import hashlib
import json

import numpy as np

_audit = []
_installed = False


def validate_regular3(segment_ids, num_segments):
    if not isinstance(segment_ids, np.ndarray) or segment_ids.ndim != 1:
        raise ValueError("Aggregation requires a static one-dimensional numpy index array")
    if segment_ids.dtype.kind not in "iu" or len(segment_ids) != 3 * num_segments:
        raise ValueError("Index dtype/count does not prove three complete incoming edges")
    if not np.all(segment_ids.reshape(num_segments, 3) == np.arange(num_segments)[:, None]):
        raise ValueError("Indices must be exactly 0,0,0,1,1,1,... through the final segment")


def regular3_sum(data, num_segments, xp):
    if data.shape[0] != 3 * num_segments:
        raise ValueError("Message count does not match segment count")
    return xp.sum(xp.reshape(data, (num_segments, 3) + data.shape[1:]), axis=1)


def install():
    global _installed
    if _installed:
        raise RuntimeError("Candidate hook already installed")
    import jax.numpy as jnp
    from graphcast import deep_typed_graph_net

    original_get = deep_typed_graph_net._get_aggregate_edges_for_nodes_fn

    def get(name):
        original = original_get(name)
        if name != "segment_sum":
            return original

        def aggregate(data, segment_ids, num_segments, *args, **kwargs):
            # Only the fixed Operational 0.25deg mesh2grid receiver aggregation.
            # Other mesh/grid directions retain their original jraph function.
            if int(num_segments) == 1038240 and data.shape[0] == 3114720:
                if (
                    args
                    or kwargs
                    or data.shape != (3114720, 1, 512)
                    or str(data.dtype) != "float32"
                ):
                    raise ValueError(
                        "Unexpected batch/channel/dtype/arguments for fixed mesh2grid case"
                    )
                validate_regular3(segment_ids, int(num_segments))
                result = regular3_sum(data, int(num_segments), jnp)
                assert result.shape == (1038240, 1, 512) and str(result.dtype) == "float32"
                _audit.append(
                    dict(
                        selected=True,
                        input_shape=list(data.shape),
                        output_shape=list(result.shape),
                        num_segments=int(num_segments),
                        batch_axis_preserved=True,
                        complete_ordered_coverage=True,
                        indices_sha256=hashlib.sha256(segment_ids.tobytes()).hexdigest(),
                        dtype=str(data.dtype),
                    )
                )
                return result
            _audit.append(
                dict(
                    selected=False,
                    input_shape=list(data.shape),
                    num_segments=int(num_segments),
                    path="original_jraph_segment_sum",
                )
            )
            return original(data, segment_ids, num_segments, *args, **kwargs)

        return aggregate

    deep_typed_graph_net._get_aggregate_edges_for_nodes_fn = get
    _installed = True


def save_audit(path):
    selected = [row for row in _audit if row["selected"]]
    if not _installed or not selected:
        raise RuntimeError("Required fixed mesh2grid substitution did not execute during tracing")
    path.write_text(
        json.dumps(
            dict(
                status="verified",
                selected_trace_calls=len(selected),
                scope="Only static mesh2grid receivers, all three incoming messages preserved; other aggregation unchanged.",
                entries=_audit,
            ),
            indent=2,
        )
        + "\n"
    )
