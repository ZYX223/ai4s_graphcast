"""Cross-check runtime GEMMs against a frozen checkpoint and optimized HLO.

Pure Python, including Python 3.6 on the host. No accelerator is initialized.
"""

import collections
import json
import re


def build_catalog(trace_rows, hlo_text, weights, label):
    """Return one-step operation shapes; reject partial or inconsistent evidence."""
    layers = {x["module"]: x for x in weights["layers"]}
    rows = [x for x in trace_rows if x.get("label") == label]
    if not rows or len(rows) != len(layers) or len(layers) != len(weights["layers"]):
        raise ValueError("Runtime step must contain every frozen layer exactly once")
    if collections.Counter(x.get("layer") for x in rows) != collections.Counter(layers.keys()):
        raise ValueError("Duplicate or missing runtime layer")
    shapes = collections.defaultdict(list)
    for row in rows:
        layer = layers[row["layer"]]
        m, n, k = (row[x] for x in ("m", "n", "k"))
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (m, n, k)):
            raise ValueError("Invalid runtime GEMM dimensions")
        if row.get("status") != 0 or row.get("unknown", False):
            raise ValueError("Unsuccessful or unknown runtime GEMM")
        if layer["shape"] != [k, m]:
            raise ValueError("Runtime GEMM does not match checkpoint dimensions")
        if row.get("weight_sha256") not in layer["sha256"].values():
            raise ValueError("Unknown runtime weight fingerprint")
        shapes[(m, n, k)].append(row["layer"])
    hlo_shapes = collections.Counter()
    for line in hlo_text.splitlines():
        if 'custom_call_target="__cublas$gemm"' not in line:
            continue
        match = re.search(r"= \(f32\[(\d+),(\d+)\]\{1,0\}", line)
        if match is None or "backend_config=" not in line:
            raise ValueError("Unsupported HLO GEMM layout")
        n, m = map(int, match.groups())
        cfg = json.loads(line.split("backend_config=", 1)[1])["gemm_backend_config"]
        lhs, rhs = int(cfg["lhs_stride"]), int(cfg["rhs_stride"])
        if lhs % n or rhs != m * (lhs // n):
            raise ValueError("Inconsistent HLO operand dimensions")
        hlo_shapes[(m, n, lhs // n)] += 1
    if hlo_shapes != collections.Counter({key: len(v) for key, v in shapes.items()}):
        raise ValueError("HLO and runtime GEMM dimensions/counts differ")
    entries = []
    for (m, n, k), names in sorted(shapes.items(), key=lambda x: (-len(x[1]), x[0])):
        entries.append(
            {
                "id": "m{}_n{}_k{}".format(m, n, k),
                "m": m,
                "n": n,
                "k": k,
                "activation_shape": [n, k],
                "weight_shape": [k, m],
                "output_shape": [n, m],
                "dtype": "float32",
                "native_layout": "column-major NN: A[m,k], B[k,n], C[m,n]",
                "alpha": 1,
                "beta": 0,
                "calls_per_step": len(names),
                "layers": names,
                "flops_per_call": 2 * m * n * k,
                "weight_bytes": 4 * m * k,
                "activation_bytes": 4 * n * k,
                "output_bytes": 4 * m * n,
                "rne_read_write_bytes_per_call": 8 * (m * k + n * k),
            }
        )
    return {
        "schema": "graphcast.operator_catalog.v1",
        "trace_label": label,
        "calls_per_step": len(rows),
        "unique_gemm_shapes": len(entries),
        "hlo_runtime_shapes_match": True,
        "gemm_shapes": entries,
        "input_policy": "Deterministic synthetic inputs; actual model shapes and frequencies. Not captured activations.",
        "timing_scope": "Standalone operator timings and frequency-weighted estimates are not model profiler timings.",
    }
