"""Check that JAX captured GPU work for every requested formal interval."""

import gzip
import json
import math
from pathlib import Path


def validate_jax_trace(root, expected_labels):
    files = sorted(Path(root).rglob("*.trace.json.gz"))
    if not files:
        raise ValueError("Missing JAX trace")
    captured = {}
    for path in files:
        with gzip.open(str(path), "rt") as stream:
            events = json.load(stream)["traceEvents"]
        devices = {e["pid"] for e in events if e.get("ph") == "M"
                   and e.get("name") == "process_name"
                   and e.get("args", {}).get("name", "").startswith("/device:GPU:")}
        if len(devices) != 1:
            raise ValueError("Missing or ambiguous GPU in JAX trace")
        ranges = [e for e in events if e.get("ph") == "X"
                  and e.get("name", "").startswith(("train-update-", "graphcast-inference-"))]
        kernels = [e for e in events if e.get("ph") == "X" and e.get("pid") in devices
                   and not e.get("name", "").lower().startswith(("memcpy", "memset"))]
        for interval in ranges:
            label = interval["name"]
            start, duration = interval["ts"], interval["dur"]
            if label in captured or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Duplicate or invalid formal trace range")
            count = sum(1 for e in kernels if start <= e["ts"]
                        and e["ts"] + e["dur"] <= start + duration and e["dur"] > 0)
            if not count:
                raise ValueError("Formal interval has no GPU kernels: " + label)
            captured[label] = count
    if set(captured) != set(expected_labels):
        raise ValueError("JAX trace labels differ from requested formal intervals")
    return dict(status="passed", kernel_counts=captured)
