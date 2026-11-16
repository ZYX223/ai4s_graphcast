"""Validate complete physical GPU streams from XSpace, independent of Chrome caps."""
import hashlib
import importlib.util
import re
from pathlib import Path

_XSPACE = None


def xspace_type():
    """Load only the protobuf schema; importing TensorFlow could initialize a GPU."""
    global _XSPACE
    if _XSPACE is None:
        spec = importlib.util.find_spec("tensorflow")
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("Pinned BW environment lacks the XSpace protobuf schema")
        root = Path(next(iter(spec.submodule_search_locations)))
        path = root / "tsl/profiler/protobuf/xplane_pb2.py"
        module_spec = importlib.util.spec_from_file_location(
            "graphcast_xplane_pb2", str(path)
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        _XSPACE = module.XSpace
    return _XSPACE


def interval_union(intervals):
    end, total = None, 0
    for start, stop in sorted(intervals):
        if stop <= start:
            raise ValueError("Invalid GPU compute duration")
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def validate_training_trace(root, expected_labels, expected_gemm_calls=None):
    files = sorted(Path(root).rglob("*.xplane.pb"))
    if len(files) != 1:
        raise ValueError("Expected one complete XSpace trace per training stage")
    data = files[0].read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    space = xspace_type()()
    space.ParseFromString(data)
    del data
    if space.errors or any(
        re.search(r"drop|lost|truncat", warning, re.I) for warning in space.warnings
    ):
        raise ValueError(
            "XSpace reports incomplete capture: "
            + repr(list(space.errors) + list(space.warnings))
        )
    devices = [p for p in space.planes if p.name.startswith("/device:GPU:")]
    if len(devices) != 1:
        raise ValueError("Expected one physical GPU in XSpace")
    ranges = {}
    for plane in space.planes:
        if plane.name != "/host:CPU":
            continue
        labels = {
            key: value.name
            for key, value in plane.event_metadata.items()
            if value.name.startswith(("train-update-", "graphcast-inference-"))
        }
        for line in plane.lines:
            for event in line.events:
                if event.metadata_id not in labels:
                    continue
                label = labels[event.metadata_id]
                if label in ranges or event.duration_ps <= 0:
                    raise ValueError("Duplicate or invalid formal XSpace range")
                start = line.timestamp_ns * 1000 + event.offset_ps
                ranges[label] = (start, start + event.duration_ps)
    if set(ranges) != set(expected_labels):
        raise ValueError("XSpace labels differ from requested formal updates")
    previous_end = None
    for label in expected_labels:
        start, stop = ranges[label]
        if previous_end is not None and start < previous_end:
            raise ValueError(
                "XSpace formal ranges must be chronological and nonoverlapping"
            )
        previous_end = stop
    plane = devices[0]
    kernels = []
    for line in plane.lines:
        if not line.name.startswith("Stream #"):
            continue  # Derived XLA/op timelines duplicate physical execution.
        for event in line.events:
            name = plane.event_metadata[event.metadata_id].name
            if name.lower().startswith(("memcpy", "memset")):
                continue
            start = line.timestamp_ns * 1000 + event.offset_ps
            kernels.append((start, start + event.duration_ps, name))
    counts, gemms, auxiliaries, active = {}, {}, {}, {}
    for label, (lo, hi) in ranges.items():
        selected = [(a, b, n) for a, b, n in kernels if lo <= a and b <= hi]
        if not selected:
            raise ValueError("Formal interval has no physical GPU kernels: " + label)
        counts[label] = len(selected)
        # The pinned rocBLAS split-K path launches PostGSU after its main GEMM.
        # Count the native GEMM once, but retain every helper in active GPU time.
        auxiliaries[label] = sum(n == "Cijk_S_PostGSU" for a, b, n in selected)
        gemms[label] = sum(
            n != "Cijk_S_PostGSU"
            and (n.startswith("Cijk")
                 or any(token in n.lower() for token in ("gemm", "matmul")))
            for a, b, n in selected
        )
        active[label] = interval_union([(a, b) for a, b, n in selected]) / 1e12
    if expected_gemm_calls is not None:
        expected = dict(zip(expected_labels, expected_gemm_calls))
        if len(expected_gemm_calls) != len(expected_labels) or gemms != expected:
            raise ValueError(
                "XSpace GEMM launches differ from the per-update native call audit: "
                + repr(dict(trace=gemms, native=expected))
            )
    return dict(
        status="passed",
        source="xspace_physical_streams",
        trace=str(files[0]),
        trace_sha256=digest,
        kernel_counts=counts,
        gemm_counts=gemms,
        gemm_auxiliary_counts=auxiliaries,
        gpu_active_seconds=active,
        warnings=list(space.warnings),
        scope="Formal compute kernel interval union; copies, sets, derived timelines and between-update checkpoint work excluded. Not end-to-end time.",
    )


def validate_legacy_trace(root, labels):
    """Keep historical basic checks, but reject known capped Chrome exports."""
    import gzip
    import json
    from .traces import validate_jax_trace

    for path in Path(root).rglob("*.trace.json.gz"):
        with gzip.open(str(path), "rt") as stream:
            if len(json.load(stream)["traceEvents"]) >= 1000000:
                raise ValueError(
                    "Chrome trace may be truncated; original XSpace is required"
                )
    return validate_jax_trace(root, labels)
