"""Summarize HIPPROF Chrome traces without mixing host and device time."""
import collections
import math
import re


def hlo_kernel_tags(text):
    bodies = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"^(?:ENTRY )?%([\w.\-]+) .*\{$", line)
        if match:
            current = match.group(1)
            bodies[current] = []
        elif line == "}":
            current = None
        elif current:
            bodies[current].append(line)
    tags = {}
    for line in text.splitlines():
        match = re.search(r"%([\w.\-]+) = .* fusion\(.*calls=%([\w.\-]+)", line)
        if not match:
            continue
        body = "\n".join(bodies.get(match.group(2), []))
        values = []
        if " scatter(" in body: values.append("scatter")
        if " gather(" in body: values.append("gather")
        if "layer_norm" in body: values.append("layer_norm")
        if "logistic(" in body or "swish" in body: values.append("activation")
        if " reduce(" in body: values.append("reduction")
        tags[re.sub(r"\W", "_", match.group(1))] = values
    return tags


def category(name, tags):
    lower = name.lower()
    if "round_rne" in lower:
        return "rne"
    if "gemm" in lower or name.startswith(("Cijk_", "Cij_")) or "tensile" in lower:
        return "gemm"
    if "scatter" in tags or "scatter" in lower:
        return "scatter_fusion"
    if "gather" in tags or "gather" in lower:
        return "gather_fusion"
    if "layer_norm" in tags:
        return "layer_norm_fusion"
    if "reduction" in tags or "reduce" in lower:
        return "other_reduction"
    return "other_device_kernel"


def union_duration(intervals):
    end = None
    total = 0.0
    for begin, finish in sorted(intervals):
        if end is None or begin >= end:
            total += finish - begin
            end = finish
        elif finish > end:
            total += finish - end
            end = finish
    return total


def summarize_events(events, hlo_text=""):
    tags = hlo_kernel_tags(hlo_text)
    kernels, host, copies = [], [], []
    for e in events:
        if e.get("ph") != "X":
            continue
        start, duration = float(e["ts"]), float(e["dur"])
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid trace timestamp/duration")
        if e.get("cat") == "HIPOPS": kernels.append(e)
        elif e.get("cat") == "HIP": host.append(e)
        elif "COPY" in e.get("cat", "").upper(): copies.append(e)
    if not kernels:
        raise ValueError("Trace contains no device kernels; host API success is insufficient")
    groups = collections.defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
    names = {}
    for e in kernels:
        name = e["name"]
        found = tags.get(name, tags.get(name.removeprefix("wrapped_") if hasattr(str, "removeprefix") else name.replace("wrapped_", "", 1), []))
        group = category(name, found)
        ms = e["dur"] / 1000
        groups[group]["calls"] += 1
        groups[group]["total_ms"] += ms
        if name not in names:
            names[name] = {"name": name, "category": group, "hlo_tags": found, "calls": 0, "total_ms": 0.0}
        names[name]["calls"] += 1
        names[name]["total_ms"] += ms
    total = sum(e["dur"] for e in kernels) / 1000
    for row in list(groups.values()) + list(names.values()):
        row["percent_of_kernel_sum"] = row["total_ms"] / total * 100 if total else 0
    host_names = collections.defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
    for e in host:
        host_names[e["name"]]["calls"] += 1
        host_names[e["name"]]["total_ms"] += e["dur"] / 1000
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
    return {
        "gpu_kernel_count": len(kernels), "gpu_kernel_sum_ms": total,
        "gpu_kernel_busy_union_ms": union_duration(intervals) / 1000,
        "gpu_kernel_span_ms": (max(x[1] for x in intervals) - min(x[0] for x in intervals)) / 1000,
        "host_api_count": len(host), "host_api_sum_ms": sum(e["dur"] for e in host) / 1000,
        "device_copy_count": len(copies), "device_copy_sum_ms": sum(e["dur"] for e in copies) / 1000,
        "categories": dict(sorted(groups.items(), key=lambda x: -x[1]["total_ms"])),
        "kernels": sorted(names.values(), key=lambda x: -x["total_ms"]),
        "host_apis": dict(sorted(host_names.items(), key=lambda x: -x[1]["total_ms"])),
        "classification": "Exclusive precedence: GEMM/RNE, then fusion containing scatter, gather, LayerNorm, other reduction, other. Fusion labels include all operations fused into that kernel; HLO tags retain overlapping semantic labels.",
        "time_units": "Trace timestamps/durations are microseconds; all report times milliseconds.",
        "limits": "Host API times can overlap device time; never add them. Kernel sum percentages describe GPU kernel work, not end-to-end latency. Busy time is the union across device kernel intervals.",
    }
