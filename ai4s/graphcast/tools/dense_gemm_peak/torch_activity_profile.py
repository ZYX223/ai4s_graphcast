#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--activity", choices=("cuda", "privateuse1"), required=True)
    args, workload_args = parser.parse_known_args()
    if workload_args and workload_args[0] == "--":
        workload_args = workload_args[1:]
    if not workload_args:
        parser.error("workload arguments are required after --")
    return args, workload_args


def main() -> int:
    args, workload_args = parse_args()
    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            f"source SHA-256 mismatch: {source_sha256} != {args.expected_source_sha256}"
        )
    for path in (args.trace, args.summary):
        if path.exists():
            raise RuntimeError(f"output already exists: {path}")

    spec = importlib.util.spec_from_file_location("profiled_dense_gemm", args.source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source: {args.source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    device_activity = {
        "cuda": torch.profiler.ProfilerActivity.CUDA,
        "privateuse1": torch.profiler.ProfilerActivity.PrivateUse1,
    }[args.activity]
    original_argv = sys.argv
    sys.argv = [str(args.source), *workload_args]
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                device_activity,
            ],
            record_shapes=True,
        ) as profile:
            return_code = int(module.main())
    finally:
        sys.argv = original_argv
    if return_code != 0:
        raise RuntimeError(f"workload returned {return_code}")

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(args.trace))
    events = []
    for event in profile.key_averages():
        events.append(
            {
                "key": event.key,
                "count": int(event.count),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_device_time_total_us": float(event.self_device_time_total),
                "device_time_total_us": float(event.device_time_total),
            }
        )
    payload = {
        "schema": "dense_gemm_torch_activity_profile.v1",
        "source": {"path": str(args.source), "sha256": source_sha256},
        "activities": ["CPU", args.activity],
        "workload_argv": workload_args,
        "events": events,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
