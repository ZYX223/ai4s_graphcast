#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--runtime-library", type=Path, required=True)
    parser.add_argument("--start-symbol", required=True)
    parser.add_argument("--stop-symbol", required=True)
    args, workload_args = parser.parse_known_args()
    if workload_args and workload_args[0] == "--":
        workload_args = workload_args[1:]
    if not workload_args:
        parser.error("workload arguments are required after --")
    return args, workload_args


def value_after(arguments: list[str], flag: str) -> str:
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"missing required workload argument: {flag}") from error


def main() -> int:
    args, workload_args = parse_args()
    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            f"source SHA-256 mismatch: {source_sha256} != {args.expected_source_sha256}"
        )
    device_text = value_after(workload_args, "--device")
    device = torch.device(device_text)
    if device.type != "cuda":
        raise RuntimeError(f"runtime profiler range requires CUDA API: {device}")

    spec = importlib.util.spec_from_file_location("profiled_dense_gemm", args.source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source: {args.source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.cuda.set_device(device)
    runtime = ctypes.CDLL(str(args.runtime_library))
    start = getattr(runtime, args.start_symbol)
    stop = getattr(runtime, args.stop_symbol)
    start.restype = ctypes.c_int
    stop.restype = ctypes.c_int
    original_argv = sys.argv
    sys.argv = [str(args.source), *workload_args]
    started = False
    try:
        start_result = int(start())
        if start_result != 0:
            raise RuntimeError(f"{args.start_symbol} failed with code {start_result}")
        started = True
        return int(module.main())
    finally:
        if started:
            torch.cuda.synchronize(device)
            stop_result = int(stop())
            if stop_result != 0:
                raise RuntimeError(f"{args.stop_symbol} failed with code {stop_result}")
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
