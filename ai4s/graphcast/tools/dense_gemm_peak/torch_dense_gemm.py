#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

MODES = ("strict_fp32", "fast_fp32", "bf16")
NRMSE_LIMITS = {
    "strict_fp32": 1.0e-5,
    "fast_fp32": 2.0e-3,
    "bf16": 2.0e-2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 3:
        parser.error("warmup must be >=1 and iterations must be >=3")
    if len(set(args.sizes)) != len(args.sizes) or any(n < 256 for n in args.sizes):
        parser.error("sizes must be unique integers >=256")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "musa":
        torch.musa.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        raise RuntimeError(f"unsupported accelerator device type: {device.type}")


def device_api(device: torch.device) -> Any:
    if device.type == "npu":
        import torch_npu  # noqa: F401

        return torch.npu
    if device.type == "musa":
        import torch_musa  # noqa: F401

        return torch.musa
    if device.type == "cuda":
        return torch.cuda
    raise RuntimeError(f"unsupported accelerator device type: {device.type}")


def configure_precision(mode: str, device: torch.device) -> dict[str, Any]:
    fast = mode == "fast_fp32"
    evidence: dict[str, Any] = {
        "requested_mode": mode,
        "fast_fp32_requested": fast,
        "controls": {},
    }

    if mode == "bf16":
        evidence["status"] = "bf16_dtype_selected"
        return evidence

    if device.type == "npu":
        import torch_npu  # noqa: F401

        matmul = getattr(torch.npu, "matmul", None)
        conv = getattr(torch.npu, "conv", None)
        if matmul is None or not hasattr(matmul, "allow_hf32"):
            raise RuntimeError("torch.npu.matmul.allow_hf32 is unavailable")
        if conv is None or not hasattr(conv, "allow_hf32"):
            raise RuntimeError("torch.npu.conv.allow_hf32 is unavailable")
        matmul.allow_hf32 = fast
        conv.allow_hf32 = fast
        evidence["controls"] = {
            "torch.npu.matmul.allow_hf32": bool(matmul.allow_hf32),
            "torch.npu.conv.allow_hf32": bool(conv.allow_hf32),
        }
        if any(value is not fast for value in evidence["controls"].values()):
            raise RuntimeError("Ascend HF32 control read-back mismatch")
        evidence["status"] = "hf32_enabled" if fast else "hf32_disabled"
        return evidence

    if device.type == "musa":
        import torch_musa  # noqa: F401

        mudnn = getattr(torch.backends, "mudnn", None)
        if mudnn is None or not hasattr(mudnn, "allow_tf32"):
            raise RuntimeError("torch.backends.mudnn.allow_tf32 is unavailable")
        mudnn.allow_tf32 = fast
        value = bool(mudnn.allow_tf32)
        evidence["controls"] = {"torch.backends.mudnn.allow_tf32": value}
        if value is not fast:
            raise RuntimeError("MUSA TF32 control read-back mismatch")
        evidence["status"] = "tf32_enabled" if fast else "tf32_disabled"
        return evidence

    if device.type == "cuda":
        matmul = getattr(torch.backends.cuda, "matmul", None)
        if matmul is None:
            raise RuntimeError("torch.backends.cuda.matmul is unavailable")
        # Hygon's HIP-compatible PyTorch exposes the new property but its
        # current BLAS bridge still consumes the legacy flag.  Selecting both
        # APIs makes the runtime reject the GEMM as an ambiguous request.
        use_new_cuda_control = getattr(torch.version, "hip", None) is None and hasattr(
            matmul, "fp32_precision"
        )
        if use_new_cuda_control:
            requested = "tf32" if fast else "ieee"
            matmul.fp32_precision = requested
            value = str(matmul.fp32_precision)
            evidence["controls"] = {
                "torch.backends.cuda.matmul.fp32_precision": value,
            }
            if value != requested:
                raise RuntimeError("CUDA FP32 precision control read-back mismatch")
        elif hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = fast
            value = bool(matmul.allow_tf32)
            evidence["controls"] = {
                "torch.backends.cuda.matmul.allow_tf32": value,
            }
            if value is not fast:
                raise RuntimeError("CUDA-compatible TF32 control read-back mismatch")
        else:
            raise RuntimeError("CUDA-compatible TF32 control is unavailable")
        evidence["status"] = "tf32_enabled" if fast else "tf32_disabled"
        return evidence

    raise RuntimeError(f"unsupported accelerator device type: {device.type}")


def scalar(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def runtime_identity(device: torch.device) -> dict[str, Any]:
    api = device_api(device)
    props = api.get_device_properties(device)
    versions = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_musa": getattr(torch.version, "musa", None),
        "torch_npu": None,
    }
    try:
        import torch_npu

        versions["torch_npu"] = getattr(torch_npu, "__version__", None)
    except ImportError:
        pass
    prop_fields = {}
    for name in ("name", "total_memory", "major", "minor", "gcnArchName"):
        if hasattr(props, name):
            prop_fields[name] = scalar(getattr(props, name))
    return {
        "device": str(device),
        "device_type": device.type,
        "device_count_visible": int(api.device_count()),
        "device_name": scalar(api.get_device_name(device)),
        "properties": prop_fields,
        "versions": versions,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def bootstrap_median_ci(values: list[float], *, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    boot = []
    for _ in range(5000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(statistics.median(sample))
    boot.sort()
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def summarize(values: list[float], *, seed: int) -> dict[str, float]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(value - median) for value in values])
    ci_low, ci_high = bootstrap_median_ci(values, seed=seed)
    return {
        "count": len(values),
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean if mean else math.inf,
        "median": median,
        "mad": mad,
        "median_ci95_low": ci_low,
        "median_ci95_high": ci_high,
        "min": min(values),
        "max": max(values),
    }


def correctness(mode: str, device: torch.device, dtype: torch.dtype, seed: int) -> dict[str, Any]:
    n = 256
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a64 = torch.randn((n, n), generator=generator, dtype=torch.float64)
    b64 = torch.randn((n, n), generator=generator, dtype=torch.float64)
    ref = a64 @ b64
    a = a64.to(device=device, dtype=dtype)
    b = b64.to(device=device, dtype=dtype)
    out = torch.mm(a, b)
    synchronize(device)
    actual = out.to(device="cpu").to(dtype=torch.float64)
    diff = actual - ref
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    ref_rms = float(torch.sqrt(torch.mean(ref * ref)).item())
    normalized_rmse = rmse / max(ref_rms, sys.float_info.min)
    result = {
        "n": n,
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(torch.max(torch.abs(diff)).item()),
        "rmse": rmse,
        "reference_rms": ref_rms,
        "normalized_rmse": normalized_rmse,
        "limit": NRMSE_LIMITS[mode],
        "passed": normalized_rmse <= NRMSE_LIMITS[mode],
    }
    del a, b, out, actual
    return result


def benchmark_size(
    *,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    api = device_api(device)
    api.manual_seed_all(seed + n)
    a = torch.empty((n, n), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b = torch.empty((n, n), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    out = torch.empty((n, n), device=device, dtype=dtype)
    for _ in range(warmup):
        torch.mm(a, b, out=out)
    synchronize(device)

    seconds = []
    tflops = []
    flop_count = 2 * n * n * n
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter_ns()
        torch.mm(a, b, out=out)
        synchronize(device)
        elapsed = (time.perf_counter_ns() - start) / 1.0e9
        seconds.append(elapsed)
        tflops.append(flop_count / elapsed / 1.0e12)

    finite = bool(torch.isfinite(out).all().item())
    checksum = float(out[:16, :16].to(device="cpu").to(torch.float64).sum().item())
    result = {
        "n": n,
        "flop_count": flop_count,
        "warmup": warmup,
        "iterations": iterations,
        "dtype": str(dtype),
        "output_finite": finite,
        "output_sample_checksum": checksum,
        "seconds": seconds,
        "tflops": tflops,
        "seconds_summary": summarize(seconds, seed=seed + n),
        "tflops_summary": summarize(tflops, seed=seed + n + 1),
    }
    del a, b, out
    api.empty_cache()
    synchronize(device)
    return result


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    api = device_api(device)
    api.set_device(device)
    dtype = torch.bfloat16 if args.mode == "bf16" else torch.float32
    precision = configure_precision(args.mode, device)
    identity = runtime_identity(device)
    correctness_result = correctness(args.mode, device, dtype, args.seed)
    if not correctness_result["finite"] or not correctness_result["passed"]:
        raise RuntimeError(f"correctness gate failed: {correctness_result}")

    size_results = []
    for n in args.sizes:
        size_results.append(
            benchmark_size(
                n=n,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
            )
        )
        if not size_results[-1]["output_finite"]:
            raise RuntimeError(f"non-finite output at n={n}")

    source_path = Path(__file__).resolve()
    payload = {
        "schema": "dense_gemm_runtime_roof.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "device": args.device,
        "dtype": str(dtype),
        "precision": precision,
        "runtime_identity": identity,
        "correctness": correctness_result,
        "sizes": size_results,
        "source": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
