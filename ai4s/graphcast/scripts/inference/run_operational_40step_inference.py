#!/usr/bin/env python3
"""Run the operational GraphCast model as a device-resident 40-step rollout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
import xarray as xr


GRAPHCAST_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = GRAPHCAST_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    load_normalization_stats,
)
from graphcast_compat.autoregressive_inference import (  # noqa: E402
    advance_device_rollout,
    normalized_output_to_prediction,
    prepare_device_rollout,
)
from graphcast_compat.backend import (  # noqa: E402
    configure_float32_math,
    memory_text,
    reset_peak_memory,
    resolve_runtime_backend,
    synchronize as synchronize_runtime,
)


FORWARD_FLOPS = 28_857_094_475_776


@dataclass(frozen=True)
class RolloutMeasurement:
    steps: int
    warmup_calls: int
    e2e_seconds: float
    final_output: torch.Tensor
    finite: bool


def calculate_throughput(
    *,
    forward_flops: int,
    steps: int,
    e2e_seconds: float,
    peak_tflops_per_device: float,
    device_count: int,
) -> dict[str, int | float | None]:
    if forward_flops <= 0 or steps <= 0:
        raise ValueError("forward FLOPs and steps must be positive")
    if not math.isfinite(e2e_seconds) or e2e_seconds <= 0:
        raise ValueError("E2E seconds must be finite and positive")
    if device_count <= 0:
        raise ValueError("device count must be positive")
    useful_flops = int(forward_flops) * int(steps)
    effective_tflops = useful_flops / e2e_seconds / 1.0e12
    mfu = None
    if peak_tflops_per_device > 0:
        mfu = effective_tflops / (peak_tflops_per_device * device_count)
    return {
        "useful_flops": useful_flops,
        "effective_tflops": effective_tflops,
        "mfu": mfu,
    }


def run_tensor_resident_rollout(
    model: torch.nn.Module,
    prepared: Any,
    *,
    steps: int,
    warmup_calls: int,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> RolloutMeasurement:
    """Warm up outside the timer, then measure all dependent rollout steps."""

    if steps != 40:
        raise ValueError(f"formal operational rollout requires exactly 40 steps, got {steps}")
    if int(prepared.steps) != steps:
        raise ValueError(
            f"40-step rollout requires exactly 40 forcing frames, got {prepared.steps}"
        )
    if warmup_calls < 1:
        raise ValueError("at least one warmup call is required")

    state = prepared.initial_state.clone()
    first_forcing = prepared.forcing_features[0]
    with torch.inference_mode():
        for _ in range(warmup_calls):
            warmup_output = model(torch.cat((state, first_forcing), dim=-1))
            if tuple(warmup_output.shape[-1:]) != (int(prepared.output_channels),):
                raise RuntimeError("warmup output does not match the configured output channels")
            del warmup_output
        synchronize()

        started = clock()
        normalized_output: torch.Tensor | None = None
        for step_index in range(steps):
            forcing = prepared.forcing_features[step_index]
            prediction = model(torch.cat((state, forcing), dim=-1))
            final_step = step_index + 1 == steps
            normalized_output = advance_device_rollout(
                state,
                forcing,
                prediction,
                prepared,
                advance_state=not final_step,
                return_absolute_output=final_step,
            )
            del prediction
        if normalized_output is None:
            raise RuntimeError("the final rollout step did not produce an absolute output")
        finite = bool(torch.isfinite(normalized_output).all().item())
        final_output = normalized_output.detach().to(device="cpu", dtype=torch.float32)
        synchronize()
        e2e_seconds = clock() - started

    return RolloutMeasurement(
        steps=steps,
        warmup_calls=warmup_calls,
        e2e_seconds=e2e_seconds,
        final_output=final_output,
        finite=finite,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(tensor, torch.Tensor) for tensor in value.values()
            ):
                return value
    raise ValueError("checkpoint does not contain a supported model state dictionary")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--rollout-forcings", type=Path, required=True)
    parser.add_argument("--target-template", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--forward-flops", type=int, default=FORWARD_FLOPS)
    parser.add_argument("--peak-tflops-per-device", type=float, default=0.0)
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"report already exists: {args.report}")
    for path in (
        args.config,
        args.weights,
        args.inputs,
        args.rollout_forcings,
        args.target_template,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    runtime = resolve_runtime_backend(args.device)
    float32_math_mode = configure_float32_math(False, runtime.profile.device_type)
    config = GoogleSmallConfig.from_yaml(args.config)
    inputs = xr.open_dataset(args.inputs).load()
    forcings = xr.open_dataset(args.rollout_forcings).load()
    target_template = xr.open_dataset(args.target_template).load()
    stats = load_normalization_stats(args.stats_dir)

    model = GoogleSmallCompatibleModel(config)
    model.load_state_dict(
        _select_state_dict(torch.load(args.weights, map_location="cpu")),
        strict=True,
    )
    model.init_graph(
        latitudes=inputs.coords["lat"].values,
        longitudes=inputs.coords["lon"].values,
    )
    model.to(runtime.device)
    model.eval()
    prepared = prepare_device_rollout(
        inputs,
        target_template,
        forcings,
        config=config,
        stats=stats,
        device=runtime.device,
    )

    reset_peak_memory(runtime)
    measurement = run_tensor_resident_rollout(
        model,
        prepared,
        steps=args.steps,
        warmup_calls=args.warmup_calls,
        synchronize=lambda: synchronize_runtime(runtime),
    )
    prediction = normalized_output_to_prediction(
        measurement.final_output,
        target_template,
        stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_netcdf(args.output)
    metrics = calculate_throughput(
        forward_flops=args.forward_flops,
        steps=args.steps,
        e2e_seconds=measurement.e2e_seconds,
        peak_tflops_per_device=args.peak_tflops_per_device,
        device_count=args.device_count,
    )
    report = {
        "schema": "graphcast.operational_40step.v1",
        "status": "passed" if measurement.finite else "failed_nonfinite",
        "source": {
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "weights": str(args.weights.resolve()),
            "weights_sha256": _sha256(args.weights),
            "inputs_sha256": _sha256(args.inputs),
            "rollout_forcings_sha256": _sha256(args.rollout_forcings),
            "target_template_sha256": _sha256(args.target_template),
        },
        "runtime": runtime.to_metadata(),
        "precision": {
            "requested": args.precision,
            "storage_dtype": "float32",
            "float32_math_mode": float32_math_mode,
            "tensor_core_enabled": False,
        },
        "workload": {
            "steps": measurement.steps,
            "warmup_calls": measurement.warmup_calls,
            "warmup_excluded_from_e2e": True,
            "final_output_d2h_included_in_e2e": True,
            "device_count": args.device_count,
        },
        "measurement": {
            "e2e_seconds": measurement.e2e_seconds,
            "forward_flops_per_step": args.forward_flops,
            **metrics,
            "peak_tflops_per_device": args.peak_tflops_per_device,
            "finite": measurement.finite,
            "memory": memory_text(runtime),
        },
        "output": str(args.output.resolve()),
    }
    _write_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if measurement.finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
