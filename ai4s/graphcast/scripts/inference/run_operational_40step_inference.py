#!/usr/bin/env python3
"""Run the operational GraphCast model as a device-resident 40-step rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
    require_complete_flaggems_dispatch,
    reset_peak_memory,
    resolve_runtime_backend,
    suspend_prevalidated_flaggems_index_add_bounds,
)
from graphcast_compat.backend import (  # noqa: E402
    synchronize as synchronize_runtime,
)
from graphcast_compat.inference_precision import (  # noqa: E402
    InferencePrecisionPolicy,
    resolve_inference_precision,
    run_precision_forward,
)

FORWARD_FLOPS = 28_857_094_475_776


@dataclass(frozen=True)
class RolloutDtypeEvidence:
    state: str
    forcing: str
    model_input: str
    model_output: str
    feedback: str
    final_output: str


@dataclass(frozen=True)
class RolloutMeasurement:
    steps: int
    warmup_calls: int
    e2e_seconds: float
    final_output: torch.Tensor
    finite: bool
    dtype_evidence: RolloutDtypeEvidence


@dataclass(frozen=True)
class ModelExecutionProfile:
    name: str
    platform: str
    runtime_backend: str
    edge_input_projection: str
    grid2mesh_edge_chunk_size: int | None
    mesh2grid_edge_chunk_size: int | None
    mesh2grid_full_grid_fast_path: bool
    index_add_execution: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "platform": self.platform,
            "runtime_backend": self.runtime_backend,
            "configuration_overrides": {
                "edge_input_projection": self.edge_input_projection,
                "grid2mesh_edge_chunk_size": self.grid2mesh_edge_chunk_size,
                "mesh2grid_edge_chunk_size": self.mesh2grid_edge_chunk_size,
                "mesh2grid_full_grid_fast_path": self.mesh2grid_full_grid_fast_path,
                "index_add_execution": self.index_add_execution,
            },
        }


DOMESTIC_PLATFORMS = frozenset(("c550", "bw3000", "s5000", "910c", "810e"))
PLATFORM_RUNTIME_BACKENDS = {
    "h100": "nvidia",
    "c550": "metax",
    "bw3000": "hygon",
    "s5000": "mthreads",
    "910c": "ascend",
    "810e": "ptg",
}


def resolve_model_execution_profile(platform: str, name: str) -> ModelExecutionProfile:
    if platform == "h100" and name == "baseline":
        return ModelExecutionProfile(
            name=name,
            platform=platform,
            runtime_backend=PLATFORM_RUNTIME_BACKENDS[platform],
            edge_input_projection="concatenated",
            grid2mesh_edge_chunk_size=None,
            mesh2grid_edge_chunk_size=None,
            mesh2grid_full_grid_fast_path=False,
            index_add_execution="inline",
        )
    if platform in DOMESTIC_PLATFORMS and name == "domestic-v1":
        return ModelExecutionProfile(
            name=name,
            platform=platform,
            runtime_backend=PLATFORM_RUNTIME_BACKENDS[platform],
            edge_input_projection="factorized",
            grid2mesh_edge_chunk_size=131072,
            mesh2grid_edge_chunk_size=131072,
            mesh2grid_full_grid_fast_path=True,
            index_add_execution="inline",
        )
    raise ValueError(f"model execution profile {name!r} is not valid for platform {platform!r}")


def validate_runtime_platform(platform: str, backend: str) -> None:
    expected = PLATFORM_RUNTIME_BACKENDS[platform]
    if backend != expected:
        raise RuntimeError(
            f"platform {platform!r} requires runtime backend {expected!r}, found {backend!r}"
        )


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
    precision_policy: InferencePrecisionPolicy | None = None,
    device_type: str | None = None,
) -> RolloutMeasurement:
    """Warm up outside the timer, then measure all dependent rollout steps."""

    if steps <= 0:
        raise ValueError(f"rollout steps must be positive, got {steps}")
    if int(prepared.steps) != steps:
        raise ValueError(
            f"requested {steps} rollout steps but received {prepared.steps} forcing frames"
        )
    if warmup_calls < 1:
        raise ValueError("at least one warmup call is required")

    policy = precision_policy or resolve_inference_precision("fp32")
    state = prepared.initial_state.clone()
    resolved_device_type = device_type or state.device.type
    first_forcing = prepared.forcing_features[0]
    model_input_dtype: str | None = None
    model_output_dtype: str | None = None
    feedback_dtype: str | None = None

    def observe_forward(result: Any) -> None:
        nonlocal model_input_dtype, model_output_dtype, feedback_dtype
        if model_input_dtype is not None and model_input_dtype != result.model_input_dtype:
            raise RuntimeError("model input dtype changed during rollout")
        if model_output_dtype is not None and model_output_dtype != result.model_output_dtype:
            raise RuntimeError("model output dtype changed during rollout")
        model_input_dtype = result.model_input_dtype
        model_output_dtype = result.model_output_dtype
        feedback_dtype = str(result.prediction.dtype)

    with torch.inference_mode():
        for _ in range(warmup_calls):
            warmup = run_precision_forward(
                model,
                state,
                first_forcing,
                policy=policy,
                device_type=resolved_device_type,
            )
            observe_forward(warmup)
            if tuple(warmup.prediction.shape[-1:]) != (int(prepared.output_channels),):
                raise RuntimeError("warmup output does not match the configured output channels")
            del warmup
        synchronize()

        started = clock()
        normalized_output: torch.Tensor | None = None
        for step_index in range(steps):
            forcing = prepared.forcing_features[step_index]
            forward = run_precision_forward(
                model,
                state,
                forcing,
                policy=policy,
                device_type=resolved_device_type,
            )
            observe_forward(forward)
            prediction = forward.prediction
            final_step = step_index + 1 == steps
            normalized_output = advance_device_rollout(
                state,
                forcing,
                prediction,
                prepared,
                advance_state=not final_step,
                return_absolute_output=final_step,
            )
            del forward, prediction
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
        dtype_evidence=RolloutDtypeEvidence(
            state=str(state.dtype),
            forcing=str(prepared.forcing_features.dtype),
            model_input=str(model_input_dtype),
            model_output=str(model_output_dtype),
            feedback=str(feedback_dtype),
            final_output=str(final_output.dtype),
        ),
    )


def precision_metadata(
    policy: InferencePrecisionPolicy,
    measurement: RolloutMeasurement,
    *,
    model: torch.nn.Module,
    device_type: str,
    float32_math_mode: str,
) -> dict[str, Any]:
    parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if len(parameter_dtypes) != 1:
        raise RuntimeError(f"model parameters require one dtype, found {sorted(parameter_dtypes)}")
    evidence = measurement.dtype_evidence
    return {
        "requested": policy.requested,
        "effective": policy.effective,
        "autocast": {
            "enabled": policy.autocast_dtype is not None,
            "device_type": device_type,
            "dtype": str(policy.autocast_dtype) if policy.autocast_dtype is not None else None,
        },
        "parameter_dtype": next(iter(parameter_dtypes)),
        "state_dtype": evidence.state,
        "forcing_dtype": evidence.forcing,
        "model_input_dtype": evidence.model_input,
        "model_output_dtype": evidence.model_output,
        "feedback_dtype": evidence.feedback,
        "final_output_dtype": evidence.final_output,
        "float32_math_mode": float32_math_mode,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if (
        isinstance(payload, dict)
        and payload
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    ):
        return payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            value = payload.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
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


def build_model_for_run(
    config: GoogleSmallConfig,
    *,
    edge_input_projection: str,
    mesh2grid_full_grid_fast_path: bool = False,
    index_add_execution: str = "inline",
) -> GoogleSmallCompatibleModel:
    return GoogleSmallCompatibleModel(
        config,
        edge_input_projection_mode=edge_input_projection,
        mesh2grid_full_grid_fast_path=mesh2grid_full_grid_fast_path,
        index_add_execution=index_add_execution,
    )


def apply_edge_chunk_overrides(
    config: GoogleSmallConfig,
    *,
    grid2mesh: int | None,
    mesh2grid: int | None,
) -> GoogleSmallConfig:
    for value in (grid2mesh, mesh2grid):
        if value is not None and value <= 0:
            raise ValueError("edge chunk size must be positive")
    return replace(
        config,
        grid2mesh_edge_chunk_size=(
            grid2mesh if grid2mesh is not None else config.grid2mesh_edge_chunk_size
        ),
        mesh2grid_edge_chunk_size=(
            mesh2grid if mesh2grid is not None else config.mesh2grid_edge_chunk_size
        ),
    )


def model_execution_metadata(
    model: GoogleSmallCompatibleModel,
    *,
    profile: ModelExecutionProfile | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "edge_input_projection": model.edge_input_projection_mode,
        "mesh2grid_full_grid_fast_path": model.mesh2grid_full_grid_fast_path,
        "index_add_execution": model.index_add_execution,
        "grid2mesh_edge_chunk_size": model.config.grid2mesh_edge_chunk_size,
        "mesh2grid_edge_chunk_size": model.config.mesh2grid_edge_chunk_size,
    }
    if profile is not None:
        metadata["model_execution_profile"] = profile.to_metadata()
    return metadata


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
    parser.add_argument(
        "--platform",
        choices=("h100", "c550", "bw3000", "s5000", "910c", "810e"),
        required=True,
    )
    parser.add_argument(
        "--model-execution-profile",
        dest="model_execution_profile_name",
        choices=("baseline", "domestic-v1"),
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--forward-flops", type=int, default=FORWARD_FLOPS)
    parser.add_argument("--peak-tflops-per-device", type=float, default=0.0)
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.model_execution_profile = resolve_model_execution_profile(
            args.platform, args.model_execution_profile_name
        )
    except ValueError as error:
        parser.error(str(error))
    return args


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

    model_execution_profile = args.model_execution_profile
    runtime = resolve_runtime_backend(
        args.device,
        expected_profile=model_execution_profile.runtime_backend,
    )
    validate_runtime_platform(args.platform, runtime.profile.name)
    precision_policy = resolve_inference_precision(args.precision)
    float32_math_mode = configure_float32_math(False, runtime.profile.device_type)
    config = apply_edge_chunk_overrides(
        GoogleSmallConfig.from_yaml(args.config),
        grid2mesh=model_execution_profile.grid2mesh_edge_chunk_size,
        mesh2grid=model_execution_profile.mesh2grid_edge_chunk_size,
    )
    inputs = xr.open_dataset(args.inputs).load()
    forcings = xr.open_dataset(args.rollout_forcings).load()
    target_template = xr.open_dataset(args.target_template).load()
    stats = load_normalization_stats(args.stats_dir)

    model = build_model_for_run(
        config,
        edge_input_projection=model_execution_profile.edge_input_projection,
        mesh2grid_full_grid_fast_path=(
            model_execution_profile.mesh2grid_full_grid_fast_path
        ),
        index_add_execution=model_execution_profile.index_add_execution,
    )
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
    static_index_add_contract = None
    if model_execution_profile.index_add_execution == "prevalidated-static":
        static_index_add_contract = model.prevalidate_static_index_add()
    prepared = prepare_device_rollout(
        inputs,
        target_template,
        forcings,
        config=config,
        stats=stats,
        device=runtime.device,
    )

    reset_peak_memory(runtime)
    static_runtime_evidence: dict[str, Any] = {"status": "not_requested"}

    if static_index_add_contract is None:
        measurement = run_tensor_resident_rollout(
            model,
            prepared,
            steps=args.steps,
            warmup_calls=args.warmup_calls,
            synchronize=lambda: synchronize_runtime(runtime),
            precision_policy=precision_policy,
            device_type=runtime.device.type,
        )
    else:
        with suspend_prevalidated_flaggems_index_add_bounds(runtime) as runtime_evidence:
            measurement = run_tensor_resident_rollout(
                model,
                prepared,
                steps=args.steps,
                warmup_calls=args.warmup_calls,
                synchronize=lambda: synchronize_runtime(runtime),
                precision_policy=precision_policy,
                device_type=runtime.device.type,
            )
        model.verify_static_index_add(static_index_add_contract)
        static_runtime_evidence = runtime_evidence

    dispatch_audit = require_complete_flaggems_dispatch(runtime)
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
        "runtime": {
            **runtime.to_metadata(),
            "flaggems_dispatch": (
                dispatch_audit.to_metadata()
                if dispatch_audit is not None
                else {"status": "not_requested"}
            ),
        },
        "precision": precision_metadata(
            precision_policy,
            measurement,
            model=model,
            device_type=runtime.device.type,
            float32_math_mode=float32_math_mode,
        ),
        "workload": {
            "steps": measurement.steps,
            "warmup_calls": measurement.warmup_calls,
            "warmup_excluded_from_e2e": True,
            "final_output_d2h_included_in_e2e": True,
            "device_count": args.device_count,
            **model_execution_metadata(model, profile=model_execution_profile),
            "static_index_add": (
                {
                    **static_index_add_contract.to_metadata(),
                    "runtime_patch": static_runtime_evidence,
                }
                if static_index_add_contract is not None
                else {"status": "not_requested"}
            ),
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
