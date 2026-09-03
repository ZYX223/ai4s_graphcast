#!/usr/bin/env python3
"""Locate BF16 error growth across the three GraphCast network stages."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
for path in (GRAPHCAST_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphcast_compat import GoogleSmallConfig, load_normalization_stats  # noqa: E402
from graphcast_compat.autoregressive_inference import prepare_device_rollout  # noqa: E402
from graphcast_compat.backend import (  # noqa: E402
    configure_float32_math,
    resolve_runtime_backend,
)
from scripts.inference.run_operational_40step_inference import (  # noqa: E402
    _select_state_dict,
    _write_json,
    build_model_for_run,
)


def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    """Compare one candidate tensor with an FP32 reference on its current device."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have matching shapes")
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(reference_float.square())).item())
    finite = bool(
        torch.isfinite(reference_float).all().item()
        and torch.isfinite(candidate_float).all().item()
    )
    return {
        "shape": [int(value) for value in reference.shape],
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "finite": finite,
        "max_abs": float(torch.max(torch.abs(difference)).item()),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": rmse / max(reference_rms, math.ulp(0.0)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--rollout-forcings", type=Path, required=True)
    parser.add_argument("--target-template", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = resolve_runtime_backend(args.device)
    configure_float32_math(False, runtime.profile.device_type)
    config = GoogleSmallConfig.from_yaml(args.config)
    inputs = xr.open_dataset(args.inputs).load()
    forcings = xr.open_dataset(args.rollout_forcings).load()
    target_template = xr.open_dataset(args.target_template).load()
    stats = load_normalization_stats(args.stats_dir)

    model = build_model_for_run(
        config,
        edge_input_projection="concatenated",
        mesh2grid_full_grid_fast_path=False,
        index_add_execution="inline",
    )
    model.load_state_dict(
        _select_state_dict(torch.load(args.weights, map_location="cpu")),
        strict=True,
    )
    model.init_graph(
        latitudes=inputs.coords["lat"].values,
        longitudes=inputs.coords["lon"].values,
    )
    model.to(runtime.device).eval()
    prepared = prepare_device_rollout(
        inputs,
        target_template,
        forcings,
        config=config,
        stats=stats,
        device=runtime.device,
    )
    model_input = torch.cat(
        (prepared.initial_state, prepared.forcing_features[0]),
        dim=-1,
    )

    with torch.inference_mode():
        fp32_grid2mesh, fp32_mesh = model.forward_pre_mesh2grid(model_input)
        fp32_prediction = model.forward_mesh2grid_partition(
            mesh_nodes=fp32_mesh["mesh_nodes"],
            grid_nodes=fp32_grid2mesh["grid_nodes"],
            grid_start=0,
            grid_end=int(fp32_grid2mesh["grid_nodes"].shape[1]),
        )
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
        ):
            bf16_grid2mesh, bf16_mesh = model.forward_pre_mesh2grid(model_input)
            bf16_prediction = model.forward_mesh2grid_partition(
                mesh_nodes=bf16_mesh["mesh_nodes"],
                grid_nodes=bf16_grid2mesh["grid_nodes"],
                grid_start=0,
                grid_end=int(bf16_grid2mesh["grid_nodes"].shape[1]),
            )

    report = {
        "schema": "graphcast.bf16_stage_diagnostics.v1",
        "runtime": runtime.to_metadata(),
        "stages": {
            "grid2mesh_grid_nodes": tensor_error_metrics(
                fp32_grid2mesh["grid_nodes"],
                bf16_grid2mesh["grid_nodes"],
            ),
            "grid2mesh_mesh_nodes": tensor_error_metrics(
                fp32_grid2mesh["mesh_nodes"],
                bf16_grid2mesh["mesh_nodes"],
            ),
            "mesh_processor_mesh_nodes": tensor_error_metrics(
                fp32_mesh["mesh_nodes"],
                bf16_mesh["mesh_nodes"],
            ),
            "mesh2grid_prediction": tensor_error_metrics(
                fp32_prediction,
                bf16_prediction,
            ),
        },
    }
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
