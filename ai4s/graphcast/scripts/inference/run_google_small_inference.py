#!/usr/bin/env python3
"""Run Google Small compatible PyTorch inference.

Inputs are Google-style xarray NetCDF files:

- inputs: variables with batch/time/lat/lon/(level) plus static vars.
- forcings: target-time forcing variables.
- target-template: target variables and output coordinates/shapes.

When those three input files are omitted, the script runs a tiny synthetic
self-check and still writes a prediction NetCDF plus a report.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    RuntimeBackend,
    backend_report_lines,
    build_synthetic_google_small_datasets,
    dataset_to_stacked,
    inputs_to_grid_node_features,
    load_normalization_stats,
    memory_text,
    predict_with_inputs_and_residuals,
    reset_peak_memory,
    resolve_runtime_backend,
    synchronize,
)
from graphcast_compat.graph import default_latitudes, default_longitudes  # noqa: E402

DEFAULT_CONFIG = "configs/inference/google_graphcast_small.yaml"
DEFAULT_WEIGHTS = "data/checkpoints/google_graphcast_small_compat_state.pt"
DEFAULT_OUTPUT = "outputs/predictions/google_small_prediction.nc"
DEFAULT_REPORT = "outputs/reports/google_small_inference_report.md"


def load_or_build_datasets(
    args: argparse.Namespace,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, str]:
    paths = [args.inputs, args.forcings, args.target_template]
    if any(paths) and not all(paths):
        raise ValueError("--inputs, --forcings, and --target-template must be provided together")

    if all(paths):
        return (
            xr.open_dataset(args.inputs).load(),
            xr.open_dataset(args.target_template).load(),
            xr.open_dataset(args.forcings).load(),
            "netcdf",
        )

    latitudes = None
    longitudes = None
    if args.synthetic_grid == "full":
        config = GoogleSmallConfig.from_yaml(args.config)
        latitudes = default_latitudes(config.resolution)
        longitudes = default_longitudes(config.resolution)

    return (
        *build_synthetic_google_small_datasets(
            args.config,
            batch_size=args.batch_size,
            latitudes=latitudes,
            longitudes=longitudes,
        ),
        f"synthetic-{args.synthetic_grid}",
    )


def choose_config(args: argparse.Namespace, data_mode: str) -> GoogleSmallConfig:
    config = GoogleSmallConfig.from_yaml(args.config)
    mesh_size = args.mesh_size
    if mesh_size is None and data_mode == "synthetic-tiny":
        mesh_size = 1
    if mesh_size is not None:
        config = replace(config, mesh_size=mesh_size)
    return config


def finite_dataset(dataset: xr.Dataset) -> bool:
    return all(
        bool(np.isfinite(data_array.values).all()) for data_array in dataset.data_vars.values()
    )


def dataset_mean_std(dataset: xr.Dataset) -> tuple[float, float]:
    stacked = dataset_to_stacked(dataset)
    values = np.asarray(stacked.data, dtype=np.float32)
    return float(values.mean()), float(values.std())


def build_report(
    *,
    args: argparse.Namespace,
    data_mode: str,
    config: GoogleSmallConfig,
    runtime: RuntimeBackend,
    output_path: Path,
    report_path: Path,
    grid_features_shape: tuple[int, ...],
    prediction_variables: list[str],
    prediction_channels: int,
    finite: bool,
    prediction_mean: float,
    prediction_std: float,
    load_seconds: float,
    graph_seconds: float,
    inference_seconds: float,
    total_seconds: float,
    memory: str,
    normalization_mode: str,
) -> str:
    lines = [
        "# Google Small Inference Report",
        "",
        f"- config: `{args.config}`",
        f"- weights: `{args.weights}`",
        f"- output: `{output_path}`",
        f"- report: `{report_path}`",
        f"- data mode: `{data_mode}`",
        f"- device: `{runtime.device}`",
        f"- mesh size: `{config.mesh_size}`",
        f"- normalization mode: `{normalization_mode}`",
        *backend_report_lines(runtime),
        f"- grid feature tensor shape: `{grid_features_shape}`",
        f"- prediction variables: `{prediction_variables}`",
        f"- prediction stack channels: `{prediction_channels}`",
        f"- finite: `{finite}`",
        f"- prediction mean: `{prediction_mean}`",
        f"- prediction std: `{prediction_std}`",
        f"- load seconds: `{load_seconds:.3f}`",
        f"- graph seconds: `{graph_seconds:.3f}`",
        f"- inference seconds: `{inference_seconds:.3f}`",
        f"- total seconds: `{total_seconds:.3f}`",
        f"- memory: `{memory}`",
        "",
        f"Overall status: `{'OK' if finite else 'FAIL'}`",
        "",
    ]
    if data_mode.startswith("synthetic"):
        lines.extend(
            [
                "This run used synthetic data for pipeline validation. It is not",
                "a real ERA5 sample or JAX numerical parity check.",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--inputs", default=None, help="NetCDF inputs Dataset")
    parser.add_argument("--forcings", default=None, help="NetCDF forcings Dataset")
    parser.add_argument("--target-template", default=None, help="NetCDF target template Dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="prediction NetCDF path")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="markdown report path")
    parser.add_argument(
        "--stats-dir",
        default=None,
        help=(
            "optional directory containing stddev_by_level.nc, mean_by_level.nc, "
            "and diffs_stddev_by_level.nc for Google InputsAndResiduals"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--synthetic-grid",
        choices=("tiny", "full"),
        default="tiny",
        help="grid used when NetCDF inputs are omitted",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--mesh-size",
        type=int,
        default=None,
        help="optional mesh_size override; default uses 1 for synthetic-tiny, config otherwise",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    runtime = resolve_runtime_backend(args.device)
    device = runtime.device
    inputs, targets_template, forcings, data_mode = load_or_build_datasets(args)
    config = choose_config(args, data_mode)
    output_path = Path(args.output)
    report_path = Path(args.report)

    t1 = time.perf_counter()
    model = GoogleSmallCompatibleModel(config)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - t1

    reset_peak_memory(runtime)

    grid_features = inputs_to_grid_node_features(
        inputs,
        forcings,
        input_variables=config.input_variables or None,
        forcing_variables=config.forcing_variables or None,
    )
    latitudes = inputs.coords["lat"].values
    longitudes = inputs.coords["lon"].values
    t2 = time.perf_counter()
    model.init_graph(latitudes=latitudes, longitudes=longitudes)
    graph_seconds = time.perf_counter() - t2

    synchronize(runtime)
    t3 = time.perf_counter()
    if args.stats_dir:
        stats = load_normalization_stats(args.stats_dir)
        prediction = predict_with_inputs_and_residuals(
            model,
            inputs,
            targets_template,
            forcings,
            stats,
        )
        normalization_mode = f"inputs_and_residuals:{args.stats_dir}"
    else:
        prediction = model.predict_xarray(inputs, targets_template, forcings)
        normalization_mode = "none"
    synchronize(runtime)
    inference_seconds = time.perf_counter() - t3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_netcdf(output_path)

    prediction_stacked = dataset_to_stacked(prediction)
    finite = finite_dataset(prediction)
    prediction_mean, prediction_std = dataset_mean_std(prediction)
    total_seconds = time.perf_counter() - t0
    memory = memory_text(runtime)

    report = build_report(
        args=args,
        data_mode=data_mode,
        config=config,
        runtime=runtime,
        output_path=output_path,
        report_path=report_path,
        grid_features_shape=tuple(grid_features.shape),
        prediction_variables=sorted(prediction.data_vars.keys()),
        prediction_channels=int(prediction_stacked.sizes["channels"]),
        finite=finite,
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        load_seconds=load_seconds,
        graph_seconds=graph_seconds,
        inference_seconds=inference_seconds,
        total_seconds=total_seconds,
        memory=memory,
        normalization_mode=normalization_mode,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"data_mode: {data_mode}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"backend: {runtime.profile.name}", flush=True)
    print(f"flagos_mode: {runtime.flagos.mode}", flush=True)
    print(f"flagos_status: {runtime.flagos.status}", flush=True)
    print(f"flaggems_variant: {runtime.flagos.variant}", flush=True)
    print(f"flaggems_ops: {list(runtime.flagos.requested_ops)}", flush=True)
    print(f"runtime_versions: {runtime.versions}", flush=True)
    if runtime.flagos.warnings:
        print(f"flagos_warnings: {list(runtime.flagos.warnings)}", flush=True)
    if runtime.flagos.error:
        print(f"flagos_error: {runtime.flagos.error}", flush=True)
    print(f"mesh_size: {config.mesh_size}", flush=True)
    print(f"normalization_mode: {normalization_mode}", flush=True)
    print(f"grid_features_shape: {tuple(grid_features.shape)}", flush=True)
    print(f"prediction_variables: {sorted(prediction.data_vars.keys())}", flush=True)
    print(f"prediction_stack_channels: {prediction_stacked.sizes['channels']}", flush=True)
    print(f"finite: {finite}", flush=True)
    print(f"prediction_mean: {prediction_mean}", flush=True)
    print(f"prediction_std: {prediction_std}", flush=True)
    print(f"load_seconds: {load_seconds:.3f}", flush=True)
    print(f"graph_seconds: {graph_seconds:.3f}", flush=True)
    print(f"inference_seconds: {inference_seconds:.3f}", flush=True)
    print(f"total_seconds: {total_seconds:.3f}", flush=True)
    print(f"memory: {memory}", flush=True)
    print(f"output: {output_path}", flush=True)
    print(f"report: {report_path}", flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
