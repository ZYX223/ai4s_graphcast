#!/usr/bin/env python3
"""Train the real 25km GraphCast case with a single-process model-parallel layout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
LIB_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, LIB_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphcast_compat import (  # noqa: E402
    GoogleSmallCompatibleModel,
    GoogleSmallConfig,
    inputs_to_grid_node_features,
    load_normalization_stats,
)
from graphcast_compat.backend import (  # noqa: E402
    backend_report_lines,
    configure_flagos_environment,
    graphcast_gather_mode,
    resolve_runtime_backend,
)
from graphcast_compat.backend import (  # noqa: E402
    memory_text as _memory_text,
)
from graphcast_compat.backend import (  # noqa: E402
    reset_peak_memory as _reset_peak_memory,
)
from graphcast_compat.backend import (  # noqa: E402
    synchronize as _synchronize,
)
from graphcast_compat.normalization import normalize_dataset  # noqa: E402
from graphcast_compat.operator_profile import (  # noqa: E402
    OperatorEventTimer,
    OperatorProfiler,
)
from train_ddp import (  # noqa: E402
    load_real_case,
    set_train_scope,
    target_to_training_tensor,
)

DEFAULT_CASE_DIR = REPO_ROOT / "data" / "cases" / "operational_25km"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference" / "google_graphcast_operational_25km.yaml"
DEFAULT_WEIGHTS = (
    REPO_ROOT / "data" / "checkpoints" / "google_graphcast_operational_25km_compat_state.pt"
)
DEFAULT_STATS_DIR = REPO_ROOT / "data" / "stats"
DEFAULT_REPORT = (
    REPO_ROOT / "outputs" / "training" / "reports" / "model_parallel_25km_train_report.md"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "outputs" / "training" / "checkpoints" / "model_parallel_25km_train.pt"
)
DEFAULT_FLAGGEMS_CONFIG = REPO_ROOT / "configs/flaggems/graphcast_training.yaml"
ACCELERATOR_TYPES = {"npu", "cuda"}


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_devices(value: str) -> tuple[torch.device, ...]:
    devices = tuple(torch.device(item.strip()) for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("--devices must contain at least one device")
    normalized = []
    for device in devices:
        if device.type == "npu" and device.index is None:
            normalized.append(torch.device("npu:0"))
        else:
            normalized.append(device)
    return tuple(normalized)


def init_accelerators(runtime: Any, devices: tuple[torch.device, ...]) -> None:
    """Validate and initialize accelerator devices via the unified backend."""
    if devices[0].type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(int(devices[0].index or 0))
    elif devices[0].type == "cuda":
        torch.cuda.set_device(int(devices[0].index or 0))


def distribute_mesh_steps(
    *,
    devices: tuple[torch.device, ...],
    steps: int,
    explicit: str | None,
) -> tuple[torch.device, ...]:
    if explicit:
        step_devices = parse_devices(explicit)
        if len(step_devices) != steps:
            raise ValueError("--mesh-step-devices length must equal gnn_msg_steps")
        return step_devices
    return tuple(
        devices[min(step * len(devices) // steps, len(devices) - 1)] for step in range(steps)
    )


def unique_devices(devices: tuple[torch.device, ...]) -> tuple[torch.device, ...]:
    seen = set()
    result = []
    for device in devices:
        key = str(device)
        if key in seen:
            continue
        seen.add(key)
        result.append(device)
    return tuple(result)


def synchronize(device: torch.device) -> None:
    _synchronize(device)


def reset_peak_memory(devices: tuple[torch.device, ...]) -> None:
    for device in unique_devices(devices):
        _reset_peak_memory(device)


def memory_summary(devices: tuple[torch.device, ...]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for device in unique_devices(devices):
        _synchronize(device)
        text = _memory_text(device)
        summary[str(device)] = {"memory": text}
    return summary


def partition_bounds(total: int, partitions: int) -> list[tuple[int, int]]:
    if partitions < 1:
        raise ValueError("partitions must be >= 1")
    return [
        (index * total // partitions, (index + 1) * total // partitions)
        for index in range(partitions)
    ]


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tensor_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }


def flatten_tensor_summaries(value: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if len(summaries) >= limit:
            return
        summary = tensor_summary(item)
        if summary is not None:
            summaries.append(summary)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
                if len(summaries) >= limit:
                    return
        elif isinstance(item, list | tuple):
            for child in item:
                visit(child)
                if len(summaries) >= limit:
                    return

    visit(value)
    return summaries


def has_accelerator_tensor(value: Any) -> bool:
    summary = tensor_summary(value)
    if summary is not None:
        return str(summary["device"]).split(":", 1)[0] in ACCELERATOR_TYPES
    if isinstance(value, dict):
        return any(has_accelerator_tensor(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(has_accelerator_tensor(item) for item in value)
    return False


class OperatorTrace:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.current_phase = "unscoped"
        self.ops: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, str]] = []

    @contextmanager
    def mode(self):
        if not self.enabled:
            yield
            return
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except Exception:
            self.enabled = False
            yield
            return

        tracer = self

        class _TraceMode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func,
                types,
                args=(),
                kwargs=None,
            ):
                call_kwargs = kwargs or {}
                try:
                    result = func(*args, **call_kwargs)
                except Exception as exc:
                    tracer.record_error(func, args, call_kwargs, exc)
                    raise
                tracer.record(func, args, call_kwargs, result)
                return result

        with _TraceMode():
            yield

    @contextmanager
    def phase(self, name: str):
        previous = self.current_phase
        self.current_phase = name
        try:
            yield
        finally:
            self.current_phase = previous

    def record(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        if not (
            has_accelerator_tensor(args)
            or has_accelerator_tensor(kwargs)
            or has_accelerator_tensor(result)
        ):
            return
        name = str(func)
        entry = self.ops.setdefault(
            name,
            {
                "count": 0,
                "phases": {},
                "examples": [],
            },
        )
        entry["count"] += 1
        phases = entry["phases"]
        phases[self.current_phase] = phases.get(self.current_phase, 0) + 1
        if len(entry["examples"]) < 5:
            entry["examples"].append(
                {
                    "phase": self.current_phase,
                    "inputs": flatten_tensor_summaries((args, kwargs)),
                    "outputs": flatten_tensor_summaries(result),
                }
            )

    def record_error(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        exc: BaseException,
    ) -> None:
        if not (has_accelerator_tensor(args) or has_accelerator_tensor(kwargs)):
            return
        self.errors.append(
            {
                "op": str(func),
                "phase": self.current_phase,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    def to_json(self) -> dict[str, Any]:
        ops = [
            {"op": name, **entry}
            for name, entry in sorted(
                self.ops.items(),
                key=lambda item: (-int(item[1]["count"]), item[0]),
            )
        ]
        return {
            "enabled": self.enabled,
            "total_ops": len(ops),
            "ops": ops,
            "errors": self.errors,
        }

    def top_lines(self, limit: int = 20) -> list[str]:
        if not self.enabled:
            return ["- operator trace enabled: `False`"]
        lines = ["- operator trace enabled: `True`"]
        for item in self.to_json()["ops"][:limit]:
            phases = ", ".join(
                f"{phase}:{count}" for phase, count in sorted(item["phases"].items())
            )
            lines.append(f"- `{item['op']}`: `{item['count']}` ({phases})")
        if self.errors:
            lines.append(f"- operator trace errors: `{self.errors}`")
        return lines


def write_operator_trace(
    trace: OperatorTrace,
    *,
    json_path: Path | None,
    markdown_path: Path | None,
) -> None:
    data = trace.to_json()
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if markdown_path is None:
        return
    lines = [
        "# GraphCast 25km Training Operator Trace",
        "",
        f"- enabled: `{data['enabled']}`",
        f"- total ops: `{data['total_ops']}`",
        "",
        "| ATen op | Count | Phases | Example input shapes |",
        "| --- | ---: | --- | --- |",
    ]
    for item in data["ops"]:
        phases = "<br>".join(f"{phase}: {count}" for phase, count in sorted(item["phases"].items()))
        examples = item["examples"][:1]
        input_shapes = []
        if examples:
            for tensor in examples[0]["inputs"]:
                input_shapes.append(f"{tensor['shape']} {tensor['dtype']} {tensor['device']}")
        lines.append(
            f"| `{item['op']}` | {item['count']} | {phases} | {'<br>'.join(input_shapes)} |"
        )
    if data["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in data["errors"]:
            lines.append(f"- `{error['phase']}` `{error['op']}`: `{error['error']}`")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_environment_info(runtime: Any) -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backend": runtime.profile.name,
        "device_type": runtime.profile.device_type,
        "runtime_family": runtime.profile.runtime_family,
        "versions": str(runtime.versions),
        "visible_devices": os.environ.get(
            "ASCEND_RT_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        ),
    }


def summarize_flaggems_record(path: str | None) -> dict[str, int | str]:
    if not path:
        return {"path": "None", "exists": 0}
    record_path = Path(path)
    summary: dict[str, int | str] = {
        "path": str(record_path),
        "exists": int(record_path.is_file()),
    }
    if not record_path.is_file():
        return summary
    patterns = {
        "ADDMM": "ADDMM",
        "SILU": "SILU",
        "LAYERNORM": "LAYERNORM",
        "CAT": "CAT",
        "INDEX": "INDEX",
        "INDEX ADD_": "INDEX ADD_",
        "ADD": "GEMS ADD",
    }
    counts = dict.fromkeys(patterns, 0)
    line_count = 0
    try:
        with record_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line_count += 1
                for name, pattern in patterns.items():
                    if pattern in line:
                        counts[name] += 1
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary
    summary["lines"] = line_count
    summary.update(counts)
    return summary


def build_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    name: str,
    lr: float,
) -> torch.optim.Optimizer | None:
    if lr == 0.0:
        return None
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.95))
    raise ValueError(f"unsupported optimizer: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--stats-dir", default=str(DEFAULT_STATS_DIR))
    parser.add_argument("--inputs", default=str(DEFAULT_CASE_DIR / "inputs.nc"))
    parser.add_argument("--target", default=str(DEFAULT_CASE_DIR / "targets.nc"))
    parser.add_argument("--forcings", default=str(DEFAULT_CASE_DIR / "forcings.nc"))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--train-scope", choices=("decoder-grid", "all"), default="all")
    parser.add_argument("--devices", default="npu:0,npu:1")
    parser.add_argument("--mesh-step-devices", default=None)
    parser.add_argument("--activation-checkpointing", default="1")
    parser.add_argument("--grid2mesh-node-chunk-size", type=int, default=8192)
    parser.add_argument("--mesh2grid-edge-chunk-size", type=int, default=2048)
    parser.add_argument("--mesh2grid-node-chunk-size", type=int, default=8192)
    parser.add_argument("--mesh2grid-decoder-chunk-size", type=int, default=8192)
    parser.add_argument("--mesh2grid-grid-partitions", type=int, default=1)
    parser.add_argument("--save-checkpoint", default="0")
    parser.add_argument(
        "--flagos-mode",
        default=os.environ.get("GRAPHCAST_FLAGOS_MODE", "off"),
        help="FlagOS backend mode: off or flaggems",
    )
    parser.add_argument(
        "--flaggems-ops",
        default=os.environ.get("GRAPHCAST_FLAGGEMS_OPS", ""),
    )
    parser.add_argument(
        "--flaggems-config",
        default=os.environ.get("GRAPHCAST_FLAGGEMS_CONFIG", str(DEFAULT_FLAGGEMS_CONFIG)),
    )
    parser.add_argument(
        "--flaggems-record-path",
        default=os.environ.get("GRAPHCAST_FLAGGEMS_RECORD_PATH"),
    )
    parser.add_argument(
        "--flaggems-strict",
        default=os.environ.get(
            "GRAPHCAST_FLAGGEMS_STRICT",
            os.environ.get("GRAPHCAST_STRICT_FLAGOS", "1"),
        ),
    )
    parser.add_argument(
        "--bishengir-opt-dir",
        default=os.environ.get("GRAPHCAST_BISHENGIR_OPT_DIR"),
    )
    parser.add_argument(
        "--flaggems-src",
        default=os.environ.get(
            "GRAPHCAST_FLAGGEMS_SOURCE_ROOT",
            os.environ.get("GRAPHCAST_FLAGGEMS_SRC"),
        ),
    )
    parser.add_argument(
        "--flaggems-variant",
        choices=("installed", "official", "internal"),
        default=os.environ.get("GRAPHCAST_FLAGGEMS_VARIANT", "installed"),
    )
    parser.add_argument(
        "--flaggems-runtime-dest",
        default=os.environ.get("FLAGGEMS_RUNTIME_DEST"),
    )
    parser.add_argument(
        "--trace-operators",
        default=os.environ.get("GRAPHCAST_TRACE_OPERATORS", "0"),
    )
    parser.add_argument(
        "--operator-trace-path",
        default=os.environ.get("GRAPHCAST_OPERATOR_TRACE_PATH"),
    )
    parser.add_argument(
        "--operator-trace-report",
        default=os.environ.get("GRAPHCAST_OPERATOR_TRACE_REPORT"),
    )
    parser.add_argument(
        "--profile-step",
        type=int,
        default=0,
        help="1-based training step to capture with torch.profiler; 0 disables profiling",
    )
    parser.add_argument(
        "--operator-profile-path",
        default=os.environ.get("GRAPHCAST_OPERATOR_PROFILE_PATH"),
    )
    parser.add_argument(
        "--operator-profile-ops",
        default="",
        help="optional comma-separated operators retained from a Kineto trace",
    )
    parser.add_argument("--operator-event-path", default=None)
    parser.add_argument(
        "--operator-event-ops",
        default="",
        help="comma-separated dispatcher ops measured with device events",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.mesh2grid_grid_partitions < 1:
        raise ValueError("--mesh2grid-grid-partitions must be >= 1")
    if args.profile_step < 0 or args.profile_step > args.steps:
        raise ValueError("--profile-step must be 0 or within the configured training steps")
    capture_paths = [args.operator_profile_path, args.operator_event_path]
    if sum(bool(path) for path in capture_paths) > 1:
        raise ValueError("Kineto and event profiling cannot be enabled together")
    if bool(args.profile_step) != any(bool(path) for path in capture_paths):
        raise ValueError("--profile-step and one operator profile path must be provided together")
    if args.operator_event_path and not args.operator_event_ops:
        raise ValueError("--operator-event-ops is required with --operator-event-path")
    devices = parse_devices(args.devices)

    configure_flagos_environment(
        mode=args.flagos_mode,
        ops=args.flaggems_ops,
        config=args.flaggems_config,
        source_root=args.flaggems_src,
        record_path=args.flaggems_record_path,
        strict=args.flaggems_strict,
        bishengir_opt_dir=args.bishengir_opt_dir,
        variant=args.flaggems_variant,
        runtime_dest=args.flaggems_runtime_dest,
    )

    runtime = resolve_runtime_backend(devices[0].type)
    init_accelerators(runtime, devices)
    print(
        f"backend={runtime.profile.name} flagos_mode={runtime.flagos.mode} "
        f"flagos_status={runtime.flagos.status} "
        f"ops={list(runtime.flagos.requested_ops)} "
        f"gather_mode={graphcast_gather_mode()} "
        f"record={runtime.flagos.record_path}",
        flush=True,
    )
    t0 = time.perf_counter()

    inputs, targets, forcings = load_real_case(args)
    config = GoogleSmallConfig.from_yaml(args.config)
    if args.grid2mesh_node_chunk_size is not None:
        if args.grid2mesh_node_chunk_size <= 0:
            raise ValueError("--grid2mesh-node-chunk-size must be positive")
        config = replace(
            config,
            grid2mesh_node_chunk_size=args.grid2mesh_node_chunk_size,
        )
    if args.mesh2grid_edge_chunk_size is not None:
        if args.mesh2grid_edge_chunk_size <= 0:
            raise ValueError("--mesh2grid-edge-chunk-size must be positive")
        config = replace(
            config,
            mesh2grid_edge_chunk_size=args.mesh2grid_edge_chunk_size,
        )
    if args.mesh2grid_node_chunk_size is not None:
        if args.mesh2grid_node_chunk_size <= 0:
            raise ValueError("--mesh2grid-node-chunk-size must be positive")
        config = replace(
            config,
            mesh2grid_node_chunk_size=args.mesh2grid_node_chunk_size,
        )
    if args.mesh2grid_decoder_chunk_size is not None:
        if args.mesh2grid_decoder_chunk_size <= 0:
            raise ValueError("--mesh2grid-decoder-chunk-size must be positive")
        config = replace(
            config,
            mesh2grid_decoder_chunk_size=args.mesh2grid_decoder_chunk_size,
        )
    mesh_step_devices = distribute_mesh_steps(
        devices=devices,
        steps=config.gnn_msg_steps,
        explicit=args.mesh_step_devices,
    )
    stats = load_normalization_stats(args.stats_dir)

    norm_inputs = normalize_dataset(inputs, stats.stddev_by_level, stats.mean_by_level)
    norm_forcings = normalize_dataset(forcings, stats.stddev_by_level, stats.mean_by_level)
    input_device = devices[0]
    output_device = devices[-1]
    grid_features = inputs_to_grid_node_features(
        norm_inputs,
        norm_forcings,
        input_variables=config.input_variables or None,
        forcing_variables=config.forcing_variables or None,
    ).to(input_device)
    target_tensor = target_to_training_tensor(
        inputs=inputs,
        targets=targets,
        stats=stats,
        target_variables=config.target_variables,
    ).to(output_device)

    model = GoogleSmallCompatibleModel(
        config,
        activation_checkpointing=parse_bool(args.activation_checkpointing),
    )
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.init_graph(
        latitudes=inputs.coords["lat"].values,
        longitudes=inputs.coords["lon"].values,
    )
    trainable, total, trainable_names = set_train_scope(model, args.train_scope)
    model.set_model_parallel_devices(
        grid2mesh_device=input_device,
        mesh_step_devices=mesh_step_devices,
        mesh2grid_device=output_device,
    )
    model.train()

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"train scope {args.train_scope!r} selected no parameters")
    optimizer = build_optimizer(trainable_parameters, name=args.optimizer, lr=args.lr)

    reset_peak_memory(devices)
    trace_enabled = parse_bool(args.trace_operators)
    trace_json_path = (
        Path(args.operator_trace_path)
        if args.operator_trace_path
        else Path(args.report).with_suffix(".operators.json")
        if trace_enabled
        else None
    )
    trace_markdown_path = (
        Path(args.operator_trace_report)
        if args.operator_trace_report
        else Path(args.report).with_suffix(".operators.md")
        if trace_enabled
        else None
    )
    operator_trace = OperatorTrace(enabled=trace_enabled)
    operator_profiler = OperatorProfiler(
        args.operator_profile_path,
        args.operator_profile_ops,
    )
    operator_event_timer = OperatorEventTimer(
        args.operator_event_path,
        args.operator_event_ops,
    )
    step_losses = []
    train_t0 = time.perf_counter()
    try:
        with operator_trace.mode():
            for step in range(1, args.steps + 1):
                for device in unique_devices(devices):
                    synchronize(device)
                profile_this_step = step == args.profile_step
                if profile_this_step:
                    if operator_event_timer.enabled:
                        operator_event_timer.start()
                    else:
                        operator_profiler.start()
                with operator_trace.phase(f"step_{step}:zero_grad"):
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        model.zero_grad(set_to_none=True)
                step_t0 = time.perf_counter()
                if args.mesh2grid_grid_partitions == 1:
                    with operator_trace.phase(f"step_{step}:forward"):
                        prediction = model(grid_features).to(output_device)
                        loss = torch.mean((prediction.float() - target_tensor.float()) ** 2)
                    with operator_trace.phase(f"step_{step}:backward"):
                        loss.backward()
                    loss_value = float(loss.detach().cpu())
                else:
                    total_grid_nodes = int(target_tensor.shape[1])
                    total_elements = int(target_tensor.numel())
                    loss_value = 0.0
                    for partition_index, (start, end) in enumerate(
                        partition_bounds(total_grid_nodes, args.mesh2grid_grid_partitions),
                        start=1,
                    ):
                        with operator_trace.phase(
                            f"step_{step}:partition_{partition_index}:forward"
                        ):
                            grid2mesh_nodes, mesh_nodes_out = model.forward_pre_mesh2grid(
                                grid_features,
                                grid_output_slice=(start, end),
                            )
                            grid_part = grid2mesh_nodes["grid_nodes"]
                            prediction_part = model.forward_mesh2grid_partition(
                                mesh_nodes=mesh_nodes_out["mesh_nodes"],
                                grid_nodes=grid_part,
                                grid_start=start,
                                grid_end=end,
                            ).to(output_device)
                            target_part = target_tensor[:, start:end, :]
                            loss_part = (
                                torch.sum((prediction_part.float() - target_part.float()) ** 2)
                                / total_elements
                            )
                        with operator_trace.phase(
                            f"step_{step}:partition_{partition_index}:backward"
                        ):
                            loss_part.backward()
                        partition_loss = float(loss_part.detach().cpu())
                        loss_value += partition_loss
                        print(
                            f"step={step} mesh2grid_partition="
                            f"{partition_index}/{args.mesh2grid_grid_partitions} "
                            f"grid=[{start},{end}) loss_part={partition_loss:.8g}",
                            flush=True,
                        )
                        del (
                            grid2mesh_nodes,
                            mesh_nodes_out,
                            grid_part,
                            prediction_part,
                            target_part,
                            loss_part,
                        )
                with operator_trace.phase(f"step_{step}:optimizer_step"):
                    if optimizer is not None:
                        optimizer.step()
                for device in unique_devices(devices):
                    synchronize(device)
                step_losses.append(loss_value)
                print(
                    f"step={step} loss={loss_value:.6f} "
                    f"step_seconds={time.perf_counter() - step_t0:.3f}",
                    flush=True,
                )
                if profile_this_step:
                    profile_metadata = {
                        "stage": "training",
                        "step": step,
                        "flagos_mode": args.flagos_mode,
                        "flaggems_ops": args.flaggems_ops,
                        "devices": [str(device) for device in devices],
                    }
                    if operator_event_timer.enabled:
                        operator_event_timer.stop(metadata=profile_metadata)
                    else:
                        operator_profiler.stop(metadata=profile_metadata)
    except Exception:
        write_operator_trace(
            operator_trace,
            json_path=trace_json_path,
            markdown_path=trace_markdown_path,
        )
        raise

    train_seconds = time.perf_counter() - train_t0
    total_seconds = time.perf_counter() - t0
    checkpoint_path = Path(args.checkpoint)
    saved_checkpoint = None
    if parse_bool(args.save_checkpoint):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": state_dict_to_cpu(model),
                "train_scope": args.train_scope,
                "step_losses": step_losses,
                "devices": [str(device) for device in devices],
                "mesh_step_devices": [str(device) for device in mesh_step_devices],
                "activation_checkpointing": parse_bool(args.activation_checkpointing),
                "grid2mesh_node_chunk_size": config.grid2mesh_node_chunk_size,
                "mesh2grid_edge_chunk_size": config.mesh2grid_edge_chunk_size,
                "mesh2grid_node_chunk_size": config.mesh2grid_node_chunk_size,
                "mesh2grid_decoder_chunk_size": config.mesh2grid_decoder_chunk_size,
                "mesh2grid_grid_partitions": args.mesh2grid_grid_partitions,
                "backend": runtime.to_metadata(),
            },
            checkpoint_path,
        )
        saved_checkpoint = str(checkpoint_path)

    environment_info = collect_environment_info(runtime)
    record_summary = summarize_flaggems_record(runtime.flagos.record_path)
    write_operator_trace(
        operator_trace,
        json_path=trace_json_path,
        markdown_path=trace_markdown_path,
    )
    report_lines = [
        "# GraphCast 25km Model-Parallel Training Report",
        "",
        "- status: `OK`",
        f"- devices: `{[str(device) for device in devices]}`",
        f"- mesh step devices: `{[str(device) for device in mesh_step_devices]}`",
        f"- train scope: `{args.train_scope}`",
        f"- activation checkpointing: `{parse_bool(args.activation_checkpointing)}`",
        f"- grid2mesh node chunk size: `{config.grid2mesh_node_chunk_size}`",
        f"- mesh2grid edge chunk size: `{config.mesh2grid_edge_chunk_size}`",
        f"- mesh2grid node chunk size: `{config.mesh2grid_node_chunk_size}`",
        f"- mesh2grid decoder chunk size: `{config.mesh2grid_decoder_chunk_size}`",
        f"- mesh2grid grid partitions: `{args.mesh2grid_grid_partitions}`",
        f"- optimizer: `{args.optimizer}`",
        f"- lr: `{args.lr}`",
        f"- trainable parameters: `{trainable}`",
        f"- total parameters: `{total}`",
        f"- trainable parameter prefixes/sample: `{trainable_names[:8]}`",
        f"- grid feature shape: `{tuple(grid_features.shape)}`",
        f"- target tensor shape: `{tuple(target_tensor.shape)}`",
        f"- steps: `{args.steps}`",
        f"- losses: `{step_losses}`",
        f"- train seconds: `{train_seconds:.3f}`",
        f"- total seconds: `{total_seconds:.3f}`",
        f"- memory: `{memory_summary(devices)}`",
        f"- checkpoint: `{saved_checkpoint}`",
        f"- operator trace path: `{trace_json_path}`",
        f"- operator trace report: `{trace_markdown_path}`",
        "",
        "## Backend",
        *backend_report_lines(runtime),
        f"- flaggems record summary: `{record_summary}`",
        "",
        "## Operator Trace",
        *operator_trace.top_lines(),
        "",
        "## Environment",
        *[f"- {key}: `{value}`" for key, value in environment_info.items()],
        "",
        "Overall status: `OK`",
    ]
    write_report(Path(args.report), report_lines)
    print(f"report: {args.report}", flush=True)
    if saved_checkpoint:
        print(f"checkpoint: {saved_checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
