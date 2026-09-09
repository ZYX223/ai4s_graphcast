"""Cross-platform PyTorch operator hotspot profiling for GraphCast."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, ContextManager
import warnings

import torch

from .profiled_mfu import extract_profiled_step_busy, iter_trace_events


@dataclass(frozen=True)
class HotspotProfilerConfig:
    enabled: bool = False
    output_dir: Path = Path("outputs/hotspot_profile")
    wait: int = 0
    warmup: int = 0
    active: int = 1
    repeat: int = 1
    top_k: int = 50
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = True
    staging_dir: Path | None = None
    split_active_traces: bool = True
    npu_profiler_level: str = "Level0"
    npu_op_attr: bool = False

    def validate(self) -> None:
        if self.wait < 0 or self.warmup < 0:
            raise ValueError("profiler wait and warmup must be non-negative")
        if self.active <= 0 or self.repeat <= 0 or self.top_k <= 0:
            raise ValueError("profiler active, repeat, and top_k must be positive")
        if self.npu_profiler_level not in {"Level0", "Level1"}:
            raise ValueError(
                "NPU profiler level must be either Level0 or Level1"
            )


def _profiler_api_for_device(device: torch.device) -> tuple[Any, str]:
    if device.type == "npu":
        try:
            import torch_npu
        except ImportError as error:
            raise RuntimeError(
                "NPU profiling requires the torch_npu.profiler backend"
            ) from error
        return torch_npu.profiler, "torch_npu.profiler"
    return torch.profiler, "torch.profiler"


def activities_for_device(
    device: torch.device,
    profiler_api: Any | None = None,
) -> list[Any]:
    api = profiler_api or _profiler_api_for_device(device)[0]
    activities = [api.ProfilerActivity.CPU]
    activity_names = {
        "cuda": ("CUDA",),
        "musa": ("MUSA",),
        "xpu": ("XPU",),
        "hpu": ("HPU",),
        "mtia": ("MTIA",),
        "npu": ("NPU", "PRIVATEUSE1", "PrivateUse1"),
    }
    for name in activity_names.get(device.type, ()):
        activity = getattr(api.ProfilerActivity, name, None)
        if activity is not None:
            activities.append(activity)
            break
    return activities


def _split_active_schedule(
    profiler_api: Any,
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
):
    """Save every active update as its own bounded trace."""

    action = profiler_api.ProfilerAction
    cycle_length = wait + warmup + active

    def schedule(step: int):
        if step < 0:
            raise ValueError("profiler step must be non-negative")
        cycle = step // cycle_length
        if cycle >= repeat:
            return action.NONE
        offset = step % cycle_length
        if offset < wait:
            return action.NONE
        if offset < wait + warmup:
            return action.WARMUP
        return action.RECORD_AND_SAVE

    return schedule


def _first_number(value: Any, *names: str) -> float | int | None:
    for name in names:
        result = getattr(value, name, None)
        if result is not None:
            return result
    return None


def _npu_experimental_config(api: Any, config: HotspotProfilerConfig) -> Any | None:
    """Build the pinned torch_npu DB-export contract.

    Level0 remains the default used by cross-platform timing runs.  Level1 is
    opt-in for execution audits that need CANN AI Core metadata such as
    ``hf32Eligible``; it deliberately retains the unsimplified DB.
    """

    experimental_config_type = getattr(api, "_ExperimentalConfig", None)
    export_type = getattr(getattr(api, "ExportType", None), "Db", None)
    if experimental_config_type is None or export_type is None:
        return None
    options: dict[str, Any] = {"export_type": export_type}
    if config.npu_profiler_level == "Level1":
        profiler_level = getattr(getattr(api, "ProfilerLevel", None), "Level1", None)
        aic_metrics = getattr(
            getattr(api, "AiCMetrics", None),
            "PipeUtilization",
            None,
        )
        if profiler_level is None or aic_metrics is None:
            raise RuntimeError(
                "torch_npu Level1/PipeUtilization profiling is unavailable"
            )
        options.update(
            profiler_level=profiler_level,
            aic_metrics=aic_metrics,
            op_attr=config.npu_op_attr,
            data_simplification=False,
        )
    return experimental_config_type(**options)


class HotspotProfiler:
    """Small lifecycle wrapper around ``torch.profiler``.

    Call ``start`` before a training/inference loop, ``step`` once per iteration,
    and ``stop`` in a ``finally`` block. ``record`` adds readable regions to the
    exported trace and is a no-op when profiling is disabled.
    """

    def __init__(self, config: HotspotProfilerConfig, device: torch.device):
        config.validate()
        self.config = config
        self.device = device
        self._profiler: Any | None = None
        self._started = False
        self._trace_paths: list[Path] = []
        self._profiler_api: Any | None = None
        self._profiler_backend = "disabled"
        self._activities: list[Any] = []
        self._start_succeeded = False
        self._step_calls = 0
        self._npu_raw_profiles: list[dict[str, Any]] = []
        self._npu_profiled_steps: dict[int, dict[str, Any]] = {}
        self._npu_postprocess_state = "not_applicable"
        self._npu_postprocess_error: str | None = None
        self._completion_reported = False
        self._hotspot_rows_by_trace: list[list[dict[str, Any]]] = []

    def start(self) -> None:
        if not self.config.enabled or self._started:
            return
        self._trace_paths = []
        self._step_calls = 0
        self._npu_raw_profiles = []
        self._npu_profiled_steps = {}
        self._completion_reported = False
        self._hotspot_rows_by_trace = []
        self._npu_postprocess_state = (
            "waiting_for_captures" if self.device.type == "npu" else "not_applicable"
        )
        self._npu_postprocess_error = None
        api, backend = _profiler_api_for_device(self.device)
        self._profiler_api = api
        self._profiler_backend = backend
        self._activities = activities_for_device(self.device, api)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.device.type == "npu":
            raw_root = output_dir / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)
            initializer = getattr(api, "tensorboard_trace_handler", None)
            if initializer is None:
                raise RuntimeError(
                    "torch_npu.profiler.tensorboard_trace_handler is unavailable"
                )
            # torch_npu uses this call to configure ProfPathCreator.  The
            # returned handler is deliberately not registered: its analyse()
            # path invokes CANNTimeline/Relation/TraceView during training and
            # can abort the next active capture when JSON export fails.
            initializer(
                dir_name=str(raw_root),
                worker_name=output_dir.parent.name,
                analyse_flag=False,
            )
        schedule = (
            _split_active_schedule(
                api,
                wait=self.config.wait,
                warmup=self.config.warmup,
                active=self.config.active,
                repeat=self.config.repeat,
            )
            if self.config.split_active_traces
            else api.schedule(
                wait=self.config.wait,
                warmup=self.config.warmup,
                active=self.config.active,
                repeat=self.config.repeat,
            )
        )
        profile_options: dict[str, Any] = {}
        if self.device.type == "npu":
            experimental_config = _npu_experimental_config(api, self.config)
            if experimental_config is not None:
                # The pinned runtime otherwise warns that export_type=None is
                # invalid and silently selects text.  The capture remains raw;
                # this merely fixes its declared offline export contract.
                profile_options["experimental_config"] = experimental_config
        self._profiler = api.profile(
            activities=self._activities,
            schedule=schedule,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            on_trace_ready=self._on_trace_ready,
            **profile_options,
        )
        self._profiler.start()
        self._started = True
        self._start_succeeded = True
        self._write_backend_metadata()

    def step(self) -> None:
        if self._started and self._profiler is not None:
            try:
                self._profiler.step()
            finally:
                self._step_calls += 1

    def stop_if_trace_ready(self, *, suppress_errors: bool = False) -> bool:
        """Stop after the configured schedule has exported every trace.

        Every active update is stopped and exported before the next trace is
        prepared, so its event buffers do not accumulate across active updates.
        The wrapper itself is released once all requested cycles are complete.
        """

        expected_traces = self.config.repeat * (
            self.config.active if self.config.split_active_traces else 1
        )
        completed = (
            len(self._npu_raw_profiles)
            if self.device.type == "npu"
            else len(self._trace_paths)
        )
        if completed < expected_traces or self._completion_reported:
            return False
        self.stop(suppress_errors=suppress_errors)
        self._completion_reported = True
        return True

    def stop(self, *, suppress_errors: bool = False) -> None:
        if not self._started or self._profiler is None:
            return
        profiler = self._profiler
        self._started = False
        self._profiler = None
        try:
            profiler.stop()
        except Exception as error:
            if not suppress_errors:
                raise
            warnings.warn(
                f"hotspot profiler cleanup failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def record(self, name: str) -> ContextManager[Any]:
        if not self.config.enabled or not self._started:
            return nullcontext()
        return torch.profiler.record_function(name)

    def trace_paths(self) -> tuple[Path, ...]:
        """Return completed profiler-cycle traces in creation order."""

        return tuple(self._trace_paths)

    def latest_trace_path(self) -> Path | None:
        """Return the newest completed trace, if a profiling cycle has closed."""

        return self._trace_paths[-1] if self._trace_paths else None

    def raw_profile_paths(self) -> tuple[Path, ...]:
        """Return finalized raw torch_npu capture directories in update order."""

        return tuple(Path(row["path"]) for row in self._npu_raw_profiles)

    def postprocess_ready_profiles(
        self,
        *,
        expected_devices: tuple[str | int, ...],
    ) -> bool:
        """Export finalized Ascend captures to SQLite and calculate statistics.

        This method is called at an optimizer boundary after heartbeat and
        checkpoint persistence.  It never invokes torch_npu's Chrome/TraceView
        parser and is idempotent after a successful export.
        """

        if not self.config.enabled or self.device.type != "npu":
            return False
        if self._npu_postprocess_state == "complete":
            return True
        expected = self.config.repeat * (
            self.config.active if self.config.split_active_traces else 1
        )
        if len(self._npu_raw_profiles) < expected:
            return False
        from .ascend_profile import postprocess_ascend_profiles

        self._npu_postprocess_state = "running"
        self._npu_postprocess_error = None
        self._write_backend_metadata()
        try:
            self._npu_profiled_steps = postprocess_ascend_profiles(
                self._npu_raw_profiles,
                expected_devices=expected_devices,
                output_dir=self.config.output_dir,
                top_k=self.config.top_k,
            )
        except Exception as error:
            self._npu_postprocess_state = "failed"
            self._npu_postprocess_error = f"{type(error).__name__}: {error}"
            self._write_backend_metadata()
            raise
        self._npu_postprocess_state = "complete"
        self._write_backend_metadata()
        return True

    def profiled_steps(
        self,
        *,
        expected_devices: tuple[str | int, ...],
    ) -> dict[int, dict[str, Any]]:
        """Return completed per-update busy summaries for this process."""

        if self.device.type == "npu":
            self.postprocess_ready_profiles(expected_devices=expected_devices)
            return dict(self._npu_profiled_steps)
        result: dict[int, dict[str, Any]] = {}
        for trace_path in self._trace_paths:
            result.update(
                extract_profiled_step_busy(
                    trace_path,
                    expected_devices=expected_devices,
                )
            )
        return result

    def _on_trace_ready(self, profiler: Any) -> None:
        """Export a cycle synchronously when ``profiler.step`` closes it."""

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.device.type == "npu":
            prof_if = getattr(profiler, "prof_if", None)
            raw_path_value = getattr(prof_if, "prof_path", None)
            if not raw_path_value:
                raise RuntimeError(
                    "torch_npu finalized a trace without a raw profiling path"
                )
            raw_path = Path(str(raw_path_value)).expanduser().resolve()
            raw_root = (output_dir / "raw").resolve()
            try:
                raw_path.relative_to(raw_root)
            except ValueError as error:
                raise RuntimeError(
                    f"torch_npu raw profile escaped configured root: {raw_path}"
                ) from error
            if not raw_path.is_dir() or not any(raw_path.glob("PROF_*")):
                raise RuntimeError(
                    f"torch_npu raw profile is incomplete: {raw_path}"
                )
            if any(Path(row["path"]) == raw_path for row in self._npu_raw_profiles):
                raise RuntimeError(
                    f"torch_npu reused a raw profile directory: {raw_path}"
                )
            self._npu_raw_profiles.append(
                {
                    "capture_index": len(self._npu_raw_profiles) + 1,
                    "local_step": self._step_calls + 1,
                    "path": str(raw_path),
                }
            )
            self._write_npu_raw_manifest()
            self._write_backend_metadata()
            return
        cycle = len(self._trace_paths)
        trace_path = (
            output_dir / "trace.json"
            if cycle == 0
            else output_dir / f"trace_cycle_{cycle + 1:03d}.json"
        )
        self._export_trace(profiler, trace_path)
        self._trace_paths.append(trace_path)
        self._write_backend_metadata()
        self._write_outputs(profiler)

    @staticmethod
    def _looks_like_complete_json(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size < 2:
            return False
        with path.open("rb") as stream:
            first = stream.read(4096).lstrip()
            stream.seek(max(0, path.stat().st_size - 4096))
            last = stream.read().rstrip()
        if not first or not last:
            return False
        return (first[:1], last[-1:]) in {(b"{", b"}"), (b"[", b"]")}

    @classmethod
    def _completed_export(cls, requested: Path) -> Path | None:
        if cls._looks_like_complete_json(requested):
            return requested
        temporary = Path(f"{requested}.tmp")
        if cls._looks_like_complete_json(temporary):
            return temporary
        return None

    @staticmethod
    def _copy_completed_trace(source: Path, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite profiler trace: {destination}"
            )
        try:
            with source.open("rb") as input_stream, destination.open("xb") as output:
                shutil.copyfileobj(input_stream, output, length=16 * 1024 * 1024)
            if destination.stat().st_size != source.stat().st_size:
                raise OSError("copied profiler trace size does not match staging file")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    def _export_trace(self, profiler: Any, destination: Path) -> None:
        staging_root = self.config.staging_dir
        if staging_root is None:
            requested = destination
            export_error: Exception | None = None
            try:
                profiler.export_chrome_trace(str(requested))
            except Exception as error:
                export_error = error
            completed = self._completed_export(requested)
            if completed is None:
                if export_error is not None:
                    raise export_error
                raise RuntimeError(
                    f"profiler did not complete Chrome trace export: {requested}"
                )
            if completed != destination:
                self._copy_completed_trace(completed, destination)
            return

        staging_root = Path(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="graphcast-profile-",
            dir=staging_root,
        ) as temporary_dir:
            requested = Path(temporary_dir) / destination.name
            export_error = None
            try:
                profiler.export_chrome_trace(str(requested))
            except Exception as error:
                export_error = error
            completed = self._completed_export(requested)
            if completed is None:
                if export_error is not None:
                    raise export_error
                raise RuntimeError(
                    f"profiler did not complete staged trace export: {requested}"
                )
            self._copy_completed_trace(completed, destination)

    def _cycle_output_path(self, stem: str, suffix: str) -> Path:
        cycle = len(self._trace_paths)
        name = (
            f"{stem}{suffix}"
            if cycle == 1
            else f"{stem}_cycle_{cycle:03d}{suffix}"
        )
        return Path(self.config.output_dir) / name

    def _write_npu_trace_outputs(self, trace_path: Path) -> None:
        aggregate: dict[str, dict[str, Any]] = {}
        compute_task_types = {
            "AI_CORE",
            "AI_CPU",
            "AI_VECTOR_CORE",
            "FFTS_PLUS",
            "MIX_AIC",
            "MIX_AIV",
        }
        for event in iter_trace_events(trace_path):
            args = event.get("args")
            if (
                event.get("ph") != "X"
                or not isinstance(args, dict)
                or str(args.get("Task Type")) not in compute_task_types
            ):
                continue
            name = str(event.get("name", "<unnamed>"))
            duration = float(event.get("dur", 0.0))
            row = aggregate.setdefault(
                name,
                {
                    "name": name,
                    "calls": 0,
                    "cpu_time_total_us": None,
                    "self_cpu_time_total_us": None,
                    "device_time_total_us": 0.0,
                    "self_device_time_total_us": 0.0,
                    "cpu_memory_usage_bytes": None,
                    "device_memory_usage_bytes": None,
                    "flops": None,
                    "profiler_source": "torch_npu Ascend Hardware trace",
                },
            )
            row["calls"] += 1
            row["device_time_total_us"] += duration
            row["self_device_time_total_us"] += duration
        if not aggregate:
            raise RuntimeError(
                "torch_npu trace contains no Ascend Hardware compute events"
            )
        rows = sorted(
            aggregate.values(),
            key=lambda row: float(row["self_device_time_total_us"]),
            reverse=True,
        )[: self.config.top_k]
        self._cycle_output_path("hotspots", ".json").write_text(
            json.dumps(rows, indent=2) + "\n",
            encoding="utf-8",
        )
        table = ["name\tcalls\tdevice_time_total_us"]
        table.extend(
            f"{row['name']}\t{row['calls']}\t{row['device_time_total_us']:.3f}"
            for row in rows
        )
        self._cycle_output_path("hotspots", ".txt").write_text(
            "\n".join(table) + "\n",
            encoding="utf-8",
        )

    def _write_backend_metadata(self) -> None:
        payload = {
            "schema_version": 2,
            "backend": self._profiler_backend,
            "device_type": self.device.type,
            "activities": [str(activity) for activity in self._activities],
            "profiler_start_succeeded": self._start_succeeded,
            "split_active_traces": self.config.split_active_traces,
            "wait": self.config.wait,
            "warmup": self.config.warmup,
            "active": self.config.active,
            "repeat": self.config.repeat,
            "staging_dir": (
                str(self.config.staging_dir)
                if self.config.staging_dir is not None
                else None
            ),
            "completed_traces": [
                {"path": str(path), "bytes": path.stat().st_size}
                for path in self._trace_paths
            ],
            "raw_profile_capture_mode": (
                "torch_npu_raw_only_msprof_sqlite_offline"
                if self.device.type == "npu"
                else None
            ),
            "npu_declared_export_type": (
                "db" if self.device.type == "npu" else None
            ),
            "npu_profiler_level": (
                self.config.npu_profiler_level
                if self.device.type == "npu"
                else None
            ),
            "npu_op_attr": (
                self.config.npu_op_attr if self.device.type == "npu" else None
            ),
            "completed_raw_profiles": list(self._npu_raw_profiles),
            "npu_postprocess_state": self._npu_postprocess_state,
            "npu_postprocess_error": self._npu_postprocess_error,
            "profiled_steps": sorted(self._npu_profiled_steps),
        }
        (Path(self.config.output_dir) / "profiler_backend.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_npu_raw_manifest(self) -> None:
        payload = {
            "schema_version": 1,
            "capture_contract": "one finalized raw CANN profile per active update",
            "traceview_export_during_training": False,
            "captures": list(self._npu_raw_profiles),
        }
        (Path(self.config.output_dir) / "npu_raw_profiles.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _capture_runtime_step(self, capture_index: int) -> int | None:
        if not self.config.split_active_traces:
            return None
        cycle_length = self.config.wait + self.config.warmup + self.config.active
        cycle = capture_index // self.config.active
        active_offset = capture_index % self.config.active
        return (
            cycle * cycle_length
            + self.config.wait
            + self.config.warmup
            + active_offset
            + 1
        )

    def _write_aggregate_hotspots(self) -> None:
        """Write one comparable top-k across all independent active traces."""

        aggregate: dict[str, dict[str, Any]] = {}
        additive_fields = (
            "cpu_time_total_us",
            "self_cpu_time_total_us",
            "device_time_total_us",
            "self_device_time_total_us",
            "flops",
        )
        for capture_index, rows in enumerate(self._hotspot_rows_by_trace):
            runtime_step = self._capture_runtime_step(capture_index)
            for source in rows:
                name = str(source["name"])
                row = aggregate.setdefault(
                    name,
                    {
                        "name": name,
                        "calls": 0,
                        **{field: None for field in additive_fields},
                        "cpu_memory_usage_bytes": None,
                        "device_memory_usage_bytes": None,
                        "profiled_runtime_steps": [],
                        "profiler_source": (
                            "torch.profiler key_averages across independent "
                            "active-step traces"
                        ),
                    },
                )
                row["calls"] += int(source.get("calls", 0))
                for field in additive_fields:
                    value = source.get(field)
                    if value is None:
                        continue
                    row[field] = float(row[field] or 0.0) + float(value)
                if runtime_step is not None:
                    row["profiled_runtime_steps"].append(runtime_step)
        for row in aggregate.values():
            row["profiled_runtime_steps"] = sorted(
                set(row["profiled_runtime_steps"])
            )
        rows = sorted(
            aggregate.values(),
            key=lambda row: float(
                row["self_device_time_total_us"]
                or row["device_time_total_us"]
                or row["self_cpu_time_total_us"]
                or 0.0
            ),
            reverse=True,
        )[: self.config.top_k]
        output_dir = Path(self.config.output_dir)
        (output_dir / "hotspots_aggregate.json").write_text(
            json.dumps(rows, indent=2) + "\n",
            encoding="utf-8",
        )
        table = [
            "rank\tname\tcalls\tself_device_time_total_us\t"
            "device_time_total_us\tprofiled_runtime_steps"
        ]
        table.extend(
            f"{rank}\t{row['name']}\t{row['calls']}\t"
            f"{float(row['self_device_time_total_us'] or 0.0):.3f}\t"
            f"{float(row['device_time_total_us'] or 0.0):.3f}\t"
            f"{','.join(str(step) for step in row['profiled_runtime_steps'])}"
            for rank, row in enumerate(rows, start=1)
        )
        (output_dir / "hotspots_aggregate.txt").write_text(
            "\n".join(table) + "\n",
            encoding="utf-8",
        )

    def _write_outputs(self, profiler: Any) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.device.type == "npu":
            self._write_npu_trace_outputs(self._trace_paths[-1])
            return
        if not hasattr(profiler, "key_averages"):
            return
        averages = profiler.key_averages()
        sort_candidates = {
            "musa": "self_musa_time_total",
            "cuda": "self_cuda_time_total",
            "xpu": "self_xpu_time_total",
        }
        sort_by = sort_candidates.get(self.device.type, "self_cpu_time_total")
        try:
            table = averages.table(sort_by=sort_by, row_limit=self.config.top_k)
        except (AttributeError, KeyError, RuntimeError):
            table = averages.table(
                sort_by="self_cpu_time_total", row_limit=self.config.top_k
            )
        self._cycle_output_path("hotspots", ".txt").write_text(
            table + "\n", encoding="utf-8"
        )

        rows = []
        for event in averages:
            rows.append(
                {
                    "name": str(event.key),
                    "calls": int(event.count),
                    "cpu_time_total_us": _first_number(event, "cpu_time_total"),
                    "self_cpu_time_total_us": _first_number(
                        event, "self_cpu_time_total"
                    ),
                    "device_time_total_us": _first_number(
                        event,
                        "device_time_total",
                        "musa_time_total",
                        "cuda_time_total",
                        "xpu_time_total",
                    ),
                    "self_device_time_total_us": _first_number(
                        event,
                        "self_device_time_total",
                        "self_musa_time_total",
                        "self_cuda_time_total",
                        "self_xpu_time_total",
                    ),
                    "cpu_memory_usage_bytes": _first_number(
                        event, "cpu_memory_usage"
                    ),
                    "device_memory_usage_bytes": _first_number(
                        event,
                        "device_memory_usage",
                        "musa_memory_usage",
                        "cuda_memory_usage",
                    ),
                    "flops": _first_number(event, "flops"),
                }
            )
        rows.sort(
            key=lambda row: float(
                row["self_device_time_total_us"]
                or row["self_cpu_time_total_us"]
                or 0
            ),
            reverse=True,
        )
        self._hotspot_rows_by_trace.append(rows)
        self._write_aggregate_hotspots()
        self._cycle_output_path("hotspots", ".json").write_text(
            json.dumps(rows[: self.config.top_k], indent=2) + "\n",
            encoding="utf-8",
        )
