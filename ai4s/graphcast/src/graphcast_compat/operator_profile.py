"""Persist top-level PyTorch operator timings from a device profiler trace."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch


def _normalized_operator_name(name: str) -> str:
    return name.replace("aten::", "", 1).removeprefix("aten.")


def _matches_operator(name: str, selected_ops: set[str]) -> bool:
    normalized = _normalized_operator_name(name)
    base_name = normalized.split(".", 1)[0]
    selected_bases = {item.split(".", 1)[0] for item in selected_ops}
    return normalized in selected_ops or base_name in selected_bases


def _first_cuda_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device if value.device.type == "cuda" else None
    if isinstance(value, dict):
        for item in value.values():
            device = _first_cuda_device(item)
            if device is not None:
                return device
    if isinstance(value, list | tuple):
        for item in value:
            device = _first_cuda_device(item)
            if device is not None:
                return device
    return None


class OperatorEventTimer:
    """Measure selected dispatcher calls with device-stream event pairs."""

    def __init__(self, output_path: str | Path | None, selected_ops: str | None) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.selected_ops = {
            _normalized_operator_name(item.strip())
            for item in (selected_ops or "").split(",")
            if item.strip()
        }
        self._mode: Any | None = None
        self._records: list[tuple[str, str, Any, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.output_path is not None and bool(self.selected_ops)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._mode is not None:
            raise RuntimeError("operator event timer is already running")

        from torch.utils._python_dispatch import TorchDispatchMode

        timer = self

        class _EventMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                call_kwargs = kwargs or {}
                name = str(func)
                if not _matches_operator(name, timer.selected_ops):
                    return func(*args, **call_kwargs)

                device = _first_cuda_device((args, call_kwargs))
                if device is None:
                    return func(*args, **call_kwargs)
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    result = func(*args, **call_kwargs)
                    end.record(stream)
                timer._records.append((name, str(device), start, end))
                return result

        self._records = []
        self._mode = _EventMode()
        self._mode.__enter__()

    def stop(self, *, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        if self._mode is None:
            raise RuntimeError("operator event timer is not running")

        self._mode.__exit__(None, None, None)
        self._mode = None
        devices = sorted({device for _, device, _, _ in self._records})
        for device in devices:
            torch.cuda.synchronize(torch.device(device))

        totals: dict[str, dict[str, Any]] = {}
        for name, device, start, end in self._records:
            current = totals.setdefault(
                name,
                {"name": name, "count": 0, "device_time_total_ms": 0.0, "devices": {}},
            )
            elapsed_ms = float(start.elapsed_time(end))
            current["count"] += 1
            current["device_time_total_ms"] += elapsed_ms
            current["devices"][device] = current["devices"].get(device, 0.0) + elapsed_ms

        payload = {
            "schema_version": 1,
            "timer": "dispatcher_cuda_event",
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "metadata": metadata or {},
            "selected_ops": sorted(self.selected_ops),
            "events": sorted(totals.values(), key=lambda event: event["name"]),
        }
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class OperatorProfiler:
    """Collect one bounded region without changing the model execution path."""

    def __init__(
        self,
        output_path: str | Path | None,
        selected_ops: str | None = None,
    ) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.selected_ops = {
            _normalized_operator_name(item.strip())
            for item in (selected_ops or "").split(",")
            if item.strip()
        }
        self._profile: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.output_path is not None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._profile is not None:
            raise RuntimeError("operator profiler is already running")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            # MetaX and other CUDA-compatible runtimes expose device events as CUDA.
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profile = torch.profiler.profile(
            activities=activities,
            # OperatorTrace records representative shapes separately without
            # retaining every profiled tensor until post-processing.
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        )
        self._profile.start()

    def stop(self, *, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        if self._profile is None:
            raise RuntimeError("operator profiler is not running")

        profile = self._profile
        self._profile = None
        profile.stop()

        events = []
        if self.selected_ops:
            # Filtering raw events avoids grouping every operator in a full model
            # step while preserving the same totals as key_averages().
            totals: dict[tuple[str, str], dict[str, Any]] = {}
            for event in profile.events():
                if not event.name.startswith("aten::") or not _matches_operator(
                    event.name,
                    self.selected_ops,
                ):
                    continue
                key = (event.name, event.overload_name)
                current = totals.setdefault(
                    key,
                    {
                        "name": event.name,
                        "overload_name": event.overload_name,
                        "count": 0,
                        "cpu_time_total_us": 0.0,
                        "self_cpu_time_total_us": 0.0,
                        "device_time_total_us": 0.0,
                        "self_device_time_total_us": 0.0,
                    },
                )
                current["count"] += int(event.count)
                current["cpu_time_total_us"] += float(event.cpu_time_total)
                current["self_cpu_time_total_us"] += float(event.self_cpu_time_total)
                current["device_time_total_us"] += float(event.device_time_total)
                current["self_device_time_total_us"] += float(event.self_device_time_total)
            events = list(totals.values())
        else:
            for event in profile.key_averages():
                if not event.key.startswith("aten::"):
                    continue
                events.append(
                    {
                        "name": event.key,
                        "overload_name": event.overload_name,
                        "count": int(event.count),
                        "cpu_time_total_us": float(event.cpu_time_total),
                        "self_cpu_time_total_us": float(event.self_cpu_time_total),
                        "device_time_total_us": float(event.device_time_total),
                        "self_device_time_total_us": float(event.self_device_time_total),
                    }
                )

        payload = {
            "schema_version": 1,
            "profiler": "kineto",
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "metadata": metadata or {},
            "selected_ops": sorted(self.selected_ops),
            "events": events,
        }
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
