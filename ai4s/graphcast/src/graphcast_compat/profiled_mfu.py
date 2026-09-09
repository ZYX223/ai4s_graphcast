"""Per-step, per-device busy-time extraction and MFU calculations.

PyTorch ``key_averages`` attributes device time to nested operators and therefore
cannot be used as hardware busy time.  This module instead reads complete kernel
events from the Chrome trace, clips them to each ``ProfilerStep`` window, and
unions overlapping intervals independently for every device.
"""

from __future__ import annotations

import json
import io
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterator, MutableMapping, Sequence


BUSY_MFU_FORMULA = (
    "global useful FLOPs / "
    "sum(per-device peak FLOPs/s * per-device kernel-busy seconds)"
)
SYSTEM_MFU_FORMULA = (
    "global useful FLOPs / "
    "(step wall seconds * participating devices * peak FLOPs/s/device)"
)
HFU_SYSTEM_FORMULA = (
    "(global useful FLOPs + measured activation-recompute Linear FLOPs) / "
    "(step wall seconds * participating devices * peak FLOPs/s/device)"
)
E2E_MFU_FORMULA = (
    "global useful FLOPs / "
    "(end-to-end step seconds * participating devices * peak FLOPs/s/device)"
)
DEVICE_BUSY_DEFINITION = (
    "mean across participating devices of each device's unioned kernel "
    "intervals inside the matching ProfilerStep"
)
DEVICE_SYS_DEFINITION = (
    "per-device optimizer-update compute wall time; total_device_sys_ms is "
    "that wall time multiplied by participating_device_count"
)
DEVICE_E2E_DEFINITION = (
    "per-device optimizer-update data-plus-compute wall time; "
    "total_device_e2e_ms is that wall time multiplied by "
    "participating_device_count"
)
_PROFILER_STEP = re.compile(r"^ProfilerStep#(?P<index>[0-9]+)$")
_MUSA_EMPTY_PROCESS_GROUP_DESCRIPTION = (
    '"Process Group Description": ,'
)
_TRACE_EVENTS_KEY = b'"traceEvents"'
_TRACE_READ_CHUNK_BYTES = 8 * 1024 * 1024
_ASCEND_COMPUTE_TASK_TYPES = {
    "AI_CORE",
    "AI_CPU",
    "AI_VECTOR_CORE",
    "FFTS_PLUS",
    "MIX_AIC",
    "MIX_AIV",
}


class _RepairingBinaryReader:
    """Bounded reader that repairs the torch_musa token across chunk edges."""

    def __init__(self, raw: BinaryIO):
        self._raw = raw
        self._pending = bytearray()
        self._carry = b""
        self._eof = False
        self._invalid = _MUSA_EMPTY_PROCESS_GROUP_DESCRIPTION.encode("utf-8")
        self._replacement = b'"Process Group Description": null,'

    def _fill(self, requested_bytes: int = 64 * 1024) -> None:
        if self._eof:
            return
        block = self._raw.read(max(64 * 1024, requested_bytes))
        combined = self._carry + block
        if block:
            carry_size = 0
            for candidate_size in range(
                min(len(self._invalid) - 1, len(combined)),
                0,
                -1,
            ):
                if combined.endswith(self._invalid[:candidate_size]):
                    carry_size = candidate_size
                    break
            safe_end = len(combined) - carry_size
            safe = combined[:safe_end]
            self._carry = combined[safe_end:]
        else:
            safe = combined
            self._carry = b""
            self._eof = True
        self._pending.extend(safe.replace(self._invalid, self._replacement))

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            while not self._eof:
                self._fill(_TRACE_READ_CHUNK_BYTES)
            result = bytes(self._pending)
            self._pending.clear()
            return result
        while len(self._pending) < size and not self._eof:
            self._fill(size - len(self._pending))
        result = bytes(self._pending[:size])
        del self._pending[:size]
        return result


def _seek_trace_event_array(stream: BinaryIO) -> None:
    """Position a binary stream immediately after the trace-event ``[``."""

    stream.seek(0)
    initial = stream.read(_TRACE_READ_CHUNK_BYTES)
    first_non_space = next(
        (index for index, value in enumerate(initial) if not chr(value).isspace()),
        None,
    )
    if first_non_space is None:
        raise ValueError("profiler trace is empty")
    if initial[first_non_space] == ord("["):
        stream.seek(first_non_space + 1)
        return

    stream.seek(0)
    carry = b""
    absolute_end = 0
    while True:
        block = stream.read(_TRACE_READ_CHUNK_BYTES)
        if not block:
            raise ValueError("profiler trace has no traceEvents array")
        combined = carry + block
        marker_index = combined.find(_TRACE_EVENTS_KEY)
        if marker_index >= 0:
            marker_end = (
                absolute_end - len(carry) + marker_index + len(_TRACE_EVENTS_KEY)
            )
            stream.seek(marker_end)
            while True:
                suffix = stream.read(_TRACE_READ_CHUNK_BYTES)
                if not suffix:
                    raise ValueError("traceEvents key has no array value")
                bracket_index = suffix.find(b"[")
                if bracket_index >= 0:
                    stream.seek(stream.tell() - len(suffix) + bracket_index + 1)
                    return
        carry = combined[-(len(_TRACE_EVENTS_KEY) - 1) :]
        absolute_end += len(block)


def _iter_trace_events_stdlib(
    trace_path: str | Path,
) -> Iterator[dict[str, Any]]:
    """Standard-library fallback with bounded torch_musa token repair.

    Both Kineto's ``{"traceEvents": [...]}`` root and torch_npu's top-level
    event list are accepted.
    """

    resolved = Path(trace_path)
    repair = _MUSA_EMPTY_PROCESS_GROUP_DESCRIPTION.encode("utf-8")
    repair_text = repair.decode("utf-8")
    replacement_text = '"Process Group Description": null,'
    with resolved.open("rb") as stream:
        _seek_trace_event_array(stream)
        text_stream = io.TextIOWrapper(stream, encoding="utf-8")
        decoder = json.JSONDecoder()
        buffer = ""
        position = 0
        end_of_file = False
        while True:
            if position >= _TRACE_READ_CHUNK_BYTES:
                buffer = buffer[position:]
                position = 0

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return

            try:
                event, event_end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                if end_of_file:
                    if buffer[position:].strip():
                        raise ValueError(
                            "profiler trace ends inside a trace event"
                        ) from error
                    raise ValueError(
                        "profiler trace ends before traceEvents closes"
                    ) from error
                block = text_stream.read(_TRACE_READ_CHUNK_BYTES)
                if block:
                    # Repair only when reading new text, retaining the incomplete
                    # event so a malformed token split across reads is repaired.
                    # Scanning per event repeatedly traverses megabytes of text.
                    buffer = (buffer[position:] + block).replace(
                        repair_text, replacement_text
                    )
                    position = 0
                else:
                    end_of_file = True
                continue
            if not isinstance(event, dict):
                raise ValueError("each trace event must be an object")
            yield event
            position = event_end


def iter_trace_events(trace_path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream Chrome/Ascend trace events without loading the trace in RAM.

    The C-backed ijson path is used when the runtime provides it. Ascend's
    pinned Python environment does not, so an incremental standard-library
    decoder remains available. Both paths repair the one known torch_musa
    metadata defect without modifying the original trace.
    """

    resolved = Path(trace_path)
    try:
        import ijson
    except ImportError:
        yield from _iter_trace_events_stdlib(resolved)
        return

    with resolved.open("rb") as stream:
        prefix_probe = stream.read(4096).lstrip()
        if not prefix_probe:
            raise ValueError("profiler trace is empty")
        item_prefix = "item" if prefix_probe.startswith(b"[") else "traceEvents.item"
        stream.seek(0)
        reader = _RepairingBinaryReader(stream)
        try:
            for event in ijson.items(reader, item_prefix, use_float=True):
                if not isinstance(event, dict):
                    raise ValueError("each trace event must be an object")
                yield event
        except ijson.JSONError as error:
            raise ValueError(
                "profiler trace ends inside a trace event or is invalid"
            ) from error


def _load_trace_events(trace_path: str | Path) -> list[Any]:
    """Compatibility helper for callers that explicitly need a small list."""

    return list(iter_trace_events(trace_path))


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"{field} must be a finite number") from error
    else:
        raise ValueError(f"{field} must be a finite number")
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _union_duration_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    duration = 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            duration += end - start
            start, end = next_start, next_end
    return duration + end - start


def _normalized_devices(expected_devices: Sequence[str | int]) -> tuple[str, ...]:
    devices = tuple(str(device) for device in expected_devices)
    if not devices or any(not device for device in devices):
        raise ValueError("expected_devices must contain at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError("expected_devices must be unique")
    return devices


def extract_profiled_step_busy(
    trace_path: str | Path,
    *,
    expected_devices: Sequence[str | int],
) -> dict[int, dict[str, Any]]:
    """Return kernel busy intervals keyed by one-based runtime step number.

    ``ProfilerStep#0`` maps to runtime step 1, ``ProfilerStep#2`` maps to
    runtime step 3, and unprofiled steps are deliberately absent.  Transfer and
    memset events are excluded because this metric describes compute busy time.
    """

    devices = _normalized_devices(expected_devices)
    windows: list[tuple[int, float, float]] = []
    ascend_hardware_pids: set[str] = set()
    ascend_raw_device_ids: set[str] = set()
    for event in iter_trace_events(trace_path):
        args = event.get("args")
        if (
            event.get("ph") == "M"
            and event.get("name") == "process_name"
            and isinstance(args, dict)
            and args.get("name") == "Ascend Hardware"
            and event.get("pid") is not None
        ):
            ascend_hardware_pids.add(str(event["pid"]))
        if (
            event.get("ph") == "X"
            and isinstance(args, dict)
            and str(args.get("Task Type")) in _ASCEND_COMPUTE_TASK_TYPES
            and args.get("Device Id") is not None
        ):
            ascend_raw_device_ids.add(str(args["Device Id"]))
        if (
            event.get("ph") != "X"
            or event.get("cat") == "gpu_user_annotation"
            or (
                isinstance(args, dict)
                and args.get("device") is not None
            )
        ):
            continue
        match = _PROFILER_STEP.match(str(event.get("name", "")))
        if match is None:
            continue
        start = _finite_number(event.get("ts"), field="ProfilerStep.ts")
        duration = _finite_number(
            event.get("dur"), field="ProfilerStep.dur", positive=True
        )
        windows.append((int(match.group("index")), start, start + duration))
    windows.sort()
    if not windows:
        raise ValueError("profiler trace contains no ProfilerStep window")
    if len({index for index, _, _ in windows}) != len(windows):
        raise ValueError("profiler trace contains duplicate ProfilerStep windows")

    ascend_device_by_raw_id: dict[str, str] = {}
    ascend_device_by_pid: dict[str, str] = {}
    if ascend_raw_device_ids:
        if len(ascend_raw_device_ids) != len(devices):
            raise ValueError(
                "Ascend trace device count does not match expected devices: "
                f"{len(ascend_raw_device_ids)} != {len(devices)}"
            )
        ascend_device_by_raw_id = dict(
            zip(
                sorted(ascend_raw_device_ids, key=lambda value: int(value)),
                sorted(devices, key=lambda value: int(value)),
                strict=True,
            )
        )
    elif ascend_hardware_pids:
        if len(ascend_hardware_pids) != len(devices):
            raise ValueError(
                "Ascend Hardware process count does not match expected devices: "
                f"{len(ascend_hardware_pids)} != {len(devices)}"
            )
        ascend_device_by_pid = dict(
            zip(
                sorted(ascend_hardware_pids),
                sorted(devices, key=lambda value: int(value)),
                strict=True,
            )
        )

    intervals: dict[int, dict[str, list[tuple[float, float]]]] = {
        index: {device: [] for device in devices}
        for index, _, _ in windows
    }
    expected_set = set(devices)
    for event in iter_trace_events(trace_path):
        if event.get("ph") != "X":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        if event.get("cat") == "kernel" and args.get("device") is not None:
            device = str(args["device"])
        elif str(args.get("Task Type")) in _ASCEND_COMPUTE_TASK_TYPES:
            raw_device = str(args.get("Device Id"))
            if raw_device in ascend_device_by_raw_id:
                device = ascend_device_by_raw_id[raw_device]
            elif str(event.get("pid")) in ascend_device_by_pid:
                device = ascend_device_by_pid[str(event["pid"])]
            else:
                continue
        else:
            continue
        if device not in expected_set:
            continue
        start = _finite_number(event.get("ts"), field="kernel.ts")
        duration = _finite_number(
            event.get("dur"), field="kernel.dur", positive=True
        )
        end = start + duration
        for index, window_start, window_end in windows:
            clipped_start = max(start, window_start)
            clipped_end = min(end, window_end)
            if clipped_end > clipped_start:
                intervals[index][device].append((clipped_start, clipped_end))

    missing_devices = [
        device
        for device in devices
        if not any(intervals[index][device] for index, _, _ in windows)
    ]
    if missing_devices:
        raise ValueError(
            "profiler trace has no compute events for expected devices: "
            + ",".join(missing_devices)
        )

    result: dict[int, dict[str, Any]] = {}
    for index, window_start, window_end in windows:
        busy_by_device = {
            device: _union_duration_us(intervals[index][device])
            for device in devices
        }
        total_busy_us = float(sum(busy_by_device.values()))
        profile_wall_us = window_end - window_start
        average_busy_us = total_busy_us / len(devices)
        # Clipping and per-device union make this a hard invariant.
        if average_busy_us > profile_wall_us + 1.0e-6:
            raise ValueError(
                "mean device busy time cannot exceed the ProfilerStep wall time"
            )
        runtime_step = index + 1
        result[runtime_step] = {
            "profile_step_index": index,
            "profile_step_wall_ms": profile_wall_us / 1.0e3,
            "device_busy_us_by_device": busy_by_device,
            "device_busy_ms": average_busy_us / 1.0e3,
            "total_device_busy_us": total_busy_us,
            "total_device_busy_ms": total_busy_us / 1.0e3,
            "device_busy_definition": DEVICE_BUSY_DEFINITION,
        }
    return result


def enrich_step_metrics(
    metrics: Sequence[MutableMapping[str, Any]],
    *,
    profiled_steps: dict[int, dict[str, Any]],
    device_count: int,
    peak_tflops_per_device: float,
) -> None:
    """Add auditable system and busy MFU fields to per-step metric rows.

    System MFU is available for every timed step.  Busy MFU is available only
    for steps present in the profiler trace; an unprofiled step receives
    ``None`` rather than reusing a cumulative value from another step.
    """

    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise ValueError("device_count must be an integer")
    if device_count <= 0:
        raise ValueError("device_count must be positive")
    peak_tflops_input = _finite_number(
        peak_tflops_per_device,
        field="peak_tflops_per_device",
    )
    if peak_tflops_input < 0:
        raise ValueError("peak_tflops_per_device must be non-negative")
    # A zero peak is the established "unknown" sentinel used by the
    # compatibility trainer.  Keep the measured FLOPs and device-time fields,
    # but do not manufacture MFU values until the customer supplies an
    # authoritative platform peak.
    peak_tflops = peak_tflops_input if peak_tflops_input > 0 else None
    aggregate_peak_tflops = (
        peak_tflops * device_count if peak_tflops is not None else None
    )

    for metric in metrics:
        step = int(metric["step"])
        step_seconds = _finite_number(
            metric["step_seconds"], field=f"step_seconds[{step}]", positive=True
        )
        useful_flops = _finite_number(
            metric["training_flops"],
            field=f"training_flops[{step}]",
            positive=True,
        )
        recompute_flops = _finite_number(
            metric.get("recompute_flops", 0.0),
            field=f"recompute_flops[{step}]",
        )
        if recompute_flops < 0:
            raise ValueError("recompute FLOPs must be non-negative")
        end_to_end_step_seconds = _finite_number(
            metric.get("end_to_end_step_seconds", step_seconds),
            field=f"end_to_end_step_seconds[{step}]",
            positive=True,
        )
        if end_to_end_step_seconds + 1.0e-12 < step_seconds:
            raise ValueError("end-to-end step time cannot be less than step wall time")
        achieved_tflops_aggregate = useful_flops / step_seconds / 1.0e12
        achieved_tflops_per_device = achieved_tflops_aggregate / device_count
        mfu_sys = (
            useful_flops
            / (step_seconds * aggregate_peak_tflops * 1.0e12)
            if aggregate_peak_tflops is not None
            else None
        )
        hfu_sys = (
            (useful_flops + recompute_flops)
            / (step_seconds * aggregate_peak_tflops * 1.0e12)
            if aggregate_peak_tflops is not None
            and not bool(metric.get("inference_only", False))
            else None
        )
        mfu_e2e = (
            useful_flops
            / (end_to_end_step_seconds * aggregate_peak_tflops * 1.0e12)
            if aggregate_peak_tflops is not None
            else None
        )

        profile = profiled_steps.get(step)
        if profile is None:
            device_busy_ms = None
            device_busy_ms_by_device: dict[str, float] = {}
            total_device_busy_ms = None
            total_device_busy_us = None
            mfu_busy = None
            profile_step_index = None
            profile_step_wall_ms = None
            device_busy_definition = DEVICE_BUSY_DEFINITION
            profile_source = None
        else:
            raw_by_device = profile["device_busy_us_by_device"]
            if not isinstance(raw_by_device, dict) or len(raw_by_device) != device_count:
                raise ValueError(
                    f"profiled step {step} device count does not match "
                    f"participating device count"
                )
            device_busy_us_by_device = {
                str(device): _finite_number(
                    busy_us,
                    field=f"device_busy_us[{step}][{device}]",
                )
                for device, busy_us in raw_by_device.items()
            }
            if any(value < 0 for value in device_busy_us_by_device.values()):
                raise ValueError("device busy time must be non-negative")
            total_device_busy_us = float(sum(device_busy_us_by_device.values()))
            expected_total = _finite_number(
                profile["total_device_busy_us"],
                field=f"total_device_busy_us[{step}]",
            )
            if not math.isclose(
                total_device_busy_us, expected_total, rel_tol=1.0e-12, abs_tol=1.0e-6
            ):
                raise ValueError(
                    f"profiled step {step} total busy time is inconsistent"
                )
            device_busy_ms_by_device = {
                device: value / 1.0e3
                for device, value in device_busy_us_by_device.items()
            }
            device_busy_ms = total_device_busy_us / device_count / 1.0e3
            total_device_busy_ms = total_device_busy_us / 1.0e3
            if device_busy_ms > step_seconds * 1.0e3 + 1.0e-6:
                raise ValueError(
                    f"profiled step {step} mean device busy time exceeds step wall time"
                )
            mfu_busy = (
                useful_flops
                / (peak_tflops * 1.0e12 * total_device_busy_us / 1.0e6)
                if peak_tflops is not None and total_device_busy_us > 0
                else None
            )
            profile_step_index = int(profile["profile_step_index"])
            profile_step_wall_ms = float(profile["profile_step_wall_ms"])
            device_busy_definition = str(
                profile.get("device_busy_definition", DEVICE_BUSY_DEFINITION)
            )
            profile_source = profile.get("profile_source")

        device_sys_ms = step_seconds * 1.0e3
        total_device_sys_ms = device_sys_ms * device_count
        device_e2e_ms = end_to_end_step_seconds * 1.0e3
        total_device_e2e_ms = device_e2e_ms * device_count

        metric.update(
            {
                "achieved_tflops": achieved_tflops_aggregate,
                "achieved_tflops_aggregate": achieved_tflops_aggregate,
                "achieved_tflops_per_device": achieved_tflops_per_device,
                "peak_tflops": peak_tflops,
                "peak_tflops_per_device": peak_tflops,
                "aggregate_peak_tflops": aggregate_peak_tflops,
                "participating_device_count": device_count,
                "device_busy_ms": device_busy_ms,
                "device_busy_ms_by_device": device_busy_ms_by_device,
                "total_device_busy_ms": total_device_busy_ms,
                "total_device_busy_us": total_device_busy_us,
                "device_busy_definition": device_busy_definition,
                "device_sys_ms": device_sys_ms,
                "total_device_sys_ms": total_device_sys_ms,
                "device_sys_definition": DEVICE_SYS_DEFINITION,
                "device_e2e_ms": device_e2e_ms,
                "total_device_e2e_ms": total_device_e2e_ms,
                "device_e2e_definition": DEVICE_E2E_DEFINITION,
                "busy_profiled": profile is not None,
                "profile_step_index": profile_step_index,
                "profile_step_wall_ms": profile_step_wall_ms,
                "profile_source": profile_source,
                "mfu_busy": mfu_busy,
                "mfu_sys": mfu_sys,
                "mfu_model_sys": mfu_sys,
                "hfu_sys": hfu_sys,
                "mfu_e2e": mfu_e2e,
                "mfu_task_busy": mfu_busy,
                "mfu_system": mfu_sys,
                "mfu_wall_per_device": mfu_sys,
                "mfu": mfu_sys,
                "recompute_flops": recompute_flops,
                "executed_training_flops": useful_flops + recompute_flops,
                "mfu_busy_formula": BUSY_MFU_FORMULA,
                "mfu_sys_formula": SYSTEM_MFU_FORMULA,
                "mfu_model_sys_formula": SYSTEM_MFU_FORMULA,
                "hfu_sys_formula": HFU_SYSTEM_FORMULA,
                "mfu_e2e_formula": E2E_MFU_FORMULA,
            }
        )
