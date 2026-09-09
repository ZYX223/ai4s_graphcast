"""Offline statistics for raw ``torch_npu``/CANN profiling captures.

The pinned Ascend runtime can collect complete profiling data but its
``TraceViewParser`` fails while building a multi-gigabyte Chrome JSON trace.
GraphCast does not need that lossy intermediate for performance accounting:
``msprof`` can export the supported SQLite representation. ``TASK`` is the
authoritative device timeline. ``COMPUTE_TASK_INFO`` adds operator metadata,
but CANN can omit those rows for graph/FFTS tasks even when ``TASK.taskType``
unambiguously identifies an AI compute task. Busy time therefore uses both
sources; hotspot names fall back to the explicit task type when operator
metadata is unavailable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Any, Mapping, Sequence


ASCEND_DB_BUSY_DEFINITION = (
    "mean across participating devices of each device's unioned CANN TASK "
    "intervals classified as compute by COMPUTE_TASK_INFO or TASK.taskType, "
    "clipped to SESSION_TIME_INFO"
)
_ASCEND_DB_COMPUTE_TASK_TYPES = (
    "AI_CORE",
    "AI_CPU",
    "AI_VECTOR_CORE",
    "FFTS_PLUS",
    "MIX_AIC",
    "MIX_AIV",
    "KERNEL_AICORE",
    "KERNEL_AICPU",
    "KERNEL_AIVEC",
    "KERNEL_MIX_AIC",
    "KERNEL_MIX_AIV",
)
_REQUIRED_TABLES = {
    "COMPUTE_TASK_INFO",
    "SESSION_TIME_INFO",
    "STRING_IDS",
    "TASK",
}


def _atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalized_devices(devices: Sequence[str | int]) -> tuple[str, ...]:
    normalized = tuple(str(device) for device in devices)
    if not normalized or any(not device for device in normalized):
        raise ValueError("expected_devices must contain at least one device")
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected_devices must be unique")
    return normalized


def _profile_container(raw_profile_path: Path) -> Path:
    raw = raw_profile_path.expanduser().resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"Ascend raw profile directory is absent: {raw}")
    candidates = sorted(path for path in raw.glob("PROF_*") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one CANN PROF_* directory below {raw}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _database_candidates(profile_container: Path) -> list[Path]:
    return sorted(
        path
        for path in profile_container.rglob("msprof_*.db")
        if path.is_file() and path.stat().st_size > 0
    )


def _validate_database(database: Path) -> None:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = sorted(_REQUIRED_TABLES - tables)
        if missing:
            raise RuntimeError(
                f"CANN database {database} is missing tables: {','.join(missing)}"
            )
        task_count = int(connection.execute("SELECT COUNT(*) FROM TASK").fetchone()[0])
        compute_count = int(
            connection.execute("SELECT COUNT(*) FROM COMPUTE_TASK_INFO").fetchone()[0]
        )
        session_count = int(
            connection.execute("SELECT COUNT(*) FROM SESSION_TIME_INFO").fetchone()[0]
        )
        if task_count <= 0 or compute_count <= 0 or session_count <= 0:
            raise RuntimeError(
                "CANN database has no TASK, COMPUTE_TASK_INFO, or session rows"
            )
    finally:
        connection.close()


def ensure_msprof_database(
    raw_profile_path: str | Path,
    *,
    export_log: str | Path,
) -> Path:
    """Return a validated CANN SQLite export, creating it when necessary."""

    container = _profile_container(Path(raw_profile_path))
    existing = _database_candidates(container)
    if existing:
        database = max(existing, key=lambda path: (path.stat().st_mtime_ns, path.name))
        _validate_database(database)
        return database

    executable = shutil.which("msprof")
    if executable is None:
        raise RuntimeError("msprof is not available in the Ascend runtime PATH")
    log_path = Path(export_log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--export=on",
        "--type=db",
        f"--output={container}",
    ]
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("command=" + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=str(container),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        stream.write(f"returncode={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"msprof SQLite export failed with code {completed.returncode}; "
            f"see {log_path}"
        )
    created = _database_candidates(container)
    if not created:
        raise RuntimeError(
            f"msprof reported success but created no SQLite database below {container}"
        )
    database = max(created, key=lambda path: (path.stat().st_mtime_ns, path.name))
    _validate_database(database)
    return database


def _unioned_busy_ns(
    connection: sqlite3.Connection,
    *,
    session_start_ns: int,
    session_end_ns: int,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Stream globally start-ordered compute tasks and union them per device."""

    busy_by_device: dict[int, int] = defaultdict(int)
    event_count_by_device: dict[int, int] = defaultdict(int)
    fallback_event_count_by_device: dict[int, int] = defaultdict(int)
    active_interval: dict[int, tuple[int, int]] = {}
    placeholders = ",".join("?" for _ in _ASCEND_DB_COMPUTE_TASK_TYPES)
    query = f"""
        SELECT
          task.deviceId,
          task.startNs,
          task.endNs,
          compute.globalTaskId IS NOT NULL AS has_operator_metadata
        FROM TASK AS task
        LEFT JOIN COMPUTE_TASK_INFO AS compute
          ON compute.globalTaskId = task.globalTaskId
        LEFT JOIN STRING_IDS AS task_types
          ON task_types.id = task.taskType
        WHERE compute.globalTaskId IS NOT NULL
           OR task_types.value IN ({placeholders})
        ORDER BY task.startNs, task.endNs
    """
    for raw_device, raw_start, raw_end, has_metadata in connection.execute(
        query, _ASCEND_DB_COMPUTE_TASK_TYPES
    ):
        device = int(raw_device)
        start = max(int(raw_start), session_start_ns)
        end = min(int(raw_end), session_end_ns)
        if end <= start:
            continue
        event_count_by_device[device] += 1
        if not bool(has_metadata):
            fallback_event_count_by_device[device] += 1
        current = active_interval.get(device)
        if current is None:
            active_interval[device] = (start, end)
        elif start <= current[1]:
            active_interval[device] = (current[0], max(current[1], end))
        else:
            busy_by_device[device] += current[1] - current[0]
            active_interval[device] = (start, end)
    for device, (start, end) in active_interval.items():
        busy_by_device[device] += end - start
    return (
        dict(busy_by_device),
        dict(event_count_by_device),
        dict(fallback_event_count_by_device),
    )


def _hotspots(
    connection: sqlite3.Connection,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in _ASCEND_DB_COMPUTE_TASK_TYPES)
    query = f"""
        SELECT
          CASE
            WHEN compute.globalTaskId IS NOT NULL THEN
              COALESCE(operator_names.value, CAST(compute.name AS TEXT))
            ELSE
              '[task_type] ' || COALESCE(
                task_types.value, CAST(task.taskType AS TEXT)
              )
          END AS operator_name,
          CASE
            WHEN compute.globalTaskId IS NOT NULL
              THEN 'COMPUTE_TASK_INFO'
            ELSE 'TASK.taskType fallback'
          END AS metadata_source,
          CASE
            WHEN compute.globalTaskId IS NOT NULL THEN
              COALESCE(
                hf32_flags.value,
                CAST(compute.hf32Eligible AS TEXT),
                'N/A'
              )
            ELSE 'METADATA_UNAVAILABLE'
          END AS hf32_eligible,
          COUNT(*) AS calls,
          SUM(task.endNs - task.startNs) AS device_time_ns
        FROM TASK AS task
        LEFT JOIN COMPUTE_TASK_INFO AS compute
          ON compute.globalTaskId = task.globalTaskId
        LEFT JOIN STRING_IDS AS operator_names
          ON operator_names.id = compute.name
        LEFT JOIN STRING_IDS AS task_types
          ON task_types.id = task.taskType
        LEFT JOIN STRING_IDS AS hf32_flags
          ON hf32_flags.id = compute.hf32Eligible
        WHERE task.endNs > task.startNs
          AND (
            compute.globalTaskId IS NOT NULL
            OR task_types.value IN ({placeholders})
          )
        GROUP BY operator_name, metadata_source, hf32_eligible
        ORDER BY device_time_ns DESC, calls DESC, operator_name ASC
    """
    aggregate: dict[str, dict[str, Any]] = {}
    for name, source, hf32_eligible, calls, duration_ns in connection.execute(
        query, _ASCEND_DB_COMPUTE_TASK_TYPES
    ):
        operator_name = str(name)
        row = aggregate.setdefault(
            operator_name,
            {
                "name": operator_name,
                "calls": 0,
                "device_time_total_us": 0.0,
                "self_device_time_total_us": 0.0,
                "metadata_sources": set(),
                "hf32_eligible_task_counts": defaultdict(int),
                "hf32_eligible_device_time_us": defaultdict(float),
                "profiler_source": (
                    "torch_npu CANN SQLite TASK plus optional "
                    "COMPUTE_TASK_INFO metadata"
                ),
            },
        )
        duration_us = float(duration_ns) / 1.0e3
        row["calls"] += int(calls)
        row["device_time_total_us"] += duration_us
        row["self_device_time_total_us"] += duration_us
        row["metadata_sources"].add(str(source))
        row["hf32_eligible_task_counts"][str(hf32_eligible)] += int(calls)
        row["hf32_eligible_device_time_us"][str(hf32_eligible)] += duration_us
    rows = sorted(
        aggregate.values(),
        key=lambda row: (
            -float(row["device_time_total_us"]),
            -int(row["calls"]),
            str(row["name"]),
        ),
    )[: int(limit)]
    for row in rows:
        row["metadata_sources"] = sorted(row["metadata_sources"])
        row["hf32_eligible_task_counts"] = dict(
            sorted(row["hf32_eligible_task_counts"].items())
        )
        row["hf32_eligible_device_time_us"] = dict(
            sorted(row["hf32_eligible_device_time_us"].items())
        )
    return rows


def _hf32_eligibility(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, float | int]]:
    query = """
        SELECT
          COALESCE(
            hf32_flags.value,
            CAST(compute.hf32Eligible AS TEXT),
            'N/A'
          ) AS hf32_eligible,
          COUNT(*) AS calls,
          SUM(task.endNs - task.startNs) AS device_time_ns
        FROM TASK AS task
        JOIN COMPUTE_TASK_INFO AS compute
          ON compute.globalTaskId = task.globalTaskId
        LEFT JOIN STRING_IDS AS hf32_flags
          ON hf32_flags.id = compute.hf32Eligible
        WHERE task.endNs > task.startNs
        GROUP BY hf32_eligible
        ORDER BY hf32_eligible
    """
    counts: dict[str, int] = {}
    times: dict[str, float] = {}
    for flag, calls, duration_ns in connection.execute(query):
        counts[str(flag)] = int(calls)
        times[str(flag)] = float(duration_ns) / 1.0e3
    return {"task_counts": counts, "device_time_us": times}


def analyze_ascend_profile(
    raw_profile_path: str | Path,
    *,
    expected_devices: Sequence[str | int],
    runtime_step: int,
    output_dir: str | Path,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Export and analyze one raw profile corresponding to one active update."""

    if runtime_step <= 0:
        raise ValueError("runtime_step must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    devices = _normalized_devices(expected_devices)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    label = f"update_{runtime_step:08d}"
    database = ensure_msprof_database(
        raw_profile_path,
        export_log=destination / f"msprof_export_{label}.log",
    )
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        session_row = connection.execute(
            "SELECT MIN(startTimeNs), MAX(endTimeNs) FROM SESSION_TIME_INFO"
        ).fetchone()
        if session_row is None or session_row[0] is None or session_row[1] is None:
            raise RuntimeError("CANN database contains no session time interval")
        session_start_ns, session_end_ns = map(int, session_row)
        if session_end_ns <= session_start_ns:
            raise RuntimeError("CANN profiling session has a non-positive duration")
        (
            busy_ns_by_raw_device,
            event_counts,
            fallback_event_counts,
        ) = _unioned_busy_ns(
            connection,
            session_start_ns=session_start_ns,
            session_end_ns=session_end_ns,
        )
        raw_devices = tuple(sorted(busy_ns_by_raw_device))
        if len(raw_devices) != len(devices):
            raise RuntimeError(
                "CANN compute device count does not match the requested topology: "
                f"raw={raw_devices}, expected={devices}"
            )
        ordered_expected = tuple(sorted(devices, key=int))
        device_map = dict(zip(raw_devices, ordered_expected, strict=True))
        busy_us_by_device = {
            device_map[raw]: float(busy_ns_by_raw_device[raw]) / 1.0e3
            for raw in raw_devices
        }
        event_count_by_device = {
            device_map[raw]: int(event_counts[raw]) for raw in raw_devices
        }
        fallback_event_count_by_device = {
            device_map[raw]: int(fallback_event_counts.get(raw, 0))
            for raw in raw_devices
        }
        total_busy_us = float(sum(busy_us_by_device.values()))
        profile_wall_ms = (session_end_ns - session_start_ns) / 1.0e6
        average_busy_ms = total_busy_us / len(devices) / 1.0e3
        if average_busy_ms > profile_wall_ms + 1.0e-3:
            raise RuntimeError(
                "mean CANN device busy time exceeds the profiling session wall time"
            )
        rows = _hotspots(connection, limit=top_k)
        hf32_eligibility = _hf32_eligibility(connection)
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("CANN database contains no compute hotspot rows")

    profile = {
        "profile_step_index": runtime_step - 1,
        "profile_step_wall_ms": profile_wall_ms,
        "device_busy_us_by_device": busy_us_by_device,
        "device_busy_ms": average_busy_ms,
        "total_device_busy_us": total_busy_us,
        "total_device_busy_ms": total_busy_us / 1.0e3,
        "device_busy_definition": ASCEND_DB_BUSY_DEFINITION,
        "profile_source": "torch_npu raw capture -> msprof SQLite",
        "raw_profile_path": str(Path(raw_profile_path).expanduser().resolve()),
        "msprof_database": str(database),
        "session_start_ns": session_start_ns,
        "session_end_ns": session_end_ns,
        "compute_event_count_by_device": event_count_by_device,
        "task_type_fallback_compute_event_count_by_device": (
            fallback_event_count_by_device
        ),
        "compute_task_metadata_coverage": {
            "total_compute_events": sum(event_count_by_device.values()),
            "events_with_compute_task_info": (
                sum(event_count_by_device.values())
                - sum(fallback_event_count_by_device.values())
            ),
            "events_classified_by_task_type_fallback": sum(
                fallback_event_count_by_device.values()
            ),
        },
        "hf32_eligible_task_counts": hf32_eligibility["task_counts"],
        "hf32_eligible_device_time_us": hf32_eligibility["device_time_us"],
        "raw_to_runtime_device": {
            str(raw): mapped for raw, mapped in device_map.items()
        },
    }
    _atomic_json_dump(destination / f"profile_{label}.json", profile)
    _atomic_json_dump(destination / f"hotspots_{label}.json", rows)
    _write_hotspot_table(destination / f"hotspots_{label}.txt", rows)
    return profile, rows


def _write_hotspot_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["rank\tname\tcalls\tdevice_time_total_us"]
    lines.extend(
        f"{rank}\t{row['name']}\t{row['calls']}\t"
        f"{float(row['device_time_total_us']):.3f}"
        for rank, row in enumerate(rows, start=1)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def postprocess_ascend_profiles(
    captures: Sequence[Mapping[str, Any]],
    *,
    expected_devices: Sequence[str | int],
    output_dir: str | Path,
    top_k: int,
) -> dict[int, dict[str, Any]]:
    """Analyze independent active-update captures and write aggregate hotspots."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    profiles: dict[int, dict[str, Any]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for capture in captures:
        runtime_step = int(capture["local_step"])
        if runtime_step in profiles:
            raise ValueError(f"duplicate Ascend profile runtime step {runtime_step}")
        profile, rows = analyze_ascend_profile(
            capture["path"],
            expected_devices=expected_devices,
            runtime_step=runtime_step,
            output_dir=destination,
            top_k=top_k,
        )
        profiles[runtime_step] = profile
        for row in rows:
            name = str(row["name"])
            merged = aggregate.setdefault(
                name,
                {
                    "name": name,
                    "calls": 0,
                    "device_time_total_us": 0.0,
                    "self_device_time_total_us": 0.0,
                    "profiled_updates": [],
                    "metadata_sources": set(),
                    "hf32_eligible_task_counts": defaultdict(int),
                    "hf32_eligible_device_time_us": defaultdict(float),
                    "profiler_source": row["profiler_source"],
                },
            )
            merged["calls"] += int(row["calls"])
            merged["device_time_total_us"] += float(row["device_time_total_us"])
            merged["self_device_time_total_us"] += float(
                row["self_device_time_total_us"]
            )
            merged["profiled_updates"].append(runtime_step)
            merged["metadata_sources"].update(row.get("metadata_sources", ()))
            for flag, calls in row.get("hf32_eligible_task_counts", {}).items():
                merged["hf32_eligible_task_counts"][str(flag)] += int(calls)
            for flag, duration_us in row.get(
                "hf32_eligible_device_time_us", {}
            ).items():
                merged["hf32_eligible_device_time_us"][str(flag)] += float(
                    duration_us
                )
    if not profiles:
        return {}
    aggregate_rows = sorted(
        aggregate.values(),
        key=lambda row: (
            -float(row["device_time_total_us"]),
            -int(row["calls"]),
            str(row["name"]),
        ),
    )[:top_k]
    for row in aggregate_rows:
        row["profiled_updates"] = sorted(set(row["profiled_updates"]))
        row["metadata_sources"] = sorted(row["metadata_sources"])
        row["hf32_eligible_task_counts"] = dict(
            sorted(row["hf32_eligible_task_counts"].items())
        )
        row["hf32_eligible_device_time_us"] = dict(
            sorted(row["hf32_eligible_device_time_us"].items())
        )
    _atomic_json_dump(destination / "profile_metrics.json", profiles)
    _atomic_json_dump(destination / "hotspots.json", aggregate_rows)
    _write_hotspot_table(destination / "hotspots.txt", aggregate_rows)
    _atomic_json_dump(destination / "hotspots_aggregate.json", aggregate_rows)
    _write_hotspot_table(
        destination / "hotspots_aggregate.txt",
        aggregate_rows,
    )
    top_twenty = aggregate_rows[:20]
    _atomic_json_dump(destination / "hotspots_top20.json", top_twenty)
    _write_hotspot_table(destination / "hotspots_top20.txt", top_twenty)
    return profiles


def _parse_devices(raw: str) -> tuple[str, ...]:
    return _normalized_devices(tuple(part.strip() for part in raw.split(",")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-profile", action="append", required=True)
    parser.add_argument("--runtime-steps", required=True)
    parser.add_argument("--expected-devices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args(argv)
    raw_profiles = tuple(args.raw_profile)
    runtime_steps = tuple(
        int(part.strip()) for part in args.runtime_steps.split(",") if part.strip()
    )
    if len(raw_profiles) != len(runtime_steps):
        raise ValueError("--raw-profile count must match --runtime-steps count")
    result = postprocess_ascend_profiles(
        [
            {"path": path, "local_step": step}
            for path, step in zip(raw_profiles, runtime_steps, strict=True)
        ],
        expected_devices=_parse_devices(args.expected_devices),
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    if len(result) != len(raw_profiles) or any(
        not math.isfinite(float(row["device_busy_ms"])) for row in result.values()
    ):
        raise RuntimeError("Ascend profile postprocessing self-check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
