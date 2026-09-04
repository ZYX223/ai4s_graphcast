from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools" / "bf16_inference"
SCRIPT = TOOLS_ROOT / "compare_bf16_trajectories.py"
SPEC = importlib.util.spec_from_file_location("compare_bf16_trajectories", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DRIVER_PATH = TOOLS_ROOT / "run_with_trajectory_reference.py"
DRIVER_SPEC = importlib.util.spec_from_file_location(
    "run_with_trajectory_reference", DRIVER_PATH
)
assert DRIVER_SPEC is not None and DRIVER_SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(DRIVER)
LOCAL_RUNNER = TOOLS_ROOT / "run_bf16_case_local_then_publish.sh"
PERFORMANCE_RUNNER = TOOLS_ROOT / "run_bf16_performance_matrix.sh"


def _trajectory(values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {
            "temperature": (
                ("init_time", "lead_time", "latitude", "longitude"), values
            )
        },
        coords={
            "init_time": np.asarray(["2022-01-01T00"], dtype="datetime64[h]"),
            "lead_time": np.asarray([6, 12], dtype="timedelta64[h]"),
            "latitude": [-60.0, 0.0],
            "longitude": [0.0, 180.0],
        },
    )


def test_reference_reader_does_not_decode_timedelta_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_path = tmp_path / "reference.nc"
    reference_path.touch()
    expected = xr.Dataset()
    observed: dict[str, object] = {}

    def fake_open_dataset(path: Path, **kwargs: object) -> xr.Dataset:
        observed["path"] = path
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(DRIVER.xr, "open_dataset", fake_open_dataset)

    actual = DRIVER._open_reference_dataset(reference_path)

    assert actual is expected
    assert observed == {"path": reference_path, "decode_timedelta": False}


def test_reports_parity_without_claiming_weather_skill(tmp_path: Path) -> None:
    reference = _trajectory(np.zeros((1, 2, 2, 2), dtype=np.float32))
    candidate = reference.copy(deep=True)
    candidate["temperature"].values[:, 1] = 2.0
    stddev = xr.Dataset({"temperature": xr.DataArray(4.0)})
    reference_path = tmp_path / "reference.nc"
    candidate_path = tmp_path / "candidate.nc"
    stddev_path = tmp_path / "stddev.nc"
    output = tmp_path / "diagnostics.json"
    reference.to_netcdf(reference_path)
    candidate.to_netcdf(candidate_path)
    stddev.to_netcdf(stddev_path)

    assert MODULE.main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--reference", str(reference_path),
            "--candidate", str(candidate_path),
            "--stddev", str(stddev_path),
            "--output", str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["methodology"]["weather_skill"].startswith("not_measured")
    assert payload["lead_count"] == 2
    assert payload["worst"]["normalized_max_abs"] == pytest.approx(0.5)
    assert payload["worst"]["normalized_rmse"] == pytest.approx(0.5)
    assert payload["inputs"]["reference"]["sha256"]


def test_streaming_writer_reduces_each_candidate_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = xr.Dataset(
        {
            "temperature": (
                ("batch", "lead_time", "lat", "lon"),
                np.zeros((1, 2, 2, 2), dtype=np.float32),
            )
        },
        coords={
            "batch": [0],
            "lead_time": np.asarray([6, 12], dtype="timedelta64[h]"),
            "lat": [-60.0, 0.0],
            "lon": [0.0, 180.0],
        },
    )
    stddev = xr.Dataset({"temperature": xr.DataArray(4.0)})
    reference_path = tmp_path / "reference.nc"
    stddev_path = tmp_path / "stddev.nc"
    output = tmp_path / "streaming.json"
    reference.to_netcdf(reference_path)
    stddev.to_netcdf(stddev_path)
    monkeypatch.setenv("GRAPHCAST_TRAJECTORY_REFERENCE", str(reference_path))
    monkeypatch.setenv("GRAPHCAST_TRAJECTORY_STDDEV", str(stddev_path))
    valid_times = xr.DataArray(
        np.asarray(["2022-01-01T06", "2022-01-01T12"], dtype="datetime64[h]"),
        dims=("time",),
    )
    writer = DRIVER.TrajectoryReferenceWriter(
        output, steps=2, step_hours=6, valid_times=valid_times
    )
    for step, value in enumerate((1.0, 2.0)):
        prediction = xr.Dataset(
            {
                "temperature": (
                    ("batch", "time", "lat", "lon"),
                    np.full((1, 1, 2, 2), value, dtype=np.float32),
                )
            },
            coords={
                "batch": [0],
                "time": [step],
                "lat": [-60.0, 0.0],
                "lon": [0.0, 180.0],
            },
        )
        writer.append(step, prediction)
    staged = writer.seal()
    assert staged.is_file()
    writer.publish()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["lead_count"] == 2
    assert payload["worst"]["normalized_max_abs"] == pytest.approx(0.5)
    assert payload["worst"]["normalized_rmse"] == pytest.approx(0.5)
    assert payload["implementation_parity"]["rmse"] == pytest.approx(np.sqrt(2.5))


def test_local_publisher_keeps_executable_cache_outside_noexec_staging(
    tmp_path: Path,
) -> None:
    generic = tmp_path / "generic.sh"
    generic.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "mkdir -p \"${OUTPUT_ROOT}/${RUN_NAME}\"\n"
        "printf '%s\\n' \"${EXECUTABLE_CACHE_PARENT}\" "
        "> \"${OUTPUT_ROOT}/${RUN_NAME}/cache-parent.txt\"\n",
        encoding="utf-8",
    )
    publication = tmp_path / "publication"
    local_staging = tmp_path / "local-staging"
    environment = {
        **os.environ,
        "GENERIC_CASE_RUNNER": str(generic),
        "OUTPUT_ROOT": str(publication),
        "RUN_NAME": "candidate",
        "PLATFORM": "test-platform",
        "LOCAL_STAGING_PARENT": str(local_staging),
        "EXECUTABLE_CACHE_PARENT": str(tmp_path / "executable-cache"),
    }

    subprocess.run(["bash", str(LOCAL_RUNNER)], env=environment, check=True)

    assert (publication / "candidate" / "cache-parent.txt").read_text().strip() == str(
        tmp_path / "executable-cache"
    )
    assert not (local_staging / "test-platform" / "candidate").exists()


def test_performance_matrix_can_resume_flaggems_only(tmp_path: Path) -> None:
    generic = tmp_path / "case.sh"
    generic.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "printf '%s %s\\n' \"${MODE}\" \"${RUN_NAME}\" >> \"${CALL_LOG}\"\n",
        encoding="utf-8",
    )
    log = tmp_path / "calls.log"
    environment = {
        **os.environ,
        "CASE_RUNNER": str(generic),
        "PLATFORM": "test-platform",
        "OUTPUT_ROOT": str(tmp_path / "results"),
        "STEPS": "4",
        "RUN_NATIVE": "0",
        "RUN_FLAGGEMS": "1",
        "CALL_LOG": str(log),
    }

    subprocess.run(["bash", str(PERFORMANCE_RUNNER)], env=environment, check=True)

    assert log.read_text(encoding="utf-8").splitlines() == [
        "flaggems test-platform-flaggems-bf16-4step-r1",
        "flaggems test-platform-flaggems-bf16-4step-r2",
        "flaggems test-platform-flaggems-bf16-4step-r3",
    ]
