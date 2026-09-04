from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
for path in (GRAPHCAST_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphcast_compat.google_small import GoogleSmallConfig  # noqa: E402
from graphcast_compat.inference_precision import resolve_inference_precision  # noqa: E402
from scripts.inference import run_operational_40step_inference as operational_runner  # noqa: E402
from scripts.inference.run_operational_40step_inference import (  # noqa: E402
    RolloutMeasurement,
    calculate_throughput,
    parse_args,
    run_tensor_resident_rollout,
)


def _trajectory_writer():
    writer = getattr(operational_runner, "TrajectoryNetCDFWriter", None)
    assert writer is not None, "trajectory NetCDF writer is missing"
    return writer


class CountingModel(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.calls = 0

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.events.append(f"model:{self.calls}")
        return model_input[..., :1] * 0.0 + 1.0


def _required_cli_args() -> list[str]:
    return [
        "--config",
        "config.yaml",
        "--weights",
        "weights.pt",
        "--inputs",
        "inputs.nc",
        "--rollout-forcings",
        "forcings.nc",
        "--target-template",
        "target.nc",
        "--stats-dir",
        "stats",
        "--output",
        "prediction.nc",
        "--report",
        "report.json",
    ]


def _profiled_cli_args(
    *, platform: str = "h100", profile: str = "baseline", precision: str = "fp32"
) -> list[str]:
    return [
        *_required_cli_args(),
        "--platform",
        platform,
        "--model-execution-profile",
        profile,
        "--precision",
        precision,
    ]


def _profile_resolver():
    resolver = getattr(operational_runner, "resolve_model_execution_profile", None)
    assert callable(resolver), "model execution profile resolver is missing"
    return resolver


def _runtime_platform_validator():
    validator = getattr(operational_runner, "validate_runtime_platform", None)
    assert callable(validator), "runtime platform validator is missing"
    return validator


def test_h100_baseline_expands_to_the_conservative_reference_path() -> None:
    profile = _profile_resolver()("h100", "baseline")

    assert profile.to_metadata() == {
        "name": "baseline",
        "platform": "h100",
        "runtime_backend": "nvidia",
        "configuration_overrides": {
            "edge_input_projection": "concatenated",
            "grid2mesh_edge_chunk_size": None,
            "mesh2grid_edge_chunk_size": None,
            "mesh2grid_full_grid_fast_path": False,
            "index_add_execution": "inline",
        },
    }


@pytest.mark.parametrize("platform", ("c550", "bw3000", "s5000", "910c", "810e"))
def test_domestic_v1_expands_identically_on_all_five_platforms(platform: str) -> None:
    profile = _profile_resolver()(platform, "domestic-v1")

    assert profile.to_metadata() == {
        "name": "domestic-v1",
        "platform": platform,
        "runtime_backend": operational_runner.PLATFORM_RUNTIME_BACKENDS[platform],
        "configuration_overrides": {
            "edge_input_projection": "factorized",
            "grid2mesh_edge_chunk_size": 131072,
            "mesh2grid_edge_chunk_size": 131072,
            "mesh2grid_full_grid_fast_path": True,
            "index_add_execution": "inline",
        },
    }


@pytest.mark.parametrize(
    ("platform", "profile"),
    (
        ("h100", "domestic-v1"),
        ("c550", "baseline"),
        ("bw3000", "baseline"),
        ("s5000", "baseline"),
        ("910c", "baseline"),
        ("810e", "baseline"),
    ),
)
def test_model_execution_profile_rejects_cross_platform_use(
    platform: str, profile: str
) -> None:
    with pytest.raises(ValueError, match="not valid"):
        _profile_resolver()(platform, profile)


@pytest.mark.parametrize(
    ("platform", "profile"),
    (
        ("h100", "baseline"),
        ("c550", "domestic-v1"),
        ("bw3000", "domestic-v1"),
        ("s5000", "domestic-v1"),
        ("910c", "domestic-v1"),
        ("810e", "domestic-v1"),
    ),
)
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_formal_cli_applies_profile_independently_of_precision(
    platform: str, profile: str, precision: str
) -> None:
    args = parse_args(
        _profiled_cli_args(platform=platform, profile=profile, precision=precision)
    )

    assert args.precision == precision
    assert args.model_execution_profile.name == profile
    assert args.model_execution_profile.platform == platform


@pytest.mark.parametrize(
    "missing_pair",
    (("--platform", "h100"), ("--model-execution-profile", "baseline")),
)
def test_formal_cli_requires_platform_and_model_execution_profile(
    missing_pair: tuple[str, str],
) -> None:
    arguments = _profiled_cli_args()
    index = arguments.index(missing_pair[0])
    del arguments[index : index + 2]
    with pytest.raises(SystemExit):
        parse_args(arguments)


@pytest.mark.parametrize(
    ("platform", "backend"),
    (
        ("h100", "nvidia"),
        ("c550", "metax"),
        ("bw3000", "hygon"),
        ("s5000", "mthreads"),
        ("910c", "ascend"),
        ("810e", "ptg"),
    ),
)
def test_runtime_platform_validator_accepts_only_the_bound_backend(
    platform: str, backend: str
) -> None:
    _runtime_platform_validator()(platform, backend)


@pytest.mark.parametrize(
    ("platform", "backend"),
    (("h100", "metax"), ("c550", "nvidia"), ("910c", "ptg"), ("810e", "ascend")),
)
def test_runtime_platform_validator_rejects_swapped_backends(
    platform: str, backend: str
) -> None:
    with pytest.raises(RuntimeError, match="requires runtime backend"):
        _runtime_platform_validator()(platform, backend)


@pytest.mark.parametrize(
    "retired_args",
    (
        ("--edge-input-projection", "factorized"),
        ("--grid2mesh-edge-chunk-size", "131072"),
        ("--mesh2grid-edge-chunk-size", "131072"),
        ("--mesh2grid-full-grid-fast-path",),
        ("--index-add-execution", "prevalidated-static"),
    ),
)
def test_formal_cli_rejects_low_level_model_optimization_overrides(
    retired_args: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        parse_args([*_profiled_cli_args(), *retired_args])


@pytest.mark.parametrize(
    "removed_args",
    (
        ("--encoder-input-projection", "factorized"),
        ("--node-input-projection", "factorized"),
        ("--reuse-inference-aggregation-buffers",),
        ("--silu-execution", "opaque-flaggems"),
        ("--graph-capture", "platform-graph"),
        ("--index-add-execution", "opaque-flaggems-static"),
        ("--index-add-execution", "opaque-flaggems-functional-static"),
    ),
)
def test_formal_runner_rejects_retired_experimental_paths(
    removed_args: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        parse_args([*_profiled_cli_args(), *removed_args])


def _prepared(steps: int):
    feedback = SimpleNamespace(
        name="state",
        input_previous=slice(0, 1),
        input_current=slice(1, 2),
        output=slice(0, 1),
        residual_to_absolute=torch.tensor([1.0]),
    )
    forcing = SimpleNamespace(
        name="forcing",
        input_previous=slice(2, 3),
        input_current=slice(3, 4),
        forcing=slice(0, 1),
    )
    return SimpleNamespace(
        initial_state=torch.zeros((1, 1, 4)),
        forcing_features=torch.arange(steps, dtype=torch.float32).reshape(steps, 1, 1, 1),
        feedback_updates=(feedback,),
        forcing_updates=(forcing,),
        output_channels=1,
        steps=steps,
    )


def test_warmup_is_before_timer_and_all_40_dependent_steps_are_measured() -> None:
    events: list[str] = []
    model = CountingModel(events)
    clock_values = iter((100.0, 108.0))

    def clock() -> float:
        events.append("clock")
        return next(clock_values)

    def synchronize() -> None:
        events.append("sync")

    result = run_tensor_resident_rollout(
        model,
        _prepared(40),
        steps=40,
        warmup_calls=1,
        synchronize=synchronize,
        clock=clock,
    )

    assert isinstance(result, RolloutMeasurement)
    assert model.calls == 41
    assert events.index("model:1") < events.index("clock")
    assert result.steps == 40
    assert result.warmup_calls == 1
    assert result.e2e_seconds == 8.0
    assert result.final_output.device.type == "cpu"


def test_rollout_supports_any_positive_autoregressive_step_count() -> None:
    model = CountingModel([])

    result = run_tensor_resident_rollout(
        model,
        _prepared(3),
        steps=3,
        warmup_calls=1,
        synchronize=lambda: None,
    )

    assert result.steps == 3
    assert model.calls == 4


def test_diagnostic_callback_receives_every_dependent_absolute_output() -> None:
    captured: list[tuple[int, torch.Tensor]] = []

    result = run_tensor_resident_rollout(
        CountingModel([]),
        _prepared(3),
        steps=3,
        warmup_calls=1,
        synchronize=lambda: None,
        trajectory_callback=lambda step, value: captured.append((step, value.clone())),
    )

    assert [step for step, _ in captured] == [0, 1, 2]
    assert [value.item() for _, value in captured] == [1.0, 2.0, 3.0]
    assert all(value.device.type == "cpu" for _, value in captured)
    assert all(value.dtype == torch.float32 for _, value in captured)
    assert result.final_output.item() == 3.0


def test_trajectory_writer_publishes_one_atomic_multilead_netcdf(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.nc"
    valid_times = xr.DataArray(
        np.asarray([["2022-01-01T06", "2022-01-01T12"]], dtype="datetime64[h]"),
        dims=("batch", "time"),
        coords={"batch": [0], "time": [6, 12]},
    )
    writer = _trajectory_writer()(
        output,
        steps=2,
        step_hours=6,
        valid_times=valid_times,
    )
    first = xr.Dataset(
        {
            "temperature": (
                ("batch", "time", "lat", "lon"),
                torch.ones((1, 1, 2, 2)).numpy(),
            )
        },
        coords={"batch": [0], "time": [6], "lat": [-45.0, 45.0], "lon": [0.0, 180.0]},
    )
    second = first.copy(deep=True)
    second["temperature"] *= 2.0

    writer.append(0, first)
    assert not output.exists()
    writer.append(1, second)
    writer.finalize()

    with xr.open_dataset(output) as trajectory:
        assert trajectory.attrs["schema"] == "graphcast.inference_trajectory.v1"
        assert trajectory.attrs["performance_eligible"] == 0
        assert (
            trajectory["lead_time"].values / np.timedelta64(1, "h")
        ).tolist() == [6.0, 12.0]
        assert trajectory["temperature"].dims == ("batch", "lead_time", "lat", "lon")
        assert "valid_time" in trajectory.coords
        assert trajectory["valid_time"].dims == ("batch", "lead_time")
        assert np.array_equal(
            trajectory["forecast_reference_time"].values.astype("datetime64[h]"),
            np.asarray(["2022-01-01T00"], dtype="datetime64[h]"),
        )
        assert np.array_equal(
            trajectory["valid_time"].values.astype("datetime64[h]"),
            valid_times.values,
        )
        assert trajectory["temperature"].isel(lead_time=0).mean().item() == 1.0
        assert trajectory["temperature"].isel(lead_time=1).mean().item() == 2.0


def test_trajectory_writer_rejects_valid_times_inconsistent_with_step_hours(
    tmp_path: Path,
) -> None:
    irregular = xr.DataArray(
        np.asarray([["2022-01-01T06", "2022-01-01T18"]], dtype="datetime64[h]"),
        dims=("batch", "time"),
        coords={"batch": [0], "time": [6, 18]},
    )

    with pytest.raises(ValueError, match="regular autoregressive leads"):
        _trajectory_writer()(
            tmp_path / "trajectory.nc",
            steps=2,
            step_hours=6,
            valid_times=irregular,
        )


def test_trajectory_writer_requires_auditable_valid_times(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require forcing valid times"):
        _trajectory_writer()(
            tmp_path / "trajectory.nc",
            steps=2,
            step_hours=6,
        )


def test_incomplete_trajectory_is_not_published(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.nc"
    valid_times = xr.DataArray(
        np.asarray([["2022-01-01T06", "2022-01-01T12"]], dtype="datetime64[h]"),
        dims=("batch", "time"),
        coords={"batch": [0], "time": [6, 12]},
    )
    writer = _trajectory_writer()(
        output,
        steps=2,
        step_hours=6,
        valid_times=valid_times,
    )
    first = xr.Dataset(
        {"temperature": (("batch", "time", "lat", "lon"), np.ones((1, 1, 1, 1)))},
        coords={"batch": [0], "time": [6], "lat": [0.0], "lon": [0.0]},
    )
    writer.append(0, first)

    with pytest.raises(RuntimeError, match="wrote 1 of 2"):
        writer.finalize()

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_formal_cli_accepts_optional_diagnostic_trajectory_output() -> None:
    args = parse_args([*_profiled_cli_args(), "--trajectory-output", "trajectory.nc"])

    assert args.trajectory_output == Path("trajectory.nc")


def test_main_rejects_nonfinite_peak_before_loading_assets() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        operational_runner.main(
            [*_profiled_cli_args(), "--peak-tflops-per-device", "nan"]
        )


def test_runner_outputs_cannot_alias_normalization_stats(tmp_path: Path) -> None:
    assets = {
        name: tmp_path / name
        for name in ("config.yaml", "weights.pt", "inputs.nc", "forcings.nc", "target.nc")
    }
    for path in assets.values():
        path.write_bytes(b"fixture")
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    for name in operational_runner.REQUIRED_STATS_FILES:
        (stats_dir / name).write_bytes(b"stats")

    with pytest.raises(ValueError, match="must be distinct"):
        operational_runner.main(
            [
                "--config", str(assets["config.yaml"]),
                "--weights", str(assets["weights.pt"]),
                "--inputs", str(assets["inputs.nc"]),
                "--rollout-forcings", str(assets["forcings.nc"]),
                "--target-template", str(assets["target.nc"]),
                "--stats-dir", str(stats_dir),
                "--output", str(stats_dir / "stddev_by_level.nc"),
                "--report", str(tmp_path / "report.json"),
                "--platform", "h100",
                "--model-execution-profile", "baseline",
            ]
        )


def test_overwrite_backups_restore_the_previous_artifact_set(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("prediction.nc", "trajectory.nc", "report.json")]
    for index, path in enumerate(paths):
        path.write_text(f"old-{index}", encoding="utf-8")

    backups = operational_runner._backup_existing_outputs(paths)
    paths[0].write_text("new", encoding="utf-8")
    for path in paths:
        path.unlink(missing_ok=True)
    operational_runner._restore_output_backups(backups)

    assert [path.read_text(encoding="utf-8") for path in paths] == [
        "old-0",
        "old-1",
        "old-2",
    ]


def test_report_staging_rejects_nonstandard_json_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        operational_runner._stage_json(tmp_path / "report.json", {"value": float("inf")})
    assert list(tmp_path.iterdir()) == []


def test_normalization_stats_reject_nonfinite_or_nonpositive_scales() -> None:
    valid = xr.Dataset({"temperature": xr.DataArray([1.0])})
    invalid = xr.Dataset({"temperature": xr.DataArray([float("inf")])})
    stats = SimpleNamespace(
        mean_by_level=valid,
        stddev_by_level=invalid,
        diffs_stddev_by_level=valid,
    )

    with pytest.raises(ValueError, match="stddev_by_level.*positive finite"):
        operational_runner.validate_normalization_stats(
            stats,
            inputs=valid,
            forcings=xr.Dataset(),
            target_template=valid,
        )


def test_normalization_stats_allow_unused_official_placeholder_values() -> None:
    active = xr.Dataset({"temperature": xr.DataArray([1.0])})
    scales = xr.Dataset(
        {
            "temperature": xr.DataArray([1.0]),
            "seconds_since_epoch": xr.DataArray([float("nan")]),
        }
    )
    residual_scales = xr.Dataset(
        {
            "temperature": xr.DataArray([1.0]),
            "seconds_since_epoch": xr.DataArray([0.0]),
        }
    )
    stats = SimpleNamespace(
        mean_by_level=scales,
        stddev_by_level=scales,
        diffs_stddev_by_level=residual_scales,
    )

    operational_runner.validate_normalization_stats(
        stats,
        inputs=active,
        forcings=xr.Dataset(),
        target_template=active,
    )


def test_physical_prediction_finiteness_is_checked_after_denormalization() -> None:
    prediction = xr.Dataset({"temperature": xr.DataArray([1.0, float("inf")])})

    assert operational_runner.dataset_is_finite(prediction) is False


@pytest.mark.parametrize("steps", [0, -1])
def test_rollout_rejects_non_positive_step_counts(steps: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_tensor_resident_rollout(
            CountingModel([]),
            _prepared(1),
            steps=steps,
            warmup_calls=1,
            synchronize=lambda: None,
        )


def test_rollout_rejects_a_forcing_count_mismatch() -> None:
    with pytest.raises(ValueError, match="requested 4 rollout steps.*3 forcing frames"):
        run_tensor_resident_rollout(
            CountingModel([]),
            _prepared(3),
            steps=4,
            warmup_calls=1,
            synchronize=lambda: None,
        )


def test_throughput_uses_all_40_forward_flops_and_measured_e2e() -> None:
    metrics = calculate_throughput(
        forward_flops=28_857_094_475_776,
        steps=40,
        e2e_seconds=8.71864111876736,
        peak_tflops_per_device=495.0,
        device_count=1,
    )
    assert metrics["useful_flops"] == 1_154_283_779_031_040
    assert metrics["effective_tflops"] == pytest.approx(132.39262441326777)
    assert metrics["mfu"] == pytest.approx(0.26745984729953084)


def test_formal_h100_baseline_defaults_to_ieee_fp32() -> None:
    args = parse_args(_profiled_cli_args())

    assert args.precision == "fp32"
    assert args.steps == 40
    assert args.model_execution_profile.name == "baseline"
    assert args.model_execution_profile.platform == "h100"


def test_formal_runner_accepts_bf16_mixed_precision() -> None:
    args = parse_args(_profiled_cli_args(precision="bf16"))

    assert args.precision == "bf16"


def test_bf16_rollout_keeps_state_and_feedback_in_fp32() -> None:
    model = torch.nn.Linear(5, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(0.25)

    result = run_tensor_resident_rollout(
        model,
        _prepared(3),
        steps=3,
        warmup_calls=1,
        synchronize=lambda: None,
        precision_policy=resolve_inference_precision("bf16"),
        device_type="cpu",
    )

    assert next(model.parameters()).dtype == torch.float32
    assert result.dtype_evidence.state == "torch.float32"
    assert result.dtype_evidence.forcing == "torch.float32"
    assert result.dtype_evidence.model_input == "torch.bfloat16"
    assert result.dtype_evidence.model_output == "torch.bfloat16"
    assert result.dtype_evidence.feedback == "torch.float32"
    assert result.dtype_evidence.final_output == "torch.float32"
    assert result.final_output.dtype == torch.float32


def test_precision_report_distinguishes_request_from_observed_dtypes() -> None:
    model = torch.nn.Linear(5, 1, bias=False).eval()
    measurement = run_tensor_resident_rollout(
        model,
        _prepared(1),
        steps=1,
        warmup_calls=1,
        synchronize=lambda: None,
        precision_policy=resolve_inference_precision("bf16"),
        device_type="cpu",
    )

    metadata = operational_runner.precision_metadata(
        resolve_inference_precision("bf16"),
        measurement,
        model=model,
        device_type="cpu",
        float32_math_mode="ieee_fp32",
    )

    assert metadata == {
        "requested": "bf16",
        "effective": "mixed_precision",
        "autocast": {
            "enabled": True,
            "device_type": "cpu",
            "dtype": "torch.bfloat16",
        },
        "parameter_dtype": "torch.float32",
        "state_dtype": "torch.float32",
        "forcing_dtype": "torch.float32",
        "model_input_dtype": "torch.bfloat16",
        "model_output_dtype": "torch.bfloat16",
        "feedback_dtype": "torch.float32",
        "final_output_dtype": "torch.float32",
        "float32_math_mode": "ieee_fp32",
    }


def test_operational_model_builder_threads_factorized_edge_projection() -> None:
    model = operational_runner.build_model_for_run(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        ),
        edge_input_projection="factorized",
    )

    assert model.edge_input_projection_mode == "factorized"
    assert model.grid2mesh_gnn.edge_input_projection_mode == "factorized"
    assert model.mesh_gnn.edge_input_projection_mode == "factorized"
    assert model.mesh2grid_gnn.edge_input_projection_mode == "factorized"


def test_report_model_metadata_binds_domestic_profile_to_effective_settings() -> None:
    profile = _profile_resolver()("c550", "domestic-v1")
    config = operational_runner.apply_edge_chunk_overrides(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        ),
        grid2mesh=profile.grid2mesh_edge_chunk_size,
        mesh2grid=profile.mesh2grid_edge_chunk_size,
    )
    model = operational_runner.build_model_for_run(
        config,
        edge_input_projection=profile.edge_input_projection,
        mesh2grid_full_grid_fast_path=profile.mesh2grid_full_grid_fast_path,
        index_add_execution=profile.index_add_execution,
    )

    metadata = operational_runner.model_execution_metadata(model, profile=profile)

    assert metadata.get("model_execution_profile") == profile.to_metadata()


def test_baseline_report_separates_profile_overrides_from_effective_chunks() -> None:
    profile = _profile_resolver()("h100", "baseline")
    config = GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=1,
        node_input_dim=4,
        edge_input_dim=1,
        output_dim=1,
        grid2mesh_edge_chunk_size=131072,
        mesh2grid_edge_chunk_size=131072,
    )
    model = operational_runner.build_model_for_run(
        config,
        edge_input_projection=profile.edge_input_projection,
        mesh2grid_full_grid_fast_path=profile.mesh2grid_full_grid_fast_path,
        index_add_execution=profile.index_add_execution,
    )

    metadata = operational_runner.model_execution_metadata(model, profile=profile)

    assert metadata["model_execution_profile"]["configuration_overrides"] == {
        "edge_input_projection": "concatenated",
        "grid2mesh_edge_chunk_size": None,
        "mesh2grid_edge_chunk_size": None,
        "mesh2grid_full_grid_fast_path": False,
        "index_add_execution": "inline",
    }
    assert metadata["grid2mesh_edge_chunk_size"] == 131072
    assert metadata["mesh2grid_edge_chunk_size"] == 131072


@pytest.mark.parametrize(
    ("physical_finite", "expected_exit", "expected_status"),
    ((True, 0, "passed"), (False, 1, "failed_nonfinite")),
)
def test_main_binds_runtime_builder_and_report_to_one_domestic_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    physical_finite: bool,
    expected_exit: int,
    expected_status: str,
) -> None:
    assets = {
        name: tmp_path / name
        for name in ("config.yaml", "weights.pt", "inputs.nc", "forcings.nc", "target.nc")
    }
    for path in assets.values():
        path.write_bytes(b"fixture")
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    for name in operational_runner.REQUIRED_STATS_FILES:
        (stats_dir / name).write_bytes(b"stats")
    output = tmp_path / "prediction.nc"
    trajectory_output = tmp_path / "trajectory.nc"
    report_path = tmp_path / "report.json"
    events: list[str] = []
    captured: dict[str, object] = {}

    class FakeDataset:
        coords = {
            "lat": SimpleNamespace(values=[0.0]),
            "lon": SimpleNamespace(values=[0.0]),
            "datetime": xr.DataArray(
                np.asarray(
                    [["2022-01-01T12", "2022-01-02T00", "2022-01-02T12"]],
                    dtype="datetime64[h]",
                ),
                dims=("batch", "time"),
            ),
        }

        def load(self):
            return self

    class FakeRuntime:
        profile = SimpleNamespace(name="metax", device_type="cuda")
        device = torch.device("cpu")

        def to_metadata(self):
            return {"backend": "metax"}

    class FakeModel:
        def __init__(self, config, **settings) -> None:
            self.config = config
            self.edge_input_projection_mode = settings["edge_input_projection"]
            self.mesh2grid_full_grid_fast_path = settings[
                "mesh2grid_full_grid_fast_path"
            ]
            self.index_add_execution = settings["index_add_execution"]

        def load_state_dict(self, state_dict, strict=True) -> None:
            assert strict

        def init_graph(self, **coordinates) -> None:
            assert set(coordinates) == {"latitudes", "longitudes"}

        def to(self, device):
            return self

        def eval(self):
            return self

    config = GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=1,
        node_input_dim=4,
        edge_input_dim=1,
        output_dim=1,
        grid2mesh_edge_chunk_size=17,
        mesh2grid_edge_chunk_size=19,
    )

    def fake_resolve_runtime(device, *, expected_profile):
        events.append("runtime")
        captured["expected_backend"] = expected_profile
        return FakeRuntime()

    def fake_build_model(effective_config, **settings):
        events.append("model")
        captured["effective_config"] = effective_config
        captured["settings"] = settings
        return FakeModel(effective_config, **settings)

    monkeypatch.setattr(operational_runner, "resolve_runtime_backend", fake_resolve_runtime)
    monkeypatch.setattr(operational_runner, "configure_float32_math", lambda *args: "ieee_fp32")
    monkeypatch.setattr(GoogleSmallConfig, "from_yaml", lambda path: config)
    monkeypatch.setattr(operational_runner.xr, "open_dataset", lambda path: FakeDataset())
    monkeypatch.setattr(operational_runner, "load_normalization_stats", lambda path: object())
    monkeypatch.setattr(
        operational_runner,
        "validate_normalization_stats",
        lambda stats, **datasets: None,
    )
    monkeypatch.setattr(operational_runner, "build_model_for_run", fake_build_model)
    monkeypatch.setattr(
        operational_runner.torch,
        "load",
        lambda *args, **kwargs: {"weight": torch.zeros(1)},
    )
    monkeypatch.setattr(
        operational_runner,
        "prepare_device_rollout",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(operational_runner, "reset_peak_memory", lambda runtime: None)

    def fake_rollout(*args, **kwargs):
        assert kwargs["steps"] == 3
        callback = kwargs.get("trajectory_callback")
        assert callable(callback)
        for step in range(3):
            callback(step, torch.full((1,), float(step)))
        return RolloutMeasurement(
            steps=3,
            warmup_calls=1,
            e2e_seconds=10.0,
            final_output=torch.zeros(1),
            finite=True,
            dtype_evidence=SimpleNamespace(),
        )

    monkeypatch.setattr(operational_runner, "run_tensor_resident_rollout", fake_rollout)
    monkeypatch.setattr(
        operational_runner,
        "require_complete_flaggems_dispatch",
        lambda runtime: None,
    )

    class FakePrediction:
        def to_netcdf(self, path) -> None:
            Path(path).write_bytes(b"prediction")

    monkeypatch.setattr(
        operational_runner,
        "normalized_output_to_prediction",
        lambda *args, **kwargs: FakePrediction(),
    )
    monkeypatch.setattr(
        operational_runner,
        "dataset_is_finite",
        lambda dataset: physical_finite,
    )
    monkeypatch.setattr(operational_runner, "precision_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(operational_runner, "memory_text", lambda runtime: {})

    class FakeTrajectoryWriter:
        def __init__(self, path, *, steps, step_hours, valid_times=None) -> None:
            assert valid_times.sizes["time"] == 3
            captured["trajectory_writer"] = (path, steps, step_hours)

        def append(self, step, prediction) -> None:
            captured.setdefault("trajectory_appends", []).append(
                (step, type(prediction).__name__)
            )

        def seal(self) -> Path:
            captured["trajectory_sealed"] = True
            staged = tmp_path / "staged-trajectory.nc"
            staged.write_bytes(b"trajectory")
            return staged

        def publish(self) -> None:
            captured["trajectory_published"] = True
            trajectory_output.write_bytes(b"trajectory")

        def abort(self) -> None:
            captured["trajectory_aborted"] = True

    monkeypatch.setattr(operational_runner, "TrajectoryNetCDFWriter", FakeTrajectoryWriter)

    result = operational_runner.main(
        [
            "--config", str(assets["config.yaml"]),
            "--weights", str(assets["weights.pt"]),
            "--inputs", str(assets["inputs.nc"]),
            "--rollout-forcings", str(assets["forcings.nc"]),
            "--target-template", str(assets["target.nc"]),
            "--stats-dir", str(stats_dir),
            "--output", str(output),
            "--trajectory-output", str(trajectory_output),
            "--report", str(report_path),
            "--platform", "c550",
            "--model-execution-profile", "domestic-v1",
            "--steps", "3",
            "--step-hours", "12",
        ]
    )

    assert result == expected_exit
    assert events == ["runtime", "model"]
    assert captured["expected_backend"] == "metax"
    assert captured["settings"] == {
        "edge_input_projection": "factorized",
        "mesh2grid_full_grid_fast_path": True,
        "index_add_execution": "inline",
    }
    effective_config = captured["effective_config"]
    assert effective_config.grid2mesh_edge_chunk_size == 131072
    assert effective_config.mesh2grid_edge_chunk_size == 131072
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == expected_status
    assert report["measurement"]["physical_output_finite"] is physical_finite
    assert report["runtime"]["backend"] == "metax"
    assert set(report["source"]["normalization_stats"]) == set(
        operational_runner.REQUIRED_STATS_FILES
    )
    assert report["workload"]["model_execution_profile"]["name"] == "domestic-v1"
    assert report["workload"]["edge_input_projection"] == "factorized"
    assert report["workload"]["grid2mesh_edge_chunk_size"] == 131072
    assert report["workload"]["mesh2grid_edge_chunk_size"] == 131072
    assert report["schema"] == "graphcast.operational_rollout.v2"
    assert report["workload"]["steps"] == 3
    assert report["workload"]["step_hours"] == 12
    assert captured["trajectory_writer"] == (trajectory_output, 3, 12)
    assert captured["trajectory_appends"] == [
        (0, "FakePrediction"),
        (1, "FakePrediction"),
        (2, "FakePrediction"),
    ]
    assert captured["trajectory_sealed"] is True
    assert captured["trajectory_published"] is True
    assert "trajectory_aborted" not in captured
    assert report["artifacts"]["publication"] == "report_terminal_last"
    assert report["artifacts"]["prediction"]["sha256"]
    assert report["artifacts"]["trajectory"]["sha256"]
    assert report["diagnostics"] == {
        "trajectory": str(trajectory_output.resolve()),
        "all_leads_captured": True,
        "performance_eligible": False,
    }


def test_operational_report_metadata_uses_effective_edge_projection_mode() -> None:
    model = operational_runner.build_model_for_run(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        ),
        edge_input_projection="factorized",
    )

    assert operational_runner.model_execution_metadata(model) == {
        "edge_input_projection": "factorized",
        "mesh2grid_full_grid_fast_path": False,
        "index_add_execution": "inline",
        "grid2mesh_edge_chunk_size": None,
        "mesh2grid_edge_chunk_size": None,
    }


@pytest.mark.parametrize("value", [0, -1])
def test_edge_chunk_override_rejects_non_positive_values(value: int) -> None:
    config = GoogleSmallConfig(
        grid2mesh_edge_chunk_size=131072,
        mesh2grid_edge_chunk_size=131072,
    )

    with pytest.raises(ValueError, match="edge chunk size must be positive"):
        operational_runner.apply_edge_chunk_overrides(
            config,
            grid2mesh=value,
            mesh2grid=None,
        )


def test_edge_chunk_overrides_return_new_config_and_drive_model_metadata() -> None:
    original = GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=1,
        node_input_dim=4,
        edge_input_dim=1,
        output_dim=1,
        grid2mesh_edge_chunk_size=131072,
        mesh2grid_edge_chunk_size=131072,
    )

    effective = operational_runner.apply_edge_chunk_overrides(
        original,
        grid2mesh=262144,
        mesh2grid=196608,
    )
    model = operational_runner.build_model_for_run(
        effective,
        edge_input_projection="factorized",
    )

    assert original.grid2mesh_edge_chunk_size == 131072
    assert original.mesh2grid_edge_chunk_size == 131072
    assert operational_runner.model_execution_metadata(model) == {
        "edge_input_projection": "factorized",
        "mesh2grid_full_grid_fast_path": False,
        "index_add_execution": "inline",
        "grid2mesh_edge_chunk_size": 262144,
        "mesh2grid_edge_chunk_size": 196608,
    }


def test_operational_model_builder_threads_full_grid_fast_path() -> None:
    model = operational_runner.build_model_for_run(
        GoogleSmallConfig(
            latent_size=4,
            hidden_layers=1,
            gnn_msg_steps=1,
            node_input_dim=4,
            edge_input_dim=1,
            output_dim=1,
        ),
        edge_input_projection="factorized",
        mesh2grid_full_grid_fast_path=True,
    )

    assert model.mesh2grid_full_grid_fast_path is True
