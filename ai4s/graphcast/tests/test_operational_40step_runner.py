from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
for path in (GRAPHCAST_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphcast_compat.google_small import GoogleSmallConfig  # noqa: E402
from scripts.inference import run_operational_40step_inference as operational_runner  # noqa: E402
from scripts.inference.run_operational_40step_inference import (  # noqa: E402
    RolloutMeasurement,
    calculate_throughput,
    parse_args,
    run_tensor_resident_rollout,
)


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
        parse_args([*_required_cli_args(), *removed_args])


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


def test_formal_five_platform_runner_defaults_to_ieee_fp32() -> None:
    args = parse_args(
        [
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
    )

    assert args.precision == "fp32"
    assert args.steps == 40
    assert args.edge_input_projection == "concatenated"
    assert args.mesh2grid_full_grid_fast_path is False
    assert args.grid2mesh_edge_chunk_size is None
    assert args.mesh2grid_edge_chunk_size is None
    assert args.index_add_execution == "inline"


def test_formal_runner_accepts_explicit_prevalidated_static_index_add() -> None:
    args = parse_args(
        [
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
            "--index-add-execution",
            "prevalidated-static",
        ]
    )

    assert args.index_add_execution == "prevalidated-static"


def test_formal_five_platform_runner_accepts_factorized_edge_projection() -> None:
    args = parse_args(
        [
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
            "--edge-input-projection",
            "factorized",
        ]
    )

    assert args.edge_input_projection == "factorized"


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


def test_formal_runner_accepts_edge_chunk_overrides() -> None:
    args = parse_args(
        [
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
            "--grid2mesh-edge-chunk-size",
            "262144",
            "--mesh2grid-edge-chunk-size",
            "196608",
        ]
    )

    assert args.grid2mesh_edge_chunk_size == 262144
    assert args.mesh2grid_edge_chunk_size == 196608


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


def test_formal_runner_accepts_full_grid_fast_path() -> None:
    args = parse_args(
        [
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
            "--mesh2grid-full-grid-fast-path",
        ]
    )

    assert args.mesh2grid_full_grid_fast_path is True


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
