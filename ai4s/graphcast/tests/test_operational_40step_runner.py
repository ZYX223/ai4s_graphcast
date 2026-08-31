from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
for path in (GRAPHCAST_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.inference.run_operational_40step_inference import (
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


def test_rollout_rejects_a_non_40_step_contract() -> None:
    try:
        run_tensor_resident_rollout(
            CountingModel([]),
            _prepared(39),
            steps=40,
            warmup_calls=1,
            synchronize=lambda: None,
        )
    except ValueError as exc:
        assert "40" in str(exc)
    else:
        raise AssertionError("39 forcings must not satisfy a 40-step run")


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
