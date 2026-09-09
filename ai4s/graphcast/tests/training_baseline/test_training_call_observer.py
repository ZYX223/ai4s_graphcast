"""Observed log calls use real Torch forward/backward and AdamW updates."""
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validation/trace_training_flaggems.py"


def observer_module():
    assert SCRIPT.is_file(), "opt-in source observer has not been integrated"
    spec = importlib.util.spec_from_file_location("training_call_observer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logger = logging.getLogger("flag_gems.fixture")


def observed(value):
    logger.debug("GEMS fixture actual function")
    return value * 2


class LoggedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return observed(value)

    @staticmethod
    def backward(ctx, grad):
        from graphcast_compat.recompute_flops import _RECOMPUTE_DEPTH
        token = _RECOMPUTE_DEPTH.set(1)
        try:
            observed(grad)
        finally:
            _RECOMPUTE_DEPTH.reset(token)
        return observed(grad)


class EpochAutoregressiveModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self):
        return LoggedBackward.apply(self.weight)


def test_observer_separates_startup_forward_recompute_backward_optimizer_and_restores(tmp_path, monkeypatch):
    module = observer_module()
    original_step = torch.optim.AdamW.step
    def logged_step(optimizer, *args, **kwargs):
        observed(next(iter(optimizer.param_groups[0]["params"])))
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", logged_step)
    monkeypatch.setenv("RANK", "2")
    before_backward = torch.autograd.backward
    before_call = torch.nn.Module._call_impl
    handler_count = len(logging.getLogger("flag_gems").handlers)
    model = EpochAutoregressiveModule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    before_step = torch.optim.AdamW.step  # optimizer construction installs Torch's hook wrapper
    observer = module.CallObserver(tmp_path, trainer=Path(__file__))
    with observer:
        observed(model.weight)  # startup must never become forward coverage
        for _ in range(2):
            optimizer.zero_grad()
            model().backward()
            optimizer.step()
    rows = [json.loads(line) for line in (tmp_path / "rank_2.jsonl").read_text().splitlines()]
    assert [r["optimizer_update"] for r in rows] == [1, 2]
    first = rows[0]
    assert first["rank"] == 2 and first["count_kind"] == "logged_function_calls"
    phases = {r["phase"] for r in first["calls"]}
    assert phases == {"initialization", "forward", "recompute", "backward", "optimizer"}
    assert all(r["source"] == str(Path(__file__).resolve()) and r["device"] == "cpu" for r in first["calls"])
    assert all(r["calls"] == 1 for r in first["calls"])
    assert next(r for r in first["calls"] if r["phase"] == "initialization")["optimizer_update"] is None
    assert "initialization" not in {r["phase"] for r in rows[1]["calls"]}
    assert torch.autograd.backward is before_backward
    assert torch.nn.Module._call_impl is before_call
    assert torch.optim.AdamW.step is before_step
    assert len(logging.getLogger("flag_gems").handlers) == handler_count


def test_observer_restores_instrumentation_on_training_exception(tmp_path):
    module = observer_module()
    before = torch.optim.AdamW.step
    with pytest.raises(RuntimeError, match="training failure"):
        with module.CallObserver(tmp_path, trainer=Path(__file__)):
            raise RuntimeError("training failure")
    assert torch.optim.AdamW.step is before


def test_wrapper_rejects_stale_source_switch_without_target_backend_mode(tmp_path):
    observer_module()
    env = {k: v for k, v in os.environ.items() if k != "GRAPHCAST_FLAGOS_MODE"}
    env["FLAGOS_REPLACE"] = "1"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--trace-dir", str(tmp_path), "--", "--help"],
        env=env, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "GRAPHCAST_FLAGOS_MODE=flaggems" in completed.stderr
    assert not list(tmp_path.glob("rank_*.jsonl"))
