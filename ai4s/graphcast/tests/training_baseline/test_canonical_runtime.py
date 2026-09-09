import importlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))


def runner():
    assert (ROOT / "scripts/training/run_official_epoch_training.py").is_file(), "canonical runner not integrated"
    return importlib.import_module("run_official_epoch_training")


def test_public_contract_is_accepted_by_actual_source_parser():
    runner_module = runner()
    import run_training
    for mp in (2, 4):
        for dp in (1, 2):
            args = run_training.parser().parse_args(["--mp", str(mp), "--dp", str(dp), "--device-type", "npu"])
            metadata = run_training.resolve_contract(args)
            native = runner_module.parse_args(metadata["engine_argv"])
            schedule, _ = runner_module.resolve_training_schedule(native)
            assert schedule.to_dict() == metadata["schedule"]
            assert schedule.value_for_update(300000).rollout_steps == 2


def test_model_only_initialization_accepts_raw_weights_but_resume_rejects_them(tmp_path):
    runner()
    from official_epoch_training_file_validation import load_model_only_checkpoint, load_resume_checkpoint
    path = tmp_path / "raw.pt"
    torch.save({"weight": torch.ones(2, 3)}, path)
    state, metadata, digest = load_model_only_checkpoint(path, formal=True,
        training_stage="era5_pretrain", tensor_core_enabled=True)
    torch.testing.assert_close(state["weight"], torch.ones(2, 3))
    assert metadata["initialization_format"] == "raw_state_dict"
    assert len(digest) == 64
    with pytest.raises(ValueError):
        load_resume_checkpoint(path, resume_contract={})


def test_unknown_peak_keeps_mfu_unavailable():
    runner()
    from graphcast_compat.profiled_mfu import enrich_step_metrics
    rows = [{"step": 1, "step_seconds": 1., "training_flops": 1e9}]
    enrich_step_metrics(rows, profiled_steps={}, device_count=4, peak_tflops_per_device=0.)
    assert rows[0]["mfu_sys"] is None
    assert rows[0]["mfu_busy"] is None
