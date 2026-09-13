import importlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "tests"))


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


def test_source_batch_aliases_keep_global_units_and_schedule_identity():
    module = runner()
    import run_training
    contract = run_training.resolve_contract(run_training.parser().parse_args([
        "--device-type", "npu", "--dp", "2", "--global-batch-size", "16", "--micro-batch-size", "2",
    ]))
    # Strip either spelling so this test runs against both old and new adapters.
    batch_flags = {"--batch-size", "--micro-batch-size", "--global-batch-size", "--global-micro-batch-size"}
    arguments = iter(contract["engine_argv"])
    common = []
    for value in arguments:
        if value in batch_flags:
            next(arguments)
        else:
            common.append(value)
    legacy = module.parse_args(common + ["--batch-size", "16", "--micro-batch-size", "4"])
    explicit = module.parse_args(common + ["--global-batch-size", "16", "--global-micro-batch-size", "4"])
    assert vars(explicit) == vars(legacy)
    assert explicit.global_batch_size == 16
    assert explicit.global_micro_batch_size == 4
    assert module.resolve_training_schedule(explicit)[0].digest == contract["schedule_digest"]


def test_source_microbatch_aliases_reject_conflicting_units(capsys):
    module = runner()
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--global-micro-batch-size", "4", "--micro-batch-size", "2"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


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


def test_validation_diagnostics_do_not_change_default_training_loss_or_gradients():
    from test_autoregressive_training import ScalarResidualModel, _scalar_batch
    module = runner()
    models = [ScalarResidualModel(.5), ScalarResidualModel(.5)]
    wrappers = [module.EpochAutoregressiveModule(model,
        checkpoint_segments=0, checkpoint_single_step=False) for model in models]
    plain_loss, plain_leads, plain_variables = wrappers[0](_scalar_batch())
    diagnostic_loss, diagnostic_leads, diagnostic_variables = wrappers[1](
        _scalar_batch(), collect_variable_diagnostics=True)
    plain_loss.backward()
    diagnostic_loss.backward()
    assert plain_variables.shape == (2, 0)
    assert diagnostic_variables.shape == (2, 1)
    torch.testing.assert_close(plain_loss, diagnostic_loss, rtol=0, atol=0)
    torch.testing.assert_close(plain_leads, diagnostic_leads, rtol=0, atol=0)
    torch.testing.assert_close(diagnostic_variables[:, 0], plain_leads, rtol=0, atol=0)
    torch.testing.assert_close(models[0].residual.grad, models[1].residual.grad, rtol=0, atol=0)
