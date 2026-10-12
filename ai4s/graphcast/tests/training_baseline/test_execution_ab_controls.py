"""Opt-in policies must reach the real engine and reject unsafe combinations."""
import pytest
import torch
from test_canonical_runtime import runner


def test_shared_launcher_passes_both_experiment_controls():
    import run_training
    args = run_training.parser().parse_args([
        "--mp", "1", "--device-type", "cuda", "--platform", "h100",
        "--edge-gather-group-size", "4", "--resident-training-inputs", "enabled",
    ])
    engine = runner().parse_args(run_training.resolve_contract(args)["engine_argv"])
    assert engine.edge_gather_group_size == 4
    assert engine.resident_training_inputs == "enabled"


@pytest.mark.parametrize("group", [0, -1])
def test_engine_rejects_invalid_gather_group(group):
    import run_training
    argv = run_training.resolve_contract(run_training.parser().parse_args([]))["engine_argv"]
    with pytest.raises(ValueError, match="edge_gather_group_size"):
        runner().parse_args(argv + ["--edge-gather-group-size", str(group)])


def test_resident_wrapper_keeps_complete_feedback_gradient():
    from test_autoregressive_training import ScalarResidualModel, _scalar_batch
    module = runner()
    model = ScalarResidualModel(.5)
    wrapper = module.EpochAutoregressiveModule(model, checkpoint_segments=2,
        checkpoint_single_step=True, checkpoint_implementation="reentrant", resident_training_inputs=True)
    loss, leads, _ = wrapper(_scalar_batch())
    loss.backward()
    assert loss.item() == pytest.approx(2.125)
    assert model.residual.grad.item() == pytest.approx(-4.5)
