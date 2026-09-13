"""Shared MP1 recipe, actual same-device gradients, and epoch-loop coverage."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))

from test_canonical_entry import run
from test_canonical_cpu_integration import cpu_runner, inputs, command
from graphcast_compat.training_model import GoogleSmallCompatibleModel, GoogleSmallConfig
from graphcast_compat.recompute_flops import tracked_reentrant_checkpoint


PLATFORMS = (
    ("h100", "CUDA_VISIBLE_DEVICES"),
    ("ascend910c", "ASCEND_RT_VISIBLE_DEVICES"),
    ("hygon", "HIP_VISIBLE_DEVICES"),
    ("mthreads", "MUSA_VISIBLE_DEVICES"),
    ("metax", "CUDA_VISIBLE_DEVICES"),
    ("thead", "CUDA_VISIBLE_DEVICES"),
)


@pytest.mark.parametrize("platform,visibility", PLATFORMS)
@pytest.mark.parametrize("backend", ["native", "flaggems"])
def test_single_device_recipe_reaches_engine_without_platform_drift(platform, visibility, backend):
    result = run("--platform", platform, "--backend", backend,
                 "--mp", "1", "--dp", "1",
                 "--global-batch-size", "2", "--micro-batch-size", "1",
                 "--physical-devices", "8", "--devices", "0")
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["rank_device_groups"] == [[0]]
    assert row["accumulation_steps"] == 2
    assert row["requested_precision"] == "tf32_hf32"
    assert row["storage_dtype"] == "float32"
    assert row["environment"][visibility] == "8"
    assert row["environment"]["FLAGOS_REPLACE"] == ("0" if backend == "native" else "1")
    import run_official_epoch_training as engine
    args = engine.parse_args(row["engine_argv"])
    assert args.mp_devices_per_replica == 1
    assert args.mesh_size is None
    # Literal recipe, independently taken from the qualified 910C MP1 run.
    expected = dict(
        edge_chunk_size=8192, mesh_edge_chunk_size=8192,
        grid_node_chunk_size=4096, loss_grid_chunk_size=65536,
        mesh_split_step=16, mesh_checkpoint_granularity="step",
        mesh_edge_input_projection="concatenated",
        rollout_checkpoint_implementation="reentrant", gradient_sync="manual",
        grid2mesh_stage_checkpoint="enabled", mesh2grid_stage_checkpoint="enabled",
        single_step_rollout_checkpoint="enabled",
    )
    for key, value in expected.items():
        assert row["execution"][key] == value, key
        assert getattr(args, key) == value, key


@pytest.mark.parametrize("platform,visibility", PLATFORMS)
def test_explicit_checkpoint_override_still_wins(platform, visibility):
    result = run("--platform", platform, "--mp", "1",
                 "--rollout-checkpoint-implementation", "non_reentrant")
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["execution"]["rollout_checkpoint_implementation"] == "non_reentrant"
    argv = row["engine_argv"]
    assert argv[argv.index("--rollout-checkpoint-implementation") + 1] == "non_reentrant"


def test_mp1_ddp_guard_does_not_silently_switch_checkpoint():
    assert run("--mp", "1", "--gradient-sync", "ddp").returncode != 0
    result = run("--mp", "1", "--gradient-sync", "ddp",
                 "--rollout-checkpoint-implementation", "non_reentrant")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("outer", [False, True])
def test_colocated_checkpointed_stages_match_all_gradients_and_updates(outer):
    torch.manual_seed(27)
    model = GoogleSmallCompatibleModel(GoogleSmallConfig(
        mesh_size=0, latent_size=4, hidden_layers=1, gnn_msg_steps=2,
        node_input_dim=6, edge_input_dim=4, output_dim=3,
        mesh2grid_edge_chunk_size=8, mesh_edge_chunk_size=8,
        grid_node_chunk_size=4,
    ))
    model.init_graph(latitudes=np.array([-45., 0., 45.]),
                     longitudes=np.array([0., 90., 180., 270.]))
    reference = deepcopy(model).double().train()
    candidate = model.double().train()
    device = torch.device("cpu")
    candidate.configure_four_stage_pipeline(device, device, device, device,
        split_step=2, activation_checkpoint=True, checkpoint_mesh_per_step=True,
        checkpoint_grid2mesh_stage=True, checkpoint_mesh2grid_stage=True)
    optimizers = [torch.optim.SGD(m.parameters(), lr=.001) for m in (reference, candidate)]
    for _ in range(2):
        source = torch.randn(1, 12, 3, dtype=torch.double, requires_grad=True)
        copied = source.detach().clone().requires_grad_()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        expected = reference(source)
        actual = tracked_reentrant_checkpoint(candidate, copied) if outer else candidate(copied)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for left, right in zip((source, *reference.parameters()),
                               (copied, *candidate.parameters()), strict=True):
            assert (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-8, atol=1e-9)
        for optimizer in optimizers:
            optimizer.step()
        for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-9)


def test_mp1_dp2_is_not_silently_enabled():
    assert run("--mp", "1", "--dp", "2").returncode != 0


def test_single_device_uses_real_epoch_loop(cpu_runner, tmp_path, monkeypatch):
    from graphcast_compat.distributed_training import DistributedContext
    monkeypatch.setattr(DistributedContext, "from_environment", classmethod(
        lambda cls, **kw: cls(0, 0, 1, 1, 1, (0,), "gloo", True, "cpu")))
    argv = command(inputs(tmp_path), tmp_path / "single", stop=2)
    argv[argv.index("--mp-devices-per-replica") + 1] = "1"
    assert cpu_runner.train(argv) == 0
    checkpoint = torch.load(tmp_path / "single/checkpoints/step_00000002.pt", weights_only=False)
    assert checkpoint["optimizer_state"]["state"]
