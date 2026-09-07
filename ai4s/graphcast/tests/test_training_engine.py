"""FP32 scalar integration fixtures, not full-model acceptance."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphcast_compat.autoregressive_training import AutoregressiveBatch, FeedbackUpdate
from graphcast_compat.distributed_training import DistributedContext
from graphcast_compat.training_engine import run_training_update
from graphcast_compat.training_recipe import build_graphcast_adamw


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def forward(self, features):
        return self.weight * features[..., 1:2]


def batch(values, weights=None, rollout=2):
    x = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
    return AutoregressiveBatch(
        torch.cat((x * 0, x), -1), torch.empty(rollout, len(values), 1, 0),
        torch.stack([x * (lead + 2) for lead in range(rollout)]),
        (FeedbackUpdate("state", slice(0, 1), slice(1, 2), slice(0, 1), torch.ones(1)),),
        (), torch.ones(1), torch.ones(1), {"state": slice(0, 1)}, 1, 1,
        sample_weights=torch.tensor(weights or [1.] * len(values)),
    )


def context():
    return DistributedContext(0, 0, 1, 1, 1, (), None, False, "cpu")


def test_hand_derived_bptt_update_and_detached_diagnostics():
    model = Residual()
    result = run_training_update(model, torch.optim.SGD(model.parameters(), lr=.01),
                                 [(batch([1.]), 1.)], context=context())
    assert result["loss"] == pytest.approx(1.314453125)
    assert result["gradient_norm"] == pytest.approx(4.34375)
    assert model.weight.item() == pytest.approx(.2934375)
    assert isinstance(result["lead_losses"][0], float)
    assert model.weight.dtype == torch.float32


def test_lazy_unequal_padded_accumulation_equals_global_valid_batch():
    full, accumulated = Residual(), Residual()
    expected = run_training_update(full, torch.optim.SGD(full.parameters(), lr=.01),
                                   [(batch([1., 2., 3.]), 1.)], context=context())
    def parts():
        yield batch([1., 2.]), 2 / 3
        # Converting the generator to a list would fail this memory contract.
        assert accumulated.weight.grad is not None
        yield batch([3., 999.], [1., 0.]), 1 / 3
    actual = run_training_update(accumulated, torch.optim.SGD(accumulated.parameters(), lr=.01),
                                parts(), context=context(), checkpoint_segments=1, loss_grid_chunk_size=1)
    assert actual["loss"] == pytest.approx(6.134114583333333)
    assert actual["loss"] == pytest.approx(expected["loss"])
    assert accumulated.weight.item() == pytest.approx(.4527083333333333)
    torch.testing.assert_close(full.weight, accumulated.weight)


def test_zero_lr_update_advances_adamw_without_changing_parameters():
    model = Residual()
    optimizer = build_graphcast_adamw(model, lr=0.)
    run_training_update(model, optimizer, [(batch([1.]), 1.)], context=context())
    assert model.weight.item() == .25
    assert optimizer.state[model.weight]["step"].item() == 1
    assert optimizer.state[model.weight]["exp_avg"].abs().item() > 0


def test_nonfinite_update_does_not_step():
    model = Residual()
    optimizer = build_graphcast_adamw(model, lr=.01)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_training_update(model, optimizer, [(batch([float("nan")]), 1.)], context=context())
    assert not optimizer.state
    assert model.weight.item() == .25


def test_real_dp_engine_global_mean_all_padding_rank_and_collective_failures(tmp_path):
    from test_training_data_integration import _write_source
    from test_official_training_corpus import _write_manifest
    output = tmp_path / "dp.json"
    source = _write_source(tmp_path / "source.nc", start="2017-01-01", time_count=8)
    manifest = _write_manifest(tmp_path / "corpus.json", [source])
    env = {**os.environ, "TRAINING_TEST_OUTPUT": str(output), "TRAINING_TEST_CORPUS": str(manifest)}
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=2", str(Path(__file__).resolve()), "--worker"],
        env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())
    for row in rows:
        assert row["weight"] == pytest.approx(.4527083333333333)
        assert row["loss"] == pytest.approx(6.134114583333333)
        assert row["all_padding_weight"] == pytest.approx(.2934375)
        assert row["syncs"] == 2
        assert row["aborted"] == [True, True]
        assert row["resumed_exact"] is True
        assert row["second_effective_samples"] == 1


def _dp_worker():
    ctx = DistributedContext.from_environment(dp_replicas=2, mp_devices_per_replica=1,
        available_device_count=2, backend="gloo", accelerator_type="cpu")
    syncs = 0
    original_sync = ctx.synchronize_gradients
    def synchronize(model):
        nonlocal syncs
        syncs += 1
        return original_sync(model)
    ctx.synchronize_gradients = synchronize
    try:
        model = Residual()
        values = [1., 2.] if ctx.rank == 0 else [3., 999.]
        weights = [1., 1.] if ctx.rank == 0 else [1., 0.]
        parts = ((batch([x], [w]), 2 * w / 3) for x, w in zip(values, weights))
        result = run_training_update(model, torch.optim.SGD(model.parameters(), lr=.01), parts, context=ctx)
        padded = Residual()
        w = 1. if ctx.rank == 0 else 0.
        run_training_update(padded, torch.optim.SGD(padded.parameters(), lr=.01),
                            [(batch([1.], [w]), 2 * w)], context=ctx)
        aborted = []
        for failure in ("prediction", "gradient"):
            bad = Residual()
            optimizer = build_graphcast_adamw(bad, lr=.01)
            x = float("nan") if failure == "prediction" and ctx.rank == 1 else 1.
            if failure == "gradient" and ctx.rank == 1:
                bad.weight.register_hook(lambda grad: grad * float("nan"))
            try:
                run_training_update(bad, optimizer, [(batch([x]), 1.)], context=ctx)
                aborted.append(False)
            except FloatingPointError:
                aborted.append(not optimizer.state and bad.weight.item() == .25)
        rows = ctx.gather_objects({"weight": model.weight.item(), "loss": result["loss"],
            "all_padding_weight": padded.weight.item(), "syncs": syncs, "aborted": aborted})
        from graphcast_compat.training_runtime import TrainingSession
        from graphcast_compat.training_schedule import TrainingSchedule
        from graphcast_compat.training_state import state_object_sha256
        from graphcast_compat.official_training_corpus import OfficialTrainingCorpus
        from test_training_data_integration import CONFIG_PATH
        # Real corpus extraction must receive a global microbatch before slicing
        # the all-padding rank; a rank-local all-zero call would be rejected.
        with OfficialTrainingCorpus(os.environ["TRAINING_TEST_CORPUS"], config_path=CONFIG_PATH, rollout_steps=2) as corpus:
            ctx.device_indices = ()  # CPU-only rank RNG capture, no accelerator.
            def prepare(indices, weights, rank, replicas, rollout):
                sample = corpus.sample_indices(window_indices=indices, sample_weights=weights,
                    rank=rank, replicas=replicas, target_rollout_steps=rollout)
                values = [float(i) * .1 + torch.rand(()).item() * .01 for i in sample.window_indices]
                return batch(values, list(sample.sample_weights), rollout)
            def session():
                m = Residual()
                return TrainingSession(m, build_graphcast_adamw(m, lr=.01), context=ctx,
                    schedule=TrainingSchedule.constant(name="dp", learning_rate=.01, rollout_steps=2),
                    window_indices=corpus.window_indices, global_batch_size=4, micro_batch_size=1,
                    seed=5, shuffle=False, prepare_batch=prepare, resume_contract={"fixture": 1})
            first = session()
            first.update()
            checkpoint = Path(os.environ["TRAINING_TEST_OUTPUT"]).with_suffix(".pt")
            first.save(checkpoint)
            second = first.update()
            expected = state_object_sha256(first.optimizer.state_dict())
            restored = session()
            restored.restore(checkpoint)
            replay = restored.update()
            exact = replay["loss"] == second["loss"] and state_object_sha256(restored.optimizer.state_dict()) == expected
            extra = ctx.gather_objects({"resumed_exact": exact, "second_effective_samples": second["effective_samples"]})
            for row, additional in zip(rows, extra):
                row.update(additional)
        if ctx.is_primary:
            Path(os.environ["TRAINING_TEST_OUTPUT"]).write_text(json.dumps(rows))
    finally:
        ctx.close()


if __name__ == "__main__" and "--worker" in sys.argv:
    _dp_worker()
