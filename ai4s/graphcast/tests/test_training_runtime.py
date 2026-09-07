from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphcast_compat.training_runtime import TrainingSession, build_resume_contract, validate_operational_grid
from graphcast_compat.training_schedule import TrainingSchedule
from graphcast_compat.training_recipe import build_graphcast_adamw
from graphcast_compat.training_state import state_object_sha256
from test_training_engine import Residual, batch, context


def schedule():
    return TrainingSchedule.from_dict({"schema_version": 1, "name": "fixture", "phases": [
        {"name": "warmup", "updates": 1, "learning_rate": {"kind": "constant", "value": 0.},
         "rollout": {"kind": "constant", "value": 1}},
        {"name": "curriculum", "updates": 3, "learning_rate": {"kind": "constant", "value": .01},
         "rollout": {"kind": "increment", "start": 2, "maximum": 3, "every_updates": 1}},
    ]})


def make_session(seen):
    model = Residual()
    def prepare(indices, weights, rank, replicas, rollout):
        seen.extend(i for i, w in zip(indices, weights) if w)
        # Include all three rank-local RNG sources in the replay comparison.
        noise = random.random() + np.random.rand() + torch.rand(()).item()
        return batch([i + 1 + noise * .01 for i in indices], list(weights), rollout)
    return TrainingSession(model, build_graphcast_adamw(model, lr=0.), context=context(),
        schedule=schedule(), window_indices=tuple(range(5)), global_batch_size=3,
        micro_batch_size=2, seed=7, shuffle=False, prepare_batch=prepare,
        resume_contract={"fixture": "v1"})


def test_two_updates_equal_stop_reconstruct_restore_with_exact_cursor_and_rng(tmp_path):
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    seen = []
    continuous = make_session(seen)
    first = continuous.update()
    assert first["rollout_steps"] == 1
    checkpoint = tmp_path / "step1.pt"
    continuous.save(checkpoint)
    second = continuous.update()
    expected_state = state_object_sha256(continuous.optimizer.state_dict())
    expected_weight = continuous.model.weight.detach().clone()
    replay_seen = []
    resumed = make_session(replay_seen)
    resumed.restore(checkpoint)
    replay = resumed.update()
    assert seen == [0, 1, 2, 3, 4]
    assert replay_seen == [3, 4]
    assert second["rollout_steps"] == replay["rollout_steps"] == 2
    assert replay["loss"] == second["loss"]
    assert state_object_sha256(resumed.optimizer.state_dict()) == expected_state
    torch.testing.assert_close(resumed.model.weight, expected_weight, rtol=0, atol=0)
    assert resumed.cursor.epoch == 1 and resumed.cursor.next_batch_index == 0
    assert resumed.cursor.global_optimizer_step == 2


def test_resume_identity_ignores_paths_but_detects_changed_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"original"); b.write_bytes(b"original")
    kwargs = dict(config={"layers": 16}, stats_files=[a], policy={"precision": "tf32"})
    first = build_resume_contract(data_files=[a], **kwargs)
    assert first == build_resume_contract(data_files=[b], **kwargs)
    b.write_bytes(b"changed")
    assert first != build_resume_contract(data_files=[b], **kwargs)
    assert first != build_resume_contract(data_files=[a], **{**kwargs, "policy": {"precision": "ieee_fp32"}})


def test_formal_grid_rejects_crops_and_wrong_levels():
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    data = SimpleNamespace(latitudes=np.linspace(90, -90, 721), longitudes=np.arange(1440) / 4,
                           pressure_levels=np.array(levels))
    validate_operational_grid(data, levels)
    data.latitudes = data.latitudes[:2]
    with pytest.raises(ValueError, match="721"):
        validate_operational_grid(data, levels)


def test_checkpoint_is_forbidden_before_update_or_after_failed_update(tmp_path):
    session = make_session([])
    with pytest.raises(RuntimeError, match="completed optimizer"):
        session.save(tmp_path / "unsafe.pt")
    session.update()
    session.prepare_batch = lambda *args: batch([float("nan")])
    with pytest.raises(FloatingPointError):
        session.update()
    with pytest.raises(RuntimeError, match="completed optimizer"):
        session.save(tmp_path / "unsafe.pt")
    assert not (tmp_path / "unsafe.pt").exists()


def test_resume_rejects_changed_order_before_restoring_optimizer(tmp_path):
    session = make_session([])
    session.update()
    path = tmp_path / "one.pt"
    session.save(path)
    changed = make_session([])
    changed.resume_contract["seed"] = 99
    with pytest.raises(ValueError, match="seed"):
        changed.restore(path)
    assert changed.optimizer.state_dict()["state"] == {}


def test_epoch_planning_work_is_not_repeated_within_an_epoch(monkeypatch):
    session = make_session([])
    original = session._plan
    rebuilt = []
    def record(epoch):
        rebuilt.append(epoch)
        return original(epoch)
    monkeypatch.setattr(session, "_plan", record)
    session.update()
    assert rebuilt == []  # Re-sorting a full corpus per optimizer step is costly.
    session.update()
    assert rebuilt == [1]


def test_real_corpus_normalization_and_padded_updates_equal_full_batch(tmp_path):
    import xarray as xr
    from test_training_data_integration import _write_source, CONFIG_PATH
    from test_official_training_corpus import _write_manifest
    from graphcast_compat.official_training_corpus import OfficialTrainingCorpus
    from graphcast_compat.google_small import GoogleSmallConfig
    from graphcast_compat.autoregressive_training import prepare_autoregressive_batch
    from graphcast_compat.training_runtime import PER_VARIABLE_WEIGHTS
    source = _write_source(tmp_path / "source.nc", start="2017-01-01", time_count=8)
    manifest = _write_manifest(tmp_path / "corpus.json", [source])
    config = GoogleSmallConfig.from_yaml(CONFIG_PATH)
    names = set(config.input_variables) | set(config.target_variables) | set(config.forcing_variables)
    stats = SimpleNamespace(
        mean_by_level=xr.Dataset({name: xr.DataArray(np.float32(0)) for name in names}),
        stddev_by_level=xr.Dataset({name: xr.DataArray(np.float32(1e6)) for name in names}),
        diffs_stddev_by_level=xr.Dataset({name: xr.DataArray(np.float32(1e6)) for name in names}),
    )
    with OfficialTrainingCorpus(manifest, config_path=CONFIG_PATH, rollout_steps=2) as corpus:
        def prepare(indices, weights, rank, replicas, rollout):
            sample = corpus.sample_indices(window_indices=indices, sample_weights=weights,
                rank=rank, replicas=replicas, target_rollout_steps=rollout)
            return prepare_autoregressive_batch(sample.inputs, sample.targets, sample.forcings,
                config=config, stats=stats, input_device=torch.device("cpu"),
                per_variable_weights=PER_VARIABLE_WEIGHTS, sample_weights=sample.sample_weights)
        probe = prepare((1,), (1.,), 0, 1, 2)
        width = probe.initial_input_features.shape[-1] + probe.forcing_features.shape[-1]
        torch.manual_seed(91)
        first = torch.nn.Linear(width, config.output_dim)
        second = torch.nn.Linear(width, config.output_dim)
        second.load_state_dict(first.state_dict())
        def session(model, micro):
            return TrainingSession(model, build_graphcast_adamw(model, lr=.001), context=context(),
                schedule=TrainingSchedule.constant(name="fixture", learning_rate=.001, rollout_steps=2),
                window_indices=corpus.window_indices, global_batch_size=4, micro_batch_size=micro,
                seed=0, shuffle=False, prepare_batch=prepare, resume_contract={"fixture": "real-data"})
        full, accumulated = session(first, 4), session(second, 2)
        for valid in (4, 1):
            expected, actual = full.update(), accumulated.update()
            assert expected["effective_samples"] == actual["effective_samples"] == valid
            assert actual["loss"] == pytest.approx(expected["loss"], rel=1e-6)
            for a, b in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
