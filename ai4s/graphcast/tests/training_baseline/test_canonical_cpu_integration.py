"""Actual epoch loop/model/data on CPU; only hardware/full-size gates supplied.

This fixture is not full-model or multi-device acceptance. No second loop exists.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture
def cpu_runner(monkeypatch, tmp_path):
    import run_official_epoch_training as runner
    import official_epoch_training_file_validation as validation
    from graphcast_compat.distributed_training import DistributedContext
    from graphcast_compat import training_state
    original_placement = runner.GoogleSmallCompatibleModel.configure_four_stage_pipeline
    def cpu_placement(model, *args, **kwargs):
        original_placement(model, *args, **kwargs)
        # CPU tensors discard logical CPU indices; preserve the source stage
        # setup while making its input residency check describe actual tensors.
        model._primary_device = torch.device("cpu")
    monkeypatch.setattr(runner.GoogleSmallCompatibleModel, "configure_four_stage_pipeline", cpu_placement)
    torch.distributed.init_process_group("gloo", init_method="file://" + str(tmp_path / "gloo"), rank=0, world_size=1)
    monkeypatch.setenv("GRAPHCAST_FLAGOS_MODE", "off")
    monkeypatch.setenv("GRAPHCAST_BACKEND", "cpu")
    monkeypatch.setenv("GRAPHCAST_ACCELERATOR_TYPE", "cpu")
    monkeypatch.setattr(runner, "configure_float32_math", lambda enabled: "ieee_fp32")
    monkeypatch.setattr(DistributedContext, "from_environment", classmethod(
        lambda cls, **kw: cls(0, 0, 1, 1, kw["mp_devices_per_replica"], (0, 1), "gloo", True, "cpu")))
    monkeypatch.setattr(runner, "accelerator_device", lambda index, *args: torch.device("cpu", index))
    monkeypatch.setattr(runner, "accelerator_environment", lambda **kw: {"accelerator_type": "cpu"})
    monkeypatch.setattr(runner, "reset_peak_memory", lambda indices: None)
    monkeypatch.setattr(runner, "synchronize", lambda indices: None)
    monkeypatch.setattr(runner, "memory_by_device", lambda indices: {
        str(i): {"allocated_gib": 0., "reserved_gib": 0.} for i in indices})
    monkeypatch.setattr(runner, "validate_formal_dataset", lambda *args, **kw: None)
    monkeypatch.setattr(runner, "validate_full_operational_model_config", lambda config: None)
    monkeypatch.setattr(runner, "validate_canonical_operational_graph_contract", lambda contract: None)
    monkeypatch.setattr(validation, "capture_rng_state", lambda **kw:
        training_state.capture_rng_state(cuda_device_indices=(), accelerator_type="cpu"))
    monkeypatch.setattr(runner, "restore_rng_state", lambda state, **kw:
        training_state.restore_rng_state(state, cuda_device_indices=(), accelerator_type="cpu"))
    def forbid_manual_sync(*args, **kwargs):
        raise AssertionError("DDP must not manually average gradients a second time")
    monkeypatch.setattr(DistributedContext, "synchronize_gradients", forbid_manual_sync)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield runner
    torch.set_num_threads(before)
    torch.distributed.destroy_process_group()


def inputs(tmp_path):
    from test_training_data_integration import _write_source, CONFIG_PATH
    from test_official_training_corpus import _write_manifest
    data = _write_source(tmp_path / "source.nc", start="2017-01-01", time_count=20)
    manifest = _write_manifest(tmp_path / "corpus.json", [data])
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["model_config"].update(mesh_size=0, latent_size=4, gnn_msg_steps=2)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    with xr.open_dataset(data) as source:
        names = tuple(source.data_vars)
    for filename, value in (("mean_by_level.nc", 0.), ("stddev_by_level.nc", 1e6), ("diffs_stddev_by_level.nc", 1e6)):
        xr.Dataset({name: xr.DataArray(np.float32(value)) for name in names}).to_netcdf(stats_dir / filename)
    return config, manifest, stats_dir


def command(paths, run_dir, *, stop, resume=None, prefetch=0, cache=0.):
    config, manifest, stats = paths
    argv = ["--config", str(config), "--corpus-manifest", str(manifest), "--stats-dir", str(stats),
        "--dataset-kind", "era5", "--training-stage", "era5_pretrain",
        "--run-dir", str(run_dir), "--save-checkpoint", str(run_dir / "final.pt"),
        "--init-mode", "random", "--tensor-core", "disabled", "--schedule-recipe", "official_2_to_12",
        "--warmup-updates", "2", "--training-updates", "2", "--updates", "11",
        "--batch-size", "2", "--micro-batch-size", "1", "--mp-devices-per-replica", "2",
        "--mesh-split-step", "1", "--gradient-sync", "ddp", "--activation-checkpoint",
        "--rollout-checkpoint-segments", "12", "--checkpoint-every-steps", "1",
        "--stop-after-global-step", str(stop), "--data-prefetch-depth", str(prefetch),
        "--split-cache-max-gib", str(cache), "--formal"]
    if resume:
        argv += ["--resume-training-checkpoint", str(resume)]
    return argv


@pytest.mark.parametrize("prefetch", [0, 1])
def test_actual_epoch_loop_replays_model_optimizer_samples_and_metrics(cpu_runner, tmp_path, prefetch):
    from graphcast_compat.training_state import state_object_sha256
    paths = inputs(tmp_path)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert cpu_runner.train(command(paths, whole, stop=6, prefetch=prefetch)) == 0
    assert cpu_runner.train(command(paths, resumed, stop=1, prefetch=prefetch)) == 0
    assert cpu_runner.train(command(paths, resumed, stop=6, prefetch=prefetch,
        resume=resumed / "checkpoints/step_00000001.pt")) == 0
    expected = torch.load(whole / "checkpoints/step_00000006.pt", weights_only=False)
    actual = torch.load(resumed / "checkpoints/step_00000006.pt", weights_only=False)
    assert actual["cursor"] == expected["cursor"]
    assert actual["cursor"]["global_optimizer_step"] == 6
    assert state_object_sha256(actual["model_state"]) == state_object_sha256(expected["model_state"])
    assert state_object_sha256(actual["optimizer_state"]) == state_object_sha256(expected["optimizer_state"])
    rows = cpu_runner.read_metric_segments(whole)
    replay = cpu_runner.read_metric_segments(resumed)
    assert [row["loss"] for row in rows] == [row["loss"] for row in replay]
    assert all(row["mfu_sys"] is None for row in rows)
    assert expected["metadata"]["resume_contract"]["training_sampling"] == "trajectory_with_replacement_per_optimizer_update"
    assert expected["metadata"]["resume_contract"]["gradient_accumulation_steps"] == 2
    assert [row["rollout_steps"] for row in rows] == [1, 1, 1, 1, 2, 3]
    changed = command(paths, tmp_path / "changed", stop=7, prefetch=prefetch,
                      resume=resumed / "checkpoints/step_00000006.pt")
    changed += ["--seed", "9"]
    with pytest.raises(ValueError, match="seed"):
        cpu_runner.train(changed)


def test_exact_resume_accepts_rank_device_and_record_path_changes(cpu_runner, tmp_path, monkeypatch):
    from dataclasses import replace
    original = cpu_runner.enable_network_replacements_from_env
    evidence = {"device": "cuda:0", "flaggems_record_path": "/records/process_101.log"}
    def with_runtime_evidence(**kwargs):
        runtime = original(**kwargs)
        return replace(runtime, metadata={**runtime.metadata, **evidence})
    monkeypatch.setattr(cpu_runner, "enable_network_replacements_from_env", with_runtime_evidence)
    paths = inputs(tmp_path)
    run_dir = tmp_path / "run"
    assert cpu_runner.train(command(paths, run_dir, stop=1)) == 0
    evidence.update(device="cuda:2", flaggems_record_path="/records/process_202.log")
    assert cpu_runner.train(command(paths, run_dir, stop=2,
        resume=run_dir / "checkpoints/step_00000001.pt")) == 0
    observations = [json.loads((run_dir / f"processes/from_{step:08d}/rank_0/runtime.json").read_text())
                    for step in (1, 2)]
    assert [row["operator_replacement"]["runtime"]["device"] for row in observations] == ["cuda:0", "cuda:2"]
    assert observations[0]["operator_replacement"]["runtime"]["flaggems_record_path"] == "/records/process_101.log"


@pytest.mark.parametrize("prefetch", [0, 1])
def test_cache_enabled_exact_resume_keeps_configuration_and_separate_observations(cpu_runner, tmp_path, prefetch):
    from graphcast_compat.training_state import state_object_sha256
    paths = inputs(tmp_path)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert cpu_runner.train(command(paths, whole, stop=2, cache=.01, prefetch=prefetch)) == 0
    assert cpu_runner.train(command(paths, resumed, stop=1, cache=.01, prefetch=prefetch)) == 0
    immutable_before = (resumed / "run_config.json").read_bytes()
    assert cpu_runner.train(command(paths, resumed, stop=2, cache=.01, prefetch=prefetch,
        resume=resumed / "checkpoints/step_00000001.pt")) == 0
    assert (resumed / "run_config.json").read_bytes() == immutable_before
    assert json.loads(immutable_before)["data_pipeline"]["split_cache"] == {"max_gib": .01}
    expected = torch.load(whole / "checkpoints/step_00000002.pt", weights_only=False)
    actual = torch.load(resumed / "checkpoints/step_00000002.pt", weights_only=False)
    for key in ("model_state", "optimizer_state", "cursor"):
        assert state_object_sha256(actual[key]) == state_object_sha256(expected[key])
    for step in (1, 2):
        observation = json.loads((resumed / f"processes/from_{step:08d}/rank_0/runtime.json").read_text())
        cache = observation["split_cache"]
        assert cache["enabled"] is True
        assert cache["loaded_bytes"] > 0
        assert cache["preload_seconds"] > 0
        assert cache["finite_validation_seconds"] > 0


def test_source_native_validation_computes_per_variable_lead_losses(cpu_runner, tmp_path):
    from test_training_data_integration import _write_source
    paths = inputs(tmp_path)
    validation_source = _write_source(tmp_path / "hres.nc", start="2022-01-01", time_count=20)
    validation_manifest = tmp_path / "validation.json"
    validation_manifest.write_text(json.dumps({"schema_version": 1, "name": "validation-fixture",
        "dataset_kind": "hres", "training_stage": "validation", "data_role": "validation",
        "shards": [{"path": validation_source.name}]}))
    run_dir = tmp_path / "validated"
    # One real source-native epoch update, followed by the exposed validation path.
    argv = command(paths, run_dir, stop=0) + ["--schedule-recipe", "manifest",
        "--schedule-manifest", str(ROOT / "configs/training/epoch_rehearsal_v1.json"),
        "--batch-size", "8", "--micro-batch-size", "8",
        "--validation-corpus-manifest", str(validation_manifest),
        "--validation-mode", "quick", "--validation-steps", "1"]
    assert cpu_runner.train(argv) == 0
    report = json.loads((run_dir / "validation.json").read_text())
    assert report["validation_self_check"] is True
    assert report["optimizer_updates"] == 0
    assert report["gradient_enabled"] is False
    assert len(report["steps"]) == 1
    variables = report["steps"][0]["per_variable_lead_losses"]
    assert set(variables) == set(yaml.safe_load(paths[0].read_text())["task_config"]["target_variables"])
    assert all(len(values) == 12 and np.isfinite(values).all() for values in variables.values())


def test_final_canonical_artifacts_qualify_and_resume_sidecars_keep_actual_state(cpu_runner, tmp_path):
    import diagnose_training_loss as diagnostic
    from graphcast_compat.training_diagnostics import file_sha256
    from graphcast_compat.training_state import state_object_sha256
    paths = inputs(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    for run in (first, second):
        assert cpu_runner.train(command(paths, run, stop=2)) == 0
    result = diagnostic.compare_losses(first, second, start_step=1, end_step=2)
    assert result["status"] == "passed", result["comparison_issues"]
    initial_sidecar = first / "metrics/segment_00000001.start.json"
    before = initial_sidecar.read_bytes()
    metadata = json.loads(before)
    assert metadata["data_sha256"] == [file_sha256(tmp_path / "source.nc")]
    assert metadata["config_sha256"] == file_sha256(paths[0])
    assert metadata["stats_sha256"]["mean_by_level.nc"] == file_sha256(paths[2] / "mean_by_level.nc")
    checkpoint = torch.load(first / "checkpoints/step_00000002.pt", weights_only=False)
    assert cpu_runner.train(command(paths, first, stop=3,
        resume=first / "checkpoints/step_00000002.pt")) == 0
    assert initial_sidecar.read_bytes() == before
    resumed = json.loads((first / "metrics/segment_00000003.start.json").read_text())
    assert resumed["start_optimizer_step"] == 2
    # AdamW's LR is set for the resumed next update before evidence is emitted.
    for group in checkpoint["optimizer_state"]["param_groups"]:
        group["lr"] = cpu_runner.resolve_training_schedule(cpu_runner.parse_args(command(paths, first, stop=3)))[0].value_for_update(2).learning_rate
    assert resumed["states_by_rank"][0]["optimizer_sha256"] == state_object_sha256(checkpoint["optimizer_state"])
    assert resumed["states_by_rank"][0]["parameters_sha256"] != metadata["states_by_rank"][0]["parameters_sha256"]
