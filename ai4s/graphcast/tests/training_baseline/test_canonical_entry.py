"""Public subprocess contracts; no hardware or training payload is needed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/training/run_training.py"


def run(*args, env=None):
    return subprocess.run([sys.executable, str(ENTRY), "--dry-run", *args],
                          capture_output=True, text=True, env=env)


@pytest.mark.parametrize("mp,dp", [(4, 1), (4, 2), (2, 1), (2, 2)])
def test_topology_maps_local_microbatch_to_canonical_global_batch(mp, dp):
    result = run("--device-type", "npu", "--mp", str(mp), "--dp", str(dp),
                 "--global-batch-size", "4", "--micro-batch-size", "1")
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["mp"] == mp
    assert row["canonical_micro_batch_size"] == dp
    assert row["accumulation_steps"] == 4 // dp
    assert row["rank_device_groups"] == [list(range(r * mp, (r + 1) * mp)) for r in range(dp)]
    assert row["requested_precision"] == "tf32_hf32"
    assert row["storage_dtype"] == "float32"
    assert row["training_mfu"] is None
    assert row["epoch_policy"] == "trajectory_with_replacement_per_optimizer_update"
    assert row["environment"]["GRAPHCAST_ACCELERATOR_TYPE"] == "npu"
    assert row["environment"]["GRAPHCAST_PLATFORM_LABEL"] == "Ascend 910C"
    assert row["environment"]["ASCEND_LAUNCH_BLOCKING"] == "1"
    argv = row["engine_argv"]
    assert argv[argv.index("--global-micro-batch-size") + 1] == str(dp)
    assert "--formal" in argv
    assert "--activation-checkpoint" in argv
    # Source initializes a process group even for DP1; torchrun supplies env://.
    assert row["launch_command"][:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert row["launch_command"][4:6] == ["--nproc-per-node", str(dp)]
    if mp == 2:
        assert row["execution"]["mesh_split_step"] == 8
        assert row["execution"]["mesh_checkpoint_granularity"] == "step"
        assert row["execution"]["rollout_checkpoint_implementation"] == "non_reentrant"
        assert row["execution"]["gradient_sync"] == "ddp"
    else:
        assert row["execution"]["mesh_split_step"] == 16
        assert row["execution"]["rollout_checkpoint_implementation"] == "reentrant"
        if dp == 1:
            # Bound mesh replay to one message step on the four-device recipe.
            assert argv[argv.index("--mesh-checkpoint-granularity") + 1] == "step"


@pytest.mark.parametrize("args", [
    ("--mp", "3"), ("--dp", "3"),
    ("--dp", "2", "--global-batch-size", "3"),
    ("--global-batch-size", "6", "--dp", "2", "--micro-batch-size", "2"),
    ("--curriculum-updates", "12"),
    ("--device-type", "npu", "--platform", "hygon"),
    ("--peak-tflops-per-device", "nan"),
    ("--peak-tflops-per-device", "100"),
])
def test_incompatible_contract_rejected_before_runtime(args):
    assert run(*args).returncode != 0


def test_explicit_execution_precision_and_backend_policy_are_forwarded():
    result = run("--device-type", "npu", "--mp", "2", "--tensor-core", "disabled",
                 "--mesh-split-step", "7", "--mesh-checkpoint-granularity", "stage",
                 "--backend", "flaggems", "--flaggems-config", "/custom/train.yaml",
                 "--flaggems-ops", "layer_norm,silu", "--flaggems-variant", "official")
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["requested_precision"] == "ieee_fp32"
    assert row["execution"]["mesh_split_step"] == 7
    assert row["execution"]["mesh_checkpoint_granularity"] == "stage"
    assert row["environment"]["GRAPHCAST_FLAGGEMS_CONFIG"] == "/custom/train.yaml"
    assert row["environment"]["GRAPHCAST_FLAGGEMS_OPS"] == "layer_norm,silu"
    assert row["environment"]["GRAPHCAST_FLAGGEMS_VARIANT"] == "official"
    assert row["environment"]["FLAGOS_REPLACE"] == "1"


def test_mesh_projection_experiment_is_opt_in_and_forwarded_to_engine():
    baseline = run("--device-type", "npu")
    assert baseline.returncode == 0, baseline.stderr
    assert json.loads(baseline.stdout)["execution"].get("mesh_edge_input_projection", "concatenated") == "concatenated"
    experiment = run("--device-type", "npu", "--mesh-edge-input-projection", "factorized")
    assert experiment.returncode == 0, experiment.stderr
    argv = json.loads(experiment.stdout)["engine_argv"]
    assert argv[argv.index("--mesh-edge-input-projection") + 1] == "factorized"


@pytest.mark.parametrize("platform,label,backend", [
    ("hygon", "Hygon BW3000", "hygon"),
    ("metax", "MetaX C550", "metax"),
    ("thead", "T-Head PPU-ZW810E", "ptg"),
])
def test_cuda_compatible_platform_identity_survives_adapter(platform, label, backend):
    result = run("--platform", platform)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["environment"]
    assert env["GRAPHCAST_PLATFORM_LABEL"] == label
    assert env["GRAPHCAST_BACKEND"] == backend
    assert env["FLAGOS_REPLACE"] == "0"


def test_prefix_stop_preserves_full_source_schedule():
    result = run("--warmup-updates", "6", "--one-step-updates", "10",
                 "--curriculum-updates", "22", "--max-updates", "3")
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["stop_at_update"] == 3
    assert [phase["updates"] for phase in row["schedule"]["phases"]] == [6, 10, 22]
    assert row["schedule"]["phases"][2]["rollout"]["every_updates"] == 2


def test_dry_run_is_stdlib_only_even_when_optional_backends_are_requested():
    code = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'numpy', 'torch_npu', 'torch_musa', 'xarray', 'flag_gems'}:
            raise RuntimeError('unexpected payload import ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--dry-run', '--backend', 'flaggems', '--device-type', 'npu', '--mp', '2']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", code, str(ENTRY)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shell_entry_preserves_python_platform_contract():
    env = {**os.environ, "PYTHON_BIN": sys.executable, "GRAPHCAST_PLATFORM": "ascend910c"}
    result = subprocess.run(["bash", str(ENTRY.with_name("train_pipeline.sh")), "--dry-run", "--mp", "2"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["environment"]["GRAPHCAST_PLATFORM_LABEL"] == "Ascend 910C"
    assert row["execution"]["gradient_sync"] == "ddp"


def test_top_level_help_exposes_formal_training_not_legacy_sgd():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train.sh"), "--help"],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--global-batch-size" in result.stdout
    assert "--corpus" in result.stdout


@pytest.mark.parametrize("entry", [
    "scripts/train.sh", "scripts/training/train_official.sh", "scripts/training/train_pipeline.sh",
])
def test_formal_shell_entries_preserve_batch_and_paths_from_external_cwd(entry, tmp_path):
    output = tmp_path / "output with spaces"
    result = subprocess.run(
        ["bash", str(ROOT / entry), "--dry-run", "--device-type", "npu",
         "--mp", "4", "--dp", "2", "--global-batch-size", "16",
         "--micro-batch-size-per-rank", "2", "--output-dir", str(output)],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["global_micro_batch_size"] == 4
    assert row["gradient_accumulation_steps"] == 4
    argv = row["engine_argv"]
    assert argv[argv.index("--run-dir") + 1] == str(output)
    assert not output.exists()


@pytest.mark.parametrize("script", ["train_mp.sh", "train_ddp.sh", "train_25km_single.sh", "train_100km_single.sh"])
def test_legacy_shell_aliases_keep_original_help_and_show_migration(script, tmp_path):
    env = {**os.environ, "STEPS": "3"}
    original = subprocess.run(
        ["bash", str(ROOT / "scripts/training" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )
    relocated = subprocess.run(
        ["bash", str(ROOT / "scripts/training/legacy" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )
    assert original.returncode == relocated.returncode == 0
    assert original.stdout == relocated.stdout
    assert "Deprecated probe path" in original.stderr


@pytest.mark.parametrize("variable,value", [("STEPS", "3"), ("LR", "0.001"), ("DEVICES", "npu:0,npu:1")])
def test_top_level_entry_rejects_legacy_environment_instead_of_changing_recipe(variable, value):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train.sh"), "--help"],
        env={**os.environ, "PYTHON_BIN": sys.executable, variable: value},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert variable in result.stderr
    assert "legacy" in result.stderr.lower()


@pytest.mark.parametrize("variable", ["FLAGGEMS_RUNTIME_DEST", "GRAPHCAST_FLAGGEMS_SOURCE_ROOT"])
def test_formal_entry_keeps_prepared_flaggems_source_paths(variable, tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train.sh"), "--dry-run", "--platform", "ascend910c",
         "--backend", "flaggems", "--flaggems-variant", "official"],
        env={**os.environ, "PYTHON_BIN": sys.executable, variable: str(tmp_path / "runtime")},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["environment"]["GRAPHCAST_FLAGOS_MODE"] == "flaggems"
    assert row["environment"]["GRAPHCAST_FLAGGEMS_VARIANT"] == "official"
    # The inherited source paths must not be reset by the formal adapter.
    assert variable not in row["environment"]


@pytest.mark.parametrize("dp,local_micro,global_micro,accumulation", [(1, 1, 1, 16), (1, 2, 2, 8), (2, 1, 2, 8), (2, 2, 4, 4)])
def test_explicit_per_rank_microbatch_preserves_effective_batch(dp, local_micro, global_micro, accumulation):
    result = run("--device-type", "npu", "--dp", str(dp),
                 "--global-batch-size", "16", "--micro-batch-size-per-rank", str(local_micro))
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["global_batch_size"] == 16
    assert row["micro_batch_size_per_rank"] == local_micro
    assert row["global_micro_batch_size"] == global_micro
    assert row["gradient_accumulation_steps"] == accumulation
    # Preserve existing dry-run consumers while giving the units explicit names.
    assert row["canonical_micro_batch_size"] == global_micro
    assert row["accumulation_steps"] == accumulation
    argv = row["engine_argv"]
    assert argv[argv.index("--global-micro-batch-size") + 1] == str(global_micro)
    assert "--micro-batch-size" not in argv


def test_public_microbatch_aliases_cannot_silently_override_each_other():
    result = run("--micro-batch-size-per-rank", "1", "--micro-batch-size", "2")
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


@pytest.mark.parametrize("script", ["train_mp.sh", "train_ddp.sh", "train_25km_single.sh", "train_100km_single.sh"])
def test_relocated_probe_help_is_available_without_training(script, tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/training/legacy" / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "STEPS" in result.stdout


@pytest.mark.parametrize("script", ["train_mp.py", "train_ddp.py", "train_single.py"])
def test_relocated_python_probe_finds_project_assets_without_pythonpath(script, tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    code = """
import json, runpy, sys
ns = runpy.run_path(sys.argv[1])
print(json.dumps({'root': str(ns['REPO_ROOT']), 'config': str(ns['DEFAULT_CONFIG'])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts/training/legacy/lib" / script)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert Path(row["root"]) == ROOT
    assert Path(row["config"]).is_file()
