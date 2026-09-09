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
    assert argv[argv.index("--micro-batch-size") + 1] == str(dp)
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
