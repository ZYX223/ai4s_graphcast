import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/training/run_training.py"


def run(*args):
    return subprocess.run([sys.executable, str(ENTRY), *args], capture_output=True, text=True)


def test_help_and_cpu_only_dry_run_do_not_load_payloads(tmp_path):
    assert run("--help").returncode == 0
    for dp in (1, 2):
        result = run("--dry-run", "--dp", str(dp), "--global-batch-size", "6",
                     "--micro-batch-size", "2", "--corpus", "/missing/corpus.json",
                     "--stats", "/missing/stats", "--output-dir", str(tmp_path / "absent"))
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["accumulation_steps"] == (3 if dp == 1 else 2)
        assert payload["rank_device_groups"] == [list(range(4 * r, 4 * r + 4)) for r in range(dp)]
        assert payload["storage_dtype"] == "float32"
        assert payload["resolved_precision"] == "unverified"
        assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("args, message", [
    (("--precision", "bf16"), "invalid choice"),
    (("--pretrained-checkpoint", "a", "--resume-checkpoint", "b"), "not allowed"),
    (("--dp", "2", "--global-batch-size", "3"), "divisible"),
    (("--devices", "1,2,3,4"), "contiguous"),
    (("--mp", "2"), "MP=4"),
])
def test_invalid_contracts_fail_before_runtime(args, message):
    result = run("--dry-run", *args)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize("device", ["cuda", "npu", "musa"])
def test_dry_run_never_imports_accelerator_or_data_modules(device):
    code = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'torch_musa', 'xarray', 'flag_gems'}:
            raise RuntimeError('dry-run imported ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--dry-run', '--backend', 'flaggems', '--device-type', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(ENTRY), device], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_formal_run_rejects_real_cropped_corpus_before_accelerator_resolution(tmp_path):
    from test_training_data_integration import _write_source
    from test_official_training_corpus import _write_manifest
    source = _write_source(tmp_path / "crop.nc", start="2017-01-01", time_count=16)
    manifest = _write_manifest(tmp_path / "corpus.json", [source])
    result = run("--corpus", str(manifest), "--stats", "/missing/stats", "--device-type", "npu")
    assert result.returncode != 0
    assert "721 x 1440" in result.stderr
    assert "torch_npu" not in result.stderr


def test_schedule_overrides_change_boundaries_without_changing_max_update_identity():
    args = ("--dry-run", "--warmup-updates", "2", "--one-step-updates", "3",
            "--curriculum-updates", "11", "--curriculum-every-updates", "1")
    full = json.loads(run(*args).stdout)
    short = json.loads(run(*args, "--max-updates", "2").stdout)
    assert full["stop_at_update"] == 16
    assert short["stop_at_update"] == 2
    assert full["schedule_digest"] == short["schedule_digest"]


def test_shipped_schedule_reaches_warmup_cosine_and_all_curriculum_boundaries():
    sys.path.insert(0, str(ROOT / "src"))
    from graphcast_compat.training_schedule import TrainingSchedule
    schedule = TrainingSchedule.from_dict(json.loads(run("--dry-run").stdout)["schedule"])
    expected = [(0, 0., 1), (999, .001, 1), (1000, .001, 1), (299999, 0., 1),
                (300000, 3e-7, 2), (300999, 3e-7, 2), (301000, 3e-7, 3),
                (310000, 3e-7, 12), (310999, 3e-7, 12)]
    for step, learning_rate, rollout in expected:
        value = schedule.value_for_update(step)
        assert value.learning_rate == pytest.approx(learning_rate)
        assert value.rollout_steps == rollout
    with pytest.raises(ValueError, match="outside"):
        schedule.value_for_update(311000)
