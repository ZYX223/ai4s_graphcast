"""CPU-only contracts for the NVIDIA baseline launcher; no GPU job is started."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/baseline/run_nvidia.py"


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("nvidia_baseline_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, mode="train"):
    args = [mode, "--gpu", "0", "--weights", str(tmp_path / "weights.pt"),
            "--stats", str(tmp_path / "stats"), "--output-dir", str(tmp_path / "new-run")]
    if mode == "train":
        args += ["--corpus", str(tmp_path / "corpus.json")]
    else:
        for name in ("inputs", "rollout-forcings", "target-template"):
            args += ["--" + name, str(tmp_path / (name + ".nc"))]
    return args


def options(plan):
    argv = plan["command"][2:]
    return dict(zip(argv[::2], argv[1::2]))


def test_training_reproduces_frozen_recipe(launcher, tmp_path):
    args = launcher.parser().parse_args(arguments(tmp_path))
    plan, env = launcher.build_plan(args, {})
    opt = options(plan)
    assert opt["--mp"] == opt["--dp"] == "1"
    assert opt["--global-batch-size"] == "2"
    assert opt["--micro-batch-size-per-rank"] == "1"
    assert [opt["--" + phase + "-updates"] for phase in ("warmup", "one-step", "curriculum")] == ["6", "10", "22"]
    assert opt["--max-updates"] == "38"
    assert opt["--edge-gather-group-size"] == "4"
    assert opt["--resident-training-inputs"] == "enabled"
    assert opt["--mesh2grid-stage-checkpoint"] == "disabled"
    assert opt["--rollout-checkpoint-implementation"] == "reentrant"
    assert opt["--mesh-edge-input-projection"] == "concatenated"
    assert opt["--backend"] == "native"
    assert opt["--tensor-core"] == "enabled"
    assert env["NUMPY_MADVISE_HUGEPAGE"] == "0"
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert len(plan["source_sha256"]) == 137
    # The unchanged adapter must accept the entire expanded command.
    result = subprocess.run(plan["command"] + ["--dry-run"], env=env, cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    contract = json.loads(result.stdout)
    assert contract["gradient_accumulation_steps"] == 2
    assert contract["requested_precision"] == "tf32_hf32"
    assert contract["stop_at_update"] == 38
    assert not args.output_dir.exists()


def test_recipe_identifies_public_source_commit(launcher):
    recipe = json.loads(launcher.RECIPE.read_text())
    assert recipe["source_commit"] == "50fa1808928ff8e751fe30b13e7659e30d686ef3"
    assert launcher.ROOT == ROOT
    assert len(launcher.source_hashes(recipe)) == 137


@pytest.mark.parametrize("steps", [1, 12, 40, 60])
def test_inference_steps_are_not_hardcoded(launcher, tmp_path, steps):
    args = launcher.parser().parse_args(arguments(tmp_path, "infer") + ["--steps", str(steps)])
    plan, _ = launcher.build_plan(args, {})
    opt = options(plan)
    assert opt["--steps"] == str(steps)
    assert opt["--model-execution-profile"] == "baseline"
    assert opt["--precision"] == "tf32"
    assert opt["--device-count"] == "1"
    assert opt["--warmup-calls"] == "3"
    assert opt["--peak-tflops-per-device"] == "495"
    assert "--trajectory-output" not in opt
    assert "--overwrite" not in opt


@pytest.mark.parametrize("mode", ["train", "infer"])
def test_dry_run_without_site_packages_or_assets(tmp_path, mode):
    result = subprocess.run([sys.executable, "-S", str(ENTRY), *arguments(tmp_path, mode), "--dry-run"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["execution"] == "native_eager"
    assert not list(tmp_path.iterdir())


def test_environment_is_native_and_parent_unchanged(launcher, tmp_path):
    inherited = {"GRAPHCAST_FLAGOS_MODE": "flaggems", "GRAPHCAST_GATHER_MODE": "index_select",
                 "FLAGGEMS_FAKE_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": "0",
                 "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
                 "CUDA_LAUNCH_BLOCKING": "1", "PYTHONPATH": "/old/runtime", "PATH": "/bin"}
    saved = dict(inherited)
    args = launcher.parser().parse_args(arguments(tmp_path))
    plan, env = launcher.build_plan(args, inherited)
    assert inherited == saved
    assert env["GRAPHCAST_FLAGOS_MODE"] == "off"
    assert env["GRAPHCAST_BACKEND"] == "nvidia"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PATH"] == "/bin"
    assert "GRAPHCAST_GATHER_MODE" not in env
    assert "NVIDIA_TF32_OVERRIDE" not in env
    assert "CUDA_LAUNCH_BLOCKING" not in env
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert "FLAGGEMS_FAKE_OVERRIDE" in plan["cleared_environment_names"]


@pytest.mark.parametrize("extra", [["--gpu", "0,1"], ["--numa-node", "-1"]])
def test_invalid_device_selection(launcher, tmp_path, extra):
    args = launcher.parser().parse_args(arguments(tmp_path) + extra)
    with pytest.raises(ValueError):
        launcher.build_plan(args, {})


def test_no_nested_torchrun(launcher, tmp_path):
    args = launcher.parser().parse_args(arguments(tmp_path))
    with pytest.raises(ValueError, match="torchrun"):
        launcher.build_plan(args, {"RANK": "0"})


def test_explicit_numa_binding(launcher, tmp_path):
    args = launcher.parser().parse_args(arguments(tmp_path) + ["--numa-node", "1"])
    plan, _ = launcher.build_plan(args, {})
    assert plan["command"][:3] == ["numactl", "--cpunodebind=1", "--membind=1"]


@pytest.mark.parametrize("reason", ["existing_output", "missing_assets"])
def test_fail_before_gpu_probe(launcher, tmp_path, monkeypatch, reason):
    def forbidden(*args, **kwargs):
        pytest.fail("Must not launch a process for invalid assets/output")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    if reason == "existing_output":
        (tmp_path / "new-run").mkdir()
    with pytest.raises(SystemExit) as error:
        launcher.main(arguments(tmp_path))
    assert error.value.code == 2


def test_mock_execution_records_failure_without_claiming_correctness(launcher, tmp_path, monkeypatch):
    for name in ("weights.pt", "corpus.json"):
        (tmp_path / name).touch()
    (tmp_path / "stats").mkdir()
    for name in launcher.STATS:
        (tmp_path / "stats" / name).touch()
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout='{"device": "mock"}', returncode=7)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(arguments(tmp_path)) == 7
    receipt = json.loads((tmp_path / "new-run/baseline-exit.json").read_text())
    assert receipt == {"returncode": 7, "source_unchanged": True, "diagnostic_status": "not_run"}
    assert len(calls) == 2


def test_source_guard_rejects_modified_files(launcher, tmp_path, monkeypatch):
    (tmp_path / "source.py").write_text("changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"ai4s/graphcast/source.py": "0" * 64}))
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Frozen source differs"):
        launcher.source_hashes({"source_manifest": "manifest.json"})


def test_every_fixed_training_option_matches_published_command(launcher):
    doc = (ROOT / "docs/training-native-baseline-20261012.md").read_text()
    command = doc.split("python scripts/training/run_training.py", 1)[1].split("```", 1)[0]
    argv = shlex.split(command.replace("\\\n", " "))
    published = dict(zip(argv[::2], argv[1::2]))
    recipe = json.loads(launcher.RECIPE.read_text())
    for key, value in recipe["train"]["arguments"].items():
        assert published["--" + key] == str(value)


def test_inference_rejects_nonpositive_steps(launcher, tmp_path):
    args = launcher.parser().parse_args(arguments(tmp_path, "infer") + ["--steps", "0"])
    with pytest.raises(ValueError, match="positive"):
        launcher.build_plan(args, {})


@pytest.mark.parametrize("device,hip,count,passes", [
    ("NVIDIA H100 80GB HBM3", None, 1, True),
    ("NVIDIA H100 PCIe", None, 1, False),
    ("NVIDIA H100 80GB HBM3", None, 2, False),
    ("NVIDIA H100 80GB HBM3", "6.0", 1, False),
])
def test_gpu_probe_has_no_assert_only_safety_gates(launcher, monkeypatch, device, hip, count, passes):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: count,
                           get_device_properties=lambda index: SimpleNamespace(name=device, total_memory=80 * 2**30))
    fake = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip=hip, cuda="test"), __version__="test")
    monkeypatch.setitem(sys.modules, "torch", fake)
    code = compile(launcher.GPU_PROBE, "<mock-gpu-probe>", "exec", optimize=2)
    if passes:
        exec(code, {})
    else:
        with pytest.raises(RuntimeError):
            exec(code, {})
