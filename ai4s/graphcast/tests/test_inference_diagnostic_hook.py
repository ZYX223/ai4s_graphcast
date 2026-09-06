"""Run the real campaign shell with an executable stand-in for the costly model."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("diagnostic_rc", [0, 1, 2])
def test_tf32_runner_calls_offline_diagnostic_and_propagates_status(tmp_path, diagnostic_rc):
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text('''#!/usr/bin/python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['TEST_CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if args[0].endswith('diagnose_inference.py'):
    raise SystemExit(int(os.environ['TEST_DIAGNOSTIC_RC']))
for flag in ('--output', '--report'):
    pathlib.Path(args[args.index(flag)+1]).write_text('{}')
''')
    fake.chmod(0o755)
    gems = tmp_path / "gems"
    (gems / "src/flag_gems").mkdir(parents=True)
    (gems / "src/flag_gems/__init__.py").touch()
    env = dict(os.environ, PROJECT_ROOT=str(ROOT), FLAGGEMS_ROOT=str(gems),
               ASSET_WEIGHTS="weights", ASSET_INPUTS="inputs", ASSET_FORCINGS="forcings",
               ASSET_TARGET="target", ASSET_STATS="stats", PLATFORM="c550",
               MODEL_EXECUTION_PROFILE="domestic-v1", BACKEND="metax", MODE="native",
               DEVICE="cpu", PRECISION="tf32", STEPS="2", RUN_NAME="test",
               OUTPUT_ROOT=str(tmp_path / "outputs"), PYTHON_BIN=str(fake),
               GRAPHCAST_ACCEPTANCE_BASELINE=str(tmp_path / "baseline.npz"),
               GRAPHCAST_ACCEPTANCE_METADATA=str(tmp_path / "baseline.json"),
               TEST_CALLS=str(calls), TEST_DIAGNOSTIC_RC=str(diagnostic_rc))
    for key in ("DIAGNOSTIC_TRAJECTORY", "GRAPHCAST_TRAJECTORY_REFERENCE",
                "GRAPHCAST_TRAJECTORY_DRIVER", "EXECUTABLE_CACHE_PARENT"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(ROOT / "tools/bf16_inference/run_bf16_case.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == diagnostic_rc, result.stderr
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(invocations) == 2
    assert invocations[0][invocations[0].index("--precision") + 1] == "tf32"
    assert invocations[1][0].endswith("diagnose_inference.py")
    run = tmp_path / "outputs/test"
    assert (run / "rc").read_text().strip() == "0"
    assert (run / "diagnostic_rc").read_text().strip() == str(diagnostic_rc)
