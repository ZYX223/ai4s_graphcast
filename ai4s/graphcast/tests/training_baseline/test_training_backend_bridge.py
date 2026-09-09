import importlib
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_runtime_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def bridge():
    assert (ROOT / "src/graphcast_compat/training_backend.py").is_file(), "training bridge missing"
    return importlib.import_module("graphcast_compat.training_backend")


def test_native_ignores_stale_flaggems_policy_without_import(monkeypatch):
    mod = bridge()
    monkeypatch.setenv("GRAPHCAST_FLAGOS_MODE", "off")
    monkeypatch.setenv("FLAGOS_REPLACE", "1")
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_OPS", "invalid")
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_CONFIG", "/missing.yaml")
    monkeypatch.setenv("GRAPHCAST_BACKEND", "cpu")
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    result = mod.enable_network_replacements_from_env(device=torch.device("cpu"))
    assert not result.enabled
    assert result.operators == ()
    assert result.metadata["flagos_status"] == "disabled"
    assert __import__("os").environ["FLAGOS_REPLACE"] == "0"


@pytest.mark.parametrize("missing", [False, True])
def test_target_yaml_alias_routes_to_actual_registration_and_fails_closed(monkeypatch, tmp_path, missing):
    mod = bridge()
    from graphcast_compat import backend
    config = tmp_path / "training.yaml"
    config.write_text("version: 1\ndefault_ops: [layer_norm, silu]\n")
    monkeypatch.setenv("GRAPHCAST_FLAGOS_MODE", "flaggems")
    monkeypatch.setenv("GRAPHCAST_BACKEND", "nvidia")
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_CONFIG", str(config))
    monkeypatch.delenv("GRAPHCAST_FLAGGEMS_OPS", raising=False)
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_VARIANT", "installed")
    for key in ("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", "GRAPHCAST_FLAGGEMS_RECORD_PATH", "GRAPHCAST_FLAGGEMS_RECORD", "GRAPHCAST_BISHENGIR_OPT_DIR"):
        monkeypatch.delenv(key, raising=False)
    registered = []
    def only_enable(*, include, **kwargs):
        # Installed runtime exposes only the native_layer_norm dispatcher name.
        registered.extend(name for name in include if name in (("silu",) if missing else ("silu", "native_layer_norm")))
    fake = types.ModuleType("flag_gems")
    fake.__file__ = __file__
    fake.only_enable = only_enable
    fake.all_registered_ops = lambda: registered
    fake.all_registered_keys = lambda: []
    fake.native_layer_norm = only_enable
    fake.silu = only_enable
    monkeypatch.setitem(sys.modules, "flag_gems", fake)
    monkeypatch.setattr(backend, "_resolve_device", lambda *args: torch.device("cpu"))
    if missing:
        with pytest.raises(RuntimeError, match="not registered"):
            mod.enable_network_replacements_from_env(device=torch.device("cpu"))
    else:
        result = mod.enable_network_replacements_from_env(device=torch.device("cpu"))
        assert result.enabled
        assert result.operators == ("layer_norm", "silu")
        assert result.metadata["flaggems_registered_ops"] == ["native_layer_norm", "silu"]
        assert result.metadata["operator_sources"]["layer_norm"]["registered_name"] == "native_layer_norm"
        assert result.metadata["operator_sources"]["silu"]["file"] == __file__


def test_fresh_process_recording_and_rank_devices_do_not_change_resume_policy(tmp_path):
    from graphcast_compat.training_state import validate_resume_contract
    program = '''
import json, os, sys, types
import torch
from graphcast_compat import backend
from graphcast_compat.training_backend import enable_network_replacements_from_env, replacement_resume_policy
fake = types.ModuleType("flag_gems")
fake.__file__ = "/fixture/flag_gems/__init__.py"
registered = []
def only_enable(*, include, **kwargs):
    registered[:] = include
fake.only_enable = only_enable
fake.all_registered_ops = lambda: registered
fake.all_registered_keys = lambda: []
sys.modules["flag_gems"] = fake
backend._resolve_device = lambda profile, requested: torch.device(requested)
runtime = enable_network_replacements_from_env(device=torch.device(sys.argv[1]))
replacement = dict(mode="flaggems", enabled=runtime.enabled, operators=list(runtime.operators), skipped=[], runtime=runtime.metadata)
print(json.dumps(dict(evidence=runtime.metadata, policy=replacement_resume_policy(replacement))))
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith("GRAPHCAST_")}
    env.update(PYTHONPATH=str(ROOT / "src"), GRAPHCAST_BACKEND="nvidia",
        GRAPHCAST_FLAGOS_MODE="flaggems", GRAPHCAST_FLAGGEMS_VARIANT="installed",
        GRAPHCAST_FLAGGEMS_OPS="silu", GRAPHCAST_FLAGGEMS_RECORD="1")
    def resolve(device):
        result = subprocess.run([sys.executable, "-c", program, device], cwd=tmp_path,
                                env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    rank0, rank1 = resolve("cuda:0"), resolve("cuda:2")
    assert rank0["evidence"]["device"] == "cuda:0"
    assert rank1["evidence"]["device"] == "cuda:2"
    assert rank0["evidence"]["flaggems_record_path"] != rank1["evidence"]["flaggems_record_path"]
    payload = {"metadata": {"resume_contract": {"operator_replacement": rank0["policy"]}}}
    validate_resume_contract(payload, {"operator_replacement": rank1["policy"]})
    from graphcast_compat.training_backend import replacement_resume_policy
    fresh_evidence = copy.deepcopy(rank1["evidence"])
    fresh_evidence.update(flaggems_registered_ops=["silu.Tensor", "silu"],
        operator_sources={"silu": {"file": "/tmp/process-local/export.py"}})
    observed_policy = replacement_resume_policy(dict(mode="flaggems", enabled=True,
        operators=["silu"], skipped=[], runtime=fresh_evidence))
    validate_resume_contract(payload, {"operator_replacement": observed_policy})
    # Keep failures closed for real backend, operator and version changes.
    for field, changed_value in (("backend", "hygon"), ("flaggems_ops", ["add"]),
                                 ("versions", {"torch": "different-runtime"})):
        changed = copy.deepcopy(rank1["evidence"])
        changed[field] = changed_value
        policy = replacement_resume_policy(dict(mode="flaggems", enabled=True,
            operators=["silu"], skipped=[], runtime=changed))
        with pytest.raises(ValueError, match="operator_replacement"):
            validate_resume_contract(payload, {"operator_replacement": policy})
