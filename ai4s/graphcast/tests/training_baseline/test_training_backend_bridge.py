import importlib
import os
from pathlib import Path
import sys
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
