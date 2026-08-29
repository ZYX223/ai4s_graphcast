from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = GRAPHCAST_ROOT / "src/graphcast_compat/backend.py"
QUICKSTART_PATH = GRAPHCAST_ROOT / "quickstart"


@pytest.fixture
def backend_module(monkeypatch: pytest.MonkeyPatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.__version__ = "test"
    torch_stub.version = types.SimpleNamespace(cuda=None, hip=None)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    module_name = "graphcast_backend_profile_test"
    spec = importlib.util.spec_from_file_location(module_name, BACKEND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def quickstart_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "graphcast_quickstart_profile_test"
    loader = importlib.machinery.SourceFileLoader(module_name, str(QUICKSTART_PATH))
    spec = importlib.util.spec_from_loader(module_name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    loader.exec_module(module)
    return module


def test_backend_profiles_represent_physical_platforms(backend_module) -> None:
    assert set(backend_module.BACKEND_PROFILES) == {
        "cpu",
        "nvidia",
        "ascend",
        "hygon",
        "ptg",
        "mthreads",
        "metax",
    }
    assert backend_module._normalize_backend_name("native_hygon", "auto") == "hygon"
    assert backend_module._normalize_backend_name("flagos_hygon", "auto") == "hygon"


def test_flaggems_operator_parser_deduplicates_and_has_no_hidden_default(
    backend_module,
) -> None:
    assert backend_module._parse_ops(None) == ()
    assert backend_module._parse_ops("addmm, layer_norm,addmm") == (
        "addmm",
        "layer_norm",
    )


@pytest.mark.parametrize(
    ("platform", "expected"),
    (
        (
            "ascend",
            (
                "addmm",
                "silu",
                "native_layer_norm",
                "cat",
                "index",
                "index_add_",
                "add",
            ),
        ),
        (
            "hygon",
            (
                "addmm",
                "silu",
                "native_layer_norm",
                "cat",
                "index",
                "index_add_",
                "add",
            ),
        ),
        (
            "ptg",
            (
                "addmm",
                "silu",
                "native_layer_norm",
                "cat",
                "index",
                "index_add_",
                "add",
            ),
        ),
        (
            "mthreads",
            (
                "addmm",
                "silu",
                "native_layer_norm",
                "cat",
                "index",
                "index_add_",
                "add",
            ),
        ),
        (
            "metax",
            (
                "addmm",
                "silu",
                "native_layer_norm",
                "cat",
                "index",
                "index_add_",
                "add",
            ),
        ),
        ("nvidia", ("silu", "add")),
    ),
)
def test_inference_profile_selects_platform_validated_ops(
    backend_module,
    platform: str,
    expected: tuple[str, ...],
) -> None:
    path = GRAPHCAST_ROOT / "configs/flaggems/graphcast_inference.yaml"

    assert backend_module._load_flaggems_ops(path, platform) == expected


def test_training_profile_is_independent_and_marks_forward_scope(backend_module) -> None:
    inference_path = GRAPHCAST_ROOT / "configs/flaggems/graphcast_inference.yaml"
    training_path = GRAPHCAST_ROOT / "configs/flaggems/graphcast_training.yaml"

    assert backend_module._load_flaggems_ops(training_path, "ascend") == (
        "addmm",
        "silu",
        "layer_norm",
        "cat",
        "index",
        "add",
    )
    assert yaml.safe_load(inference_path.read_text(encoding="utf-8"))["profile"] == (
        "graphcast_inference"
    )
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    assert training["profile"] == "graphcast_training"
    assert training["scope"] == "forward"


def test_explicit_ops_override_platform_profile(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_OPS", "index,addmm,index")
    monkeypatch.setenv(
        "GRAPHCAST_FLAGGEMS_CONFIG",
        str(GRAPHCAST_ROOT / "configs/flaggems/graphcast_inference.yaml"),
    )

    assert backend_module._resolve_flaggems_ops(backend_module.BACKEND_PROFILES["hygon"]) == (
        "index",
        "addmm",
    )


def test_hygon_cat_disables_buffer_ops_before_flag_gems_import(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS", raising=False)
    monkeypatch.delenv("GRAPHCAST_HYGON_CAT_BUFFER_OPS", raising=False)

    warnings = backend_module._configure_hygon_triton_knobs(
        backend_module.BACKEND_PROFILES["hygon"],
        ("cat",),
    )

    assert os.environ["AMDGCN_USE_BUFFER_OPS"] == "0"
    assert "TritonAMDGPUCanonicalizePointers" in warnings[0]


def test_cli_flagos_settings_use_canonical_environment_names(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "FlagGems"
    config = tmp_path / "training.yaml"
    for name in (
        "GRAPHCAST_FLAGOS_MODE",
        "GRAPHCAST_FLAGGEMS_OPS",
        "GRAPHCAST_FLAGGEMS_CONFIG",
        "GRAPHCAST_FLAGGEMS_SOURCE_ROOT",
        "GRAPHCAST_FLAGGEMS_RECORD_PATH",
        "GRAPHCAST_STRICT_FLAGOS",
        "GRAPHCAST_BISHENGIR_OPT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    backend_module.configure_flagos_environment(
        mode="none",
        ops="",
        config=str(config),
        source_root=str(source_root),
        record_path="records.log",
        strict="1",
        bishengir_opt_dir="/opt/bishengir/bin",
    )

    assert os.environ["GRAPHCAST_FLAGOS_MODE"] == "off"
    assert "GRAPHCAST_FLAGGEMS_OPS" not in os.environ
    assert os.environ["GRAPHCAST_FLAGGEMS_CONFIG"] == str(config)
    assert os.environ["GRAPHCAST_FLAGGEMS_SOURCE_ROOT"] == str(source_root)
    assert os.environ["GRAPHCAST_FLAGGEMS_RECORD_PATH"] == "records.log"
    assert os.environ["GRAPHCAST_STRICT_FLAGOS"] == "1"
    assert os.environ["GRAPHCAST_BISHENGIR_OPT_DIR"] == "/opt/bishengir/bin"


@pytest.mark.parametrize(
    ("profile", "config_name"),
    (
        ("inference", "graphcast_inference.yaml"),
        ("training", "graphcast_training.yaml"),
    ),
)
def test_quickstart_builds_profile_environment_for_both_paths(
    quickstart_module,
    profile: str,
    config_name: str,
) -> None:
    env = quickstart_module.build_flagos_environment(
        platform="ppu",
        mode="flaggems",
        ops=None,
        profile=profile,
    )

    assert env == {
        "GRAPHCAST_BACKEND": "ptg",
        "GRAPHCAST_FLAGOS_MODE": "flaggems",
        "GRAPHCAST_FLAGGEMS_CONFIG": f"configs/flaggems/{config_name}",
        "GRAPHCAST_FLAGGEMS_VARIANT": "installed",
        "GRAPHCAST_STRICT_FLAGOS": "1",
    }


def test_quickstart_explicit_ops_override_profile(quickstart_module) -> None:
    env = quickstart_module.build_flagos_environment(
        platform="ascend",
        mode="flaggems",
        ops="index,addmm",
        profile="inference",
    )

    assert env["GRAPHCAST_BACKEND"] == "ascend"
    assert env["GRAPHCAST_FLAGGEMS_OPS"] == "index,addmm"
    assert env["GRAPHCAST_STRICT_FLAGOS"] == "1"
    assert "GRAPHCAST_FLAGGEMS_CONFIG" not in env


def test_quickstart_can_select_prepared_internal_runtime(quickstart_module) -> None:
    env = quickstart_module.build_flagos_environment(
        platform="hygon",
        mode="flaggems",
        ops=None,
        profile="inference",
        variant="internal",
    )

    assert env["GRAPHCAST_FLAGGEMS_VARIANT"] == "internal"


def test_backend_resolves_prepared_variant_from_runtime_root(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_VARIANT", "official")
    monkeypatch.setenv("FLAGGEMS_RUNTIME_DEST", str(tmp_path / "runtime"))
    monkeypatch.delenv("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", raising=False)

    assert backend_module._resolve_flaggems_variant("flaggems") == "official"
    assert os.environ["GRAPHCAST_FLAGGEMS_SOURCE_ROOT"] == str(
        tmp_path / "runtime/official/FlagGems"
    )


def test_flaggems_never_falls_back_to_enabling_all_operators(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", raising=False)
    flag_gems_stub = types.ModuleType("flag_gems")
    flag_gems_stub.use_gems = lambda: None
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems_stub)

    state = backend_module._enable_flaggems(
        backend_module.BACKEND_PROFILES["nvidia"],
        ("silu",),
        False,
        "installed",
    )

    assert state.status == "failed"
    assert "only_enable" in state.error


def test_inference_shell_runs_from_graphcast_root(tmp_path: Path) -> None:
    capture_path = tmp_path / "invocation.txt"
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$PWD" > "$CAPTURE_PATH"\n'
        'printf "%s\\n" "$GRAPHCAST_FLAGGEMS_CONFIG" >> "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "PYTHON_BIN": str(python_stub),
            "GRAPHCAST_FLAGOS_MODE": "off",
            "GRAPHCAST_FLAGGEMS_CONFIG": "",
            "GRAPHCAST_FLAGGEMS_OPS": "",
            "GRAPHCAST_FLAGGEMS_SOURCE_ROOT": "",
            "GRAPHCAST_FLAGGEMS_VARIANT": "installed",
            "FLAGGEMS_RUNTIME_DEST": "",
        }
    )

    subprocess.run(
        ["bash", str(GRAPHCAST_ROOT / "scripts/inference/run_25km_inference.sh")],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    cwd, config_path = capture_path.read_text(encoding="utf-8").splitlines()
    assert Path(cwd) == GRAPHCAST_ROOT
    assert Path(config_path) == (GRAPHCAST_ROOT / "configs/flaggems/graphcast_inference.yaml")


def test_quickstart_propagates_container_command_failure(
    quickstart_module,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(quickstart_module, "detect_platform", lambda: "nvidia")
    monkeypatch.setattr(
        quickstart_module,
        "find_free_gpus",
        lambda platform, count=4, user_spec=None: [0],
    )
    monkeypatch.setattr(quickstart_module, "ensure_data", lambda: True)
    monkeypatch.setattr(
        quickstart_module,
        "start_container",
        lambda platform, device_ids, image=None: True,
    )
    monkeypatch.setattr(
        quickstart_module,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=7, stderr=""),
    )
    monkeypatch.setattr(sys, "argv", [str(QUICKSTART_PATH), "infer", "--devices", "0"])

    with pytest.raises(SystemExit) as exc_info:
        quickstart_module.main()

    assert exc_info.value.code == 1
    assert "Inference failed." in capsys.readouterr().out
