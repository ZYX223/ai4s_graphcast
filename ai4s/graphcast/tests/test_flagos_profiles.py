from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
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
INFERENCE_OPS = (
    "addmm",
    "cat",
    "native_layer_norm",
    "index_add_",
    "silu",
    "index",
    "add",
)
PTG_INFERENCE_OPS = tuple(op for op in INFERENCE_OPS if op != "addmm")


@pytest.fixture
def backend_module(monkeypatch: pytest.MonkeyPatch):
    # Runtime helpers write environment keys directly; isolate those side effects
    # as well as the keys explicitly changed through monkeypatch in each test.
    monkeypatch.setattr(os, "environ", os.environ.copy())
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


def test_runtime_backend_rejects_expected_profile_before_side_effects(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("GRAPHCAST_BACKEND", "metax")
    monkeypatch.setattr(
        backend_module,
        "_configure_flagos",
        lambda profile: events.append(f"configure:{profile.name}"),
    )
    monkeypatch.setattr(
        backend_module,
        "_resolve_device",
        lambda profile, requested: events.append(f"device:{profile.name}:{requested}"),
    )

    with pytest.raises(RuntimeError, match="expected runtime backend 'nvidia'.*'metax'"):
        backend_module.resolve_runtime_backend("cuda:0", expected_profile="nvidia")

    assert events == []


@pytest.mark.parametrize("configured_backend", (None, "auto"))
def test_expected_runtime_backend_bypasses_ambient_backend_probing(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
    configured_backend: str | None,
) -> None:
    events: list[str] = []
    if configured_backend is None:
        monkeypatch.delenv("GRAPHCAST_BACKEND", raising=False)
    else:
        monkeypatch.setenv("GRAPHCAST_BACKEND", configured_backend)
    monkeypatch.setattr(
        backend_module,
        "_infer_backend_from_device",
        lambda requested: events.append(f"infer:{requested}") or "nvidia",
    )
    monkeypatch.setattr(
        backend_module,
        "_configure_flagos",
        lambda profile: events.append(f"configure:{profile.name}") or object(),
    )
    monkeypatch.setattr(
        backend_module,
        "_resolve_device",
        lambda profile, requested: events.append(f"device:{profile.name}:{requested}")
        or object(),
    )
    monkeypatch.setattr(backend_module, "_runtime_versions", lambda: {})

    runtime = backend_module.resolve_runtime_backend("auto", expected_profile="metax")

    assert runtime.profile.name == "metax"
    assert events == ["configure:metax", "device:metax:auto"]


def test_flaggems_operator_parser_deduplicates_and_has_no_hidden_default(
    backend_module,
) -> None:
    assert backend_module._parse_ops(None) == ()
    assert backend_module._parse_ops("addmm, layer_norm,addmm") == (
        "addmm",
        "layer_norm",
    )


@pytest.mark.parametrize(
    ("platform", "expected_ops"),
    (
        ("nvidia", INFERENCE_OPS),
        ("ascend", INFERENCE_OPS),
        ("hygon", INFERENCE_OPS),
        ("ptg", PTG_INFERENCE_OPS),
        ("mthreads", INFERENCE_OPS),
        ("metax", INFERENCE_OPS),
    ),
)
def test_inference_profile_uses_one_shared_operator_contract(
    backend_module,
    platform: str,
    expected_ops: tuple[str, ...],
) -> None:
    path = GRAPHCAST_ROOT / "configs/flaggems/graphcast_inference.yaml"

    assert backend_module._load_flaggems_ops(path, platform) == expected_ops
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert profile == {
        "version": 2,
        "profile": "graphcast_inference",
        "scope": "inference",
        "ops": list(INFERENCE_OPS),
        "exclude_ops_by_backend": {"ptg": ["addmm"]},
    }


def test_gather_mode_defaults_to_the_index_interface_and_metadata_reports_it(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPHCAST_GATHER_MODE", raising=False)
    monkeypatch.delenv("GRAPHCAST_GATHER_BACKEND", raising=False)
    profile = backend_module.BACKEND_PROFILES["nvidia"]
    flagos = backend_module.FlagOSState(
        mode="off",
        strict=False,
        requested_ops=(),
        status="disabled",
    )
    runtime = backend_module.RuntimeBackend(
        profile=profile,
        device="cpu",
        flagos=flagos,
        versions={},
    )

    assert backend_module.graphcast_gather_mode() == "advanced_index"
    assert runtime.to_metadata()["gather_backend"] == "advanced_index"

    monkeypatch.setenv("GRAPHCAST_GATHER_MODE", "index_select")
    assert backend_module.graphcast_gather_mode() == "index_select"
    assert runtime.to_metadata()["gather_backend"] == "index_select"


def test_dispatch_audit_requires_all_seven_core_operators(
    backend_module,
    tmp_path: Path,
) -> None:
    record = tmp_path / "flaggems-dispatch.log"
    record.write_text(
        "\n".join(
            (
                "[DEBUG] flag_gems.ops.addmm.addmm: GEMS ADDMM, shape",
                "[DEBUG] _ascend.ops.cat.cat: GEMS_ASCEND CAT",
                "[DEBUG] flag_gems.ops.native_layer_norm.native_layer_norm: GEMS NATIVE_LAYER_NORM",
                "[DEBUG] _hygon.ops.index_add.index_add_: GEMS_HYGON INDEX_ADD_",
                "[DEBUG] _hygon.ops.silu.silu: GEMS_HYGON SILU",
                "[DEBUG] flag_gems.runtime._ascend.ops.index.index: GEMS_ASCEND INDEX",
                "[DEBUG] flag_gems.ops.add.add: GEMS ADD",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    complete = backend_module.audit_flaggems_dispatch(record, INFERENCE_OPS)

    assert complete.status == "passed"
    assert complete.observed_ops == INFERENCE_OPS
    assert complete.missing_ops == ()
    assert complete.counts == dict.fromkeys(INFERENCE_OPS, 1)

    record.write_text(
        "[DEBUG] flag_gems.ops.cat.cat: GEMS CAT\n",
        encoding="utf-8",
    )
    incomplete = backend_module.audit_flaggems_dispatch(record, INFERENCE_OPS)

    assert incomplete.status == "failed_missing_ops"
    assert incomplete.observed_ops == ("cat",)
    assert incomplete.missing_ops == (
        "addmm",
        "native_layer_norm",
        "index_add_",
        "silu",
        "index",
        "add",
    )


def test_dispatch_audit_recognizes_common_index_add_log_spelling(
    backend_module,
    tmp_path: Path,
) -> None:
    record = tmp_path / "flaggems-dispatch.log"
    record.write_text(
        "[DEBUG] flag_gems.ops.index_add.index_add_: GEMS INDEX ADD_\n",
        encoding="utf-8",
    )

    audit = backend_module.audit_flaggems_dispatch(record, ("index_add_",))

    assert audit.status == "passed"
    assert audit.observed_ops == ("index_add_",)
    assert audit.missing_ops == ()
    assert audit.counts == {"index_add_": 1}


def test_dispatch_recording_binds_short_vendor_logger_names(
    backend_module,
) -> None:
    def addmm_impl():
        return None

    addmm_impl.__module__ = "_metax.ops.addmm"
    flaggems = types.SimpleNamespace(FULL_CONFIG_BY_FUNC={"addmm": (("addmm", addmm_impl),)})
    base_logger = logging.getLogger("flag_gems")
    vendor_logger = logging.getLogger("_metax.ops.addmm")
    handler = logging.NullHandler()
    base_logger.addHandler(handler)
    try:
        backend_module._bind_flaggems_record_loggers(flaggems, ("addmm",))

        assert handler in vendor_logger.handlers
        assert vendor_logger.level == logging.DEBUG
        assert vendor_logger.propagate is False
    finally:
        if handler in vendor_logger.handlers:
            vendor_logger.removeHandler(handler)
        base_logger.removeHandler(handler)


def test_formal_flaggems_run_fails_closed_when_a_registered_op_is_not_observed(
    backend_module,
    tmp_path: Path,
) -> None:
    record = tmp_path / "flaggems-dispatch.log"
    record.write_text(
        "[DEBUG] flag_gems.ops.cat.cat: GEMS CAT\n",
        encoding="utf-8",
    )
    profile = backend_module.BACKEND_PROFILES["metax"]
    runtime = backend_module.RuntimeBackend(
        profile=profile,
        device="cuda:0",
        flagos=backend_module.FlagOSState(
            mode="flaggems",
            strict=True,
            requested_ops=INFERENCE_OPS,
            status="enabled_only_enable",
            record_path=str(record),
            registered_ops=INFERENCE_OPS,
        ),
        versions={},
    )

    with pytest.raises(RuntimeError, match="not observed.*addmm.*index_add_"):
        backend_module.require_complete_flaggems_dispatch(runtime)


def test_formal_flaggems_run_fails_before_dispatch_when_registration_is_missing(
    backend_module,
    tmp_path: Path,
) -> None:
    record = tmp_path / "flaggems-dispatch.log"
    record.write_text("", encoding="utf-8")
    profile = backend_module.BACKEND_PROFILES["metax"]
    runtime = backend_module.RuntimeBackend(
        profile=profile,
        device="cuda:0",
        flagos=backend_module.FlagOSState(
            mode="flaggems",
            strict=True,
            requested_ops=INFERENCE_OPS,
            status="enabled_only_enable",
            record_path=str(record),
            registered_ops=("cat",),
        ),
        versions={},
    )

    with pytest.raises(RuntimeError, match="not registered.*addmm.*index_add_"):
        backend_module.require_complete_flaggems_dispatch(runtime)


def test_v2_profile_rejects_an_exclusion_not_in_shared_ops(
    backend_module,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "version: 2\nops: [add]\nexclude_ops_by_backend:\n  ptg: [addmm]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is not present in shared ops"):
        backend_module._load_flaggems_ops(path, "ptg")


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


def test_explicit_flaggems_source_root_beats_editable_import_redirect(
    backend_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_root = tmp_path / "expected" / "FlagGems"
    expected_package = expected_root / "src/flag_gems"
    expected_package.mkdir(parents=True)
    (expected_package / "__init__.py").write_text(
        "ORIGIN = 'expected'\n",
        encoding="utf-8",
    )
    shadow_package = tmp_path / "editable" / "flag_gems"
    shadow_package.mkdir(parents=True)
    shadow_init = shadow_package / "__init__.py"
    shadow_init.write_text("ORIGIN = 'editable'\n", encoding="utf-8")

    class EditableRedirect:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "flag_gems":
                return None
            return importlib.util.spec_from_file_location(fullname, shadow_init)

    monkeypatch.setattr(sys, "meta_path", [EditableRedirect(), *sys.meta_path])
    monkeypatch.setenv("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", str(expected_root))
    previous_module = sys.modules.pop("flag_gems", None)
    try:
        activated = backend_module._activate_flaggems_source_root()
        imported = importlib.import_module("flag_gems")

        assert activated == expected_root / "src"
        assert imported.ORIGIN == "expected"
        assert Path(imported.__file__).resolve().is_relative_to(activated)
    finally:
        sys.modules.pop("flag_gems", None)
        if previous_module is not None:
            sys.modules["flag_gems"] = previous_module


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


def test_quickstart_training_keeps_explicit_legacy_recipe(quickstart_module, monkeypatch, capsys):
    import shlex

    monkeypatch.setattr(quickstart_module, "detect_platform", lambda: "nvidia")
    monkeypatch.setattr(quickstart_module, "find_free_gpus", lambda *a, **kw: [0, 1, 2, 3])
    monkeypatch.setattr(quickstart_module, "ensure_data", lambda: True)
    monkeypatch.setattr(quickstart_module, "start_container", lambda *a, **kw: True)
    calls = []

    def capture_container_command(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(quickstart_module, "run", capture_container_command)
    monkeypatch.setattr(sys, "argv", [str(QUICKSTART_PATH), "train", "--steps", "2"])
    quickstart_module.main()
    assert len(calls) == 1
    command = shlex.split(calls[0][-1])
    assert "scripts/training/legacy/lib/train_mp.py" in command
    assert command[command.index("--optimizer") + 1] == "sgd"
    assert command[command.index("--steps") + 1] == "2"
    assert "legacy" in capsys.readouterr().out.lower()
