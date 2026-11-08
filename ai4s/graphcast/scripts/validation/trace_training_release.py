#!/usr/bin/env python3
"""Observe selected official-release logger roots without changing dispatch.

Extends the frozen phase/call observer. Architecture-specific exports can use
logger roots different from GEMS_VENDOR; discover them from the pinned exports,
not from a platform-name alias table. The H100 export ``hopper.ops.mm``, for
example, identifies logger root ``hopper``. These remain logged calls, not kernels.
"""
import importlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = load_script("_frozen_training_call_observer",
                     ROOT / "scripts/validation/trace_training_flaggems.py")


def selected_ops():
    if os.environ.get("GRAPHCAST_FLAGOS_MODE") != "flaggems":
        raise RuntimeError("release observer must not be enabled on the Native path")
    from graphcast_compat import backend
    name = backend._normalize_configured_backend_name(os.environ.get("GRAPHCAST_BACKEND"))
    if name is None:
        raise RuntimeError("release observer requires an explicit physical backend")
    # Empty OPS is the canonical YAML route, not a request for no operators.
    # Do not let the backend's inference default silently select seven ops.
    if (not backend._parse_ops(os.environ.get("GRAPHCAST_FLAGGEMS_OPS"))
            and not os.environ.get("GRAPHCAST_FLAGGEMS_CONFIG", "").strip()):
        os.environ["GRAPHCAST_FLAGGEMS_CONFIG"] = str(ROOT / "configs/flaggems/graphcast_training_full.yaml")
    return backend._resolve_flaggems_ops(backend.BACKEND_PROFILES[name])


def selected_module():
    source = os.environ.get("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", "").strip()
    if not source:
        raise RuntimeError("release observer requires an explicit official source root")
    root = Path(source).expanduser().resolve()
    launcher = load_script("_release_observer_runtime_lock",
                           ROOT / "scripts/validation/run_flaggems_training.py")
    launcher.verify_runtime(root)
    # Use the same narrow finder as the canonical training backend, rather
    # than trusting PYTHONPATH in an image with another editable installation.
    from graphcast_compat import backend
    python_root = backend._activate_flaggems_source_root()
    if python_root != root / "src":
        raise RuntimeError("release observer source finder selected a different source")
    module = importlib.import_module("flag_gems")
    location = getattr(module, "__file__", None)
    if not isinstance(location, str) or not location.strip():
        raise RuntimeError("release observer imported FlagGems without an auditable source")
    origin = Path(location).resolve()
    try:
        origin.relative_to(python_root / "flag_gems")
    except ValueError as error:
        raise RuntimeError("release observer imported FlagGems outside selected official source") from error
    if getattr(module, "__version__", None) != "5.4.0":
        raise RuntimeError("release observer requires official FlagGems v5.4.0 metadata")
    return module, python_root / "flag_gems"


class ReleaseCallObserver(frozen.CallObserver):
    def __enter__(self):
        if os.environ.get("GRAPHCAST_FLAGOS_MODE") != "flaggems":
            raise RuntimeError("release observer must not be enabled on the Native path")
        super().__enter__()
        try:
            module, package_root = selected_module()
            existing = {logger.name for logger, _, _ in self.logger_states}
            roots = set()
            for name in selected_ops():
                function = getattr(module, name, None)
                if not callable(function):
                    raise RuntimeError("selected FlagGems export is unavailable: " + name)
                function = inspect.unwrap(function)
                source = inspect.getsourcefile(function)
                if not source:
                    raise RuntimeError("selected FlagGems export has no auditable source: " + name)
                try:
                    Path(source).resolve().relative_to(package_root)
                except ValueError as error:
                    raise RuntimeError("selected FlagGems export is outside official source: " + name) from error
                root = getattr(function, "__module__", "").split(".", 1)[0]
                if not root:
                    raise RuntimeError("selected FlagGems export has no logger root: " + name)
                roots.add(root)
            for name in sorted(roots - existing):
                logger = logging.getLogger(name)
                self.logger_states.append((logger, logger.level, logger.propagate))
                logger.setLevel(logging.DEBUG)
                logger.propagate = False
                logger.addHandler(self)
        except BaseException:
            super().__exit__(*sys.exc_info())
            raise
        return self


def main(argv=None):
    original = frozen.CallObserver
    try:
        frozen.CallObserver = ReleaseCallObserver
        return frozen.main(argv)
    finally:
        frozen.CallObserver = original


if __name__ == "__main__":
    raise SystemExit(main())
