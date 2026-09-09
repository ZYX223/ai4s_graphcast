"""Training bridge to the target backend/YAML; no independent operator policy."""
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
from typing import Any

from . import backend

DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parents[2] / "configs/flaggems/graphcast_training.yaml"


@dataclass(frozen=True)
class ReplacementRuntime:
    enabled: bool
    operators: tuple[str, ...]
    skipped: tuple[str, ...]
    metadata: dict[str, Any]


def replacement_resume_policy(replacement: dict[str, Any]) -> dict[str, Any]:
    """Keep stable policy/version identity separate from process observations.

    Rank-local devices, PID-derived recording paths, registration/export
    observations and incidental warnings stay in the complete evidence,
    never in an exact-resume equality check.
    """
    runtime = replacement["runtime"]
    stable_fields = (
        "backend", "device_type", "runtime_family", "chip",
        "flagos_mode", "flagos_status", "flagos_strict", "flaggems_variant",
        "flaggems_ops", "flaggems_ops_config", "flaggems_vendor",
        "flaggems_source_root", "flaggems_module_origin", "bishengir_opt_dir",
        "hygon_amdgcn_use_buffer_ops", "gather_backend", "flaggems_patched_ops",
        "versions",
    )
    return {
        **{name: replacement[name] for name in ("mode", "enabled", "operators", "skipped")},
        "runtime": {name: runtime.get(name) for name in stable_fields},
    }


def enable_network_replacements_from_env(*, device):
    mode = backend._flagos_mode()
    if mode not in ("off", "flaggems"):
        raise ValueError("training supports native or flaggems only")
    os.environ["FLAGOS_REPLACE"] = "1" if mode == "flaggems" else "0"
    requested = ()
    if mode == "flaggems":
        backend.configure_flagos_environment(mode=mode, strict=True,
            config=os.environ.get("GRAPHCAST_FLAGGEMS_CONFIG") or str(DEFAULT_TRAINING_CONFIG))
        profile_name = backend._normalize_configured_backend_name(os.environ.get("GRAPHCAST_BACKEND")) or backend._infer_backend_from_device(str(device))
        requested = backend._resolve_flaggems_ops(backend.BACKEND_PROFILES[profile_name])
        # The target training profile's layer_norm names the native ATen op.
        # Resolve before only_enable, then check actual dispatcher registration.
        routed = tuple("native_layer_norm" if op == "layer_norm" else op for op in requested)
        backend.configure_flagos_environment(ops=",".join(routed))
    else:
        backend.configure_flagos_environment(mode="off", ops="", variant="installed")
    runtime = backend.resolve_runtime_backend(str(device))
    metadata = runtime.to_metadata()
    metadata["training_requested_ops"] = list(requested)
    metadata["forward_backward_coverage"] = "unverified"
    metadata["operator_sources"] = {}
    if mode == "flaggems":
        registered = {str(name).rsplit("::", 1)[-1].split(".", 1)[0] for name in runtime.flagos.registered_ops}
        missing = [op for op in routed if op not in registered]
        if missing:
            raise RuntimeError("FlagGems operators not registered: " + ", ".join(missing))
        module = sys.modules["flag_gems"]
        for original, name in zip(requested, routed):
            function = getattr(module, name, None)
            try:
                source = inspect.getsourcefile(function) if callable(function) else None
            except TypeError:
                source = None
            metadata["operator_sources"][original] = {
                "registered_name": name, "module": getattr(function, "__module__", None),
                "file": source, "evidence": "exported_function_source" if source else "unavailable",
            }
    return ReplacementRuntime(mode == "flaggems", requested, (), metadata)
