"""Backend and FlagOS runtime helpers for GraphCast inference.

The model code should stay shared across chips. This module owns the runtime
differences: device import/checks, synchronization, memory stats, and optional
FlagOS operator layers.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata as metadata
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

DEFAULT_FLAGGEMS_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs/flaggems/graphcast_inference.yaml"
)
DEFAULT_FLAGGEMS_RUNTIME_DEST = Path(__file__).resolve().parents[2] / "third_party/flaggems-runtime"


class _ExplicitFlagGemsSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve only ``flag_gems`` from an explicitly selected source tree.

    Vendor images commonly contain a PEP 660 editable finder ahead of Python's
    normal path finder.  Such a finder can redirect ``import flag_gems`` to the
    image checkout even when the requested source tree is first on
    ``PYTHONPATH``.  This narrow finder affects only the ``flag_gems`` package
    and its children; all unrelated imports keep their original resolution.
    """

    def __init__(self, python_root: Path) -> None:
        self.python_root = python_root

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        if fullname == "flag_gems":
            return importlib.machinery.PathFinder.find_spec(
                fullname,
                [str(self.python_root)],
                target,
            )
        if fullname.startswith("flag_gems."):
            return importlib.machinery.PathFinder.find_spec(fullname, path, target)
        return None


@dataclass(frozen=True)
class BackendProfile:
    name: str
    device_type: str
    default_device: str
    runtime_family: str
    chip: str
    supports_flaggems: bool = False
    supports_flagtree: bool = False
    flaggems_vendor: str | None = None
    container_image: str | None = None
    status: str = "active"
    notes: str = ""


@dataclass(frozen=True)
class FlagOSState:
    mode: str
    strict: bool
    requested_ops: tuple[str, ...]
    status: str
    variant: str = "off"
    vendor: str | None = None
    record_path: str | None = None
    bishengir_opt_dir: str | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    patched_ops: tuple[str, ...] = ()
    registered_ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlagGemsDispatchAudit:
    status: str
    requested_ops: tuple[str, ...]
    required_observed_ops: tuple[str, ...]
    observed_ops: tuple[str, ...]
    missing_ops: tuple[str, ...]
    counts: dict[str, int]
    record_path: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "requested_ops": list(self.requested_ops),
            "required_observed_ops": list(self.required_observed_ops),
            "observed_ops": list(self.observed_ops),
            "missing_ops": list(self.missing_ops),
            "configured_coverage": f"{len(self.requested_ops)}/{len(self.requested_ops)}",
            "active_coverage": (
                f"{sum(self.counts[op] > 0 for op in self.required_observed_ops)}"
                f"/{len(self.required_observed_ops)}"
            ),
            "counts": dict(self.counts),
            "record_path": self.record_path,
        }


@dataclass(frozen=True)
class RuntimeBackend:
    profile: BackendProfile
    device: torch.device
    flagos: FlagOSState
    versions: dict[str, str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.profile.name,
            "device": str(self.device),
            "device_type": self.profile.device_type,
            "runtime_family": self.profile.runtime_family,
            "chip": self.profile.chip,
            "status": self.profile.status,
            "flagos_mode": self.flagos.mode,
            "flagos_status": self.flagos.status,
            "flagos_strict": self.flagos.strict,
            "flaggems_variant": self.flagos.variant,
            "flaggems_ops": list(self.flagos.requested_ops),
            "flaggems_ops_config": os.environ.get("GRAPHCAST_FLAGGEMS_CONFIG"),
            "flaggems_vendor": self.flagos.vendor,
            "flaggems_record_path": self.flagos.record_path,
            "flaggems_source_root": os.environ.get("GRAPHCAST_FLAGGEMS_SOURCE_ROOT"),
            "flaggems_module_origin": _module_origin("flag_gems"),
            "bishengir_opt_dir": self.flagos.bishengir_opt_dir,
            "hygon_amdgcn_use_buffer_ops": os.environ.get("AMDGCN_USE_BUFFER_OPS"),
            "gather_backend": graphcast_gather_mode(),
            "flagos_warnings": list(self.flagos.warnings),
            "flagos_error": self.flagos.error,
            "flaggems_patched_ops": list(self.flagos.patched_ops),
            "flaggems_registered_ops": list(self.flagos.registered_ops),
            "versions": dict(self.versions),
        }


def _dispatch_op_from_record_line(line: str) -> str | None:
    message = line.split(": ", 1)[-1].strip().upper()
    if "NATIVE_LAYER_NORM" in message:
        return "native_layer_norm"
    if "INDEX_ADD" in message or "INDEX ADD" in message:
        return "index_add_"
    if "ADDMM" in message:
        return "addmm"
    if message == "GEMS CAT" or message.endswith(" CAT"):
        return "cat"
    if "SILU FORWARD" in message or message.endswith(" SILU"):
        return "silu"
    if message.endswith(" INDEX"):
        return "index"
    if message == "GEMS ADD":
        return "add"
    return None


def audit_flaggems_dispatch(
    record_path: Path,
    requested_ops: tuple[str, ...],
) -> FlagGemsDispatchAudit:
    requested = tuple(dict.fromkeys(requested_ops))
    counts = dict.fromkeys(requested, 0)
    if not record_path.is_file():
        return FlagGemsDispatchAudit(
            status="failed_missing_record",
            requested_ops=requested,
            required_observed_ops=requested,
            observed_ops=(),
            missing_ops=requested,
            counts=counts,
            record_path=str(record_path),
        )

    for line in record_path.read_text(encoding="utf-8", errors="replace").splitlines():
        op = _dispatch_op_from_record_line(line)
        if op in counts:
            counts[op] += 1

    observed = tuple(op for op in requested if counts[op] > 0)
    missing = tuple(op for op in requested if counts[op] == 0)
    status = "failed_missing_ops" if missing else "passed"
    return FlagGemsDispatchAudit(
        status=status,
        requested_ops=requested,
        required_observed_ops=requested,
        observed_ops=observed,
        missing_ops=missing,
        counts=counts,
        record_path=str(record_path),
    )


def require_complete_flaggems_dispatch(
    runtime: RuntimeBackend,
) -> FlagGemsDispatchAudit | None:
    if runtime.flagos.mode != "flaggems":
        return None

    requested = runtime.flagos.requested_ops
    registered = {name.split(".", 1)[0] for name in runtime.flagos.registered_ops}
    missing_registered = tuple(op for op in requested if op not in registered)
    if missing_registered:
        raise RuntimeError("FlagGems operators not registered: " + ", ".join(missing_registered))

    if not runtime.flagos.record_path:
        raise RuntimeError("FlagGems dispatch record path is required for formal runs")
    audit = audit_flaggems_dispatch(Path(runtime.flagos.record_path), requested)
    if not audit.status.startswith("passed"):
        raise RuntimeError(
            "FlagGems operators not observed in the model run: " + ", ".join(audit.missing_ops)
        )
    return audit


BACKEND_PROFILES: dict[str, BackendProfile] = {
    "cpu": BackendProfile(
        name="cpu",
        device_type="cpu",
        default_device="cpu",
        runtime_family="pytorch",
        chip="host",
        notes="CPU smoke path only; not suitable for full 25km inference.",
    ),
    "nvidia": BackendProfile(
        name="nvidia",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_cuda",
        chip="nvidia",
        supports_flaggems=True,
        flaggems_vendor="nvidia",
        notes="NVIDIA platform through CUDA.",
    ),
    "hygon": BackendProfile(
        name="hygon",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_hip",
        chip="hygon",
        supports_flaggems=True,
        flaggems_vendor="hygon",
        notes=("Hygon/BW platform through the torch.cuda-compatible HIP/DAS/DTK runtime."),
    ),
    "ptg": BackendProfile(
        name="ptg",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_ppu",
        chip="pingtouge_810e",
        supports_flaggems=True,
        flaggems_vendor="thead",
        notes="PTG 810E platform through its torch.cuda-compatible PPU runtime.",
    ),
    "mthreads": BackendProfile(
        name="mthreads",
        device_type="musa",
        default_device="musa",
        runtime_family="pytorch_musa",
        chip="mthreads_s5000",
        supports_flaggems=True,
        flaggems_vendor="mthreads",
        notes="Moore Threads S5000 platform through torch_musa.",
    ),
    "metax": BackendProfile(
        name="metax",
        device_type="cuda",
        default_device="cuda",
        runtime_family="pytorch_metax",
        chip="metax_c550",
        supports_flaggems=True,
        flaggems_vendor="metax",
        notes="MetaX C550 platform through its torch.cuda-compatible runtime.",
    ),
    "ascend": BackendProfile(
        name="ascend",
        device_type="npu",
        default_device="npu",
        runtime_family="pytorch_npu",
        chip="huawei_ascend",
        supports_flaggems=True,
        flaggems_vendor="ascend",
        notes="Ascend 910C platform through torch_npu.",
    ),
}

# GRAPHCAST_FLAGOS_MODE selects the runtime layer. Legacy backend names remain
# aliases so existing launch commands resolve to the same physical platform.
BACKEND_ALIASES = {
    "auto": "auto",
    "cuda": "nvidia",
    "native_cuda": "nvidia",
    "nvidia_flagos": "nvidia",
    "flagos_cuda": "nvidia",
    "flagos_nvidia": "nvidia",
    "bw1000": "hygon",
    "hip": "hygon",
    "native_hygon": "hygon",
    "hygon_native": "hygon",
    "hygon_flagos": "hygon",
    "flagos_hygon": "hygon",
    "810e": "ptg",
    "pingtouge": "ptg",
    "native_ptg": "ptg",
    "ptg_native": "ptg",
    "ptg_flagos": "ptg",
    "flagos_ptg": "ptg",
    "flagos_pingtouge": "ptg",
    "musa": "mthreads",
    "s5000": "mthreads",
    "native_mthreads": "mthreads",
    "mthreads_native": "mthreads",
    "mthreads_flagos": "mthreads",
    "flagos_mthreads": "mthreads",
    "c550": "metax",
    "native_metax": "metax",
    "metax_native": "metax",
    "metax_flagos": "metax",
    "flagos_metax": "metax",
    "npu": "ascend",
    "native_npu": "ascend",
    "ascend_native": "ascend",
    "flagos_ascend": "ascend",
    "flagos": "ascend",
    "cpu": "cpu",
}


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _cuda_backend_profile() -> str:
    """Identify the platform behind a torch.cuda-compatible runtime."""

    torch_version = str(getattr(torch, "__version__", "")).lower()
    if "metax" in torch_version:
        return "metax"

    hip_version = getattr(torch.version, "hip", None)
    if not hip_version:
        return "nvidia"

    try:
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(device_id).strip().lower()
                if device_name.startswith("bw") or "hygon" in device_name:
                    return "hygon"
    except Exception:
        pass

    if _truthy(os.environ.get("GRAPHCAST_ASSUME_HYGON")):
        return "hygon"
    return "nvidia"


def _version(package_names: tuple[str, ...]) -> str:
    for package_name in package_names:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return "not-installed"


def _module_origin(module_name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return str(Path(spec.origin).resolve())


def _import_torch_npu() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("NPU requested, but torch_npu is not importable") from exc


def _import_torch_musa() -> None:
    try:
        import torch_musa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("MUSA requested, but torch_musa is not importable") from exc


def _npu_is_available() -> bool:
    try:
        _import_torch_npu()
    except RuntimeError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def _musa_is_available() -> bool:
    try:
        _import_torch_musa()
    except RuntimeError:
        return False
    return bool(hasattr(torch, "musa") and torch.musa.is_available())


def configure_float32_math(
    tensor_core_enabled: bool,
    device_type: str,
) -> str:
    """Keep FP32 storage while selecting the runtime's fast-FP32 mode."""

    enabled = bool(tensor_core_enabled)
    if device_type == "npu":
        _import_torch_npu()
        npu = getattr(torch, "npu", None)
        matmul = getattr(npu, "matmul", None)
        convolution = getattr(npu, "conv", None)
        if matmul is None or not hasattr(matmul, "allow_hf32"):
            raise RuntimeError("torch_npu does not expose torch.npu.matmul.allow_hf32")
        if convolution is None or not hasattr(convolution, "allow_hf32"):
            raise RuntimeError("torch_npu does not expose torch.npu.conv.allow_hf32")
        matmul.allow_hf32 = enabled
        convolution.allow_hf32 = enabled
        if bool(matmul.allow_hf32) != enabled or bool(convolution.allow_hf32) != enabled:
            raise RuntimeError("torch_npu did not apply the requested HF32 mode")
        return "hf32" if enabled else "ieee_fp32"

    if device_type == "musa":
        _import_torch_musa()
        mudnn = getattr(torch.backends, "mudnn", None)
        if mudnn is None or not hasattr(mudnn, "allow_tf32"):
            raise RuntimeError("torch_musa does not expose torch.backends.mudnn.allow_tf32")
        mudnn.allow_tf32 = enabled
        if bool(mudnn.allow_tf32) != enabled:
            raise RuntimeError("torch_musa did not apply the requested TF32 mode")
        return "tf32" if enabled else "ieee_fp32"

    if device_type != "cuda":
        if enabled:
            raise RuntimeError(f"fast FP32 math is unsupported for {device_type!r}")
        return "ieee_fp32"

    # Hygon DTK uses a CUDA-compatible Python surface over HIP. Its pinned
    # runtime requires the legacy switches even when newer names are present.
    is_hip = bool(getattr(torch.version, "hip", None))
    if is_hip:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        if (
            bool(torch.backends.cuda.matmul.allow_tf32) != enabled
            or bool(torch.backends.cudnn.allow_tf32) != enabled
        ):
            raise RuntimeError("HIP did not apply the requested TF32 mode")
        return "tf32" if enabled else "ieee_fp32"

    precision = "tf32" if enabled else "ieee"
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    if (
        hasattr(torch.backends.cuda.matmul, "fp32_precision")
        and cudnn_conv is not None
        and hasattr(cudnn_conv, "fp32_precision")
    ):
        torch.backends.cuda.matmul.fp32_precision = precision
        cudnn_conv.fp32_precision = precision
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    return "tf32" if enabled else "ieee_fp32"


def _infer_backend_from_device(requested_device: str) -> str:
    if requested_device.startswith("musa"):
        return "mthreads"
    if requested_device.startswith("cuda"):
        return _cuda_backend_profile()
    if requested_device.startswith("npu"):
        return "ascend"
    if requested_device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return _cuda_backend_profile()
    if _musa_is_available():
        return "mthreads"
    if _npu_is_available():
        return "ascend"
    return "cpu"


def _normalize_configured_backend_name(value: str | None) -> str | None:
    raw_name = (value or "").strip().lower()
    if not raw_name or raw_name == "auto":
        return None
    name = BACKEND_ALIASES.get(raw_name, raw_name)
    if name not in BACKEND_PROFILES:
        allowed = ", ".join(sorted(BACKEND_PROFILES))
        raise ValueError(f"unknown GRAPHCAST_BACKEND={value!r}; allowed: {allowed}")
    return name


def _normalize_backend_name(value: str | None, requested_device: str) -> str:
    configured = _normalize_configured_backend_name(value)
    if configured is not None:
        return configured
    return _infer_backend_from_device(requested_device)


def _resolve_device(profile: BackendProfile, requested_device: str) -> torch.device:
    device_value = requested_device if requested_device != "auto" else profile.default_device
    if profile.device_type == "musa":
        _import_torch_musa()
    device = torch.device(device_value)
    if device.type != profile.device_type:
        raise RuntimeError(
            f"backend {profile.name!r} expects device type {profile.device_type!r}, "
            f"but --device resolved to {device!s}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    if device.type == "npu":
        _import_torch_npu()
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu.is_available() is false")
    if device.type == "musa" and (not hasattr(torch, "musa") or not torch.musa.is_available()):
        raise RuntimeError("MUSA requested, but torch.musa.is_available() is false")
    return device


def _parse_ops(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(dict.fromkeys(op.strip() for op in value.split(",") if op.strip()))


def _load_flaggems_ops(config_path: Path, platform: str) -> tuple[str, ...]:
    """Load the operator allowlist for one physical platform."""

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"FlagGems operator profile does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid FlagGems operator profile: {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"FlagGems operator profile must be a mapping: {config_path}")

    version = config.get("version")
    exclusions = None
    if version == 2:
        ops = config.get("ops")
        exclusions = config.get("exclude_ops_by_backend", {})
        if not isinstance(exclusions, dict):
            raise ValueError(f"FlagGems exclude_ops_by_backend must be a mapping: {config_path}")
    elif version == 1:
        # Training remains on the legacy per-platform schema until its separate
        # device validation is complete.
        platform_ops = config.get("platform_ops", {})
        if not isinstance(platform_ops, dict):
            raise ValueError(f"FlagGems platform_ops must be a mapping: {config_path}")
        ops = platform_ops.get(platform, config.get("default_ops"))
    else:
        raise ValueError(
            f"unsupported FlagGems operator profile version {version!r}: {config_path}"
        )
    if not isinstance(ops, list) or not ops:
        raise ValueError(
            f"FlagGems operator profile has no non-empty list for {platform}: {config_path}"
        )
    if any(not isinstance(op, str) or not op.strip() for op in ops):
        raise ValueError(
            f"FlagGems operator profile entries must be non-empty strings: {config_path}"
        )
    normalized = tuple(dict.fromkeys(op.strip() for op in ops))
    if exclusions is None:
        return normalized

    for backend_name, excluded_ops in exclusions.items():
        if not isinstance(backend_name, str) or not backend_name.strip():
            raise ValueError(
                f"FlagGems exclusion backend names must be non-empty strings: {config_path}"
            )
        if not isinstance(excluded_ops, list):
            raise ValueError(
                f"FlagGems exclusions for {backend_name} must be a list: {config_path}"
            )
        for excluded_op in excluded_ops:
            if not isinstance(excluded_op, str) or not excluded_op.strip():
                raise ValueError(f"FlagGems exclusions must be non-empty strings: {config_path}")
            if excluded_op.strip() not in normalized:
                raise ValueError(
                    f"FlagGems exclusion {excluded_op!r} for {backend_name} "
                    f"is not present in shared ops: {config_path}"
                )

    platform_exclusions = {op.strip() for op in exclusions.get(platform, ())}
    resolved = tuple(op for op in normalized if op not in platform_exclusions)
    if not resolved:
        raise ValueError(f"FlagGems exclusions remove every operator for {platform}: {config_path}")
    return resolved


def _resolve_flaggems_ops(profile: BackendProfile) -> tuple[str, ...]:
    explicit_ops = _parse_ops(os.environ.get("GRAPHCAST_FLAGGEMS_OPS"))
    if explicit_ops:
        return explicit_ops

    config_path = Path(
        os.environ.get("GRAPHCAST_FLAGGEMS_CONFIG", str(DEFAULT_FLAGGEMS_CONFIG))
    ).expanduser()
    os.environ["GRAPHCAST_FLAGGEMS_CONFIG"] = str(config_path)
    return _load_flaggems_ops(config_path, profile.name)


def _flagos_mode() -> str:
    mode = os.environ.get("GRAPHCAST_FLAGOS_MODE", "off").strip().lower()
    aliases = {"native": "off", "none": "off", "0": "off", "false": "off"}
    mode = aliases.get(mode, mode)
    if mode not in ("off", "flaggems", "flagtree"):
        raise ValueError(
            f"GRAPHCAST_FLAGOS_MODE must be one of off, flaggems, flagtree; got {mode!r}"
        )
    return mode


def _resolve_flaggems_variant(mode: str) -> str:
    if mode != "flaggems":
        os.environ["GRAPHCAST_FLAGGEMS_VARIANT_RESOLVED"] = "off"
        return "off"

    source_root = os.environ.get("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", "").strip()
    if source_root:
        variant = "custom"
    else:
        variant = os.environ.get("GRAPHCAST_FLAGGEMS_VARIANT", "installed").strip().lower()
        if variant not in ("installed", "official", "internal"):
            raise ValueError(
                "GRAPHCAST_FLAGGEMS_VARIANT must be installed, official, or internal; "
                f"got {variant!r}"
            )
        if variant in ("official", "internal"):
            runtime_root = Path(
                os.environ.get("FLAGGEMS_RUNTIME_DEST", str(DEFAULT_FLAGGEMS_RUNTIME_DEST))
            ).expanduser()
            os.environ["GRAPHCAST_FLAGGEMS_SOURCE_ROOT"] = str(runtime_root / variant / "FlagGems")
    os.environ["GRAPHCAST_FLAGGEMS_VARIANT_RESOLVED"] = variant
    return variant


def configure_flagos_environment(
    *,
    mode: str | None = None,
    ops: str | None = None,
    config: str | None = None,
    source_root: str | None = None,
    record_path: str | None = None,
    strict: str | bool | None = None,
    bishengir_opt_dir: str | None = None,
    variant: str | None = None,
    runtime_dest: str | None = None,
) -> None:
    """Apply CLI FlagOS options using the canonical runtime environment names."""

    if mode is not None:
        normalized_mode = mode.strip().lower()
        aliases = {"native": "off", "none": "off", "0": "off", "false": "off"}
        os.environ["GRAPHCAST_FLAGOS_MODE"] = aliases.get(normalized_mode, normalized_mode)

    values = {
        "GRAPHCAST_FLAGGEMS_CONFIG": config,
        "GRAPHCAST_FLAGGEMS_SOURCE_ROOT": source_root,
        "GRAPHCAST_FLAGGEMS_RECORD_PATH": record_path,
        "GRAPHCAST_BISHENGIR_OPT_DIR": bishengir_opt_dir,
        "GRAPHCAST_FLAGGEMS_VARIANT": variant,
        "FLAGGEMS_RUNTIME_DEST": runtime_dest,
    }
    for name, value in values.items():
        if value is not None and str(value).strip():
            os.environ[name] = str(value)

    if ops is not None:
        if ops.strip():
            os.environ["GRAPHCAST_FLAGGEMS_OPS"] = ops
        else:
            os.environ.pop("GRAPHCAST_FLAGGEMS_OPS", None)
    if strict is not None:
        os.environ["GRAPHCAST_STRICT_FLAGOS"] = str(strict)


def _default_record_path() -> str:
    return str(Path("outputs/logs") / f"flag_gems_record_{os.getpid()}.log")


def _configure_flagos_toolchain(strict: bool) -> tuple[str | None, tuple[str, ...]]:
    """Apply optional runtime toolchain path overrides for FlagGems/Triton."""

    warnings: list[str] = []
    opt_dir = os.environ.get("GRAPHCAST_BISHENGIR_OPT_DIR")
    if not opt_dir:
        return None, ()

    opt_path = Path(opt_dir) / "bishengir-opt"
    if not opt_path.exists():
        message = f"GRAPHCAST_BISHENGIR_OPT_DIR does not contain bishengir-opt: {opt_dir}"
        if strict:
            raise RuntimeError(message)
        warnings.append(message)
        return opt_dir, tuple(warnings)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if opt_dir not in path_entries:
        os.environ["PATH"] = opt_dir + os.pathsep + os.environ.get("PATH", "")
    return opt_dir, tuple(warnings)


def _activate_flaggems_source_root() -> Path | None:
    source_root = os.environ.get("GRAPHCAST_FLAGGEMS_SOURCE_ROOT", "").strip()
    if not source_root:
        return None

    root = Path(source_root).expanduser().resolve()
    python_root = root / "src"
    if not (python_root / "flag_gems/__init__.py").is_file():
        raise RuntimeError(f"invalid GRAPHCAST_FLAGGEMS_SOURCE_ROOT: {root}")
    python_root_text = str(python_root)
    sys.path[:] = [entry for entry in sys.path if entry != python_root_text]
    sys.path.insert(0, python_root_text)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, _ExplicitFlagGemsSourceFinder)
    ]
    sys.meta_path.insert(0, _ExplicitFlagGemsSourceFinder(python_root))
    importlib.invalidate_caches()
    return python_root


def _configure_hygon_triton_knobs(
    profile: BackendProfile,
    ops: tuple[str, ...],
) -> tuple[str, ...]:
    """Avoid the Hygon buffer-ops lowering failure for GraphCast cat."""

    if profile.name != "hygon" or "cat" not in ops:
        return ()
    if _truthy(os.environ.get("GRAPHCAST_HYGON_CAT_BUFFER_OPS")):
        return ()
    if "AMDGCN_USE_BUFFER_OPS" in os.environ:
        return (
            "Hygon FlagGems cat requested; respecting existing "
            f"AMDGCN_USE_BUFFER_OPS={os.environ['AMDGCN_USE_BUFFER_OPS']}",
        )

    os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"
    warnings = [
        "disabled AMDGCN_USE_BUFFER_OPS for Hygon FlagGems cat to avoid "
        "TritonAMDGPUCanonicalizePointers failure",
    ]
    if "triton" in sys.modules or "triton.knobs" in sys.modules:
        warnings.append(
            "AMDGCN_USE_BUFFER_OPS was set after Triton import; restart the Python "
            "process if Hygon cat still hits the old buffer-ops path"
        )
    return tuple(warnings)


def _enable_flaggems(
    profile: BackendProfile,
    ops: tuple[str, ...],
    strict: bool,
    variant: str,
) -> FlagOSState:
    warnings: list[str] = []
    if not profile.supports_flaggems:
        message = f"backend {profile.name} does not declare FlagGems support"
        if strict:
            raise RuntimeError(message)
        return FlagOSState(
            "flaggems",
            strict,
            ops,
            "unsupported",
            variant=variant,
            warnings=(message,),
        )

    bishengir_opt_dir, toolchain_warnings = _configure_flagos_toolchain(strict)
    warnings.extend(toolchain_warnings)
    warnings.extend(_configure_hygon_triton_knobs(profile, ops))

    vendor = profile.flaggems_vendor
    if vendor:
        os.environ["GEMS_VENDOR"] = vendor

    record_path = os.environ.get("GRAPHCAST_FLAGGEMS_RECORD_PATH")
    if not record_path and _truthy(os.environ.get("GRAPHCAST_FLAGGEMS_RECORD")):
        record_path = _default_record_path()

    try:
        expected_python_root = _activate_flaggems_source_root()
        import flag_gems

        if expected_python_root is not None:
            module_origin = Path(_module_origin("flag_gems") or "").resolve()
            try:
                module_origin.relative_to(expected_python_root)
            except ValueError as exc:
                raise RuntimeError(
                    "loaded flag_gems does not come from GRAPHCAST_FLAGGEMS_SOURCE_ROOT: "
                    f"{module_origin}"
                ) from exc

        patched_ops: list[str] = []
        if _using_ascend_vendor() and "index" in ops:
            # GraphCast gathers node features as src[:, idx, :] (dim-1 index).
            # Ascend FlagGems' index wrapper only handles dim-0 contiguous indices,
            # so we patch it to transpose dim-1 to dim-0 before the wrapper call
            # and restore the original layout afterwards.
            if _patch_ascend_index_for_graphcast(flag_gems):
                patched_ops.append("index.Tensor:graphcast_dim1_gather")

        record_kwargs: dict[str, Any] = {}
        if record_path:
            Path(record_path).parent.mkdir(parents=True, exist_ok=True)
            record_kwargs = {"record": True, "path": record_path}
        if not hasattr(flag_gems, "only_enable"):
            raise RuntimeError("the selected FlagGems runtime does not provide only_enable")
        flag_gems.only_enable(include=list(ops), **record_kwargs)
        status = "enabled_only_enable"

        _bind_flaggems_record_loggers(flag_gems, ops)
        registered_ops = _safe_list(flag_gems, "all_registered_ops")
        registered_keys = _safe_list(flag_gems, "all_registered_keys")
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        return FlagOSState(
            "flaggems",
            strict,
            ops,
            "failed",
            variant=variant,
            vendor=vendor,
            record_path=record_path,
            bishengir_opt_dir=bishengir_opt_dir,
            warnings=tuple(warnings),
            error=f"{type(exc).__name__}: {exc}",
        )

    return FlagOSState(
        "flaggems",
        strict,
        ops,
        status,
        variant=variant,
        vendor=vendor,
        record_path=record_path,
        bishengir_opt_dir=bishengir_opt_dir,
        warnings=tuple(warnings),
        patched_ops=tuple(patched_ops),
        registered_ops=tuple(registered_ops + registered_keys),
    )


def _configure_flagos(profile: BackendProfile) -> FlagOSState:
    mode = _flagos_mode()
    variant = _resolve_flaggems_variant(mode)
    strict = _truthy(os.environ.get("GRAPHCAST_STRICT_FLAGOS"))
    if mode == "off":
        ops = _parse_ops(os.environ.get("GRAPHCAST_FLAGGEMS_OPS"))
        return FlagOSState(mode, strict, ops, "disabled", variant=variant)
    if mode == "flaggems":
        ops = _resolve_flaggems_ops(profile)
        return _enable_flaggems(profile, ops, strict, variant)

    ops = _parse_ops(os.environ.get("GRAPHCAST_FLAGGEMS_OPS"))
    message = "FlagTree is planned but not wired into the 25km package yet"
    if strict:
        raise RuntimeError(message)
    return FlagOSState(
        mode,
        strict,
        ops,
        "not_implemented",
        variant=variant,
        warnings=(message,),
    )


def _runtime_versions() -> dict[str, str]:
    versions = {
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None) or "not-available",
        "torch_hip": getattr(torch.version, "hip", None) or "not-available",
        "torch_npu": _version(("torch_npu", "torch-npu")),
        "torch_musa": _version(("torch_musa", "torch-musa")),
        "torch_metax": _version(("torch_metax", "torch-metax")),
        "flag_gems": _version(("flag_gems", "FlagGems")),
        "flagtree": _version(("flagtree", "FlagTree")),
    }
    return versions


def resolve_runtime_backend(
    requested_device: str = "auto",
    *,
    expected_profile: str | None = None,
) -> RuntimeBackend:
    """Resolve device/backend from CLI device and GRAPHCAST_* environment."""

    configured_backend = _normalize_configured_backend_name(
        os.environ.get("GRAPHCAST_BACKEND")
    )
    if expected_profile is not None and expected_profile not in BACKEND_PROFILES:
        allowed = ", ".join(sorted(BACKEND_PROFILES))
        raise ValueError(
            f"unknown expected runtime backend {expected_profile!r}; allowed: {allowed}"
        )
    if expected_profile is not None and configured_backend is None:
        backend_name = expected_profile
    elif configured_backend is not None:
        backend_name = configured_backend
    else:
        backend_name = _infer_backend_from_device(requested_device)
    if expected_profile is not None and backend_name != expected_profile:
        raise RuntimeError(
            f"expected runtime backend {expected_profile!r}, found {backend_name!r}"
        )
    profile = BACKEND_PROFILES[backend_name]
    flagos = _configure_flagos(profile)
    device = _resolve_device(profile, requested_device)
    return RuntimeBackend(
        profile=profile,
        device=device,
        flagos=flagos,
        versions=_runtime_versions(),
    )


@contextmanager
def suspend_prevalidated_flaggems_index_add_bounds(
    runtime: RuntimeBackend,
):
    """Temporarily skip a bounds check already proven for GraphCast static indices."""

    if runtime.flagos.mode != "flaggems" or not runtime.flagos.strict:
        raise RuntimeError("prevalidated static index_add requires strict FlagGems mode")

    import flag_gems

    registrations = getattr(flag_gems, "FULL_CONFIG_BY_FUNC", {}).get("index_add_")
    if not isinstance(registrations, list) or len(registrations) != 1:
        raise RuntimeError("expected exactly one active FlagGems index_add_ registration")
    registration = registrations[0]
    if not isinstance(registration, tuple) or len(registration) != 2:
        raise RuntimeError("unexpected FlagGems index_add_ registration format")
    registered_name, registered_function = registration
    if registered_name != "index_add_" or not callable(registered_function):
        raise RuntimeError("unexpected FlagGems index_add_ registration")
    active_globals = getattr(registered_function, "__globals__", None)
    if not isinstance(active_globals, dict):
        raise RuntimeError("active FlagGems index_add_ has no mutable Python globals")
    original_check = active_globals.get("_assert_index_in_bounds")
    if not callable(original_check):
        raise RuntimeError("active FlagGems index_add_ bounds check was not found")

    def already_prevalidated(_index: torch.Tensor, _upper_bound: int) -> None:
        return None

    evidence = {
        "status": "active",
        "platform_profile": runtime.profile.name,
        "active_module": str(getattr(registered_function, "__module__", "unknown")),
        "registered_name": registered_name,
        "original_check": str(getattr(original_check, "__name__", "unknown")),
        "restored": False,
    }
    active_globals["_assert_index_in_bounds"] = already_prevalidated
    try:
        yield evidence
    finally:
        replacement_unchanged = (
            active_globals.get("_assert_index_in_bounds") is already_prevalidated
        )
        active_globals["_assert_index_in_bounds"] = original_check
        evidence["restored"] = True
        if not replacement_unchanged:
            raise RuntimeError("FlagGems index_add_ bounds replacement changed during execution")


def synchronize(runtime: RuntimeBackend | torch.device) -> None:
    device = runtime.device if isinstance(runtime, RuntimeBackend) else runtime
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "musa" and hasattr(torch, "musa"):
        torch.musa.synchronize(device)


def reset_peak_memory(runtime: RuntimeBackend | torch.device) -> None:
    device = runtime.device if isinstance(runtime, RuntimeBackend) else runtime
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)
    elif device.type == "musa" and hasattr(torch, "musa"):
        torch.musa.reset_peak_memory_stats(device)


def memory_text(runtime: RuntimeBackend | torch.device) -> str:
    device = runtime.device if isinstance(runtime, RuntimeBackend) else runtime
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        return f"cuda allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    if device.type == "npu" and hasattr(torch, "npu"):
        allocated = torch.npu.max_memory_allocated(device) / 1024**3
        reserved = torch.npu.max_memory_reserved(device) / 1024**3
        return f"npu allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    if device.type == "musa" and hasattr(torch, "musa"):
        allocated = torch.musa.max_memory_allocated(device) / 1024**3
        reserved = torch.musa.max_memory_reserved(device) / 1024**3
        return f"musa allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
    return "n/a"


def backend_report_lines(runtime: RuntimeBackend) -> list[str]:
    metadata_dict = runtime.to_metadata()
    return [
        f"- backend: `{metadata_dict['backend']}`",
        f"- backend device type: `{metadata_dict['device_type']}`",
        f"- runtime family: `{metadata_dict['runtime_family']}`",
        f"- chip: `{metadata_dict['chip']}`",
        f"- FlagOS mode: `{metadata_dict['flagos_mode']}`",
        f"- FlagOS status: `{metadata_dict['flagos_status']}`",
        f"- FlagGems variant: `{metadata_dict['flaggems_variant']}`",
        f"- FlagGems ops: `{metadata_dict['flaggems_ops']}`",
        f"- FlagGems ops config: `{metadata_dict['flaggems_ops_config']}`",
        f"- FlagGems vendor: `{metadata_dict['flaggems_vendor']}`",
        f"- FlagGems record path: `{metadata_dict['flaggems_record_path']}`",
        f"- FlagGems source root: `{metadata_dict['flaggems_source_root']}`",
        f"- FlagGems module origin: `{metadata_dict['flaggems_module_origin']}`",
        f"- BishengIR opt dir: `{metadata_dict['bishengir_opt_dir']}`",
        f"- gather backend: `{metadata_dict['gather_backend']}`",
        f"- FlagGems patched ops: `{metadata_dict['flaggems_patched_ops']}`",
        f"- FlagGems registered ops: `{metadata_dict['flaggems_registered_ops']}`",
        f"- runtime versions: `{metadata_dict['versions']}`",
        f"- FlagOS warnings: `{metadata_dict['flagos_warnings']}`",
        f"- FlagOS error: `{metadata_dict['flagos_error']}`",
    ]


def graphcast_gather_mode() -> str:
    return os.environ.get("GRAPHCAST_GATHER_MODE", "advanced_index").strip().lower()


def use_advanced_index_gather() -> bool:
    return graphcast_gather_mode() in {"advanced_index", "index", "flaggems_index"}


def _using_ascend_vendor() -> bool:
    return os.environ.get("GEMS_VENDOR", "").strip().lower() in {"ascend", "npu"}


def _bind_ascend_record_loggers() -> None:
    _bind_record_logger_names(
        (
            "_ascend.ops.index",
            "_ascend.ops.index_add",
            "flag_gems.ops.index_add",
            "flag_gems.runtime.backend._ascend.ops.index",
            "flag_gems.runtime.backend._ascend.ops.index_add",
            "flag_gems.runtime._ascend.ops.index",
            "flag_gems.runtime._ascend.ops.index_add",
        )
    )


def _bind_flaggems_record_loggers(flag_gems: Any, ops: tuple[str, ...]) -> None:
    """Route dynamically-loaded vendor loggers into FlagGems' record file.

    Some vendor modules are loaded under short names such as
    ``_metax.ops.addmm``.  Those loggers are not children of ``flag_gems`` and
    therefore do not inherit the file handler installed by
    ``only_enable(record=True)``.  Bind the exact implementation modules from
    FlagGems' registration table so a missing record means a missing call, not
    merely a logging namespace mismatch.
    """

    names: set[str] = set()
    by_function = getattr(flag_gems, "FULL_CONFIG_BY_FUNC", None)
    if isinstance(by_function, dict):
        for op in ops:
            for entry in by_function.get(op, ()):
                if not isinstance(entry, tuple | list) or len(entry) < 2:
                    continue
                implementation = entry[1]
                module_name = getattr(implementation, "__module__", "")
                if module_name:
                    names.add(str(module_name))
    _bind_record_logger_names(tuple(sorted(names)))
    _bind_ascend_record_loggers()


def _bind_record_logger_names(names: tuple[str, ...]) -> None:
    flag_logger = logging.getLogger("flag_gems")
    flag_handlers = list(flag_logger.handlers)
    for name in names:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if name.startswith("flag_gems."):
            logger.propagate = True
            continue
        logger.propagate = False
        for handler in flag_handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)


def _patch_ascend_index_for_graphcast(flag_gems: Any) -> bool:
    try:
        index_module = importlib.import_module("flag_gems.runtime.backend._ascend.ops.index")
    except Exception:
        try:
            index_module = importlib.import_module("flag_gems.runtime._ascend.ops.index")
        except Exception:
            return False

    index_wrapper = getattr(index_module, "index_wrapper", None)
    if index_wrapper is None:
        return False

    original_index = getattr(index_module, "index", None)
    patched_index = _build_graphcast_ascend_index(original_index, index_wrapper)
    index_module.index = patched_index
    return _replace_flaggems_config_function(flag_gems, "index.Tensor", patched_index)


def _build_graphcast_ascend_index(
    original_index: Callable[..., torch.Tensor] | None,
    index_wrapper: Callable[..., None],
) -> Callable[..., torch.Tensor]:
    def graphcast_index(inp: torch.Tensor, indices: tuple[Any, ...]) -> torch.Tensor:
        logging.getLogger("flag_gems.runtime._ascend.ops.index").debug("GEMS_ASCEND INDEX")
        if _is_graphcast_dim1_index(inp, indices):
            index_tensor = indices[1]
            if index_tensor.device != inp.device:
                index_tensor = index_tensor.to(inp.device)
            moved = inp.movedim(1, 0).contiguous()
            out_shape = (index_tensor.numel(), inp.shape[0], *inp.shape[2:])
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            if moved.numel() != 0 and out.numel() != 0:
                index_wrapper(moved, [index_tensor], out)
            return out.movedim(0, 1).contiguous()
        if original_index is None:
            return inp[indices]
        return original_index(inp, indices)

    graphcast_index.__name__ = "index"
    return graphcast_index


def _is_graphcast_dim1_index(inp: torch.Tensor, indices: tuple[Any, ...]) -> bool:
    if inp.ndim < 3 or len(indices) != inp.ndim:
        return False
    if not isinstance(indices[1], torch.Tensor):
        return False
    if indices[1].dtype not in {torch.int64, torch.int32, torch.long, torch.int}:
        return False
    for dim, index in enumerate(indices):
        if dim == 1:
            continue
        if index is None:
            continue
        if isinstance(index, slice) and index == slice(None):
            continue
        if index is not Ellipsis:
            return False
    return True


def _replace_flaggems_config_function(
    flag_gems: Any,
    op_name: str,
    replacement: Callable[..., torch.Tensor],
) -> bool:
    changed = False
    full_config = getattr(flag_gems, "_FULL_CONFIG", None)
    if isinstance(full_config, tuple):
        items = list(full_config)
        for index, item in enumerate(items):
            if item and item[0] == op_name:
                items[index] = (item[0], replacement, *item[2:])
                changed = True
        if changed:
            flag_gems._FULL_CONFIG = tuple(items)
    by_func = getattr(flag_gems, "FULL_CONFIG_BY_FUNC", None)
    if isinstance(by_func, dict):
        for key, values in list(by_func.items()):
            if not isinstance(values, list):
                continue
            new_values = []
            for item in values:
                if item and item[0] == op_name:
                    new_values.append((item[0], replacement, *item[2:]))
                    changed = True
                else:
                    new_values.append(item)
            by_func[key] = new_values
        by_func["index"] = [
            (item[0], replacement, *item[2:])
            for item in (getattr(flag_gems, "_FULL_CONFIG", ()) or ())
            if item and item[0] == op_name
        ]
    return changed


def _safe_list(module: Any, attr: str) -> list[str]:
    try:
        value = getattr(module, attr)()
    except Exception:
        return []
    return [str(item) for item in value]
