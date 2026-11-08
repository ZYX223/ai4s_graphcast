"""Explicit device-copy routing for the opt-in Ascend training integration.

Do not register FlagGems copy_ globally: a Native CPU/NPU transfer would then
redispatch into the replacement recursively. This mode leaves the Native
registration intact and calls FlagGems only for supported same-device copies.
It is not imported by the frozen Native baseline.
"""
from collections import Counter
import importlib
import inspect
import os
from pathlib import Path
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def copy_route(dst, src, device_type):
    if dst.device != src.device:
        return "native_io"
    if dst.device.type == device_type:
        return "flaggems_device"
    if dst.device.type == "cpu":
        return "native_cpu"
    raise RuntimeError(f"unexpected copy device: {dst.device}")


class DeviceCopyMode(TorchDispatchMode):
    def __init__(self, copy_impl, supports_copy, *, device_type="npu"):
        super().__init__()
        self.copy_impl = copy_impl
        self.supports_copy = supports_copy
        self.device_type = device_type
        self.counts = Counter()
        self.lock = threading.Lock()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        route = None
        if func == torch.ops.aten.copy_.default:
            dst, src = args[:2]
            route = copy_route(dst, src, self.device_type)
            if route == "flaggems_device":
                # Reject the selected implementation's Native fallback cases.
                if (not self.supports_copy(dst, src) or dst._is_zerotensor()
                        or src._is_zerotensor() or dst.is_neg() or src.is_neg()
                        or dst.is_conj() or src.is_conj()
                        or max(dst.numel(), src.numel()) > 2**31 - 1):
                    raise RuntimeError("unsupported device copy; Native compute fallback is disabled")
                alias = torch._C._is_alias_of(dst, src)
                exact = (alias and dst.storage_offset() == src.storage_offset()
                         and dst.stride() == src.stride() and dst.size() == src.size()
                         and dst.dtype == src.dtype and dst.is_neg() == src.is_neg()
                         and dst.is_conj() == src.is_conj())
                if alias and not exact:
                    raise RuntimeError("unsupported device copy with overlapping storage")
                if exact or dst.numel() == 0:
                    if torch.broadcast_shapes(dst.shape, src.shape) != dst.shape:
                        raise RuntimeError("copy source cannot broadcast to destination")
                    result, route = dst, "no_kernel"
                else:
                    non_blocking = args[2] if len(args) > 2 else kwargs.get("non_blocking", False)
                    result = self.copy_impl(dst, src, non_blocking=non_blocking)
            else:
                result = func(*args, **kwargs)
        else:
            # to(device) is a distinct ATen interface; record its I/O, never
            # claim its internal Native copy as FlagGems copy_ coverage.
            if func == torch.ops.aten._to_copy.default and kwargs.get("device") is not None:
                if args[0].device != torch.device(kwargs["device"]):
                    route = "native_io_to_copy"
            result = func(*args, **kwargs)
        if route is not None:
            with self.lock:
                self.counts[route] += 1
        return result

    def metadata(self):
        with self.lock:
            counts = dict(self.counts)
        return dict(route="torch_dispatch_mode_not_global_registration",
                    counts=counts, count_kind="successful_dispatch_calls_not_kernel_counts",
                    cross_device_counts_as_flaggems=False,
                    source=inspect.getsourcefile(self.copy_impl))


def from_environment():
    if (os.environ.get("GRAPHCAST_FLAGOS_MODE") != "flaggems"
            or os.environ.get("GRAPHCAST_BACKEND") != "ascend"
            or os.environ.get("GRAPHCAST_TRAINING_NATIVE_COPY_IO") != "1"):
        raise RuntimeError("copy router requires the explicit Ascend FlagGems training policy")
    requested = os.environ.get("GRAPHCAST_FLAGGEMS_OPS", "").split(",")
    if "copy_" in requested or not all(requested):
        raise RuntimeError("explicit global registrations must exclude copy_ when using the router")
    from .backend import _activate_flaggems_source_root
    root = _activate_flaggems_source_root()
    if root is None:
        raise RuntimeError("copy router requires a pinned FlagGems source")
    import flag_gems
    if not Path(flag_gems.__file__).resolve().is_relative_to(root):
        raise RuntimeError("unexpected FlagGems source for copy router")
    # The official runtime has no registrar before only_enable/use_gems.
    if (getattr(flag_gems, "current_work_registrar", None) is not None
            and "copy_" in flag_gems.all_registered_ops()):
        raise RuntimeError("copy_ is already globally overridden; start a fresh process")
    copy_impl = flag_gems.copy_
    if not Path(inspect.getsourcefile(copy_impl)).resolve().is_relative_to(root):
        raise RuntimeError("copy implementation is outside the pinned source")
    module = importlib.import_module(copy_impl.__module__)
    return DeviceCopyMode(copy_impl, module._can_use_triton)
