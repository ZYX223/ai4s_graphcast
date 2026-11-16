"""Platform instrumentation; math and optimizer remain in the frozen model/recipe."""

import ctypes as C
import os
import subprocess
import time

import jax

from graphcast_bench.platforms.training_precision import Precision as HygonPrecision


class Precision:
    def __init__(self, context):
        self.native = HygonPrecision(context.path('training_router')) if context.platform == "bw1100" else None

    def select(self, mode, label):
        if self.native:
            self.native.select(mode, label)
        else:
            assert mode == "tf32" and os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "tensorfloat32"

    def counts(self):
        return self.native.counts() if self.native else dict(calls=0, rounded=0)

    def delta(self, before):
        return {k: v - before[k] for k, v in self.counts().items()}


class Runtime:
    def __init__(self, context):
        self.platform = context.platform
        self.device = context.device
        self.bus = C.create_string_buffer(64)
        lib = C.CDLL("libamdhip64.so" if self.platform == "bw1100" else "libcudart.so")
        prefix = "hip" if self.platform == "bw1100" else "cuda"
        assert getattr(lib, prefix + "DeviceGetPCIBusId")(self.bus, 64, 0) == 0
        self.hipProfilerStart = getattr(lib, prefix + "ProfilerStart")
        self.hipProfilerStop = getattr(lib, prefix + "ProfilerStop")

    def sample_memory(self):
        if self.platform == "h100":
            value = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            # Container device enumeration may expose all devices to NVML; use UUID.
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    self.device['uuid'],
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            return dict(time_ns=time.time_ns(), used_mib=float(value))
        # Host observer provides device-wide Hygon memory. JAX allocator stats are also retained in report.
        return dict(
            time_ns=time.time_ns(),
            used_mib=jax.devices()[0].memory_stats().get("bytes_in_use", 0) / 2**20,
            source="jax_allocator_not_device_wide",
        )


def check_runtime(config):
    """Fail before installing hooks if the interpreter is not the pinned runtime."""
    import importlib.metadata as metadata

    from graphcast import graphcast

    from graphcast_bench.assets import ROOT

    for package, key in (("jax", "jax_version"), ("dm-haiku", "haiku_version")):
        if metadata.version(package) != config[key]:
            raise RuntimeError("Runtime version differs from runtime configuration: " + package)
    from pathlib import Path

    if not Path(graphcast.__file__).resolve().is_relative_to(ROOT / "third_party/graphcast"):
        raise RuntimeError("The model must come from this package's pinned official source")
