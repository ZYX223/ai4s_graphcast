"""Explicit JAX/native profiling boundaries; no generated Python source."""

import ctypes
import time
from contextlib import contextmanager

from graphcast_bench.assets import write_json


@contextmanager
def capture(output, name, steps, enabled, ranges):
    if not enabled:
        yield
        return
    import jax

    runtime = ctypes.CDLL("libamdhip64.so")
    jax.profiler.start_trace(str(output / "jax-trace" / name))
    started = False
    try:
        if runtime.hipProfilerStart() != 0:
            raise RuntimeError("HIP profiler start failed")
        started = True
        with jax.profiler.TraceAnnotation("graphcast-inference-" + name):
            begin_wall, begin_mono = time.time_ns(), time.monotonic_ns()
            try:
                yield
            finally:
                end_mono, end_wall = time.monotonic_ns(), time.time_ns()
                ranges.append(
                    dict(
                        label=name,
                        wall_start_ns=begin_wall,
                        wall_end_ns=end_wall,
                        monotonic_start_ns=begin_mono,
                        monotonic_end_ns=end_mono,
                        measured_steps=steps,
                    )
                )
    finally:
        try:
            if started and runtime.hipProfilerStop() != 0:
                raise RuntimeError("HIP profiler stop failed")
        finally:
            jax.profiler.stop_trace()
            write_json(output / "capture-ranges.json", ranges)
