"""CUDA profiler capture with primary-error preservation and durable boundaries."""
from contextlib import contextmanager
import ctypes
import time

from graphcast_bench.assets import write_json
from graphcast_bench.progress import emit


@contextmanager
def capture(output, name, steps, ranges):
    runtime = ctypes.CDLL('libcudart.so')
    primary, started = None, False
    begin_wall, begin_mono = time.time_ns(), time.monotonic_ns()
    try:
        if runtime.cudaProfilerStart() != 0:
            raise RuntimeError('CUDA profiler start failed')
        started = True
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_error = None
        if started:
            try:
                if runtime.cudaProfilerStop() != 0:
                    raise RuntimeError('CUDA profiler stop failed')
            except BaseException as error:
                cleanup_error = error
            ranges.append(dict(label=name, measured_steps=steps,
                wall_start_ns=begin_wall, wall_end_ns=time.time_ns(),
                monotonic_start_ns=begin_mono, monotonic_end_ns=time.monotonic_ns()))
        try:
            write_json(output / 'capture-ranges.json', ranges)
        except BaseException as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            if primary is None:
                raise cleanup_error
            emit('Cleanup', 'Capture cleanup also failed: ' + repr(cleanup_error), 'WARN')
