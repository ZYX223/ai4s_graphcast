"""Set the Hygon process-only memory setting used in the weekly timing run."""

import ctypes
import os
import runpy

if os.environ["GRAPHCAST_PLATFORM"] == "hygon":
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError("Could not disable THP for this process")
runpy.run_module("graphcast_bench.benchmark", run_name="__main__")
