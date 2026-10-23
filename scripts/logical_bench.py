#!/usr/bin/env python3
"""Measure logical operators from the frozen single-step JAX graph."""

import argparse
import ctypes as C
from importlib import metadata
from pathlib import Path

from graphcast_bench.assets import ROOT, read_json
from graphcast_bench.logical_bench import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.iterations) < 1:
        parser.error("repeats and iterations must be positive")
    if metadata.version("jax") != read_json(ROOT / "configs/hygon.json")["jax_version"]:
        raise RuntimeError("Wrong JAX runtime")
    libc = C.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0):
        raise OSError(C.get_errno(), "Could not disable THP for this process")
    run(args.inventory, args.data, args.output, args.repeats, args.iterations, args.smoke)


if __name__ == "__main__":
    main()
