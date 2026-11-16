#!/usr/bin/env python3
"""Thin CLI; implementation lives in graphcast_bench."""

import sys
from pathlib import Path

# Keep a source delivery unchanged when its own CLI is invoked.
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from graphcast_bench.platforms.native import main

    main()
