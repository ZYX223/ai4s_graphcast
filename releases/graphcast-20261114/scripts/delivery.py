#!/usr/bin/env python3
"""Recipient CLI, runnable with host stdlib Python before Docker is loaded."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    from graphcast_bench.delivery.cli import main

    try:
        sys.exit(main(ROOT))
    except (ValueError, RuntimeError, OSError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(2)
