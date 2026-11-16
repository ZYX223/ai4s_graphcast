"""Same-process baseline/candidate screening through the explicit H100 worker."""
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from graphcast_bench.inference.h100.entry import main as run

    return run(single=True)


if __name__ == '__main__':
    raise SystemExit(main())
