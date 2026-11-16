#!/usr/bin/env python3
"""Source, release, completion and frozen-H100 comparison entrypoints."""

import argparse
import sys
from pathlib import Path

# Keep a source delivery unchanged when its own CLI is invoked.
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("release checks v1/v2 source files only; use scripts/verify_delivery.py "
                "for current delivery source, material manifests and run plans."),
    )
    parser.add_argument(
        "action",
        choices=(
            "code",
            "release",
            "completion",
            "inference",
            "training-loss",
            "training-loss-summary",
        ),
    )
    args, remaining = parser.parse_known_args()
    if args.action == "code":
        from graphcast_bench.validation.provenance import verify_maintained

        if remaining:
            parser.error("code takes no further arguments")
        manifest = verify_maintained()
        print("Verified official source and", len(manifest["files"]), "maintained files")
    elif args.action == "release":
        from graphcast_bench.validation.release_integrity import verify_release

        if len(remaining) != 1:
            parser.error("release requires exactly one directory")
        print(verify_release(Path(remaining[0])))
    elif args.action == "completion":
        from graphcast_bench.validation.completion import main as validate

        return validate(remaining)
    elif args.action == "inference":
        from graphcast_bench.validation.inference_comparison import main as validate

        return validate(remaining)
    elif args.action == "training-loss-summary":
        from graphcast_bench.validation.loss_metrics import main as validate

        return validate(remaining)
    else:
        from graphcast_bench.validation.training_loss import main as validate

        return validate(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
