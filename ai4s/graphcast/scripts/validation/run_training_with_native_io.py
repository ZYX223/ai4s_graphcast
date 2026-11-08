#!/usr/bin/env python3
"""Run the unchanged trainer/observer under scoped, audited model adapters."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-report", type=Path)
    parser.add_argument("--layernorm-report", type=Path)
    parser.add_argument("entry", type=Path)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parents[1]
    allowed = {scripts / "training/run_official_epoch_training.py",
               scripts / "validation/trace_training_flaggems.py",
               scripts / "validation/trace_training_release.py"}
    if args.entry.resolve() not in allowed:
        parser.error("only the canonical trainer or its call observer is supported")
    if args.copy_report is None and args.layernorm_report is None:
        parser.error("at least one explicit runtime adapter/report is required")
    reports = [path for path in (args.copy_report, args.layernorm_report) if path is not None]
    if len({path.resolve() for path in reports}) != len(reports):
        parser.error("runtime adapter report paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in reports):
        parser.error("runtime adapter reports must not overwrite existing evidence")
    adapters = []
    if args.copy_report is not None:
        from graphcast_compat.training_copy_routing import from_environment
        adapters.append((args.copy_report, from_environment()))
    if args.layernorm_report is not None:
        from graphcast_compat.training_layernorm_routing import ModelLayerNormRouting
        adapters.append((args.layernorm_report, ModelLayerNormRouting()))
    old_argv, original_exit = sys.argv, os._exit
    attempted = set()

    def flush_reports(exit_method, exit_code):
        failures = []
        for path, adapter in adapters:
            if path in attempted:
                continue
            attempted.add(path)
            try:
                payload = dict(adapter.metadata(), exit_method=exit_method, exit_code=exit_code)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as report:
                    json.dump(payload, report, indent=2)
                    report.flush()
            except Exception as error:
                failures.append(error)
        if failures:
            raise failures[0]

    def exit_with_reports(status):
        try:
            flush_reports("os._exit", status)
        except Exception as error:
            print(f"runtime adapter report failed before os._exit: {error}",
                  file=sys.stderr, flush=True)
            status = status if isinstance(status, int) and status else 1
        # Preserve the canonical MUSA static-destructor bypass. Do not turn
        # this into SystemExit or unwind the real training context here.
        return original_exit(status)

    try:
        sys.argv = [str(args.entry), *args.training_args]
        os._exit = exit_with_reports
        with ExitStack() as scope:
            for _, adapter in adapters:
                scope.enter_context(adapter)
            runpy.run_path(str(args.entry), run_name="__main__")
    finally:
        error = sys.exc_info()[1]
        os._exit = original_exit
        sys.argv = old_argv
        if isinstance(error, SystemExit):
            flush_reports("sys.exit", error.code)
        else:
            flush_reports("exception" if error is not None else "return",
                          None if error is not None else 0)


if __name__ == "__main__":
    main()
