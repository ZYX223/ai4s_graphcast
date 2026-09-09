#!/usr/bin/env python3
"""Opt-in logged FlagGems function calls, adapted from colleague b7667b4.

Launch under torchrun with --trace-dir DIR -- <canonical trainer arguments>.
Requires GRAPHCAST_FLAGOS_MODE=flaggems and the target backend environment.
The ordinary trainer never imports this observer. Counts are not kernel counts
or proof of coverage for functions that do not emit the observed log messages.
"""
from collections import Counter
import argparse
import inspect
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import threading


TRAINER = Path(__file__).resolve().parents[1] / "training/run_official_epoch_training.py"


class CallObserver(logging.Handler):
    """Observe source log events and scope actual forward/backward boundaries."""

    def __init__(self, trace_dir, *, trainer=TRAINER):
        super().__init__()
        self.trainer = Path(trainer).resolve()
        self.rank = int(os.environ.get("RANK", "0"))
        self.path = Path(trace_dir) / f"rank_{self.rank}.jsonl"
        self.counts = Counter()
        self.count_lock = threading.Lock()
        self.phase = "initialization"
        self.logger_states = []

    def emit(self, record):
        if "GEMS" not in record.getMessage().upper():
            return
        device = "unknown"
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_filename == record.pathname:
                    for value in frame.f_locals.values():
                        if isinstance(value, self.torch.Tensor):
                            device = str(value.device)
                            break
                    break
                frame = frame.f_back
        finally:
            del frame
        counter = sys.modules.get("graphcast_compat.recompute_flops")
        recompute = bool(counter and counter._RECOMPUTE_DEPTH.get())
        phase = "recompute" if recompute else self.phase
        with self.count_lock:
            self.counts[(record.name, record.funcName, phase, device, record.pathname)] += 1

    def flush_update(self, optimizer):
        steps = {int(state["step"]) for state in optimizer.state.values() if "step" in state}
        if len(steps) != 1:
            raise RuntimeError("observer cannot identify a single actual AdamW optimizer update")
        update = steps.pop()
        gems = sys.modules.get("flag_gems")
        with self.count_lock:
            calls = [dict(module=m, function=f, phase=p, device=d, source=s, calls=n,
                          optimizer_update=None if p == "initialization" else update)
                     for (m, f, p, d, s), n in sorted(self.counts.items())]
            self.counts.clear()
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(dict(rank=self.rank, optimizer_update=update,
                count_kind="logged_function_calls", coverage="observed_calls_only",
                canonical_trainer=str(self.trainer), flag_gems_imported=gems is not None,
                flag_gems_source=getattr(gems, "__file__", None), calls=calls)) + "\n")

    def __enter__(self):
        import torch
        self.torch = torch
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently append unrelated runs to previous call evidence.
        with self.path.open("x", encoding="utf-8"):
            pass
        names = ["flag_gems"]
        if os.environ.get("GEMS_VENDOR"):
            names.append("_" + os.environ["GEMS_VENDOR"])
        for name in names:
            logger = logging.getLogger(name)
            self.logger_states.append((logger, logger.level, logger.propagate))
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.addHandler(self)
        self.original_backward = torch.autograd.backward
        self.original_step = torch.optim.AdamW.step
        self.original_call = torch.nn.Module._call_impl

        def call(module, *values, **kwargs):
            forward = getattr(module.forward, "__func__", module.forward)
            code = getattr(forward, "__code__", None)
            scoped = (type(module).__name__ == "EpochAutoregressiveModule" and code is not None
                      and Path(code.co_filename).resolve() == self.trainer)
            if not scoped:
                return self.original_call(module, *values, **kwargs)
            previous = self.phase
            self.phase = "forward"
            try:
                return self.original_call(module, *values, **kwargs)
            finally:
                self.phase = "unscoped" if previous == "initialization" else previous

        def backward(*values, **kwargs):
            previous = self.phase
            self.phase = "backward"  # shared with autograd worker threads
            try:
                return self.original_backward(*values, **kwargs)
            finally:
                self.phase = previous

        def step(optimizer, *values, **kwargs):
            previous = self.phase
            self.phase = "optimizer"
            try:
                result = self.original_step(optimizer, *values, **kwargs)
            finally:
                self.phase = previous
            self.flush_update(optimizer)
            return result

        torch.nn.Module._call_impl = call
        torch.autograd.backward = backward
        torch.optim.AdamW.step = step
        return self

    def __exit__(self, *exc):
        self.torch.nn.Module._call_impl = self.original_call
        self.torch.autograd.backward = self.original_backward
        self.torch.optim.AdamW.step = self.original_step
        for logger, level, propagate in self.logger_states:
            logger.removeHandler(self)
            logger.setLevel(level)
            logger.propagate = propagate
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args, training_args = parser.parse_known_args(argv)
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    if os.environ.get("GRAPHCAST_FLAGOS_MODE") != "flaggems":
        parser.error("the evidence run requires GRAPHCAST_FLAGOS_MODE=flaggems")
    if not os.environ.get("GRAPHCAST_BACKEND") or not os.environ.get("GRAPHCAST_ACCELERATOR_TYPE"):
        parser.error("set GRAPHCAST_BACKEND and GRAPHCAST_ACCELERATOR_TYPE as in public-entry dry-run")
    old_argv = sys.argv
    try:
        with CallObserver(args.trace_dir):
            sys.argv = [str(TRAINER), *training_args]
            runpy.run_path(str(TRAINER), run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
