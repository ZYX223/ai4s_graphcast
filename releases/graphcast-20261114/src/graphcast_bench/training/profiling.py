"""Bound formal updates and always close capture without hiding worker failures."""
from contextlib import contextmanager
import json
import time


@contextmanager
def capture_stage(output, runtime, trace_enabled, profiler):
    native_started = trace_started = False
    primary = None
    try:
        if trace_enabled:
            profiler.start_trace(str(output / "jax-trace"))
            trace_started = True
        if runtime.hipProfilerStart() != 0:
            raise RuntimeError("Native profiler start failed")
        native_started = True
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        if native_started:
            try:
                if runtime.hipProfilerStop() != 0:
                    raise RuntimeError("Native profiler stop failed")
            except BaseException as error:
                errors.append(repr(error))
        if trace_started:
            try:
                profiler.stop_trace()
            except BaseException as error:
                errors.append(repr(error))
        if errors:
            try:
                (output / "profile-cleanup.json").write_text(
                    json.dumps(
                        dict(
                            primary_error=repr(primary) if primary else None,
                            cleanup_errors=errors,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            except OSError:
                pass
            if primary is None:
                raise RuntimeError(
                    "Training profiler stop failed: " + "; ".join(errors)
                )


@contextmanager
def update_interval(update, profiler):
    """Exclude staging, diagnostics and checkpoint publication from update timing."""
    with profiler.TraceAnnotation("train-update-%02d" % update):
        interval = dict(trace_begin_ns=time.time_ns())
        try:
            yield interval
        finally:
            interval["trace_end_ns"] = time.time_ns()


def finish_stage(report, monitor, checkpoints, save, started, primary_error=None):
    """Attempt all finalization work; secondary failures never replace the first."""
    from graphcast_bench.progress import emit

    errors = []

    def attempt(operation, action):
        try:
            action()
        except BaseException as error:
            errors.append(dict(operation=operation, error=repr(error)))

    attempt("monitor.close", monitor.close)
    attempt("monitor.summary", lambda: report.update(monitor.summary()))
    report["total_main_seconds"] = time.perf_counter() - started
    attempt(
        "checkpoint.summary", lambda: report.update(checkpointing=checkpoints.summary())
    )
    if errors:
        report["cleanup_errors"] = errors
        if primary_error is None:
            report.update(status="failed", error="Training finalization failed")
    attempt("report.save", lambda: save("report.json", report))
    if errors:
        report["cleanup_errors"] = errors
        # A full/unavailable disk can prevent report publication; keep stderr useful.
        try:
            emit("Training", "Finalization errors: " + repr(errors), "WARNING")
        except BaseException:
            pass
        if primary_error is None:
            raise RuntimeError("Training finalization failed: " + repr(errors))
