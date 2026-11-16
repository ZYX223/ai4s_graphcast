"""Record host messages from preparation through cleanup without owning the child."""
from contextlib import contextmanager
import sys
import threading
from types import SimpleNamespace

from graphcast_bench.progress import emit, plain


class RecordedStream:
    def __init__(self, terminal, file, lock, errors):
        self.terminal, self.file, self.lock, self.errors = terminal, file, lock, errors

    def write(self, text):
        with self.lock:
            try:
                self.file.write(plain(text))
                self.file.flush()
            except OSError as exc:
                # Finish owned cleanup before reporting a failed host log.
                self.errors.append(exc)
            try:
                self.terminal.write(text)
                self.terminal.flush()
            except (OSError, ValueError):
                pass
        return len(text)

    def flush(self):
        with self.lock:
            try:
                self.file.flush()
            except OSError as exc:
                self.errors.append(exc)
            try:
                self.terminal.flush()
            except (OSError, ValueError):
                pass


@contextmanager
def host_log(path):
    """Restore both streams on every exit; keep an early preparation error on disk."""
    original_out, original_err = sys.stdout, sys.stderr
    lock, errors = threading.RLock(), []
    status = SimpleNamespace(exit_code=None)
    file = path.open('w', encoding='utf-8')
    sys.stdout = RecordedStream(original_out, file, lock, errors)
    sys.stderr = RecordedStream(original_err, file, lock, errors)
    try:
        yield status
    except BaseException as exc:
        emit('Failure', 'Host run {}: {}: {}'.format(
            'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            type(exc).__name__, exc), 'ERROR')
        raise
    finally:
        try:
            file.close()
        except OSError as exc:
            errors.append(exc)
        sys.stdout, sys.stderr = original_out, original_err
        if errors:
            emit('Failure', 'Unable to save the complete host log: {}: {}'.format(path, errors[0]), 'ERROR')
    if errors:
        # A failed worker keeps its exit code; logging failure fails an otherwise successful run.
        if status.exit_code in (None, 0):
            raise RuntimeError('Unable to save the complete host log: {}: {}'.format(path, errors[0]))
