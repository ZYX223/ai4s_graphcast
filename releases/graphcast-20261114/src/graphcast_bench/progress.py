"""Small, dependency-free delivery messages shared by host and worker Python."""
from contextlib import contextmanager
import datetime
import re
import threading
import time


DISPLAY_TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))
ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')
EVENT = re.compile(r'^\[[^\]]+\] \[(INFO|WARN|ERROR)\] \[([^\]]+)\] ')


def plain(text):
    return ANSI.sub('', text).replace('\r\n', '\n').replace('\r', '\n')


def message(stage, text, level='INFO'):
    if level not in ('INFO', 'WARN', 'ERROR'):
        raise ValueError('Unknown progress level')
    stamp = datetime.datetime.now(DISPLAY_TIMEZONE).isoformat(timespec='seconds')
    return '[{}] [{}] [{}] {}'.format(stamp, level, stage, plain(str(text)).replace('\n', ' '))


def emit(stage, text, level='INFO'):
    # Terminal consumers may close their pipe; reporting must not skip cleanup.
    try:
        print(message(stage, text, level), flush=True)
    except (OSError, ValueError):
        pass


@contextmanager
def phase(stage, action, heartbeat_seconds=None):
    """Report boundaries without changing exceptions or the caller's timing scope."""
    emit(stage, action + '...')
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            emit(stage, '{}: still waiting ({:.0f}s elapsed)'.format(action, time.monotonic() - started))

    thread = None
    if heartbeat_seconds is not None:
        thread = threading.Thread(target=heartbeat, name='graphcast-phase', daemon=True)
        thread.start()
    try:
        try:
            yield
        finally:
            stopped.set()
            if thread is not None:
                thread.join()
    except BaseException as exc:
        emit(stage, '{} failed: {}: {}'.format(action, type(exc).__name__, exc), 'ERROR')
        raise
    else:
        emit(stage, '{} completed in {:.3f}s'.format(action, time.monotonic() - started))
