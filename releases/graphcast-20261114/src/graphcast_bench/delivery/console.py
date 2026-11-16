"""Drain worker output independently of slow host sampling; keep full diagnostics."""
import codecs
from collections import deque
import os
import re
import selectors
import sys
import threading
import time

from graphcast_bench.progress import EVENT, message, plain


class ConsoleRelay:
    """Own pipe reads only; the caller owns the child, log file and cleanup."""

    def __init__(self, pipe, log, terminal=None, heartbeat_seconds=30, path_map=None):
        self.pipe, self.log = pipe, log
        self.terminal = sys.stdout if terminal is None else terminal
        self.heartbeat_seconds = heartbeat_seconds
        self.stage_started = self.last_heartbeat = time.monotonic()
        self.stage = 'Startup'
        self.path_map = path_map or {}
        self.failure_reason = None
        self._failure_priority = -1
        self.diagnostics_path = str(getattr(log, 'name', 'console.log'))
        self.tail = deque(maxlen=20)
        self.error = None
        self._warnings = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, name='graphcast-console', daemon=True)

    def start(self):
        self._thread.start()

    def host_text(self, text):
        for container, host in self.path_map.items():
            pattern = r'(?<![\w/])' + re.escape(container) + r'(?=/|[\s\x27\x22:]|$)'
            text = re.sub(pattern, lambda match: str(host), text)
        return text

    def _display(self, text):
        text = self.host_text(text)
        # A closed downstream terminal must not discard the full on-disk log.
        try:
            self.terminal.write(text)
            self.terminal.flush()
        except (OSError, ValueError):
            pass

    def note(self, stage, text, level='INFO'):
        line = message(stage, text, level) + '\n'
        with self._lock:
            self.log.write(line)
            self.log.flush()
            self._display(line)

    def _heartbeat(self):
        # Diagnostics are not progress; a heartbeat never restarts this clock.
        now = time.monotonic()
        if now - self.last_heartbeat >= self.heartbeat_seconds:
            self.note(self.stage, 'Still waiting; elapsed={:.0f}s'.format(now - self.stage_started))
            self.last_heartbeat = now

    def _remember_failure(self, reason, priority):
        if priority > self._failure_priority:
            self.failure_reason = reason.strip()
            self._failure_priority = priority

    @property
    def has_direct_failure(self):
        """Wrapper errors give context but still need the diagnostic tail."""
        return self._failure_priority > 0

    @staticmethod
    def _visible_event(stage, body, level):
        if level != 'INFO':
            return True
        if stage in ('Prepare', 'Load', 'Initialize', 'Save') and body.rstrip().endswith('...'):
            return False
        if body.startswith(('Start inference worker:', 'Prepare forecast targets',
                            'Save compilation artifacts', 'Save round=')):
            return False
        return True

    def warn_once(self, key, text):
        with self._lock:
            if key not in self._warnings:
                self._warnings.add(key)
                self.note('Monitoring' if key.startswith('monitor') else 'Diagnostics', text, 'WARN')

    def _line(self, raw):
        line = plain(raw)
        with self._lock:
            self.log.write(line)
            self.log.flush()
            self.tail.append(line)
            event = EVENT.match(line)
            if event:
                level, stage = event.groups()
                body = line[event.end():]
                if level == 'INFO' and stage not in ('Diagnostics', 'Monitoring'):
                    self.stage = stage
                    self.stage_started = self.last_heartbeat = time.monotonic()
                if level == 'ERROR':
                    self._remember_failure(body, 2 if stage in ('Inference', 'Training') else 0)
                if self._visible_event(stage, body, level):
                    self._display(line if line.endswith('\n') else line + '\n')
                return
            # This pinned xarray compatibility notice remains in the detailed log.
            if 'FutureWarning:' in line and 'Dataset.dims' in line:
                return
            if re.match(r'^\d+ warnings? generated\b', line.strip()):
                return
            if re.match(r'^(?:[\w]+\.)*[\w]*(?:Error|Exception):', line.strip()):
                wrapper = 'CalledProcessError:' in line
                previous = self.failure_reason
                self._remember_failure(line, 0 if wrapper else 1)
                if not wrapper and self.failure_reason != previous:
                    self.note('Failure', line.strip(), 'ERROR')
                return
            diagnostic = re.search(r'Warning:|Error:|\bwarning\b|\bfailed\b|\bfatal\b|launch bounds|: E external/', line, re.I)
            if diagnostic:
                key = 'backend:' + re.sub(r'\d+', '#', line.strip())
                if len([k for k in self._warnings if k.startswith('backend:')]) < 5:
                    self.warn_once(key, line.strip()[:240] + '; diagnostics: ' + self.diagnostics_path)

    def _read(self):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = ''
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.pipe, selectors.EVENT_READ)
                while not self._stop.is_set():
                    ready = selector.select(min(1., self.heartbeat_seconds))
                    if ready:
                        chunk = os.read(self.pipe.fileno(), 65536)
                        if not chunk:
                            pending += decoder.decode(b'', final=True)
                            if pending:
                                self._line(pending)
                            break
                        pending += decoder.decode(chunk)
                        while '\n' in pending:
                            line, pending = pending.split('\n', 1)
                            self._line(line + '\n')
                    self._heartbeat()
        except Exception as exc:
            self.error = exc

    def check(self):
        if self.error is not None:
            raise RuntimeError('Unable to save or relay the run log: ' + repr(self.error))

    def finish(self):
        # The caller stops its container first, closing inherited compiler pipes.
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=2)
            raise RuntimeError('Run log pipe did not close; inspect container cleanup')
        self.check()
