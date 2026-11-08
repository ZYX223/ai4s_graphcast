"""Track only this launcher's descendants, including workers that call setsid.

PID birth markers prevent signaling a reused PID. Never use process names,
device ownership or global kill commands to identify tasks.
"""
import os
from pathlib import Path
import signal
import subprocess
import time


def snapshot():
    records = {}
    if Path('/proc/self/stat').exists():
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():
                continue
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] != 'Z':
                    records[int(path.name)] = (int(fields[1]), fields[19])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    else:
        # macOS CPU contract tests; GPU reproductions target Linux.
        result = subprocess.run(['ps', '-axo', 'pid=,ppid=,lstart=,stat='],
                                text=True, capture_output=True, check=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 8 and not parts[7].startswith('Z'):
                records[int(parts[0])] = (int(parts[1]), ' '.join(parts[2:7]))
    return records


class TaskTree:
    def __init__(self, process):
        self.process = process
        self.owned = {}
        table = snapshot()
        if process.pid in table:
            self.owned[process.pid] = table[process.pid][1]
        self.observe(table)

    def observe(self, table=None):
        table = snapshot() if table is None else table
        alive = {pid for pid, birth in self.owned.items()
                 if pid in table and table[pid][1] == birth}
        changed = True
        while changed:
            changed = False
            for pid, (parent, birth) in table.items():
                if parent in alive and pid not in alive:
                    self.owned[pid] = birth
                    alive.add(pid)
                    changed = True
        return alive

    def send(self, sig):
        for pid in self.observe():
            # Recheck the marker immediately before signaling this exact PID.
            current = snapshot().get(pid)
            if current is not None and current[1] == self.owned[pid]:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    def stop(self, grace=30):
        self.observe()
        self.send(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while self.observe() and time.monotonic() < deadline:
            self.process.poll()  # reap the immediate child if it has exited
            time.sleep(.1)
        if self.observe():
            self.send(signal.SIGKILL)
        self.process.wait()
        deadline = time.monotonic() + 5
        while self.observe() and time.monotonic() < deadline:
            time.sleep(.1)
        if self.observe():
            raise RuntimeError('owned descendants remain after task cleanup')

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            self.observe()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            try:
                return self.process.wait(timeout=min(2, remaining))
            except subprocess.TimeoutExpired:
                continue
