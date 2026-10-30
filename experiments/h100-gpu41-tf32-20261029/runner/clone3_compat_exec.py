#!/usr/bin/env python3
"""Preserve Docker seccomp; deny clone3 with ENOSYS to enable libc fallback.

Docker 20.10.5 cannot express an errnoRet override in its JSON profile. This
adds a process-local filter before exec. All other syscalls remain constrained
by the pre-existing Docker filter. No syscall denied before is allowed here.
"""
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import sys


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


def instructions():
    # Native Linux x86_64 seccomp_data.nr is at offset 0; SYS_clone3 = 435.
    # ALLOW here does not override the existing Docker filter's ERRNO/KILL.
    return ((0x20, 0, 0, 0), (0x15, 0, 1, 435),
            (0x06, 0, 0, 0x00050000 | errno.ENOSYS),
            (0x06, 0, 0, 0x7FFF0000))


def status_fields():
    return dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)


def install():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Qualified clone3 fallback is Linux x86_64 only")
    before_status = status_fields()
    if before_status.get("Seccomp", "").strip() != "2":
        raise RuntimeError("Existing Docker seccomp filter required; refusing unsandboxed launch")
    if before_status.get("TracerPid", "").strip() != "0":
        raise RuntimeError("Unexpected tracer; preserve and inspect")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall(435, ctypes.c_void_p(0), ctypes.c_size_t(0))
    previous_errno = ctypes.get_errno()
    filters = (SockFilter * 4)(*(SockFilter(*values) for values in instructions()))
    program = SockFprog(4, filters)
    # NoNewPrivs is irreversible and inherited across exec, as is this filter.
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")
    result = libc.syscall(435, ctypes.c_void_p(0), ctypes.c_size_t(0))
    effective_errno = ctypes.get_errno()
    if result != -1 or effective_errno != errno.ENOSYS:
        raise RuntimeError("Clone3 ENOSYS policy did not become effective")
    after_status = status_fields()
    report = dict(event="clone3_compat_qualified", pid=os.getpid(),
                  clone3_before_errno=previous_errno, clone3_after_errno=effective_errno,
                  original_docker_filter_retained=True, clone3_allowed=False,
                  no_new_privs=after_status.get("NoNewPrivs", "").strip(),
                  seccomp_mode=after_status.get("Seccomp", "").strip(),
                  seccomp_filters=after_status.get("Seccomp_filters", "").strip())
    print(json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
    return report


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    if not arguments:
        raise RuntimeError("Explicit exec argv required")
    install()
    os.execvpe(arguments[0], arguments, os.environ.copy())


if __name__ == "__main__":
    main()
