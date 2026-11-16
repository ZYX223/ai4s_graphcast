"""Apply a process-local host-memory policy before BW training imports JAX."""

import ctypes
import errno
import os

from graphcast_bench.assets import write_json
from graphcast_bench.progress import emit


PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42


def configure_training_memory(output):
    """Avoid synchronous THP compaction while the driver pins host pages.

    This affects only this worker and its future children. Global kernel policy
    is untouched. Unsupported or restricted runtimes keep their original policy
    and retain an explicit warning instead of failing an otherwise valid run.
    """
    record = dict(scope='process', requested='disable_transparent_hugepages',
                  status='not_applied', before=None, after=None)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        # Explicit widths are required for the variadic prctl ABI on 64-bit Linux.
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        prctl.restype = ctypes.c_int

        def control(option, value=0):
            result = prctl(option, value, 0, 0, 0)
            if result < 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            return result

        record['before'] = control(PR_GET_THP_DISABLE)
        if record['before'] != 1:
            control(PR_SET_THP_DISABLE, 1)
        record['after'] = control(PR_GET_THP_DISABLE)
        if record['after'] != 1:
            raise OSError(errno.EIO, 'THP disable readback did not confirm the requested policy')
        record['status'] = 'disabled'
    except (OSError, AttributeError) as error:
        record['error'] = str(error)

    write_json(output / 'host-memory-policy.json', record)
    if record['status'] == 'disabled':
        emit('Memory', 'Transparent huge pages disabled for this training process')
    else:
        emit('Memory', 'Process-local THP policy was not applied: ' + record['error'], 'WARN')
    return record
