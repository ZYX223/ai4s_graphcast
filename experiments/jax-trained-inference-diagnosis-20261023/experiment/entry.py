"""Same process-only THP setting as the existing baseline launcher."""
import ctypes, runpy
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(41,1,0,0,0)!=0 or libc.prctl(42,0,0,0,0)!=1:
    raise RuntimeError('Could not disable THP for this process')
runpy.run_module('full_model',run_name='__main__')
