import ctypes,runpy
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(41,1,0,0,0)==0 and libc.prctl(42,0,0,0,0)==1
runpy.run_module('train',run_name='__main__')
