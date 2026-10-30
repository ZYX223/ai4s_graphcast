"""Root-level entry keeps the selected platform policy ahead of experiment code."""
import ctypes,hashlib,os,runpy
from pathlib import Path
import remat_policy
actual=hashlib.sha256(Path(remat_policy.__file__).read_bytes()).hexdigest()
assert actual==os.environ['TRAINING_REMAT_SHA256'],(remat_policy.__file__,actual)
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(41,1,0,0,0)==0 and libc.prctl(42,0,0,0,0)==1
runpy.run_module('train',run_name='__main__')
