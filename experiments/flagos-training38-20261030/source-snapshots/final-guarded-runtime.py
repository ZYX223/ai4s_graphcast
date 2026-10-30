"""Platform instrumentation; math and optimizer remain in the frozen model/recipe."""
import ctypes as C, hashlib, json, os, pickle, subprocess, time
from pathlib import Path
import jax
import numpy as np
from precision import Precision as HygonPrecision

def validate_resume(path):
 path=Path(path);folder=path.parent
 report=json.loads((folder/'report.json').read_text())
 if report.get('status')!='completed' or report.get('checkpoint_reload',{}).get('status')!='passed':
  raise ValueError('Predecessor did not complete checkpoint validation')
 # H100 launcher writes exit-code; Hygon also writes python-exit-code.
 exits=[folder/n for n in ['exit-code.txt','python-exit-code.txt'] if (folder/n).exists()]
 if not exits or any(int(p.read_text())!=0 for p in exits):
  raise ValueError('Predecessor process has no successful exit')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=report.get('checkpoint_sha256'):
  raise ValueError('Checkpoint bytes differ from recorded successful state')
 return report

def tree_digest(tree):
 h=hashlib.sha256()
 for leaf in jax.tree.leaves(jax.device_get(tree)):
  a=np.asarray(leaf);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
 return h.hexdigest()

def verify_reload(path,params,state):
 with path.open('rb') as f:saved=pickle.load(f)
 expected=(params,state);actual=(saved['params'],saved['opt_state'])
 assert jax.tree.structure(actual)==jax.tree.structure(expected)
 for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
  assert np.asarray(a).dtype==np.asarray(b).dtype
  np.testing.assert_array_equal(a,b)
 device=jax.device_put(actual);jax.block_until_ready(device)
 assert tree_digest(device)==tree_digest(expected)
 # Adam counter counts all scheduled updates, including zero learning rates.
 counts=[int(np.asarray(x)) for x in jax.tree.leaves(saved['opt_state']) if np.asarray(x).shape==() and np.asarray(x).dtype.kind in 'iu']
 assert counts and all(x==saved['updates'] for x in counts),counts
 return dict(status='passed',params_sha256=tree_digest(saved['params']),optimizer_sha256=tree_digest(saved['opt_state']),optimizer_counters=counts,device_reload=True)

class Precision:
 def __init__(self):
  self.native=HygonPrecision() if os.environ['TRAINING_PLATFORM']=='hygon' else None
 def select(self,mode,label):
  if self.native:self.native.select(mode,label)
  else:assert mode=='tf32' and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='tensorfloat32'
 def counts(self):return self.native.counts() if self.native else dict(calls=0,rounded=0)
 def delta(self,before):return {k:v-before[k] for k,v in self.counts().items()}

class Runtime:
 def __init__(self):
  self.platform=os.environ['TRAINING_PLATFORM'];self.bus=C.create_string_buffer(64)
  lib=C.CDLL('libamdhip64.so' if self.platform=='hygon' else 'libcudart.so')
  prefix='hip' if self.platform=='hygon' else 'cuda'
  assert getattr(lib,prefix+'DeviceGetPCIBusId')(self.bus,64,0)==0
  self.hipProfilerStart=getattr(lib,prefix+'ProfilerStart');self.hipProfilerStop=getattr(lib,prefix+'ProfilerStop')
 def sample_memory(self):
  if self.platform=='h100':
   value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
   # Container device enumeration may expose all devices to NVML; use UUID.
   value=subprocess.check_output(['nvidia-smi','-i',os.environ['NVIDIA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
   return dict(time_ns=time.time_ns(),used_mib=float(value))
  # Host observer provides device-wide Hygon memory. JAX allocator stats are also retained in report.
  return dict(time_ns=time.time_ns(),used_mib=jax.devices()[0].memory_stats().get('bytes_in_use',0)/2**20,source='jax_allocator_not_device_wide')
