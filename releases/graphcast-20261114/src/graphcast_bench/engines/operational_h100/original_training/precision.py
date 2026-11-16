import ctypes as C,os
class Precision:
 def __init__(self):
  self.lib=C.CDLL(os.environ['TRAINING_ROUTER'])
  self.lib.training_select.argtypes=[C.c_int,C.c_char_p]
  self.lib.training_calls.restype=C.c_uint64
  self.lib.training_rounded.restype=C.c_uint64
 def select(self,mode,label):
  self.lib.training_select({'fp32':0,'rne_xf32':1}[mode],label.encode())
 def counts(self):return dict(calls=self.lib.training_calls(),rounded=self.lib.training_rounded())
 def delta(self,before):return {k:v-before[k] for k,v in self.counts().items()}
