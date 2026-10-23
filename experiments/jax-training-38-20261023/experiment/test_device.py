"""Exercise real JAX forward/backward GEMMs, with mutable device contents."""
import ctypes as C, json, os
import numpy as np
import jax, jax.numpy as jnp
from precision import Precision

def rne(x):
 b=np.asarray(x,dtype=np.float32).copy().view(np.uint32)
 finite=(b&0x7f800000)!=0x7f800000
 b[finite]=(b[finite]+0xfff+((b[finite]>>13)&1))&0xffffe000
 return b.view(np.float32)

control=Precision();rng=np.random.default_rng(81)
a=rng.normal(size=(512,256)).astype('float32')
b=rng.normal(size=(256,128)).astype('float32')
g=rng.normal(size=(512,128)).astype('float32')
aa,bb,gg=map(jax.device_put,(a,b,g))
f=jax.jit(lambda a,b:a@b)
vg=jax.jit(jax.value_and_grad(lambda a,b,g:jnp.sum((a@b)*g),argnums=(0,1)))
results=[]
for mode in ['fp32','rne_xf32','fp32','rne_xf32']:
 control.select(mode,'probe-'+mode)
 start=control.counts()
 got=f(aa,bb);jax.block_until_ready(got)
 v,(da,db)=vg(aa,bb,gg);jax.block_until_ready((v,da,db))
 q=rne if mode=='rne_xf32' else lambda x:x
 ref=[q(a)@q(b),q(g)@q(b).T,q(a).T@q(g)]
 errors=[]
 for actual,expected in zip([got,da,db],ref):
  error=float(np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected));errors.append(error)
  assert error<2e-5,(mode,error)
 delta=control.delta(start)
 assert delta['calls']>=3 and delta['rounded']==(delta['calls'] if mode=='rne_xf32' else 0),delta
 results.append(dict(mode=mode,relative_l2=errors,calls=delta))
 # Change same device buffer to expose stale rounded weight caches.
 b=rng.normal(size=b.shape).astype('float32')
 hip=C.CDLL('libamdhip64.so');hip.hipMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
 assert hip.hipMemcpy(bb.unsafe_buffer_pointer(),b.ctypes.data,b.nbytes,1)==0
print(json.dumps(dict(status='passed',results=results),indent=2))
