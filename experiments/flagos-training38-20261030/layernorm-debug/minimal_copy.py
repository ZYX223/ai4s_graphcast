"""Regression: native GPU must return a scalar and a reshaped input together."""
import jax,jax.numpy as jnp,numpy as np
@jax.jit
def copies(x):
    return jnp.float32(512),x.reshape(1,1,512)
x=jnp.arange(512,dtype=jnp.float32)
y=copies(x);jax.block_until_ready(y)
np.testing.assert_array_equal(np.asarray(y[0]),np.float32(512))
np.testing.assert_array_equal(np.asarray(y[1]),np.arange(512,dtype=np.float32).reshape(1,1,512))
print('PASS scalar-and-view copy outputs',flush=True)
