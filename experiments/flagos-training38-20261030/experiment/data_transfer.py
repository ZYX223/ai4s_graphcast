import jax
def to_device(tree):
 # GraphCast's xarray PyTree cannot reconstruct JAX flatten_axes sentinel
 # values. Transfer leaves individually and reconstruct only real arrays.
 return jax.tree.map(jax.device_put,tree)
