"""Experimental JAX producer rounding; not installed into the model baseline."""
def rne(x):
    import jax
    import jax.numpy as jnp
    if x.dtype != jnp.float32:
        raise TypeError('RNE requires float32 storage')
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    # Integer work, preserving NaN payloads/Inf and nearest-even tie behavior.
    rounded = (bits + jnp.uint32(0xfff) + ((bits >> 13) & jnp.uint32(1))) & jnp.uint32(0xffffe000)
    result = jnp.where((bits & jnp.uint32(0x7f800000)) == jnp.uint32(0x7f800000), bits, rounded)
    return jax.lax.bitcast_convert_type(result, jnp.float32)
