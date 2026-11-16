"""Pure JAX row-shifted FP32 moments, retaining XLA producer/consumer fusion.

A shift taken from the same row removes large common offsets. This is distinct
from unshifted E[x^2]-E[x]^2. No clipping or precision reduction is introduced.
Numerical screening includes a shifted outlier, because operation ordering can
still differ from Haiku's centered two-pass implementation.
"""

import functools

import jax
import jax.numpy as jnp


def shifted_stats(x):
    delta = x - x[..., 0:1]
    mean_delta = jnp.mean(delta, axis=-1, keepdims=True)
    variance = jnp.mean(delta * delta, axis=-1, keepdims=True) - mean_delta * mean_delta
    return delta, mean_delta, variance


@functools.partial(jax.jit, static_argnames=("eps",))
def shifted_layer_norm(x, scale, offset, eps=1e-5):
    delta, mean_delta, variance = shifted_stats(x)
    inv = scale * jax.lax.rsqrt(variance + jnp.float32(eps))
    return inv * (delta - mean_delta) + offset
