"""FP32 centered two-pass LayerNorm in one Pallas/Triton kernel.

Inference-only specialization, last channel=512. No fast-variance subtraction,
clamp, epsilon change or reduced precision. Equivalent Haiku operation order.
"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import gpu as plgpu


def stable_kernel(x_ref, scale_ref, offset_ref, y_ref, *, eps):
    cols = jnp.arange(512)
    x = pl.load(x_ref, (cols,)).astype(jnp.float32)
    mean = jnp.sum(x) / jnp.float32(512)
    centered = x - mean
    variance = jnp.sum(centered * centered) / jnp.float32(512)
    scale = pl.load(scale_ref, (cols,))
    offset = pl.load(offset_ref, (cols,))
    inv = scale * jax.lax.rsqrt(variance + jnp.float32(eps))
    pl.store(y_ref, (cols,), inv * centered + offset)


@functools.partial(jax.jit, static_argnames=('eps',))
def stable_layer_norm(x, scale, offset, eps=1e-5):
    if x.ndim != 3 or x.shape[1:] != (1, 512) or x.dtype != jnp.float32:
        raise ValueError('Only FP32 [rows,1,512] is supported')
    if scale.shape != (512,) or offset.shape != (512,):
        raise ValueError('Expected unchanged channel affine parameters')
    fn = pl.pallas_call(functools.partial(stable_kernel, eps=eps),
        grid=(), out_shape=jax.ShapeDtypeStruct((512,), jnp.float32),
        compiler_params=plgpu.TritonCompilerParams(num_warps=2, num_stages=1),
        name='graphcast_ln_centered_fp32')
    fn = jax.vmap(jax.vmap(fn, in_axes=(0,None,None)), in_axes=(0,None,None))
    return fn(x, scale, offset)
