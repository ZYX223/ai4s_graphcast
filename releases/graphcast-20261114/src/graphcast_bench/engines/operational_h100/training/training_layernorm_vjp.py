"""Fixed FP32 Haiku LayerNorm forward with a centered first-order reverse rule.

This isolated candidate installs no remat policy. The three statistic names are
compatible with layernorm_stats, but the two LayerNorm hooks are alternatives.
Forward-mode AD and higher-order AD are not supported by this experiment.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

STAT_NAMES = ('graphcast_ln_mean', 'graphcast_ln_variance', 'graphcast_ln_rsqrt')
HAIKU_SOURCE_SHA256 = '1dfd94b4425e68014a482a6b66595458e50d8a4a09a5a4c3f059503fe0008c79'
_installed = False
_audit = []


def validate_spec(module, inputs, scale, offset):
    if (module.axis != (-1,) or module.param_axis != (-1,)
            or not module.create_scale or not module.create_offset
            or module.use_fast_variance or float(module.eps) != 1e-5
            or scale is not None or offset is not None):
        raise ValueError('Only original GraphCast centered FP32 LayerNorm is supported')
    if inputs.ndim != 3 or inputs.shape[1:] != (1, 512) or str(inputs.dtype) != 'float32':
        raise ValueError('Expected FP32 [rows,1,512] input')


@lru_cache(maxsize=1)
def _operator():
    import jax
    import jax.numpy as jnp
    from jax.ad_checkpoint import checkpoint_name

    def parts(x, scale):
        # Keep the complete original jnp.var operation and affine ordering.
        mean = checkpoint_name(jnp.mean(x, axis=-1, keepdims=True), STAT_NAMES[0])
        variance = checkpoint_name(jnp.var(x, axis=-1, keepdims=True), STAT_NAMES[1])
        eps = jax.lax.convert_element_type(1e-5, variance.dtype)
        inverse_std = checkpoint_name(jax.lax.rsqrt(variance+eps), STAT_NAMES[2])
        y = (scale*inverse_std)*(x-mean)
        return y, (x, scale, mean, variance, inverse_std)

    @jax.custom_vjp
    def operation(x, scale):
        return parts(x, scale)[0]

    def forward(x, scale):
        return parts(x, scale)

    def backward(residual, cotangent):
        x, scale, mean, variance, inverse_std = residual
        centered = x-mean
        # Preserve the original chain-rule factor ans/(var+eps), rather than
        # replacing it with r**3, which rounds differently in FP32.
        eps = jax.lax.convert_element_type(1e-5, variance.dtype)
        grad_inverse = cotangent*centered
        grad_r = jnp.sum(scale*grad_inverse, axis=-1, keepdims=True)
        grad_variance = grad_r * (-jnp.float32(.5)*(inverse_std/(variance+eps)))
        direct = cotangent*(scale*inverse_std)
        variance_path = (jnp.float32(2)*centered)*(grad_variance/jnp.float32(512))
        # Both centering operations require their mean subtraction. In
        # particular mean(x-mean(x)) need not be zero in floating point.
        dx = (direct-jnp.mean(direct, axis=-1, keepdims=True)
              + variance_path-jnp.mean(variance_path, axis=-1, keepdims=True))
        dscale = jnp.sum(grad_inverse*inverse_std, axis=(0, 1))
        return dx, dscale

    operation.defvjp(forward, backward)
    return operation


def layer_norm(inputs, scale, offset):
    if (inputs.ndim != 3 or inputs.shape[1:] != (1, 512)
            or scale.shape != (512,) or offset.shape != (512,)
            or any(str(x.dtype) != 'float32' for x in (inputs, scale, offset))):
        raise ValueError('Only FP32 [rows,1,512] input and [512] affine parameters are supported')
    # Ordinary AD handles the final broadcast addition and its complete offset
    # gradient. Keeping offset outside the custom primitive also prevents an
    # otherwise unused offset argument from entering remat residuals.
    return _operator()(inputs, scale)+offset


def install():
    global _installed
    if _installed:
        raise RuntimeError('Analytic LayerNorm VJP installed twice')
    import haiku as hk
    import jax
    original = hk.LayerNorm
    source = Path(inspect.getsourcefile(original))
    if (hk.__version__ != '0.0.14' or jax.__version__ != '0.4.34'
            or original.__module__ != 'haiku._src.layer_norm'
            or hashlib.sha256(source.read_bytes()).hexdigest() != HAIKU_SOURCE_SHA256):
        raise RuntimeError('Expected unchanged pinned Haiku LayerNorm before installation')
    if jax.config.jax_numpy_rank_promotion != 'allow':
        raise RuntimeError('Expected original allow rank-promotion configuration')

    class AnalyticLayerNorm(original):
        def __call__(self, inputs, scale=None, offset=None):
            if not self.name.endswith('_layer_norm'):
                return super().__call__(inputs, scale, offset)
            validate_spec(self, inputs, scale, offset)
            scale = hk.get_parameter('scale', (512,), inputs.dtype, init=self.scale_init)
            offset = hk.get_parameter('offset', (512,), inputs.dtype, init=self.offset_init)
            _audit.append(dict(module_name=self.module_name, input_shape=list(inputs.shape),
                dtype=str(inputs.dtype), epsilon=float(self.eps), names=list(STAT_NAMES)))
            return layer_norm(inputs, scale, offset)

    hk.LayerNorm = AnalyticLayerNorm
    _installed = True


def save_audit(path, require_full_model=True):
    if not _installed or not _audit:
        raise RuntimeError('No actual analytic LayerNorm trace')
    modules = {}
    for row in _audit:
        old = modules.setdefault(row['module_name'], row)
        if old['input_shape'] != row['input_shape']:
            raise RuntimeError('LayerNorm module shape changed across traces')
    if require_full_model:
        expected = {40962: 19, 327660: 17, 1038240: 3, 1618746: 2, 3114720: 2}
        if len(modules) != 43 or dict(Counter(r['input_shape'][0] for r in modules.values())) != expected:
            raise RuntimeError('Expected all43 fixed GraphCast LayerNorm module shapes')
    Path(path).write_text(json.dumps(dict(status='traced', require_full_model=require_full_model,
        unique_modules=len(modules), trace_calls=len(_audit), entries=_audit,
        names=list(STAT_NAMES), policy_installed=False,
        forward='Original Haiku FP32 mean/full two-pass jnp.var/epsilon/affine order',
        backward='Centered chain rule with mean-subtraction on direct and variance paths; ans/(var+eps) factor',
        residual='Original input and scale, plus mean/variance/rsqrt; no saved centered activation',
        supported='Experiment validates first-order reverse AD only',
        unsupported='custom_vjp forward-mode JVP/jacfwd; higher-order AD unvalidated',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Isolated candidate; trace audit does not establish full model loss, memory or GPU performance'), indent=2)+'\n')
