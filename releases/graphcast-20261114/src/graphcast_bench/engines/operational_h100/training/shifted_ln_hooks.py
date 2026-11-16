"""Training snapshot of shifted FP32 LayerNorm with unchanged affine scopes.

Pure JAX differentiation, verified independently w.r.t. inputs/scale/offset.
Model training acceptance still requires corresponding-update loss checks.
"""
import json
from collections import Counter
_audit = []
_installed = False


def install():
    global _installed
    if _installed:
        raise RuntimeError('Shifted LayerNorm hook installed twice')
    import haiku as hk
    from shifted_layernorm import shifted_layer_norm
    original = hk.LayerNorm

    class StableLayerNorm(original):
        def __call__(self, inputs, scale=None, offset=None):
            if not self.name.endswith('_layer_norm'):
                return super().__call__(inputs, scale, offset)
            if self.axis != (-1,) or self.param_axis != (-1,) or not self.create_scale or not self.create_offset:
                raise ValueError('Unexpected fixed GraphCast affine/axis configuration')
            if self.use_fast_variance or scale is not None or offset is not None:
                raise ValueError('Only original centered variance and stored affine parameters supported')
            if inputs.ndim != 3 or inputs.shape[1:] != (1, 512) or str(inputs.dtype) != 'float32':
                raise ValueError('Unexpected GraphCast LayerNorm input')
            scale = hk.get_parameter('scale', (512,), inputs.dtype, init=self.scale_init)
            offset = hk.get_parameter('offset', (512,), inputs.dtype, init=self.offset_init)
            _audit.append(dict(module_name=self.module_name, input_shape=list(inputs.shape),
                               dtype=str(inputs.dtype), epsilon=float(self.eps),
                               formula='row-first-element shifted FP32 moments, no clamp',
                               backend='JAX/XLA with automatic differentiation',
                               parameters='unchanged scale/offset names'))
            return shifted_layer_norm(inputs, scale, offset, eps=self.eps)

    hk.LayerNorm = StableLayerNorm
    _installed = True


def save_audit(path):
    names = [r['module_name'] for r in _audit]
    expected_shapes = {40962:19, 327660:17, 1038240:3, 1618746:2, 3114720:2}
    if (not _installed or len(names) != 43 or len(set(names)) != 43 or
            dict(Counter(r['input_shape'][0] for r in _audit)) != expected_shapes):
        raise RuntimeError('Unexpected fixed-model shifted LayerNorm trace modules/shapes')
    path.write_text(json.dumps(dict(status='traced', trace_modules=43, entries=_audit,
        scope='Actual shifted formula calls from live modules; this audit does not alone prove an outer remat policy or model loss acceptance.'), indent=2) + '\n')
