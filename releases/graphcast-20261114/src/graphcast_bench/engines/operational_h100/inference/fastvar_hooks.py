"""Independent inference candidate: prior regular3 plus FP32 fast variance.

Only Haiku's documented variance formula option changes. Parameter names,
epsilon, FP32 reductions, affine parameters, and all GEMMs remain unchanged.
This formula can be less stable; single-step and 40-step accuracy must pass.
"""
import functools
import json
import regular3_hooks

_audit = []
_installed = False

def install(include_regular3=True):
    global _installed
    if _installed:
        raise RuntimeError('Candidate installed twice')
    import haiku as hk
    if include_regular3:
        regular3_hooks.install()
    original = hk.LayerNorm.__init__
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        name = kwargs.get('name', '')
        if name.endswith('_layer_norm'):
            if args or kwargs.get('axis') != -1 or 'use_fast_variance' in kwargs:
                raise ValueError('Unexpected official GraphCast LayerNorm configuration')
            kwargs['use_fast_variance'] = True
            _audit.append(dict(name=name, axis=-1, use_fast_variance=True,
                               epsilon=kwargs.get('eps', 1e-5)))
        return original(self, *args, **kwargs)
    hk.LayerNorm.__init__ = init
    _installed = True

def save_audit(path):
    if not _installed or not _audit:
        raise RuntimeError('Fast-variance candidate was not traced')
    regular3_hooks.save_audit(path.with_name('regular3-audit.json'))
    path.write_text(json.dumps(dict(status='traced', entries=_audit,
        formula='mean(x*x)-mean(x)**2, FP32; original epsilon and affine parameters; no clamp',
        limitation='Potential cancellation; acceptance requires frozen-reference 40-step check'), indent=2)+'\n')
