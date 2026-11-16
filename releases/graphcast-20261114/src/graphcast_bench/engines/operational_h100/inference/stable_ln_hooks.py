"""Inference-only Haiku module replacement with unchanged parameter names."""
import json
import regular3_hooks
_audit=[]
_installed=False

def install(include_regular3=True):
    global _installed
    if _installed:raise RuntimeError('Stable LayerNorm hook installed twice')
    import haiku as hk
    from pallas_layernorm import stable_layer_norm
    if include_regular3:regular3_hooks.install()
    original=hk.LayerNorm
    class StableLayerNorm(original):
        def __call__(self,inputs,scale=None,offset=None):
            if not self.name.endswith('_layer_norm'):
                return super().__call__(inputs,scale,offset)
            if self.axis!=(-1,) or self.param_axis!=(-1,) or not self.create_scale or not self.create_offset:
                raise ValueError('Unexpected fixed GraphCast affine/axis configuration')
            if self.use_fast_variance or scale is not None or offset is not None:
                raise ValueError('Only original centered variance and stored affine parameters supported')
            if inputs.ndim!=3 or inputs.shape[1:]!=(1,512) or str(inputs.dtype)!='float32':
                raise ValueError('Unexpected GraphCast LayerNorm input')
            scale=hk.get_parameter('scale',(512,),inputs.dtype,init=self.scale_init)
            offset=hk.get_parameter('offset',(512,),inputs.dtype,init=self.offset_init)
            _audit.append(dict(module_name=self.module_name,input_shape=list(inputs.shape),
                dtype=str(inputs.dtype),epsilon=float(self.eps),formula='centered two-pass FP32, no clamp',
                backend='Pallas/Triton single kernel',parameters='unchanged scale/offset names'))
            return stable_layer_norm(inputs,scale,offset,eps=self.eps)
    hk.LayerNorm=StableLayerNorm
    _installed=True

def save_audit(path):
    if not _installed or not _audit:raise RuntimeError('Stable LayerNorm was not traced')
    regular3_hooks.save_audit(path.with_name('regular3-audit.json'))
    names=[r['module_name'] for r in _audit]
    from collections import Counter
    expected_shapes={40962:19,327660:17,1038240:3,1618746:2,3114720:2}
    if len(names)!=43 or len(set(names))!=43 or dict(Counter(r['input_shape'][0] for r in _audit))!=expected_shapes:
        raise RuntimeError('Unexpected fixed-model LayerNorm trace modules/shapes')
    path.write_text(json.dumps(dict(status='traced',trace_modules=43,expected_runtime_layernorms=42,
        dead_output='mesh2grid processor mesh-node update is not returned by GraphCast._run_mesh2grid_gnn; optimized HLO/kernel count separately checked',entries=_audit,limit='Inference only; backward not implemented or tested'),indent=2)+'\n')
