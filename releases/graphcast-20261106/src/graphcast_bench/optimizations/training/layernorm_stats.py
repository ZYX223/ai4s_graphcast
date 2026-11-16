"""Name small statistics without changing pinned Haiku two-pass LayerNorm.

No rematerialization policy is installed here. The caller may explicitly use
jax.checkpoint_policies.save_only_these_names(*STAT_NAMES) in an existing remat
scope. Markers can affect compiler fusion and require GPU measurement.
"""

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

STAT_NAMES = ("graphcast_ln_mean", "graphcast_ln_variance", "graphcast_ln_rsqrt")
HAIKU_SOURCE_SHA256 = "1dfd94b4425e68014a482a6b66595458e50d8a4a09a5a4c3f059503fe0008c79"
_installed = False
_audit = []
_provenance = {}


def validate_spec(module, inputs, scale, offset):
    if (
        module.axis != (-1,)
        or module.param_axis != (-1,)
        or not module.create_scale
        or not module.create_offset
        or module.use_fast_variance
        or float(module.eps) != 1e-5
        or scale is not None
        or offset is not None
    ):
        raise ValueError("Only unchanged GraphCast centered variance/affine/epsilon is supported")
    if inputs.ndim != 3 or inputs.shape[1:] != (1, 512) or str(inputs.dtype) != "float32":
        raise ValueError("Expected FP32 [rows,1,512] LayerNorm input")


def annotated_layer_norm(inputs, scale, offset, eps=1e-5):
    """Exact fixed Haiku0.0.14 arithmetic with identity checkpoint_name tags."""
    import jax
    import jax.numpy as jnp
    from jax.ad_checkpoint import checkpoint_name

    axis = (inputs.ndim - 1,)
    mean = checkpoint_name(jnp.mean(inputs, axis=axis, keepdims=True), STAT_NAMES[0])
    # Keep jnp.var intact, including its own centered mean and dtype handling.
    variance = checkpoint_name(jnp.var(inputs, axis=axis, keepdims=True), STAT_NAMES[1])
    converted_eps = jax.lax.convert_element_type(eps, variance.dtype)
    rsqrt = checkpoint_name(jax.lax.rsqrt(variance + converted_eps), STAT_NAMES[2])
    inv = scale * rsqrt
    return inv * (inputs - mean) + offset


def install():
    global _installed
    if _installed:
        raise RuntimeError("LayerNorm statistic markers installed twice")
    import haiku as hk
    import jax

    if hk.__version__ != "0.0.14" or jax.__version__ != "0.6.0":
        raise RuntimeError("Only the isolated Haiku0.0.14/JAX0.6.0 BW port is supported")
    original = hk.LayerNorm
    source = Path(inspect.getsourcefile(original))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != HAIKU_SOURCE_SHA256 or original.__module__ != "haiku._src.layer_norm":
        raise RuntimeError(
            "Original Haiku LayerNorm implementation changed or was already replaced"
        )
    if jax.config.jax_numpy_rank_promotion != "allow":
        raise RuntimeError(
            "This fixed-case candidate requires original allow rank-promotion configuration"
        )
    _provenance.update(
        haiku_version=hk.__version__,
        jax_version=jax.__version__,
        haiku_source_sha256=actual,
        rank_promotion=jax.config.jax_numpy_rank_promotion,
    )

    class AnnotatedLayerNorm(original):
        def __call__(self, inputs, scale=None, offset=None):
            if not self.name.endswith("_layer_norm"):
                return super().__call__(inputs, scale, offset)
            validate_spec(self, inputs, scale, offset)
            scale = hk.get_parameter("scale", (512,), inputs.dtype, init=self.scale_init)
            offset = hk.get_parameter("offset", (512,), inputs.dtype, init=self.offset_init)
            _audit.append(
                dict(
                    module_name=self.module_name,
                    input_shape=list(inputs.shape),
                    dtype=str(inputs.dtype),
                    epsilon=float(self.eps),
                    names=list(STAT_NAMES),
                    named_shape=[inputs.shape[0], 1, 1],
                    bytes_if_all_three_saved=3 * inputs.shape[0] * 4,
                    formula="Original centered two-pass jnp.var; unchanged mean/eps/rsqrt/affine ordering",
                )
            )
            return annotated_layer_norm(inputs, scale, offset, self.eps)

    hk.LayerNorm = AnnotatedLayerNorm
    _installed = True


def save_audit(path, require_full_model=True):
    if not _installed or not _audit:
        raise RuntimeError("No actual annotated LayerNorm trace")
    modules = {}
    for row in _audit:
        old = modules.setdefault(row["module_name"], row)
        if old["input_shape"] != row["input_shape"]:
            raise RuntimeError("One LayerNorm module changed input shape across traces")
    if require_full_model:
        expected = {40962: 19, 327660: 17, 1038240: 3, 1618746: 2, 3114720: 2}
        if (
            len(modules) != 43
            or dict(Counter(r["input_shape"][0] for r in modules.values())) != expected
        ):
            raise RuntimeError("Expected the unchanged43 GraphCast trace modules and shapes")
    Path(path).write_text(
        json.dumps(
            dict(
                status="traced",
                require_full_model=require_full_model,
                unique_modules=len(modules),
                trace_calls=len(_audit),
                names=list(STAT_NAMES),
                provenance=_provenance,
                entries=_audit,
                policy_installed=False,
                policy_api="jax.checkpoint_policies.save_only_these_names(*layernorm_stats.STAT_NAMES)",
                derivative_note="Actual JAX0.4.34 differentiated remat uses saved variance for var+eps, saved rsqrt for affine/input/scale-gradient factors, and saved mean for centering. The rsqrt primitive JVP still recomputes rsqrt(var+eps), and jnp.var still recomputes its internal mean. In the isolated CPU Haiku probe, nested backward reduce_sum count is13 without saving versus11 with these names; this is a JAXPR count, not a GPU kernel or time claim.",
                scope="Markers only. No stage/message/time policy installed; trace names alone do not prove saved residuals or GPU performance. Fixed two-pass FP32 arithmetic retained; no shifted moments/BF16.",
            ),
            indent=2,
        )
        + "\n"
    )
