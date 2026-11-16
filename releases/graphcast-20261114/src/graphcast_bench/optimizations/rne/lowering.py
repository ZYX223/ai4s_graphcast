"""Experimental operand RNE after AD; the mathematical JAX program is intact.

Process-local registration only. Keep the rocBLAS router until runtime operands
and forward/backward numerical comparisons prove the lowering coverage.
"""

import json
import math

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.interpreters import mlir

_installed = set()
_records = []


def rne(x):
    """FP32 storage, TF32-width RNE; preserve NaN/Inf bit patterns."""
    bits = lax.bitcast_convert_type(x, jnp.uint32)
    finite = (bits & jnp.uint32(0x7F800000)) != jnp.uint32(0x7F800000)
    rounded = (bits + jnp.uint32(0xFFF) + ((bits >> jnp.uint32(13)) & jnp.uint32(1))) & jnp.uint32(
        0xFFFFE000
    )
    return lax.bitcast_convert_type(jnp.where(finite, rounded, bits), jnp.float32)


def selection(lhs, rhs, dimensions):
    (lc, rc), (lb, rb) = dimensions
    if lhs.dtype != np.dtype("float32") or rhs.dtype != np.dtype("float32"):
        return False, "non-fp32"
    if lb or rb:
        return False, "batched-dot-not-yet-validated"
    m = math.prod(v for i, v in enumerate(lhs.shape) if i not in lc)
    n = math.prod(v for i, v in enumerate(rhs.shape) if i not in rc)
    if min(m, n) <= 1:
        return False, "vector-or-scalar"
    return True, "fp32-matrix-contraction"


def install(platform="rocm", audit_path=None):
    """Register after autodiff, not a custom derivative or a primal bitcast."""
    if platform in _installed:
        return
    base = mlir._platform_specific_lowerings.get(platform, {}).get(
        lax.dot_general_p, mlir._lowerings[lax.dot_general_p]
    )

    def lowered(ctx, lhs, rhs, **params):
        selected, reason = selection(*ctx.avals_in, params["dimension_numbers"])
        record = dict(
            lhs=list(ctx.avals_in[0].shape),
            rhs=list(ctx.avals_in[1].shape),
            selected=selected,
            reason=reason,
            dimensions=params["dimension_numbers"],
            scope=str(ctx.name_stack),
            platform=platform,
        )
        _records.append(record)
        if audit_path:
            with open(audit_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if selected:
            operands = []
            for value, aval in zip((lhs, rhs), ctx.avals_in):
                rctx = ctx.replace(avals_in=[aval], avals_out=[aval])
                operands.append(mlir.lower_fun(rne, multiple_results=False)(rctx, value)[0])
            lhs, rhs = operands
        return base(ctx, lhs, rhs, **params)

    mlir.register_lowering(lax.dot_general_p, lowered, platform=platform)
    _installed.add(platform)
