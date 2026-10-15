# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.fused.post_layernorm_residual import (
    post_layer_norm_residual as _common_post_layer_norm_residual,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _post_layer_norm_residual_kernel(
    x_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    rows,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < width
    offsets = row * width + columns

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / width
    centered = x - mean
    sum_square = tl.sum(tl.where(mask, centered * centered, 0.0))
    rstd = tl.math.rsqrt(sum_square / width + eps)

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(
            tl.float32
        )
    else:
        weight = 1.0
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = 0.0

    normalized = centered * rstd * weight + bias
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    output = normalized + residual
    tl.store(output_ptr + offsets, output, mask=mask)


def _can_use_flat_fp32_kernel(
    x, residual, normalized_shape, weight=None, bias=None
):
    if not isinstance(x, torch.Tensor) or not isinstance(residual, torch.Tensor):
        return False
    normalized_shape = tuple(normalized_shape)
    if not normalized_shape:
        return False
    width = math.prod(normalized_shape)
    tensors = (x, residual)
    parameters = (weight, bias)
    return (
        not torch.is_grad_enabled()
        and 0 < width <= 4096
        and x.numel() > 0
        and x.numel() % width == 0
        and tuple(x.shape[-len(normalized_shape) :]) == normalized_shape
        and x.shape == residual.shape
        and x.device == residual.device
        and all(item.dtype == torch.float32 for item in tensors)
        and all(item.is_contiguous() for item in tensors)
        and all(
            parameter is None
            or (
                isinstance(parameter, torch.Tensor)
                and tuple(parameter.shape) == normalized_shape
                and parameter.device == x.device
                and parameter.dtype == torch.float32
                and parameter.is_contiguous()
            )
            for parameter in parameters
        )
    )


def post_layer_norm_residual(
    x, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    """Fuse inference LayerNorm and its single residual consumer on Hygon."""

    logger.debug("GEMS_HYGON POST_LAYER_NORM_RESIDUAL")
    if not _can_use_flat_fp32_kernel(
        x, residual, normalized_shape, weight, bias
    ):
        return _common_post_layer_norm_residual(
            x, residual, normalized_shape, weight, bias, eps
        )

    width = math.prod(normalized_shape)
    rows = x.numel() // width
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(width)
    with torch_device_fn.device(x.device):
        _post_layer_norm_residual_kernel[(rows,)](
            x,
            residual,
            output,
            weight,
            bias,
            rows,
            width,
            eps,
            BLOCK_SIZE=block_size,
        )
    return output


__all__ = ["post_layer_norm_residual"]
