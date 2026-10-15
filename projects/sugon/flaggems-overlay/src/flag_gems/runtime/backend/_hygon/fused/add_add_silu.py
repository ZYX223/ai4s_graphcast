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

import torch
import triton
import triton.language as tl

from flag_gems.fused.add_add_silu import add_add_silu as _common_add_add_silu
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _add_add_silu_kernel(
    addend,
    residual_a,
    residual_b,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(addend + offsets, mask=mask, other=0.0)
    value = value + tl.load(residual_a + offsets, mask=mask, other=0.0)
    value = value + tl.load(residual_b + offsets, mask=mask, other=0.0)
    value_fp32 = value.to(tl.float32)
    result = tl.fdiv(value_fp32, 1.0 + tl.exp(-value_fp32))
    tl.store(output + offsets, result, mask=mask)


def _use_flat_fp32_kernel(addend, residual_a, residual_b):
    tensors = (addend, residual_a, residual_b)
    return (
        all(isinstance(item, torch.Tensor) for item in tensors)
        and residual_a.shape == addend.shape
        and residual_b.shape == addend.shape
        and residual_a.device == addend.device
        and residual_b.device == addend.device
        and all(item.dtype == torch.float32 for item in tensors)
        and all(item.is_contiguous() for item in tensors)
        and not (
            torch.is_grad_enabled()
            and any(item.requires_grad for item in tensors)
        )
    )


def add_add_silu(addend, residual_a, residual_b):
    """Fuse the contiguous inference add/add/SiLU chain on Hygon."""

    logger.debug("GEMS_HYGON ADD_ADD_SILU")
    if not _use_flat_fp32_kernel(addend, residual_a, residual_b):
        return _common_add_add_silu(addend, residual_a, residual_b)

    output = torch.empty_like(addend)
    n_elements = addend.numel()
    if n_elements == 0:
        return output
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(addend.device):
        _add_add_silu_kernel[grid](
            addend,
            residual_a,
            residual_b,
            output,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    return output


__all__ = ["add_add_silu"]
