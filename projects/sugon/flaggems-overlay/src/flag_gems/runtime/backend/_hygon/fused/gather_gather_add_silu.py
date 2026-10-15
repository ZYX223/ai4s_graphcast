# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._hygon.ops.index_add import (
    prepare_index_for_triton_kernel,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _gather_gather_add_silu_kernel(
    hidden,
    sender_projection,
    senders,
    receiver_projection,
    receivers,
    output,
    total_rows,
    edge_count,
    sender_count,
    receiver_count,
    feature_count,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = ext.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < total_rows
    mask = row_mask & (cols < feature_count)
    edge = rows % edge_count
    prefix = rows // edge_count
    sender = tl.load(senders + edge, mask=row_mask, other=0).to(tl.int64)
    receiver = tl.load(receivers + edge, mask=row_mask, other=0).to(tl.int64)

    hidden_offsets = rows * feature_count + cols
    sender_offsets = (
        (prefix * sender_count + sender) * feature_count + cols
    )
    receiver_offsets = (
        (prefix * receiver_count + receiver) * feature_count + cols
    )
    value = tl.load(hidden + hidden_offsets, mask=mask, other=0.0)
    value += tl.load(
        sender_projection + sender_offsets,
        mask=mask,
        other=0.0,
    )
    value += tl.load(
        receiver_projection + receiver_offsets,
        mask=mask,
        other=0.0,
    )
    value = value.to(tl.float32)
    activated = value / (1.0 + tl.exp(-value))
    tl.store(output + hidden_offsets, activated, mask=mask)


def gather_gather_add_silu(
    hidden,
    sender_projection,
    senders,
    receiver_projection,
    receivers,
):
    """Fuse two graph gathers, two ordered adds, and SiLU in Triton."""

    logger.debug("GEMS_HYGON GATHER_GATHER_ADD_SILU")
    tensors = (hidden, sender_projection, receiver_projection)
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        raise TypeError("hidden and projection inputs must be tensors")
    if hidden.ndim < 2 or any(item.ndim != hidden.ndim for item in tensors[1:]):
        raise ValueError("hidden and projections must have the same rank >= 2")
    if hidden.dtype != torch.float32 or any(
        item.dtype != hidden.dtype for item in tensors[1:]
    ):
        raise TypeError("Hygon gather fusion currently requires float32 inputs")
    if not all(item.device == hidden.device for item in tensors[1:]):
        raise ValueError("hidden and projections must be on one device")
    if not all(item.is_contiguous() for item in tensors):
        raise ValueError("hidden and projections must be contiguous")
    if hidden.shape[:-2] != sender_projection.shape[:-2] or (
        hidden.shape[:-2] != receiver_projection.shape[:-2]
    ):
        raise ValueError("hidden and projections must have matching prefix shapes")
    feature_count = hidden.shape[-1]
    if (
        sender_projection.shape[-1] != feature_count
        or receiver_projection.shape[-1] != feature_count
    ):
        raise ValueError("hidden and projections must share the feature width")
    edge_count = hidden.shape[-2]
    if senders.ndim != 1 or receivers.ndim != 1 or (
        senders.numel() != edge_count or receivers.numel() != edge_count
    ):
        raise ValueError("sender and receiver indices must match the edge count")
    if torch.is_grad_enabled() and any(item.requires_grad for item in tensors):
        raise RuntimeError("Hygon gather fusion is inference-only")

    senders = prepare_index_for_triton_kernel(
        senders, sender_projection.shape[-2]
    )
    receivers = prepare_index_for_triton_kernel(
        receivers, receiver_projection.shape[-2]
    )
    output = torch.empty_like(hidden)
    if hidden.numel() == 0:
        return output

    total_rows = hidden.numel() // feature_count
    block_m = 4
    block_n = min(256, triton.next_power_of_2(feature_count))
    grid = (
        triton.cdiv(total_rows, block_m),
        triton.cdiv(feature_count, block_n),
    )
    with torch_device_fn.device(hidden.device):
        _gather_gather_add_silu_kernel[grid](
            hidden,
            sender_projection,
            senders,
            receiver_projection,
            receivers,
            output,
            total_rows,
            edge_count,
            sender_projection.shape[-2],
            receiver_projection.shape[-2],
            feature_count,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return output


__all__ = ["gather_gather_add_silu"]
