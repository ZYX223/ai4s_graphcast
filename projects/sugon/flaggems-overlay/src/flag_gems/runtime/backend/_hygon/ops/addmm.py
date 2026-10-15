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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.matmul_precision import should_use_fast_float32_matmul
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def classify_hygon_addmm(K, stride_bk, stride_bn):
    if K < 16:
        return "skinny_k"
    if K < 256:
        return "k184"
    if K == 512 and stride_bn == 1:
        return "k512_rhs_row"
    if K == 512 and stride_bk == 1:
        return "k512_rhs_k_contiguous"
    if K >= 1024:
        return "k1024_plus"
    return "general"


def select_hygon_input_precision(kernel_class, fp32_operands, fast_enabled):
    # DTK does not expose a hardware TF32 dot mode through Triton.  FP16 has
    # the same ten explicit significand bits as TF32, so lower only the two
    # K=512 GraphCast classes after the caller explicitly enables fast FP32.
    # Accumulation and output remain FP32; strict/default execution is unchanged.
    if (
        fast_enabled
        and fp32_operands
        and kernel_class in ("k512_rhs_row", "k512_rhs_k_contiguous")
    ):
        return "fp16"
    return "ieee"


_KERNEL_CLASS_IDS = {
    "skinny_k": 0,
    "k184": 1,
    "k512_rhs_row": 2,
    "k512_rhs_k_contiguous": 3,
    "k1024_plus": 4,
    "general": 5,
}


_SKINNY_K_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16},
        num_warps=8,
        num_stages=1,
    ),
]


@triton.jit
def _accumulate_dot(
    accumulator,
    a,
    b,
    IS_FP64: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    if IS_FP64:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    return accumulator + tl.dot(a, b, allow_tf32=ALLOW_TF32)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K", "stride_am", "stride_bk", "KERNEL_CLASS", "INPUT_PRECISION"],
    strategy=[
        "align32",
        "align32",
        "align32",
        "align32",
        "align32",
        "default",
        "default",
    ],
    warmup=5,
    rep=10,
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    HAS_K: tl.constexpr,
    IS_FP64: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    KERNEL_CLASS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_K:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m < M)[:, None] & (offs_k < k_remaining)[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k < k_remaining)[:, None] & (offs_n < N)[None, :],
                other=0.0,
            )
            if INPUT_PRECISION == "fp16":
                a = a.to(tl.float16)
                b = b.to(tl.float16)
            accumulator = _accumulate_dot(
                accumulator,
                a,
                b,
                IS_FP64,
                ALLOW_TF32,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if BETA_IS_ZERO:
        result = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(
                i_ptr + offs_n * stride_in,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            i_ptrs = (
                i_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
            )
            bias = tl.load(i_ptrs, mask=mask, other=0.0)
        result = accumulator * alpha + bias.to(accumulator.dtype) * beta
    if FUSE_SILU:
        value = result.to(c_ptr.dtype.element_ty).to(tl.float32)
        result = value / (1.0 + tl.exp(-value))
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_SKINNY_K_CONFIGS,
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_skinny_k_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    IS_FP64: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=(offs_m < M)[:, None] & (offs_k < K)[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
        other=0.0,
    )
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = _accumulate_dot(
        accumulator,
        a,
        b,
        IS_FP64,
        ALLOW_TF32,
    )

    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if BETA_IS_ZERO:
        result = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(
                i_ptr + offs_n * stride_in,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            i_ptrs = (
                i_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
            )
            bias = tl.load(i_ptrs, mask=mask, other=0.0)
        result = accumulator * alpha + bias.to(accumulator.dtype) * beta
    if FUSE_SILU:
        value = result.to(c_ptr.dtype.element_ty).to(tl.float32)
        result = value / (1.0 + tl.exp(-value))
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result.to(c_ptr.dtype.element_ty), mask=mask)


def _addmm_impl(bias, mat1, mat2, out, beta, alpha, *, fuse_silu=False):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"

    bias_is_vector = bias.ndim == 1 and bias.shape[0] == N
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        bias_stride_m = 0
        bias_stride_n = bias.stride(0)
    elif bias_is_scalar:
        bias_stride_m = 0
        bias_stride_n = 0
    else:
        bias = bias.broadcast_to(out.shape)
        bias_stride_m = bias.stride(0)
        bias_stride_n = bias.stride(1)

    kernel_class = classify_hygon_addmm(K, mat2.stride(0), mat2.stride(1))
    input_precision = select_hygon_input_precision(
        kernel_class,
        mat1.dtype == torch.float32 and mat2.dtype == torch.float32,
        should_use_fast_float32_matmul("hygon", mat1, mat2),
    )
    if kernel_class == "skinny_k":
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        with torch_device_fn.device(mat1.device):
            addmm_skinny_k_kernel[grid](
                mat1,
                mat2,
                bias,
                out,
                alpha,
                beta,
                M,
                N,
                K,
                mat1.stride(0),
                mat1.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                bias_stride_m,
                bias_stride_n,
                out.stride(0),
                out.stride(1),
                BIAS_IS_VECTOR=bias_is_vector,
                BIAS_IS_SCALAR=bias_is_scalar,
                BETA_IS_ZERO=beta == 0,
                FUSE_SILU=fuse_silu,
                IS_FP64=mat1.dtype == torch.float64,
                ALLOW_TF32=input_precision == "tf32",
                GROUP_M=8,
            )
        return out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias_stride_m,
            bias_stride_n,
            out.stride(0),
            out.stride(1),
            GROUP_M=8,
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
            BETA_IS_ZERO=beta == 0,
            FUSE_SILU=fuse_silu,
            HAS_K=K > 0,
            IS_FP64=mat1.dtype == torch.float64,
            ALLOW_TF32=input_precision == "tf32",
            KERNEL_CLASS=_KERNEL_CLASS_IDS[kernel_class],
            INPUT_PRECISION=input_precision,
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_HYGON ADDMM")
    return _addmm_impl(bias, mat1, mat2, None, beta, alpha)


def addmm_silu(bias, mat1, mat2, *, beta=1, alpha=1):
    """Compute SiLU(AddMM) in the Hygon-owned Triton GEMM epilogue."""

    logger.debug("GEMS_HYGON ADDMM_SILU")
    return _addmm_impl(
        bias,
        mat1,
        mat2,
        None,
        beta,
        alpha,
        fuse_silu=True,
    )


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_HYGON ADDMM_OUT")
    return _addmm_impl(bias, mat1, mat2, out, beta, alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_HYGON ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(
        bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out
    )


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_HYGON ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            "mat1 and mat2 must have the same dtype, but got "
            f"{mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    bias_c = bias.to(out_dtype)
    return addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)
