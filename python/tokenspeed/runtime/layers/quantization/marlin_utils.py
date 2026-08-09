# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Marlin weight-layout helpers for the wNa16 (w4a16 / w8a16) path.

Adapted 1:1 from SGLang's ``sglang/srt/layers/quantization/marlin_utils.py``
(Apache-2.0), which in turn derives from vLLM. Only the subset that
``compressed_tensors_wNa16`` needs is carried here; the following are
deliberately **not** ported:

- ``MarlinConfig`` / ``MarlinLinearMethod`` — the standalone ``marlin``
  quantization config. The compressed-tensors path does not use it.
- ``unified_apply_gptq_marlin_gemm*`` and the ``get_tc_piecewise_forward_context``
  branch — SGLang's ``torch.compile`` piecewise-CUDA-graph integration. This
  engine uses breakable CUDA graphs
  (:mod:`tokenspeed.runtime.execution.breakable_cuda_graph`) instead, so the
  direct-call branch is the whole story here.
- AWQ and MoE variants — no caller yet.

The GEMM itself lives behind
``tokenspeed_kernel.ops.quantization.cuda.gptq_marlin_gemm``.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy
import torch
from tokenspeed_kernel.ops.quantization.cuda import gptq_marlin_gemm

from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
    ScalarType,
    scalar_types,
)
from tokenspeed.runtime.layers.quantization.utils import pack_cols

logger = logging.getLogger(__name__)

__all__ = [
    "MarlinLinearLayerConfig",
    "apply_gptq_marlin_linear",
    "check_marlin_supported",
    "check_marlin_supports_shape",
    "marlin_is_k_full",
    "marlin_make_empty_g_idx",
    "marlin_make_empty_zp",
    "marlin_make_workspace",
    "marlin_permute_scales",
    "marlin_repeat_scales_on_all_ranks",
    "marlin_sort_g_idx",
    "marlin_zero_points",
    "query_marlin_supported_quant_types",
    "verify_marlin_supported",
    "verify_marlin_supports_shape",
]

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
# NVFP4 supports 16; MXFP4 supports 32 and 16.
FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16, 32]

# Setting this to False lets Marlin do global reductions in fp16 instead of
# fp32, saving some memory movement at the cost of precision.
USE_FP32_REDUCE_DEFAULT = True

# Marlin's experimental atomicAdd reduction. Off by default, matching SGLang and
# vLLM (which gate it behind VLLM_MARLIN_USE_ATOMIC_ADD).
USE_ATOMIC_ADD_DEFAULT = False


def _device_capability() -> int:
    """Return ``major * 10 + minor`` for the current device, or -1 if no CUDA."""
    if not torch.cuda.is_available():
        return -1
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@dataclass
class MarlinLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool


def query_marlin_supported_quant_types(
    has_zp: bool | None = None,
    include_fp_type: bool = True,
    device_capability: int | None = None,
):
    """Quant types Marlin supports on this device.

    ``has_zp`` True selects the AWQ-style types (unsigned + runtime zero-point),
    False the GPTQ-style ones (unsigned + symmetric bias), None both.
    """
    if device_capability is None:
        device_capability = _device_capability()

    if device_capability < 80:
        return []

    if has_zp is None:
        return query_marlin_supported_quant_types(
            False, include_fp_type, device_capability
        ) + query_marlin_supported_quant_types(True, include_fp_type, device_capability)

    if has_zp:
        return [scalar_types.uint4]

    res = [scalar_types.uint4b8, scalar_types.uint8b128]
    if include_fp_type:
        res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
    return res


def _check_marlin_supported(
    quant_type: ScalarType,
    group_size: int | None,
    has_zp: bool,
    device_capability: int | None = None,
) -> tuple[bool, str | None]:
    if device_capability is None:
        device_capability = _device_capability()

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability
    )

    if quant_type not in supported_types:
        return (
            False,
            f"Marlin does not support weight_bits = {quant_type}. "
            f"Only types = {supported_types} are supported (for group_size = "
            f"{group_size}, device_capability = {device_capability}, zp = {has_zp}).",
        )

    if quant_type == scalar_types.float4_e2m1f:
        allowed_group_sizes = FP4_MARLIN_SUPPORTED_GROUP_SIZES
    else:
        allowed_group_sizes = MARLIN_SUPPORTED_GROUP_SIZES

    if group_size is None or group_size not in allowed_group_sizes:
        return (
            False,
            f"Marlin does not support group_size = {group_size} for quant_type = "
            f"{quant_type}. Only group_sizes = {allowed_group_sizes} are supported.",
        )

    return True, None


def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: int | None = None,
) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)
    return cond


def verify_marlin_supported(
    quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        raise ValueError(err_msg)


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = {output_size_per_partition} is not "
            f"divisible by min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. Consider "
            "reducing tensor_parallel_size or running with --quantization gptq."
        )

    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} is not "
            f"divisible by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. Consider "
            "reducing tensor_parallel_size or running with --quantization gptq."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} is not "
            f"divisible by group_size = {group_size}. Consider reducing "
            "tensor_parallel_size or running with --quantization gptq."
        )


def check_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> tuple[bool, str | None]:
    try:
        verify_marlin_supports_shape(
            output_size_per_partition, input_size_per_partition, input_size, group_size
        )
    except ValueError as e:
        return False, str(e)
    return True, None


def marlin_make_workspace(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    """Workspace sized by threadblock count (SM count x blocks per SM)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool:
    """Scales must be repeated on every rank under act-ordering, and under
    channelwise quantization on a RowParallelLinear."""
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_make_empty_zp(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Permute zero-points like scales, but never with the "single" permutation
    — zero-points are applied on every MMA."""
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave the column dim for the dequantize code, then pack to int32.
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    return pack_cols(zp, num_bits, size_k, size_n)


@functools.cache
def _warn_marlin_bf16_before_sm90() -> None:
    logger.info(
        "Running the Marlin kernel with bf16 on a GPU before SM90. Consider fp16 "
        "for better performance if the model allows it."
    )


def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    """atomicAdd beats a global reduce only when m*n is small and k is large."""
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    if not USE_ATOMIC_ADD_DEFAULT:
        return False

    # sm8x has no native atomicAdd for bfloat16.
    if torch.cuda.get_device_capability(device)[0] < 9 and dtype == torch.bfloat16:
        if not torch.compiler.is_dynamo_compiling():
            _warn_marlin_bf16_before_sm90()
        return False

    return True


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    output = gptq_marlin_gemm(
        reshaped_x,
        None,
        weight,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    if bias is not None:
        output.add_(bias)  # in-place

    return output.reshape(out_shape)
