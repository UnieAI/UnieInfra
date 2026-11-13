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

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import (
    load_per_tensor_input_scale,
    make_weight_loader,
    round_up,
)
from tokenspeed.runtime.utils import set_weight_attrs

MXFP4_BLOCK = 32


def create_mxfp4_weight_pair(
    spec: MoELayerSpec,
    layer: nn.Module,
    *,
    with_bias: bool = False,
    solution: str | None = None,
) -> None:
    ispp = spec.intermediate_size // spec.tp_size
    platform = current_platform()
    hidden_size_padded = spec.hidden_size
    if platform.is_blackwell and solution == "flashinfer_trtllm":
        # trtllm-gen tiles are 128-granular (flashinfer TRTLLM-Gen SiTU MoE).
        ispp_padded = round_up(ispp, 128)
        hidden_size_padded = round_up(spec.hidden_size, 256)
    else:
        ispp_padded = (
            round_up(ispp, 64) if platform.is_blackwell else round_up(ispp, MXFP4_BLOCK)
        )

    # AITER pads the intermediate size to a 128 multiple and its scale shuffle
    # pads the w2 scale columns to a multiple of 8. Allocate those padded
    # buffers here and let the checkpoint load into unpadded views of them, so
    # its weight preprocessor pads and shuffles in place: building a second,
    # larger copy of every expert stack at load time strands freed device
    # memory (see tokenspeed_kernel/ops/moe/README.md, "AITER MXFP4 experts").
    if solution == "aiter":
        storage_ispp = round_up(ispp_padded, 128)
        w2_scale_cols = round_up(storage_ispp // MXFP4_BLOCK, 8)
    else:
        storage_ispp = ispp_padded
        w2_scale_cols = ispp_padded // MXFP4_BLOCK

    def padded_param(storage_shape: tuple[int, ...], shape: tuple[int, ...]):
        storage = torch.zeros(storage_shape, dtype=torch.uint8)
        if storage_shape != shape:
            storage = storage[tuple(slice(0, size) for size in shape)]
        return torch.nn.Parameter(storage, requires_grad=False)

    experts = spec.num_local_experts
    w13_weight = padded_param(
        (experts, 2 * storage_ispp, hidden_size_padded // 2),
        (experts, 2 * ispp_padded, hidden_size_padded // 2),
    )
    w13_weight_scale = padded_param(
        (experts, 2 * storage_ispp, hidden_size_padded // MXFP4_BLOCK),
        (experts, 2 * ispp_padded, hidden_size_padded // MXFP4_BLOCK),
    )
    w2_weight = padded_param(
        (experts, hidden_size_padded, storage_ispp // 2),
        (experts, hidden_size_padded, ispp_padded // 2),
    )
    w2_weight_scale = padded_param(
        (experts, hidden_size_padded, w2_scale_cols),
        (experts, hidden_size_padded, ispp_padded // MXFP4_BLOCK),
    )
    layer.register_parameter("w13_weight", w13_weight)
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight", w2_weight)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    weight_loader = make_weight_loader(spec)
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})

    if with_bias:
        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(spec.num_local_experts, 2 * ispp_padded, dtype=torch.bfloat16),
            requires_grad=False,
        )
        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(
                spec.num_local_experts, hidden_size_padded, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        bias_loader = make_weight_loader(spec, is_bias=True)
        set_weight_attrs(w13_weight_bias, {"weight_loader": bias_loader})
        set_weight_attrs(w2_weight_bias, {"weight_loader": bias_loader})


def create_mxfp4_fp8_input_scales(
    layer: nn.Module,
    num_local_experts: int,
) -> None:
    w13_input_scale = nn.Parameter(
        torch.zeros(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    w2_input_scale = nn.Parameter(
        torch.zeros(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_input_scale", w13_input_scale)
    layer.register_parameter("w2_input_scale", w2_input_scale)
    set_weight_attrs(w13_input_scale, {"weight_loader": load_per_tensor_input_scale})
    set_weight_attrs(w2_input_scale, {"weight_loader": load_per_tensor_input_scale})


__all__ = ["create_mxfp4_fp8_input_scales", "create_mxfp4_weight_pair"]
