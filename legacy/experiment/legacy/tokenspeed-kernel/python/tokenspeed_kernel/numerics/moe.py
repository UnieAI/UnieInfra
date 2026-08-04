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

"""Numerics framework hooks for MoE family ops.

The ``align_block_size`` mode covers the trtllm ``moe_align_block_size``
op (the per-block expert dispatch helper). Its raw output is three int32
tensors plus an undefined per-expert order within each block (CUDA
atomics make ``sorted_ids`` non-deterministic). We canonicalize the
output to a single int32 tensor so ``compare_outputs`` works as-is.
"""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance


def tolerance(dtype: torch.dtype, **_: Any) -> Tolerance:
    # int32 outputs — both implementations must match exactly.
    return Tolerance(atol=0.0, rtol=0.0)


set_family_tolerance("moe", tolerance)


class MoeAlignBlockSizeInputGenerator(InputGenerator):
    """Generates topk_ids for moe_align_block_size.

    Shape kwargs:
        total_tokens: number of input tokens
        top_k:        number of experts each token routes to
        num_experts:  expert pool size
        block_size:   tile width the dispatch packs into
    """

    def generate(
        self,
        *,
        total_tokens: int,
        top_k: int,
        num_experts: int,
        block_size: int,
    ) -> dict[str, Any]:
        topk_ids = torch.randint(
            low=0,
            high=num_experts,
            size=(total_tokens, top_k),
            device=self.device,
            dtype=torch.int32,
            generator=self.rng,
        )
        return {
            "topk_ids": topk_ids,
            "block_size": block_size,
            "num_experts": num_experts,
        }


set_input_generator("moe", "align_block_size", MoeAlignBlockSizeInputGenerator)


_MOE_ALIGN_STANDARD_SHAPES: list[dict[str, int]] = [
    # DSv3 routed MoE: 256 experts, top-8.
    {"total_tokens": 16, "top_k": 8, "num_experts": 256, "block_size": 64},
    {"total_tokens": 128, "top_k": 8, "num_experts": 256, "block_size": 64},
    # MiniMax / Kimi 8-expert routing.
    {"total_tokens": 16, "top_k": 8, "num_experts": 64, "block_size": 64},
    {"total_tokens": 128, "top_k": 8, "num_experts": 64, "block_size": 128},
    # Decode-shape (single token, multiple experts).
    {"total_tokens": 1, "top_k": 8, "num_experts": 64, "block_size": 64},
]

set_standard_shapes("moe", "align_block_size", _MOE_ALIGN_STANDARD_SHAPES)
set_benchmark_shapes("moe", "align_block_size", _MOE_ALIGN_STANDARD_SHAPES)


def compute_align_block_size_buffer_dims(
    pad_id: int, num_experts: int, block_size: int
) -> tuple[int, int]:
    """Output buffer dims for moe_align_block_size, block-aligned.

    Returns ``(num_blocks, sorted_ids_size)`` where
    ``sorted_ids_size == num_blocks * block_size`` so the canonical reshape
    can use ``view(num_blocks, block_size)`` without padding.
    """
    max_num_tokens_padded = pad_id + (num_experts + 1) * (block_size - 1)
    num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    return num_blocks, num_blocks * block_size


def canonicalize_align_block_size(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Pack the three moe_align_block_size outputs into a single int32 tensor.

    Within each block the ``sorted_ids`` slot order is non-deterministic
    (CUDA atomics in the trtllm impl), so we sort each block before
    concatenating. The set of token IDs assigned to each block is what
    matters for downstream MoE GEMM correctness.

    Caller must size ``sorted_ids`` to ``expert_ids.numel() * block_size``.
    """
    n_blocks = expert_ids.numel()
    assert sorted_ids.numel() == n_blocks * block_size, (
        f"sorted_ids size {sorted_ids.numel()} doesn't match "
        f"expert_ids.numel()={n_blocks} * block_size={block_size}"
    )
    blocks = sorted_ids.view(n_blocks, block_size)
    blocks_sorted, _ = blocks.sort(dim=-1)
    return torch.cat(
        [
            num_tokens_post_pad.flatten().to(torch.int32),
            expert_ids.to(torch.int32),
            blocks_sorted.flatten().to(torch.int32),
        ]
    )
