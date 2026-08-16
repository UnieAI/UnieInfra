# Copyright (c) 2026 UnieAI
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

"""Block-scale FP8 MoE on DeepGEMM grouped GEMMs, without DeepEP.

``deepep_fp8`` is the only DeepGEMM MoE path in this tree, and it declares
``supports_all_to_all_ep`` and fails with "DeepEP MoE plan is missing
deepep_group" whenever a plan carries no process group. That makes DeepGEMM
unreachable for a single-box tensor-parallel deployment, which is how
Qwen3.5-122B-A10B-FP8 is served on two cards -- and DeepGEMM is what vLLM
actually selects for that model on this hardware (`Using DEEPGEMM Fp8 MoE
backend`), so the gap is plumbing, not kernels.

The compute between DeepEP's ``dispatch_b`` and ``combine_a`` is already
collective-free: it takes FP8 tokens, their block scales, routing ids and
weights, and per-expert row counts, and returns one bf16 row per token. This
module supplies those five inputs locally -- quantize, histogram, align -- and
reuses the same permute/GEMM/unpermute core, so both paths share the numerics
and the weight preparation rather than growing a second copy of them.

What it deliberately does not do: no capacity ceiling, no masked GEMM, no
low-latency mode. Those exist in ``deepep_fp8`` to bound cross-rank traffic,
which is not a cost here.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import (
        get_mn_major_tma_aligned_tensor,
        m_grouped_fp8_gemm_nt_contiguous,
    )
except ImportError:  # pragma: no cover - DeepGEMM is an optional dependency
    get_mn_major_tma_aligned_tensor = None
    m_grouped_fp8_gemm_nt_contiguous = None


if platform.is_nvidia and m_grouped_fp8_gemm_nt_contiguous is not None:
    from tokenspeed_kernel.ops.activation.triton import fused_swiglu_fp8_ue8m0
    from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
    from tokenspeed_kernel.ops.moe.deep_gemm.deepep_fp8 import (
        _configure_deep_gemm_pdl,
        _prepare_routing_tensors,
        deep_gemm_deepep_fp8_moe_weights,
    )
    from tokenspeed_kernel.ops.moe.deep_gemm.ue8m0 import (
        deep_gemm_requires_ue8m0,
        per_token_group_quant_fp8_ue8m0,
    )
    from tokenspeed_kernel.ops.moe.triton.deepep_permute import (
        deepep_gather,
        deepep_scatter,
    )
    from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

    _FP8_BLOCK = 128
    _UE8M0_PACKED_RECIPE = (1, 1, _FP8_BLOCK)

    def _aligned_expert_counts(topk_ids: torch.Tensor, num_experts: int) -> list[int]:
        """Rows each expert block needs, padded to the grouped-GEMM tile.

        DeepEP returns these from dispatch; without it they come from a
        histogram of the routing ids. ``deepep_scatter`` requires every count to
        be a multiple of the alignment so no GEMM tile straddles two experts,
        and requires them host-side, so this costs one device sync per layer --
        the same sync ``dispatch_b`` would have imposed.

        Experts nothing routed to keep a count of zero rather than a padded
        block, which is what keeps a single decode token at 8 x 128 rows
        instead of 256 x 128.
        """
        counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
        counts = counts[:num_experts]
        padded = ((counts + (_FP8_BLOCK - 1)) // _FP8_BLOCK) * _FP8_BLOCK
        return padded.tolist()

    @register_kernel(
        "moe",
        "apply",
        name="deep_gemm_contiguous_fp8_moe_apply",
        solution="deep_gemm",
        weight_preprocessor=deep_gemm_deepep_fp8_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            # Expert parallelism is fine -- each rank owns a slice of the expert
            # dimension and runs this over its own slice. What is not supported
            # is all-to-all EP, which is exactly the distinction that makes this
            # reachable where ``deepep_fp8`` is not.
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({_FP8_BLOCK}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(_FP8_BLOCK, _FP8_BLOCK)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_contiguous_fp8_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
    ):
        """Run one FP8 MoE layer with DeepGEMM grouped GEMMs, no collectives.

        Args:
            plan: Per-layer scratch dict. Unused here; accepted so the signature
                matches the other ``moe.apply`` solutions.
            x: ``[tokens, hidden]`` bf16 activations.
            w: Layer weights: ``w13_weight`` ``[E, 2*ispp, hidden]``,
                ``w2_weight`` ``[E, hidden, ispp]`` and their block scales,
                already prepared by the shared weight preprocessor.
            router_logits: ``[tokens, E]``; only read when top-k is not supplied.
            topk_weights: Precomputed routing weights, ``[tokens, top_k]``.
            topk_ids: Precomputed expert ids, ``[tokens, top_k]``.
            num_tokens_global: Unused; no cross-rank sizing to do.
            max_num_tokens_per_gpu: Unused; there is no capacity buffer.
            do_finalize: Unused; this path always reduces before returning.
            enable_pdl: Join a Programmatic Dependent Launch chain.

        Returns:
            ``[tokens, hidden]`` bf16, top-k weighted and reduced.
        """
        _configure_deep_gemm_pdl(enable_pdl)

        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights, topk_ids = _prepare_routing_tensors(topk_weights, topk_ids)

        num_experts = w.w13_weight.shape[0]
        hidden_size = x.shape[1]
        requires_ue8m0 = deep_gemm_requires_ue8m0()

        # deepep_scatter consumes [tokens, groups] float32 scales, which is the
        # layout dispatch would have delivered -- not the packed-int32 TMA one.
        if requires_ue8m0:
            x_fp8, x_scales = per_token_group_quant_fp8_ue8m0(x, _FP8_BLOCK)
        else:
            x_fp8, x_scales = per_token_group_quant_fp8(x, _FP8_BLOCK)

        counts = _aligned_expert_counts(topk_ids, num_experts)
        gemm_x, gemm_scales, m_indices, dest_index = deepep_scatter(
            x_fp8,
            x_scales,
            topk_ids,
            counts,
            expert_alignment=_FP8_BLOCK,
            pack_ue8m0_scales=requires_ue8m0,
        )

        out = torch.zeros(
            (x.shape[0], hidden_size), dtype=torch.bfloat16, device=x.device
        )
        total_rows = gemm_x.shape[0]
        if not total_rows:
            return out

        # GEMM1: [total_rows, hidden] @ [E, 2*ispp, hidden]^T
        gateup = torch.empty(
            (total_rows, w.w13_weight.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        gemm1_scales = (
            gemm_scales
            if requires_ue8m0
            else get_mn_major_tma_aligned_tensor(gemm_scales)
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (gemm_x, gemm1_scales),
            (w.w13_weight, w.w13_weight_scale_inv),
            gateup,
            m_indices,
            recipe=_UE8M0_PACKED_RECIPE if requires_ue8m0 else None,
        )

        # Fused SiLU(gate)*up + FP8 block quantize. Padding rows inside an
        # expert block are quantized like any other row and never gathered back.
        ispp = w.w2_weight.shape[-1]
        if requires_ue8m0:
            down_in, gemm2_scales = fused_swiglu_fp8_ue8m0(
                gateup, enable_pdl=enable_pdl
            )
        else:
            down_in = torch.empty(
                (total_rows, ispp), dtype=torch.float8_e4m3fn, device=x.device
            )
            # DeepGEMM wants mn-major scales, hence the [blocks, M] allocation
            # viewed as [M, blocks].
            down_scales = torch.zeros(
                (ispp // _FP8_BLOCK, total_rows),
                dtype=torch.float32,
                device=x.device,
            ).permute(1, 0)
            silu_and_mul_fuse_block_quant(
                gateup, down_scales, down_in, enable_pdl=enable_pdl
            )
            gemm2_scales = get_mn_major_tma_aligned_tensor(down_scales)

        # GEMM2: [total_rows, ispp] @ [E, hidden, ispp]^T
        expert_out = torch.empty(
            (total_rows, hidden_size), dtype=torch.bfloat16, device=x.device
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (down_in, gemm2_scales),
            (w.w2_weight, w.w2_weight_scale_inv),
            expert_out,
            m_indices,
            recipe=_UE8M0_PACKED_RECIPE if requires_ue8m0 else None,
        )

        # Unpermute and apply the routing weights in one pass. With DeepEP this
        # had to happen before combine because the normal combine leg reduces
        # unweighted; here it is simply the reduction.
        deepep_gather(expert_out, topk_ids, topk_weights, dest_index, out=out)
        return out
