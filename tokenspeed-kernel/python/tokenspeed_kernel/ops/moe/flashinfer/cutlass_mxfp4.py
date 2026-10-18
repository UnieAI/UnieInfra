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
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe
    from flashinfer.fused_moe.core import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    _MXFP4_BLOCK_SIZE = 32

    def flashinfer_cutlass_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
        """Pack MoE expert weights into the flashinfer sm90 W4A16 MXFP4 layout.

        The stored expert weights use an OCP MXFP4 serialized checkpoint layout
        where ``w13_weight`` is ``[E, 2*N, K/2]`` uint8 (two e2m1 fp4 values per
        byte) and ``w13_weight_scale`` is ``[E, 2*N, K/32]`` uint8 e8m0. flashinfer
        ``cutlass_fused_moe`` with ``use_w4_group_scaling`` consumes W4A16
        weights interleaved in a fixed SM90 mixed-GEMM layout and with the gate
        and up halves ordered ``[up; gate]``.

        The layer gate/up rows are presented either concatenated
        (``[w1_all | w3_all]``, DeepSeek-V4) or interleaved (``[w1_0, w3_0, ...]``
        per intermediate unit, HF style). Both are converted here to the
        ``[up; gate]`` half order flashinfer expects, then byte/scale interleaved
        for the SM90 kernel. Weights are mutated in place so ``moe_apply`` reads
        the prepared tensors directly.
        """
        if getattr(w, "_cutlass_mxfp4_interleaved", False):
            return None

        names = (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        )
        if any(not hasattr(w, name) for name in names):
            raise ValueError("linear MXFP4 MoE weights are incomplete")
        tensors = tuple(getattr(w, name) for name in names)
        if any(t.dtype != torch.uint8 for t in tensors):
            raise TypeError("MXFP4 packed values and scales must use torch.uint8")
        if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
            raise ValueError("linear MXFP4 weights must be contiguous GPU tensors")
        if any(t.ndim != 3 for t in tensors):
            raise ValueError("linear MXFP4 weights must be rank-3")
        if len({t.shape[0] for t in tensors}) != 1:
            raise ValueError("linear MXFP4 weights must share an expert axis")

        num_experts, half_rows, kbytes = w.w13_weight.shape
        if half_rows % 2 != 0:
            raise ValueError("w13_weight must have an even gate/up row count")
        intermediate = half_rows // 2
        hidden = kbytes * 2
        if hidden % 128 != 0 or intermediate % 128 != 0:
            raise ValueError(
                "flashinfer SM90 MXFP4 cutlass needs hidden%128==0 and "
                f"intermediate%128==0, got hidden={hidden} intermediate={intermediate}"
            )

        layout = getattr(w, "w13_input_layout", "concatenated")
        if layout not in {"concatenated", "interleaved"}:
            raise ValueError(
                f"flashinfer SM90 MXFP4 MoE received an invalid gate/up layout: "
                f"{layout!r}"
            )

        def reorder_rows(x: torch.Tensor) -> torch.Tensor:
            """Return rows ordered ``[up; gate]`` for ``[E, 2*N, K]``."""
            if layout == "concatenated":
                first = x[:, :intermediate, :].clone()
                x[:, :intermediate, :] = x[:, intermediate:, :]
                x[:, intermediate:, :] = first
                return x
            gate = x[:, 0::2, :]
            up = x[:, 1::2, :]
            return torch.cat([up, gate], dim=1)

        w13 = reorder_rows(w.w13_weight.detach().contiguous()).contiguous()
        s13 = reorder_rows(w.w13_weight_scale.detach().contiguous()).contiguous()
        w2 = w.w2_weight.detach().contiguous()
        s2 = w.w2_weight_scale.detach().contiguous()

        w13 = interleave_moe_weights_for_sm90_mixed_gemm(w13, "fp4").contiguous()
        s13 = interleave_moe_scales_for_sm90_mixed_gemm(
            s13, group_size=_MXFP4_BLOCK_SIZE
        ).contiguous()
        w2 = interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4").contiguous()
        s2 = interleave_moe_scales_for_sm90_mixed_gemm(
            s2, group_size=_MXFP4_BLOCK_SIZE
        ).contiguous()

        w.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        w.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
        w.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        w.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)

        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None:
            # The cutlass gated activation runs post-dequant, so alpha/beta/limit
            # live in the actual-value domain (SwigluBias adaptor). DSv4 leaves
            # all None; only non-default values produce per-expert tensors.
            alpha = swiglu_arg.alpha if swiglu_arg.alpha is not None else 1.0
            beta = getattr(w, "swiglu_beta", None)

            def _per_expert(value: float) -> torch.nn.Parameter:
                return torch.nn.Parameter(
                    torch.full(
                        (num_experts,),
                        float(value),
                        dtype=torch.float32,
                        device=w.w13_weight.device,
                    ),
                    requires_grad=False,
                )

            if alpha != 1.0 or beta is not None:
                w.swiglu_alpha_t = _per_expert(alpha)
                if beta is not None:
                    w.swiglu_beta_t = _per_expert(beta)
            if swiglu_arg.limit is not None:
                w.swiglu_limit_t = _per_expert(swiglu_arg.limit)

        w._cutlass_mxfp4_interleaved = True
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_mxfp4_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(9, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "situ", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_mxfp4_moe_apply(
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
        """Apply MXFP4 experts with the flashinfer SM90 cutlass MoE kernel.

        Args:
            plan: MoE plan selecting the weight-only MXFP4 solution.
            x: Contiguous FP16/BF16 hidden states ``[tokens, hidden]``.
            w: Module holding prepared (interleaved) MXFP4 expert weights.
            router_logits: Unused; routing must be precomputed.
            topk_weights: Route weights ``[tokens, top_k]``.
            topk_ids: Expert ids ``[tokens, top_k]`` in the local expert space.
            num_tokens_global: Unused; distributed expert parallelism is
                unsupported.
            max_num_tokens_per_gpu: Unused token-capacity hint.
            do_finalize: Must be true (deferred finalize unsupported).
            enable_pdl: Unused launch hint.

        Returns:
            Finalized hidden states ``[tokens, hidden]`` with dtype matching
            ``x``, a per-rank partial sum the caller all-reduces across TP.
        """
        del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
        if not do_finalize:
            raise AssertionError("flashinfer SM90 MXFP4 cutlass cannot defer finalize")
        if topk_weights is None or topk_ids is None:
            raise ValueError("flashinfer SM90 MXFP4 cutlass requires precomputed top-k")
        if not getattr(w, "_cutlass_mxfp4_interleaved", False):
            raise RuntimeError(
                "flashinfer SM90 MXFP4 cutlass weights were not preprocessed"
            )

        output = torch.empty(
            x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device
        )
        swiglu_alpha = getattr(w, "swiglu_alpha_t", None)
        activation_type = (
            ActivationType.Swiglu
            if swiglu_alpha is None
            else ActivationType.SwigluBias
        )
        return cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.to(torch.float32),
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=[
                w.w13_weight_scale.view(torch.int32),
                w.w2_weight_scale.view(torch.int32),
            ],
            fc1_expert_biases=None,
            fc2_expert_biases=None,
            ep_size=1,
            ep_rank=0,
            tp_size=1,
            tp_rank=0,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            use_w4_group_scaling=True,
            use_mxfp8_act_scaling=False,
            activation_type=activation_type,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=getattr(w, "swiglu_beta_t", None),
            swiglu_limit=getattr(w, "swiglu_limit_t", None),
        )[0]