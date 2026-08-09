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

"""Unquantized (bf16/fp16) MoE apply on the ragged Triton grouped matmul.

Why this exists: the only unquantized MoE kernel registered for NVIDIA was
`flashinfer_cutlass_unquant_moe_apply`, whose cost is flat in token count --
measured 3.18 ms per layer at both 1 and 4096 tokens on Qwen3.6-35B-A3B shapes
(256 experts, top-8, hidden 2048, moe_intermediate 512), against vLLM's
`fused_experts` at 0.15 ms (1 token) to 1.53 ms (4096). A grouped GEMM that pays
for all 256 experts regardless of how many are routed to loses badly whenever
the batch is small.

The ragged path this module uses is the same one the fp8/mxfp4 kernels here
already run (`triton_kernels.matmul` with ragged metadata + gather/scatter
indices): tokens are grouped by expert, so work scales with the number of
tokens actually routed. Nothing new is written -- this is the existing kernel
registered for an unquantized weight dtype.

Registered alongside the cutlass kernel, not in place of it: selection decides.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from tokenspeed_kernel.ops.moe.triton.mxfp4 import (  # noqa: E402
    _local_topk_for_ep,
    _routing,
    _routing_from_topk,
    _silu_gate_up,
)
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from triton_kernels.matmul import PrecisionConfig, matmul  # noqa: E402
from triton_kernels.matmul_details.opt_flags import (  # noqa: E402
    scoped_opt_flags_constraints,
)
from triton_kernels.tensor import wrap_torch_tensor  # noqa: E402


@contextmanager
def _smem_guard(which: str = "all"):
    """Cap the tile so the chosen config fits sm_12x shared memory.

    Left unconstrained, the autoselected tile asks for 115004 bytes of shared
    memory on these shapes and the launch fails with `OutOfResources` (limit is
    101376 on sm_120); a smaller-but-still-too-large config instead trips an
    internal ptxas codegen error at 1024 tokens. The mxfp4 path here carries the
    same kind of guard for CDNA4 -- this is that guard for sm_12x.
    """
    if scoped_opt_flags_constraints is None or not current_platform().is_nvidia:
        yield
        return
    if current_platform().arch_version.major != 12:
        yield
        return
    import os

    off = os.getenv("TS_MOE_GUARD_OFF", "")
    if off and (off == "all" or which in off.split(",")):
        yield
        return
    bm = int(os.getenv("TS_MOE_BM", "64"))
    bn = int(os.getenv("TS_MOE_BN", "128"))
    bk = int(os.getenv("TS_MOE_BK", "128"))
    with scoped_opt_flags_constraints({"block_m": bm, "block_n": bn, "block_k": bk}):
        yield


def triton_bf16_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Present the expert weights to the ragged matmul.

    `matmul` wants B as ``[experts, K, N]``; the module stores
    ``w13 [E, 2I, H]`` and ``w2 [E, H, I]``, so both need a transpose of the
    last two dims. The transpose is a view -- the original parameters stay
    alive and are deliberately not released, unlike the quantized paths which
    materialise a repacked copy.
    """
    # Wrap in triton_kernels' own Tensor type, as the quantized paths here do.
    # A raw torch tensor reaches a different, less-exercised branch of the
    # kernel; on sm_120 that branch fails ptxas codegen for any batch above one
    # token, at every tile size tried.
    w.w13_weight_triton_tensor = wrap_torch_tensor(
        w.w13_weight.transpose(-2, -1).contiguous()
    )
    w.w2_weight_triton_tensor = wrap_torch_tensor(
        w.w2_weight.transpose(-2, -1).contiguous()
    )
    w.w13_precision_config = PrecisionConfig(out_dtype=torch.bfloat16)
    w.w2_precision_config = PrecisionConfig(out_dtype=torch.bfloat16)


@register_kernel(
    "moe",
    "apply",
    name="triton_bf16_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_bf16_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_bf16_moe_apply(
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
) -> torch.Tensor:
    if topk_weights is None or topk_ids is None:
        raise RuntimeError("triton bf16 MoE requires precomputed topk_weights/topk_ids")
    if topk_weights.shape != topk_ids.shape:
        raise RuntimeError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if getattr(w, "w13_weight_bias", None) is not None or (
        getattr(w, "w2_weight_bias", None) is not None
    ):
        raise RuntimeError("triton bf16 MoE does not support bias")

    top_k = getattr(w, "top_k", topk_ids.shape[1])
    n_tokens = x.shape[0]

    topk_weights, topk_ids, num_experts = _local_topk_for_ep(topk_weights, topk_ids, w)
    ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=router_logits.dtype,
    )
    return _bf16_moe_body(
        x, w, ragged_metadata, gather_indx, scatter_indx, gate_scal, top_k, n_tokens
    )


def _bf16_moe_body(
    x, w, ragged_metadata, gather_indx, scatter_indx, gate_scal, top_k, n_tokens
):

    with _smem_guard("w13"):
        intermediate = matmul(
            x,
            w.w13_weight_triton_tensor,
            None,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            precision_config=w.w13_precision_config,
        )
    # w13 is the concatenation [gate, up] on its output dim, which is what
    # `_silu_gate_up` chunks on.
    intermediate = _silu_gate_up(intermediate, output_dtype=x.dtype)

    with _smem_guard("w2"):
        out_sorted = matmul(
            intermediate,
            w.w2_weight_triton_tensor,
            None,
            a_ragged_metadata=ragged_metadata,
            precision_config=w.w2_precision_config,
        )

    # The gate scaling and the un-sort are done here rather than handed to the
    # kernel as `gammas` / `scatter_indx`. That fused path emits the TMA
    # instruction `.tile::scatter4`, which ptxas rejects for sm_120a
    # ("Feature '.tile::scatter4' not supported on .target 'sm_120a'") for any
    # batch beyond a single token, at every tile size. Doing it in torch costs
    # one extra pass over [tokens * top_k, hidden].
    out_sorted = out_sorted * gate_scal.to(out_sorted.dtype).unsqueeze(-1)
    # `scatter_indx` is the argsort permutation: sorted[i] == flat[perm[i]].
    out_flat = torch.empty_like(out_sorted)
    out_flat[scatter_indx.long()] = out_sorted
    if top_k > 1:
        return out_flat.view(n_tokens, top_k, out_flat.shape[-1]).sum(dim=1)
    return out_flat


@register_kernel(
    "moe",
    "apply",
    name="triton_bf16_routed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_bf16_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"kernel_routing"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_bf16_routed_moe_apply(
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
) -> torch.Tensor:
    """Same MoE, but routing done in the kernel instead of in torch.

    The precomputed-topk sibling above spends 0.249 ms on
    `_routing_from_topk` at one token -- an argsort plus a scatter_add plus the
    segment metadata, all as separate torch ops. That is more than vLLM spends
    on its entire MoE (0.152 ms), because vLLM sorts tokens by expert in one
    fused CUDA kernel.

    `triton_kernels.topk` already computes the same sorted layout as part of
    selecting the experts, so when the caller can hand over logits rather than
    a precomputed top-k there is no reason to redo it in torch.
    """
    if getattr(w, "ep_size", 1) > 1:
        raise RuntimeError("triton bf16 routed MoE does not support EP")
    if getattr(w, "w13_weight_bias", None) is not None or (
        getattr(w, "w2_weight_bias", None) is not None
    ):
        raise RuntimeError("triton bf16 MoE does not support bias")

    top_k = int(getattr(w, "top_k"))
    n_tokens = x.shape[0]
    ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing(
        router_logits, top_k, dtype=router_logits.dtype
    )
    return _bf16_moe_body(
        x, w, ragged_metadata, gather_indx, scatter_indx, gate_scal, top_k, n_tokens
    )
