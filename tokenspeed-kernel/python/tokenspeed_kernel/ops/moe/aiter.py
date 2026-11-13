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

"""MXFP4 experts through AMD AITER on gfx950.

This is the kernel path the vLLM reference runs. ``--moe-backend aiter`` in
vLLM maps to ``Mxfp4MoeBackend.AITER_MXFP4_BF16`` (see
``fused_moe/oracle/mxfp4.py``), which on gfx950 selects bf16 activations
against preshuffled MXFP4 weights: W4A16 with ``QuantType`` ``BLOCK_1X32``
(one e8m0 scale per 32 weights) and ``GateMode.INTERLEAVE`` gate/up packing,
dispatched by ``aiter.fused_moe`` into the FlyDSL stage-1 and stage-2 kernel
pair (``..._afp8_wfp4_bf16_t32x128x256_w2_gui`` and
``..._t32x128x128_atomic_bnt2`` at these shapes: no tuned row in
``dsv41_fp4_tuned_fmoe.csv`` carries ``cu_num=128``, so the heuristic picks
the same kernels the reference runs).

The portable Triton path (``_triton/mxfp4.py``) instead upcasts the weights to
bf16 and multiplies with ``tl.dot``, which loses to a real MFMA fp4 kernel
exactly where decode spends its time (small M). Registering AITER here lets the
two engines share the vendor GEMM, so an A/B difference measures engine
overhead rather than GEMM quality.

Weights enter the loader packed as uint8 ``[E, N, K//2]`` with raw E8M0 scales
``[E, N, K//32]``; :func:`aiter_mxfp4_moe_weights` moves them into AITER's
interleaved layout once, at load time. Both tensors carry ``is_shuffled`` so
``fused_moe`` will accept them for the blockscaled paths.

Two load-time adjustments are not optional, and neither is visible from the
call site:

* The intermediate dimension is padded up to a 128 multiple (576 -> 640 at
  TP4) with zero weights and E8M0 ``1.0`` scales. ``shuffle_weight`` needs the
  packed K of ``w2`` (``inter_dim // 2``) to be a 64 multiple, which 288 is not,
  and the stage-1 tile configs assume the aligned shape. Padding is harmless
  numerically: the extra gate/up lanes see zero weights, emit zero, and are
  multiplied by zero ``w2`` columns.
* ``AITER_BF16_FP8_MOE_BOUND`` picks the activation dtype inside
  ``fused_moe``; vLLM pins it to ``"0"`` while preparing weights, which makes
  even single-token decode use fp8 activations. Leaving it unset sends small
  batches down the bf16-activation kernel family and changes both speed and
  numerics.
"""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# AITER's e8m0 scale block: one exponent per 32 packed FP4 values.
MXFP4_BLOCK = 32

# Tile alignment AITER's interleaved W4A16 kernels require of the per-partition
# intermediate dimension, and of the hidden dimension.
INTERMEDIATE_ALIGN = 128
HIDDEN_ALIGN = 512

# E8M0 1.0, used to fill the scale entries that cover padding.
E8M0_ONE = 0x7F

# AITER picks fp8 activations over bf16 when the intermediate dimension is
# large enough; vLLM pins this to 0 for the interleaved W4A16 kernel so the
# reference and this path agree on bf16. Must be set before the dispatch runs.
os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")


def _aiter_ops():
    """AITER entry points, with an install hint instead of a bare ImportError."""
    try:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.ops.shuffle import shuffle_scale, shuffle_weight
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the 'aiter' solution runs the vendor MXFP4 MoE kernels and needs "
            "AMD's AITER package (pip install amd-aiter, or see "
            "https://github.com/ROCm/aiter). Install it into the same "
            "environment as tokenspeed, or pass --moe-backend with another "
            "solution."
        ) from exc
    return (
        fused_moe,
        ActivationType,
        QuantType,
        GateMode,
        shuffle_scale,
        shuffle_weight,
    )


def _swiglu_limit(w: torch.nn.Module) -> float:
    """The checkpoint's SwiGLU clamp as AITER's ``swiglu_limit`` (0 = none).

    AITER's blockscaled epilogue clamps the gate from above and the up branch
    on both sides, matching the checkpoint's ``swiglu_limit``. A sigmoid
    multiplier (``alpha``) or an up-branch offset (``swiglu_beta``) has no
    counterpart in the kernel, so it is rejected rather than silently dropped.
    """
    swiglu_arg = getattr(w, "swiglu_arg", None)
    alpha = None if swiglu_arg is None else getattr(swiglu_arg, "alpha", None)
    beta = getattr(w, "swiglu_beta", None)
    if alpha not in (None, 1.0) or beta not in (None, 0.0):
        raise ValueError(
            "AITER MXFP4 MoE supports only standard SwiGLU with an optional "
            f"clamp limit; got alpha={alpha!r}, swiglu_beta={beta!r}"
        )
    if swiglu_arg is None:
        return 0.0
    limit = getattr(swiglu_arg, "limit", None)
    return 0.0 if limit is None else float(limit)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _as_dtype(t: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Reinterpret packed uint8 as the typed view AITER's dtype checks expect."""
    if dtype is None:
        return t
    return t if t.dtype == dtype else t.view(dtype)


def _pad_inter_dim(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    ispp: int,
    gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the intermediate dimension up to AITER's tile alignment.

    ``gate_up`` weights (``w13``) are ``[E, 2 * ispp, hidden // 2]`` with the
    gate half first: padding inserts zero rows at the end of each half, because
    the stage-1 kernel reads gate from rows ``[0, I_pad)`` and up from
    ``[I_pad, 2 * I_pad)``. ``w2`` is ``[E, hidden, ispp // 2]``, so its packed
    K grows instead and the scales get one extra E8M0 block per 32 columns.

    Returns the padded weight plus the scales flattened to the ``[E * N, K]``
    layout ``shuffle_scale`` wants.
    """
    padded = _round_up(ispp, INTERMEDIATE_ALIGN)
    pad = padded - ispp
    if pad:
        if gate_up:
            zeros = weight.new_zeros((weight.shape[0], pad, weight.shape[-1]))
            gate, up = weight.split(ispp, dim=1)
            weight = torch.cat((gate, zeros, up, zeros), dim=1)
            ones = scale.new_full((scale.shape[0], pad, scale.shape[-1]), E8M0_ONE)
            gate_scale, up_scale = scale.split(ispp, dim=1)
            scale = torch.cat((gate_scale, ones, up_scale, ones), dim=1)
        else:
            pad_k = pad // 2
            weight = torch.cat(
                (weight, weight.new_zeros((*weight.shape[:-1], pad_k))), dim=-1
            )
            pad_blocks = pad // MXFP4_BLOCK
            scale = torch.cat(
                (scale, scale.new_full((*scale.shape[:-1], pad_blocks), E8M0_ONE)),
                dim=-1,
            )
    return weight.contiguous(), scale.contiguous().reshape(-1, scale.shape[-1])


def aiter_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Move loaded MXFP4 experts into AITER's interleaved W4A16 layout.

    Input (from the mxfp4 weight loader): ``w13_weight``/``w2_weight`` packed
    E2M1 as uint8 ``[E, N, K//2]`` and ``w13_weight_scale``/``w2_weight_scale``
    raw E8M0 as uint8 ``[E, N, K//32]``. Output: the same tensors, padded to
    AITER's tile alignment and permuted into its N-tile swizzle with gate/up
    rows interleaved for the w1 matrix, plus matching scale swizzles. The
    permutation is a pure data movement over the whole expert stack, so it
    happens once at load and is invisible to steady state.

    AITER re-reads these tensors every call, so the tensors must keep the
    ``is_shuffled`` marker: ``fused_moe`` rejects blockscaled weights without
    it, and any later copy that drops the marker has to be re-shuffled.
    """
    _, _, _, _, shuffle_scale, shuffle_weight = _aiter_ops()

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    num_experts = int(w.w13_weight.shape[0])

    # ``ispp`` is the per-partition intermediate size; ``w2`` packs its K as
    # fp4 pairs, so ``ispp == 2 * w2.shape[-1]``.
    ispp = 2 * int(w.w2_weight.shape[-1])

    weight, scale = _pad_inter_dim(
        w.w13_weight.data,
        w.w13_weight_scale.data,
        ispp=ispp,
        gate_up=True,
    )
    w13_weight = shuffle_weight(
        _as_dtype(weight, fp4_dtype), is_guinterleave=True, gate_up=True
    )
    w13_scale = shuffle_scale(_as_dtype(scale, e8m0_dtype), num_experts, True, True)
    del weight, scale

    weight, scale = _pad_inter_dim(
        w.w2_weight.data,
        w.w2_weight_scale.data,
        ispp=ispp,
        gate_up=False,
    )
    w2_weight = shuffle_weight(
        _as_dtype(weight, fp4_dtype), is_guinterleave=True, gate_up=False
    )
    w2_scale = shuffle_scale(_as_dtype(scale, e8m0_dtype), num_experts, True, False)
    del weight, scale

    w.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    # The marker has to go on the Parameter, not the tensor it wraps: building a
    # Parameter copies into a fresh tensor object, which drops custom attributes.
    # Without it fused_moe falls back to its "preshuffle_off" CK path.
    w.w13_weight.is_shuffled = True
    w.w2_weight.is_shuffled = True
    w._aiter_ispp = ispp
    w._aiter_hidden = 2 * int(w13_weight.shape[-1])
    w._aiter_intermediate_pad = _round_up(ispp, INTERMEDIATE_ALIGN) - ispp
    w._aiter_hidden_pad = _round_up(w._aiter_hidden, HIDDEN_ALIGN) - w._aiter_hidden
    w._aiter_shuffled = True


@register_kernel(
    "moe",
    "apply",
    name="aiter_mxfp4_moe_apply",
    solution="aiter",
    weight_preprocessor=aiter_mxfp4_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        # The blockscaled epilogue applies the checkpoint's gate/up clamp.
        "activation_clamped": frozenset({True, False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({64}),
        "hidden_alignment": frozenset({512}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
        # The kernel sums a token's routes into one row, so either order works.
        "combine_order": frozenset({"rank", "slot"}),
    },
    # Opt-in (``--moe-backend aiter``): below the gluon MXFP4 kernels so the
    # default selection on gfx950 is unchanged, and AITER stays an optional
    # vendor dependency.
    priority=Priority.PERFORMANT,
)
def aiter_mxfp4_moe_apply(
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
    """Apply MXFP4 experts with bf16 activations through AITER.

    Args:
        plan: MoE plan from ``moe_plan``; only ``num_experts``/``ispp`` are
            read, and only for shape validation.
        x: bf16 hidden states ``[tokens, hidden]``.
        w: Module holding AITER-shuffled ``w13_weight``/``w2_weight`` (fp4 pairs)
            and their E8M0 scale swizzles.
        router_logits: Unused; routing must be precomputed.
        topk_weights: Route weights ``[tokens, top_k]``; cast to fp32.
        topk_ids: Expert ids ``[tokens, top_k]``; cast to int32.
        num_tokens_global: Unused; expert parallelism is not wired up for this
            solution yet (`supports_ep` is false).
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        BF16 hidden states ``[tokens, hidden]``.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    (
        fused_moe,
        ActivationType,
        QuantType,
        GateMode,
        _shuffle_scale,
        _shuffle_weight,
    ) = _aiter_ops()
    if not do_finalize:
        raise ValueError("AITER MXFP4 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("AITER MXFP4 MoE requires precomputed top-k")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"AITER MXFP4 MoE requires bf16 activations, got {x.dtype}")
    if not getattr(w, "_aiter_shuffled", False):
        raise ValueError(
            "AITER MXFP4 MoE weights were not shuffled; run "
            "aiter_mxfp4_moe_weights after loading the checkpoint"
        )

    swiglu_limit = _swiglu_limit(w)
    return fused_moe(
        x if x.is_contiguous() else x.contiguous(),
        w.w13_weight,
        w.w2_weight,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=w.w13_weight_scale,
        w2_scale=w.w2_weight_scale,
        a1_scale=None,
        a2_scale=None,
        block_size_M=None,
        num_local_tokens=None,
        hidden_pad=w._aiter_hidden_pad,
        intermediate_pad=w._aiter_intermediate_pad,
        gate_mode=GateMode.INTERLEAVE.value,
        swiglu_limit=swiglu_limit,
        dtype=torch.bfloat16,
    )
