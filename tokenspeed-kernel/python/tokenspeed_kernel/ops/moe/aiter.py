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

# Experts per step of the load-time pad + shuffle (see _shuffle_in_place).
SHUFFLE_CHUNK_EXPERTS = 32

# Column alignment shuffle_scale pads the E8M0 scale rows to.
SCALE_COL_ALIGN = 8

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


def _padded_buffer(param: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """The contiguous uint8 ``shape`` buffer whose origin ``param`` views.

    The mxfp4 loader allocates AITER's padded buffers up front and loads the
    checkpoint into unpadded views of them (``create_mxfp4_weight_pair``), so
    this normally returns that buffer and nothing is allocated. Any other
    layout is copied into a fresh padded buffer once.
    """
    expected_strides = []
    stride = 1
    for size in reversed(shape):
        expected_strides.append(stride)
        stride *= size
    expected_strides.reverse()
    data = param.data.view(torch.uint8)
    storage = data.untyped_storage()
    if (
        data.storage_offset() == 0
        and data.stride() == tuple(expected_strides)
        and storage.nbytes() == stride
    ):
        return data.new_empty(0).set_(storage, 0, shape)
    buffer = data.new_zeros(shape)
    buffer[tuple(slice(0, size) for size in data.shape)].copy_(data)
    return buffer


def _shuffle_in_place(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    ispp: int,
    gate_up: bool,
) -> None:
    """Pad and shuffle a padded expert stack in place, a chunk of experts at a time.

    ``weight``/``scale`` are the padded uint8 buffers (``[E, N, K//2]`` and
    ``[E, N, K//32]``) holding the checkpoint values in their unpadded corner.
    ``gate_up`` stacks (``w13``) hold gate rows ``[0, ispp)`` and up rows
    ``[ispp, 2 * ispp)``; up moves to ``[I_pad, I_pad + ispp)`` because the
    stage-1 kernel reads gate from rows ``[0, I_pad)`` and up from
    ``[I_pad, 2 * I_pad)``. ``w2`` keeps its columns and grows packed K
    instead. Padding is zero weights and E8M0 ``1.0`` scales.

    Both AITER shuffles permute within an expert, so each chunk is shuffled
    into a temporary and copied back. The temporaries are a few percent of a
    stack and the same size every layer, so the caching allocator reuses
    them: nothing GB-sized is allocated or freed (see ``ops/moe/README.md``).
    """
    _, _, _, _, shuffle_scale, shuffle_weight = _aiter_ops()
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    num_experts = int(weight.shape[0])
    padded = _round_up(ispp, INTERMEDIATE_ALIGN)
    for start in range(0, num_experts, SHUFFLE_CHUNK_EXPERTS):
        end = min(start + SHUFFLE_CHUNK_EXPERTS, num_experts)
        chunk_weight = weight[start:end]
        chunk_scale = scale[start:end]
        if gate_up:
            for chunk, fill in ((chunk_weight, 0), (chunk_scale, E8M0_ONE)):
                up = chunk[:, ispp : 2 * ispp].clone()
                chunk[:, ispp:padded].fill_(fill)
                chunk[:, padded : padded + ispp].copy_(up)
                chunk[:, padded + ispp :].fill_(fill)
                del up
        else:
            chunk_weight[..., ispp // 2 :].zero_()
            chunk_scale[..., ispp // MXFP4_BLOCK :].fill_(E8M0_ONE)
        shuffled = shuffle_weight(
            _as_dtype(chunk_weight, fp4_dtype), is_guinterleave=True, gate_up=gate_up
        )
        chunk_weight.copy_(shuffled.view(torch.uint8))
        del shuffled
        flat_scale = chunk_scale.view(-1, chunk_scale.shape[-1])
        shuffled = shuffle_scale(
            _as_dtype(flat_scale, e8m0_dtype), end - start, True, gate_up
        )
        flat_scale.copy_(shuffled.view(torch.uint8))
        del shuffled


def aiter_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Move loaded MXFP4 experts into AITER's interleaved W4A16 layout.

    Input (from the mxfp4 weight loader): ``w13_weight``/``w2_weight`` packed
    E2M1 as uint8 ``[E, N, K//2]`` and ``w13_weight_scale``/``w2_weight_scale``
    raw E8M0 as uint8 ``[E, N, K//32]``, ideally as views of AITER's padded
    buffers. Output: the padded buffers, permuted into AITER's N-tile swizzle
    with gate/up rows interleaved for the w1 matrix, plus matching scale
    swizzles flattened to ``[E * N, K//32]``. The permutation is a pure data
    movement, so it happens once at load and is invisible to steady state.

    AITER re-reads these tensors every call, so the tensors must keep the
    ``is_shuffled`` marker: ``fused_moe`` rejects blockscaled weights without
    it, and any later copy that drops the marker has to be re-shuffled.
    """
    # ``ispp`` is the per-partition intermediate size; ``w2`` packs its K as
    # fp4 pairs, so ``ispp == 2 * w2.shape[-1]``.
    ispp = 2 * int(w.w2_weight.shape[-1])
    padded = _round_up(ispp, INTERMEDIATE_ALIGN)
    num_experts, _, packed_hidden = (int(d) for d in w.w13_weight.shape)
    hidden = int(w.w2_weight.shape[1])
    # shuffle_scale pads the scale columns up to a multiple of 8.
    w2_scale_cols = _round_up(padded // MXFP4_BLOCK, SCALE_COL_ALIGN)

    w13_weight = _padded_buffer(w.w13_weight, (num_experts, 2 * padded, packed_hidden))
    w13_scale = _padded_buffer(
        w.w13_weight_scale,
        (num_experts, 2 * padded, int(w.w13_weight_scale.shape[-1])),
    )
    w2_weight = _padded_buffer(w.w2_weight, (num_experts, hidden, padded // 2))
    w2_scale = _padded_buffer(w.w2_weight_scale, (num_experts, hidden, w2_scale_cols))
    _shuffle_in_place(w13_weight, w13_scale, ispp=ispp, gate_up=True)
    _shuffle_in_place(w2_weight, w2_scale, ispp=ispp, gate_up=False)

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    w.w13_weight = torch.nn.Parameter(
        _as_dtype(w13_weight, fp4_dtype), requires_grad=False
    )
    w.w2_weight = torch.nn.Parameter(
        _as_dtype(w2_weight, fp4_dtype), requires_grad=False
    )
    w.w13_weight_scale = torch.nn.Parameter(
        _as_dtype(w13_scale.view(-1, w13_scale.shape[-1]), e8m0_dtype),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        _as_dtype(w2_scale.view(-1, w2_scale_cols), e8m0_dtype), requires_grad=False
    )
    # The marker has to go on the Parameter, not the tensor it wraps: building a
    # Parameter copies into a fresh tensor object, which drops custom attributes.
    # Without it fused_moe falls back to its "preshuffle_off" CK path.
    w.w13_weight.is_shuffled = True
    w.w2_weight.is_shuffled = True
    w._aiter_ispp = ispp
    w._aiter_hidden = 2 * packed_hidden
    w._aiter_intermediate_pad = padded - ispp
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
