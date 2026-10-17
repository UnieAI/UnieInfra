# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fused Gated Delta Net chunked prefill for sm_120 (RTX PRO 6000 Blackwell).

Why this exists
---------------
Qwen3.5-27B / Qwen3.6-35B-A3B are 48/64 GDN linear-attention layers, so GDN
prefill is the dominant prefill cost. On sm_120 TokenSpeed selects
``triton_gdn_chunk_prefill`` (PORTABLE, 4): the stock FLA
``chunk_gated_delta_rule`` pipeline. FlashInfer's fast GDN prefill is gated on
compute-capability *major 10* (sm100/sm103) plus CUDA>=13, so sm_120 never gets
it; vLLM is on the same Triton/FLA code, so today both engines run the same
math.

The FLA pipeline is 8 kernel launches that materialize five large
intermediates. Measured on this device at the real geometry (Hk=16, Hv=48,
K=V=128, one 8192-token sequence, bf16):

    l2norm x2     43.8 us   3.6%
    cumsum        25.8 us   2.1%
    kkt           58.0 us   4.7%
    solve_tril   107.4 us   8.7%
    wy (w,u)     279.9 us  22.7%
    chunk_h      383.5 us  31.2%     <- writes h: [1, NT, 48, 128, 128] bf16
    chunk_o      332.4 us  27.0%     <- reads it straight back
    TOTAL       1230.7 us

``chunk_h`` and ``chunk_o`` are 58% of the pipeline and they communicate
through ``h``, a per-chunk state checkpoint tensor that is 201 MB for a single
8192-token sequence and is never read by anything else. Writing and re-reading
it is pure overhead: the two kernels sweep chunks in the same order, over the
same (sequence, head, V-tile) decomposition.

What this module does
---------------------
Fuses them. One kernel per ``(sequence, v-head, V-tile)`` walks the chunks
sequentially, keeps the recurrent state ``h`` in registers, and emits ``o``
directly -- ``h`` is never written to HBM. That also removes ``v_new`` (another
100 MB at 8192 tokens), which only existed to hand the corrected values from
``chunk_h`` to ``chunk_o``.

Two smaller wins come along with it:

* The in-tree Triton wrapper receives the K-last ``[N, H, V, K]`` state that
  the runtime's SSM pool allocates and calls ``.transpose(-2, -1).contiguous()``
  on it, materializing a copy every layer on every call, because the FLA
  ``.jit`` kernel has no stride support for ``h0``. This kernel addresses the
  state K-last through a block pointer with strides ``(1, K)`` and writes the
  final state back K-last the same way, so the contract layout costs nothing.
* One fewer kernel launch. At 48 GDN layers a launch is not free: the same
  pipeline at 2048 tokens costs 308 us, ~150 us of which is fixed per-launch
  cost rather than work.

Everything before ``chunk_h`` (l2norm, chunk-local cumsum, the K K^T Gram
matrix, the unit-lower-triangular inverse, and the WY ``w``/``u``
recomputation) is reused unchanged from ``tokenspeed_kernel``. Fusing those in
too would mean doing a 64x64 forward substitution inside this kernel, which is
serial and would stall the chunk loop; they are also only 40% of the cost and
are not what the split design is paying for.

Honest scope
------------
``output_h=True`` asks for the per-chunk checkpoints in FLA layout. Not
materializing them is the entire point of this kernel, so that trait is
declared ``{False}`` and such requests fall through to the in-tree Triton
kernel. ``head_dim`` is declared 128-only (the kernel unrolls K as two 64-wide
register tiles); Qwen3.5/3.6 are 128, and anything else falls back.

The decode and MTP ops are deliberately not touched here: they are
state-pool-bound, not bandwidth-bound, and the in-tree Triton kernel already
addresses the pool K-last without a transpose.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["register_gdn", "unieinfra_gdn_chunk_prefill_sm120"]

# The FLA pipeline's chunk size. Everything upstream (cumsum, the Gram matrix,
# solve_tril, wy_fast) is built around 64, so this is not a free parameter.
CHUNK_SIZE = 64

# V-tile width. The grid is (V // BV, N * Hv); with Hv=48 and BV=32 a single
# sequence fills 192 blocks against this device's 188 SMs. BV=64 would halve
# the redundant q k^T work but leaves half the SMs idle at N=1, which measured
# worse. Overridable for tuning; see _fused_chunk_ho.
BV_DEFAULT = 32
NUM_WARPS_DEFAULT = 4
# num_stages=1, i.e. no software pipelining of the chunk loop. This is not the
# usual choice and it is worth explaining: the loop body already holds the
# recurrent state, the [BT, BT] attention block, the output tile and v_new at
# once, and every chunk depends on the previous one through the state.
# Pipelining a second stage duplicates all the staged operand buffers for
# overlap the dependency does not allow, and measured ~2x SLOWER end to end
# (N=1 T=2048: 884 us at stages=2 vs 268 us at stages=1). num_stages>=3 does
# not compile here at all -- see _SMEM_BYTES.
NUM_STAGES_DEFAULT = 1

# Measured shared-memory demand of this kernel, in bytes, keyed by
# (BV, num_stages) -- num_warps does not change it. Read off
# ``kernel.metadata.shared`` on the device.
#
# This is the sm_120 trap the port-kernel skill warns about, in concrete
# numbers. sm_120 offers 101376 B opt-in per block; sm_100 offers ~227 KB. Every
# num_stages=3 config below fits datacenter Blackwell and none of them fit here:
#
#     BV     stages=1   stages=2   stages=3
#     16        34816      53764     105480   <- over 101376
#     32        36864      57860     111624   <- over
#     64        40960      66052     123912   <- over
#    128        32768      74244     140296   <- over
#
# The same cliff hits the in-tree kernel this one replaces:
# chunk_h at BV=64/stages=3 needs 107016 B and at BV=32/stages=4 needs 123660 B,
# both of which fail to launch on sm_120. vLLM autotunes chunk_h over
# BV in {32,64} x warps in {2,4} x stages in {2,3,4}; on this device a third of
# that space is silently unreachable. Selection here is therefore driven by the
# device's *reported* opt-in shared memory, never by the architecture name.
_SMEM_BYTES = {
    (16, 1): 34816,
    (16, 2): 53764,
    (16, 3): 105480,
    (32, 1): 36864,
    (32, 2): 57860,
    (32, 3): 111624,
    (64, 1): 40960,
    (64, 2): 66052,
    (64, 3): 123912,
    (128, 1): 32768,
    (128, 2): 74244,
    (128, 3): 140296,
}

# Fused-vs-split gate. The fused kernel's only parallelism is
# (V // BV) x N x Hv -- the chunk loop is strictly serial, because each chunk
# consumes the previous chunk's recurrent state. The split kernels get an extra
# NT-way parallel dimension in chunk_o. So the fusion pays off exactly when
# there are enough sequences to fill the machine without it, and does not when
# a single long sequence is the whole batch. Measured end to end at the model
# geometry (see docs/kernel-provenance.md):
#
#     N=1  T in [1024, 16384]   0.90x - 0.98x   <- fused LOSES, use the split path
#     N>=2 T in [2048, 32768]   1.08x - 1.75x   <- fused wins
#
# The threshold is on sequence count, not token count, because that is the
# mechanism. N=1 falls back to the in-tree Triton kernel unchanged.
FUSED_MIN_SEQS = 2


@triton.jit
def _exp(x):
    return tl.exp(x)


@triton.jit
def _safe_exp(x):
    # Matches tokenspeed_kernel.ops.attention.gdn._triton.op.safe_exp: the
    # gate differences fed to this are <= 0 by construction, and anything else
    # is a masked-out lane that must not blow up.
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _fused_chunk_ho_kernel(
    q,  # [1, T, Hg, K]  bf16/fp16, already L2-normalized
    k,  # [1, T, Hg, K]  bf16/fp16, already L2-normalized
    w,  # [1, T, H,  K]  bf16/fp16, from wy_fast
    u,  # [1, T, H,  V]  bf16/fp16, from wy_fast
    g,  # [1, T, H]      fp32, chunk-local cumsum of the log forget gate
    o,  # [1, T, H,  V]  out
    h0,  # [N, H, V, K]  fp32, K-LAST (the shared contract layout)
    ht,  # [N, H, V, K]  fp32, K-LAST
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program owns (sequence, v-head, V-tile) and walks every chunk.

    Holds the recurrent state for its ``[K, BV]`` slice in registers across the
    whole sequence, so the per-chunk checkpoint tensor ``h`` that the split
    FLA kernels exchange never exists. Per chunk it computes, in order:

        v_new = u - w @ h                      (h = state at chunk start)
        o     = (q * exp(g)) @ h * scale
              + tril(q k^T * exp(g_i - g_j)) @ v_new * scale
        h     = h * exp(g_last) + k^T @ (v_new * exp(g_last - g))

    which is exactly ``chunk_gated_delta_rule_fwd_h`` followed by
    ``chunk_fwd_o``, including where the intermediate rounds to the input
    dtype (``v_new`` is stored bf16 in the split version, so it is rounded here
    at the same point).
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    # Grouped-value attention: Hg query/key heads shared by H value heads.
    i_hg = i_h // (H // Hg)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    q += (bos * Hg + i_hg).to(tl.int64) * K
    k += (bos * Hg + i_hg).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    u += (bos * H + i_h).to(tl.int64) * V
    o += (bos * H + i_h).to(tl.int64) * V
    g += (bos * H + i_h).to(tl.int64)

    stride_qk = Hg * K
    stride_w = H * K
    stride_v = H * V

    # State, FLA-native [K, V] orientation in registers, K unrolled as two
    # 64-wide tiles (K == 128 is the only registered head_dim).
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        # h0 is K-last [N, H, V, K]: element (kk, vv) sits at vv * K + kk, so a
        # (K, V)-shaped block pointer with strides (1, K) reads the FLA-native
        # tile directly. No transpose, no materialized copy.
        p_h0 = h0 + (i_n * H + i_h).to(tl.int64) * V * K
        p_h0_1 = tl.make_block_ptr(
            p_h0, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1)
        )
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        p_h0_2 = tl.make_block_ptr(
            p_h0, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
        )
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # ---- q / k tiles (k also transposed, for both q k^T and k^T v_new)
        p_q1 = tl.make_block_ptr(
            q, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_q2 = tl.make_block_ptr(
            q, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        p_kt1 = tl.make_block_ptr(
            k, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_kt2 = tl.make_block_ptr(
            k, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_kt1 = tl.load(p_kt1, boundary_check=(0, 1))
        b_kt2 = tl.load(p_kt2, boundary_check=(0, 1))

        # ---- v_new = u - w @ h_start
        p_w1 = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_w2 = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_vn = tl.dot(b_w1, b_h1.to(b_w1.dtype))
        b_vn += tl.dot(b_w2, b_h2.to(b_w2.dtype))
        p_u = tl.make_block_ptr(
            u, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_vn = tl.load(p_u, boundary_check=(0, 1)) - b_vn
        # The split pipeline hands v_new to chunk_o through a bf16 tensor.
        # Round at the same point so this kernel is bit-comparable, not just
        # mathematically equal.
        b_vn_q = b_vn.to(b_q1.dtype)

        # ---- gates
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * H)

        # ---- output: inter-chunk (state) + intra-chunk (causal) contribution
        b_o = tl.dot(b_q1, b_h1.to(b_q1.dtype))
        b_o += tl.dot(b_q2, b_h2.to(b_q2.dtype))
        b_A = tl.dot(b_q1, b_kt1)
        b_A += tl.dot(b_q2, b_kt2)
        b_o = b_o * _exp(b_g)[:, None]
        b_A = b_A * _safe_exp(b_g[:, None] - b_g[None, :])
        o_i = tl.arange(0, BT)
        b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0)
        b_o = b_o * scale + tl.dot(b_A.to(b_q1.dtype), b_vn_q) * scale
        p_o = tl.make_block_ptr(
            o, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # ---- state update
        b_vd = (b_vn * _safe_exp(b_g_last - b_g)[:, None]).to(b_q1.dtype)
        b_gl = _exp(b_g_last)
        b_h1 = b_h1 * b_gl + tl.dot(b_kt1, b_vd)
        b_h2 = b_h2 * b_gl + tl.dot(b_kt2, b_vd)

    if STORE_FINAL_STATE:
        p_ht = ht + (i_n * H + i_h).to(tl.int64) * V * K
        p_ht_1 = tl.make_block_ptr(
            p_ht, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1)
        )
        tl.store(p_ht_1, b_h1.to(p_ht_1.dtype.element_ty), boundary_check=(0, 1))
        p_ht_2 = tl.make_block_ptr(
            p_ht, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
        )
        tl.store(p_ht_2, b_h2.to(p_ht_2.dtype.element_ty), boundary_check=(0, 1))


def _pick_config(device: torch.device) -> tuple[int, int, int]:
    """Largest measured-good tile config that fits this device's shared memory.

    Branches on ``shared_memory_per_block_optin`` as reported by the driver,
    not on the architecture name -- sm_100 and sm_120 are both "Blackwell" and
    differ by more than 2x here, which is exactly how a kernel tuned for the
    datacenter part fails to launch on the workstation one.

    Returns:
        ``(BV, num_warps, num_stages)``.
    """
    limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    # Preference order is measured, best first. Every entry is correct; they
    # differ only in speed and in how much shared memory they ask for.
    for bv, nw, ns in (
        (BV_DEFAULT, NUM_WARPS_DEFAULT, NUM_STAGES_DEFAULT),
        (32, 4, 1),
        (32, 2, 1),
        (16, 4, 1),
    ):
        if _SMEM_BYTES[(bv, ns)] <= limit:
            return bv, nw, ns
    raise RuntimeError(
        f"no fused GDN config fits {limit} B of opt-in shared memory; "
        f"smallest known config needs {_SMEM_BYTES[(16, 1)]} B"
    )


def _fused_chunk_ho(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    BV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the fused state-passing + output kernel.

    Args:
        q: L2-normalized queries ``[1, T, Hg, K]``.
        k: L2-normalized keys ``[1, T, Hg, K]``.
        w: WY ``w`` factor ``[1, T, H, K]`` from ``recompute_w_u_fwd``.
        u: WY ``u`` factor ``[1, T, H, V]`` from ``recompute_w_u_fwd``.
        g: Chunk-local cumsum of the log forget gate ``[1, T, H]``, fp32.
        scale: Query scale.
        initial_state: K-last state ``[N, H, V, K]`` or ``None``.
        output_final_state: Whether to allocate and fill the final state.
        cu_seqlens: ``[N+1]`` cumulative lengths, or ``None`` for a dense batch.
        BV: V-tile width; must divide ``V``. ``None`` picks by shared memory.
        num_warps: Warps per program. ``None`` picks by shared memory.
        num_stages: Chunk-loop pipeline depth. ``None`` picks by shared memory.

    Returns:
        ``(o, final_state)`` -- output ``[1, T, H, V]`` in ``u``'s dtype, and
        the final state K-last ``[N, H, V, K]`` fp32 (``None`` when not
        requested).
    """
    if BV is None or num_warps is None or num_stages is None:
        d_bv, d_nw, d_ns = _pick_config(q.device)
        BV = d_bv if BV is None else BV
        num_warps = d_nw if num_warps is None else num_warps
        num_stages = d_ns if num_stages is None else num_stages
    B, T, Hg, K = q.shape
    H, V = u.shape[-2], u.shape[-1]
    assert K == 128, "fused GDN prefill is registered for head_dim 128 only"
    assert V % BV == 0, f"BV={BV} must divide V={V}"
    N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1

    o = torch.empty_like(u)
    final_state = (
        q.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    grid = (V // BV, N * H)
    _fused_chunk_ho_kernel[grid](
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BV=BV,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, final_state


def unieinfra_gdn_chunk_prefill_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
):
    """``attention/gdn_chunk_prefill`` with ``chunk_h``/``chunk_o`` fused.

    Same contract as ``triton_gdn_chunk_prefill``: q ``[1, Tt, Hk, K]``, k
    ``[1, Tt, Hk, K]``, v ``[1, Tt, Hv, V]``, g/beta ``[1, Tt, Hv]``,
    ``initial_state`` K-last ``[N, Hv, V, K]``, ``cu_seqlens`` ``[N+1]``.

    When the batch holds fewer than ``FUSED_MIN_SEQS`` sequences the fused
    kernel has too little parallelism to pay for itself and this delegates,
    unchanged, to the in-tree ``triton_gdn_chunk_prefill``. That is a measured
    decision, not caution -- see ``FUSED_MIN_SEQS``.

    Returns:
        ``GdnChunkPrefillResult`` with ``out`` ``[1, Tt, Hv, V]`` in q's dtype
        and ``final_state`` K-last ``[N, Hv, V, K]`` fp32. ``h`` is always
        ``None`` -- this kernel is registered with ``output_h: {False}``
        because never materializing the checkpoints is the point of it.
    """
    from tokenspeed_kernel.ops.attention.gdn import (
        GdnCheckpointLayout,
        GdnChunkPrefillResult,
    )
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        triton_gdn_chunk_prefill,
    )

    num_seqs = (cu_seqlens.shape[0] - 1) if cu_seqlens is not None else q.shape[0]
    # output_h is declared {False} so selection never routes it here; delegating
    # rather than raising is a safety net for a caller that bypasses selection.
    if num_seqs < FUSED_MIN_SEQS or output_h:
        return triton_gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
            output_h=output_h,
        )
    from tokenspeed_kernel.ops.attention.gdn._triton.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from tokenspeed_kernel.ops.attention.gdn._triton.cumsum import chunk_local_cumsum
    from tokenspeed_kernel.ops.attention.gdn._triton.l2norm import l2norm_fwd
    from tokenspeed_kernel.ops.attention.gdn._triton.solve_tril import solve_tril
    from tokenspeed_kernel.ops.attention.gdn._triton.wy_fast import recompute_w_u_fwd

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("batch size must be 1 when cu_seqlens is given")

    if scale is None:
        scale = q.shape[-1] ** -0.5

    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if qk_l2norm:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    g_cumsum = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g_cumsum, cu_seqlens=cu_seqlens
    )

    out, final_state = _fused_chunk_ho(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return GdnChunkPrefillResult(
        out=out.to(q.dtype),
        final_state=final_state,
        h=None,
        h_layout=GdnCheckpointLayout.NONE,
    )


def register_gdn() -> None:
    """Register the fused GDN chunked-prefill kernel in the PLUGIN band.

    Call this from the package's ``register()``. It is a no-op with a log line
    on any platform other than NVIDIA sm_120, so it is safe to call
    unconditionally.
    """
    from tokenspeed_kernel.platform import (
        ArchVersion,
        CapabilityRequirement,
        current_platform,
    )
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures

    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version == ArchVersion(12, 0)):
        logger.info(
            "unieinfra-kernel-sm120: not registering fused GDN prefill, "
            "platform is %s sm_%s%s",
            platform.vendor,
            platform.arch_version.major,
            platform.arch_version.minor,
        )
        return

    register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="unieinfra_gdn_chunk_prefill_sm120",
        solution="unieinfra_fused",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 9),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16}
        ),
        # Beats the in-tree Triton entry (PORTABLE, 4). FlashInfer's GDN
        # prefill is major-10 only and is not a competitor here.
        priority=Priority.PLUGIN,
        traits={
            # K is unrolled as two 64-wide register tiles; nothing else fits.
            "head_dim": frozenset({128}),
            "value_head_dim": frozenset({128}),
            "head_v_eq_head_k": frozenset({True}),
            # Grouped-value attention only (Hv >= Hk), like the model.
            "num_v_gte_num_q": frozenset({True}),
            "qk_l2norm": frozenset({False, True}),
            # Not "off" -- not implemented. output_h=True must fall back to
            # the in-tree Triton kernel, which actually writes checkpoints.
            "output_h": frozenset({False}),
        },
        tags={"unieinfra", "sm120", "gdn"},
    )(unieinfra_gdn_chunk_prefill_sm120)

    logger.info(
        "unieinfra-kernel-sm120: registered fused GDN chunk prefill at " "priority %d",
        int(Priority.PLUGIN),
    )
