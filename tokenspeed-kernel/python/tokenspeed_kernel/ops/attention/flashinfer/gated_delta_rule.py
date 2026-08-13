# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FlashInfer gated delta net (GDN) prefill + decode/MTP kernels.

Prefill: fast-path for GDN chunked prefill on Blackwell (sm100/sm103) + CUDA 13
+ bf16 + head_dim==128. The caller gates on ``is_supported`` and must fail fast
when the fast-path is unavailable; this module has no Triton fallback for it.

Decode / MTP: single-token decode (``gdn_decode_step``) and multi-token
speculative-verify decode (``gdn_decode_mtp``) on Hopper+ (SM90+).

State layout convention (all three kernels): the recurrent state is K-last
(``[N, H, V, K]``, v-major), matching flashinfer's native GDN decode/MTP
layout -- this is also the layout the runtime's SSM state pool allocates, so
no transpose is needed at any of these kernels' boundaries. This differs from
the Triton FLA convention (``[N, H, K, V]``); the Triton ``gdn_chunk_prefill``
wrapper is the one that adapts (transposes) to/from FLA internally.

Other prefill conventions (verified equal to bf16 on B200):
- q, k must be L2-normalized by the caller (the sm100 kernel ignores its own
  use_qk_l2norm flag).
- g is the FLA log-space forget gate; the sm100 kernel takes log internally, so
  we pass alpha = exp(g).
- beta is cast to float32 (flashinfer passes it through without casting).
- scale defaults to head_dim**-0.5 (FLA default).
"""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel.ops.attention.gdn_utils import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
)
from tokenspeed_kernel.ops.attention.triton.linear.l2norm import l2norm_fwd
from tokenspeed_kernel.ops.layernorm.flashinfer import rmsnorm
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()
SUPPORTED_HEAD_DIM = 128

_l2norm_weights: dict[tuple, torch.Tensor] = {}


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation over the last dim.

    Routed through flashinfer's RMSNorm rather than the Triton `l2norm_fwd`,
    which runs at 380 GB/s on this part -- 29% of what a plain copy of the same
    bytes achieves (1310 GB/s) -- and costs 22.8 us per call against RMSNorm's
    9.4 us on [1024, 16, 128] bf16.

    The two are the same function up to a constant:
        rmsnorm(x, w) = x / sqrt(mean(x^2) + eps) * w
        l2norm(x)     = x / sqrt(sum(x^2)  + eps)
    and sum = mean * D, so `l2norm(x) == rmsnorm(x, w = 1/sqrt(D))`. Verified
    against `l2norm_fwd` at max|diff| 1.95e-03 (bf16 rounding), cosine
    0.9999996.
    """
    if rmsnorm is error_fn:
        return l2norm_fwd(x)
    head_dim = x.shape[-1]
    key = (head_dim, x.dtype, x.device.index)
    w = _l2norm_weights.get(key)
    if w is None:
        w = torch.full((head_dim,), head_dim**-0.5, dtype=x.dtype, device=x.device)
        _l2norm_weights[key] = w
    return rmsnorm(x.reshape(-1, head_dim), w, 1e-6).view_as(x)


_chunk_gated_delta_rule = error_fn
_chunk_gated_delta_rule_raw = error_fn
_has_blackwell_prefill = False

# `is_blackwell` is `arch_version.major == 10`, which excludes sm_120 (major
# 12) even though flashinfer ships a kernel for it. Gate on "has a kernel for
# this device" instead of on a family name.
if platform.is_blackwell_plus:
    try:
        # Ask for the arch-specific kernel this device would dispatch to, rather
        # than for a version-specific availability flag. flashinfer 0.6.15 ships
        # `chunk_gated_delta_rule_sm120` and no longer exports the
        # `_has_blackwell_prefill` symbol this used to import -- and because the
        # symbol was imported *before* the kernel, its ImportError skipped the
        # whole block and left `_chunk_gated_delta_rule` as `error_fn`. The
        # sm_120 kernel was present and working the entire time; nothing said so.
        from flashinfer import gdn_kernels as _fi_gdn
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as _fi_chunk

        _arch_major = platform.arch_version.major
        _chunk_gated_delta_rule_raw = getattr(
            _fi_gdn, f"chunk_gated_delta_rule_sm{_arch_major}0", None
        )
        _has_blackwell_prefill = _chunk_gated_delta_rule_raw is not None
        if _chunk_gated_delta_rule_raw is None:
            _chunk_gated_delta_rule_raw = error_fn
        _chunk_gated_delta_rule = _fi_chunk
    except ImportError:
        pass

# Decode / MTP (K-last, SM90+).
_gated_delta_rule_decode_pretranspose = error_fn
_gated_delta_rule_mtp = error_fn
_has_gdn_decode = False

if platform.is_hopper_plus:
    try:
        from flashinfer.gdn_decode import (
            gated_delta_rule_decode_pretranspose as _fi_decode_pretranspose,
        )
        from flashinfer.gdn_decode import gated_delta_rule_mtp as _fi_mtp

        _gated_delta_rule_decode_pretranspose = _fi_decode_pretranspose
        _gated_delta_rule_mtp = _fi_mtp
        _has_gdn_decode = True
    except ImportError:
        pass

# BF16-state MTP kernel: a separate, optional entry point. Needed so
# gdn_decode_mtp can forward the intermediate-state and per-token state-pool
# scatter arguments that are not exposed by gated_delta_rule_decode_pretranspose.
_gated_delta_rule_bf16_mtp = None

if platform.is_hopper_plus:
    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp as _fi_bf16_mtp,
        )

        _gated_delta_rule_bf16_mtp = _fi_bf16_mtp
    except ImportError:
        pass


def is_available() -> bool:
    """Whether flashinfer's fused GDN chunk-prefill kernel can run here.

    flashinfer dispatches per compute-capability major: sm100/sm103 (B200,
    B300) and sm120 (RTX PRO 6000) have separate kernels. The sm10x path is
    additionally gated on CUDA>=13 upstream and raises NotImplementedError
    otherwise, so that check is kept -- but only for sm10x. sm120 is served by
    `chunk_gated_delta_rule_sm120`, which was measured working on this box
    under a CUDA 12.9 torch build: called with the wrapper's own convention
    (decay in linear space, q/k pre-normalised) it agrees with the Triton
    chunked scan to max|diff| = 3.1e-4, far inside the 2e-2 the GDN tests use.

    The point of `_has_blackwell_prefill` is "flashinfer has a kernel for this
    device", which is now answered by probing for the arch-specific symbol
    rather than by importing a flag that later versions stopped exporting.
    """
    if _chunk_gated_delta_rule is error_fn or not _has_blackwell_prefill:
        return False
    if platform.arch_version.major == 10:
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        return cuda_major >= 13
    # sm120: the kernel EXISTS and is CORRECT -- measured against the Triton
    # chunked scan at max|diff| = 3.1e-4, well inside the 2e-2 the GDN tests
    # use -- but it is slower, and the gap widens with sequence length. Host
    # time for one prefill on Qwen3.5-27B (48 GDN layers), steady state with
    # the JIT compile excluded:
    #
    #     tokens |  Triton chunked scan  |  flashinfer sm120 fused
    #         64 |        51.2 ms        |        66.4 ms   (+30%)
    #       1024 |        73.2 ms        |       171.1 ms  (+134%)
    #
    # Nine Triton launches beat one fused launch here, so the launch count was
    # never the thing that mattered on this part. Presumably the sm120 variant
    # lacks the tensor-core/TMA path the sm10x one has. Re-measure before
    # enabling: the probe above already finds the kernel, so this is the only
    # line to revisit when flashinfer ships a faster one.
    #
    # 2026-08-08: a standalone timing loop on the *raw* kernel disagrees with
    # the table above -- flashinfer sm120 came out ~2x better on both host and
    # device time (48 calls, 5 reps, median; 1024 tokens: 9.05 ms host / 9.11 ms
    # device vs Triton's 18.18 / 18.17). The table measured the integrated path,
    # this measured the kernel, so the gap is most likely per-call conversion in
    # the wrapper below rather than the kernel.
    #
    # 2026-08-13: the integrated measurement was run, and the answer is that
    # this kernel is CORRECT and FAST but not CUDA-graph-safe. Qwen3.5-27B, RAG
    # at C=128 through the HTTP harness, 256 requests and 0 failed on all four:
    #
    #                              tok/s    mean TTFT   generated text
    #   triton,     prefill graph  511.86     6979 ms   normal
    #   triton,     no graph       472.02     7452 ms   normal
    #   flashinfer, prefill graph  579.97     5975 ms   "!!!!!!!!!!!!!!"
    #   flashinfer, no graph       574.94     6042 ms   normal
    #
    # Read the last two rows together: the kernel produces correct output the
    # moment the prefill graph is off, so the earlier "it is wrong" reading was
    # itself wrong -- the graph was breaking it. It plans from host-visible
    # batch data, so a capture bakes the dummy batch's plan and every replay
    # runs it against the real one.
    #
    # A direct A/B on engine-shaped inputs (multi-sequence cu_seqlens, live
    # non-zero initial_state, off-chunk lengths) agrees with the Triton path to
    # max|diff| 4.9e-4 on out and ~5e-3 on final_state, and both are equally
    # sensitive to initial_state. There is nothing wrong with the numerics.
    #
    # Net, on this shape: the prefill graph is worth +8.4% on Triton, and this
    # kernel is worth +21.8% over Triton without it, so flashinfer-without-graph
    # beats triton-with-graph by 12.3%. Making it a graph break point instead of
    # disabling the graph would stack both, and is the work worth doing.
    #
    # Default stays OFF because the win requires --disable-prefill-graph and
    # nothing here can enforce that; the wrapper raises under capture rather
    # than letting a graph silently corrupt every reply.
    return os.getenv("TOKENSPEED_ENABLE_FI_GDN_SM120") == "1"


def _use_raw_kernel() -> bool:
    """Whether to call the arch-specific kernel instead of the public API.

    Only sm120 for now: that is where the public wrapper's per-call Python cost
    was measured (see the call site) and where the raw entry point's contract
    was verified against the Triton scan. The sm10x path keeps the public API
    until the same two things are checked there.
    """
    return (
        _chunk_gated_delta_rule_raw is not error_fn
        and platform.arch_version.major == 12
    )


def is_supported(
    head_dim: int, dtype: torch.dtype, num_q_heads: int, num_v_heads: int
) -> bool:
    # bf16 is the verified path; fp16 is rejected (caller fails fast).
    # flashinfer reads g/beta/state with max(num_q, num_v) heads; the runtime
    # supplies them with num_v heads, so num_v < num_q (e.g. Hk=32, Hv=16) reads
    # out of bounds. Only num_v >= num_q is safe (GVA or equal heads).
    return (
        is_available()
        and head_dim == SUPPORTED_HEAD_DIM
        and dtype == torch.bfloat16
        and num_v_heads >= num_q_heads
    )


def is_decode_available() -> bool:
    """Whether the SM90+ GDN decode/MTP kernels can run on this platform."""
    return _has_gdn_decode and platform.is_hopper_plus


def is_decode_supported(head_dim: int, dtype: torch.dtype) -> bool:
    """Whether ``gdn_decode_step``/``gdn_decode_mtp`` support this shape/dtype.

    Both kernels require K == V == 128 and a float16/bfloat16 q/k/v dtype. The
    recurrent state pool may be float32 or bfloat16; FlashInfer dispatches the
    latter to its dedicated BF16-state kernel.
    """
    return (
        is_decode_available()
        and head_dim == SUPPORTED_HEAD_DIM
        and dtype in (torch.float16, torch.bfloat16)
    )


CHUNK_SIZE = 64

gdn_chunk_prefill = error_fn

if is_available():

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="flashinfer_gdn_chunk_prefill",
        solution="flashinfer",
        # No upper bound: flashinfer dispatches per arch major and ships an
        # sm120 kernel. A closed range here excluded sm_120 -- a *newer* part --
        # from a kernel that works on it, which is the failure mode this
        # repository has hit before with capability tables keyed by family.
        # `is_available()` above is what decides whether a kernel exists.
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({SUPPORTED_HEAD_DIM}),
            "head_v_dim": frozenset({SUPPORTED_HEAD_DIM}),
            "head_v_eq_head_k": frozenset({True}),
            "num_v_gte_num_q": frozenset({True}),
            "qk_l2norm": frozenset({False, True}),
            "output_h": frozenset({False, True}),
        },
        tags={"hopper", "blackwell", "latency"},
    )
    def gdn_chunk_prefill(
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
        h_dtype: torch.dtype | None = None,
    ):
        # `h_dtype` is accepted for signature parity and ignored: this
        # backend's checkpoints are float32 already, which is the widest the
        # caller can ask for.
        del h_dtype
        """Run one chunked GDN prefill on the sm100 kernel.

        q, k, v are [B, T, H, D] (B==1) or [T, H, D]; q/k already L2-normalized.
        g, beta are [B, T, H] or [T, H]; g in log space. initial_state is the
        K-last [N, H, V, K] recurrent state (flashinfer-native; matches the
        runtime's SSM state pool layout, so no transpose is needed here).

        Default returns ``(out, final_state)``: out [B, T, Hv, D] in q.dtype,
        final_state [N, H, V, K] (K-last, same layout as initial_state).

        When ``output_h=True`` returns
        ``(out, final_state, fi_checkpoints, checkpoint_cu_starts)`` where:
        - ``fi_checkpoints``: raw flashinfer checkpoint buffer, K-last
          ``[total_fi_ckpts, H, V, K]`` (float32). Checkpoint ``k`` within a
          sequence is the state *after* processing chunk ``k`` (= FLA ``h[k+1]``).
          Per-sequence count is ``L_i // CHUNK_SIZE``.
        - ``checkpoint_cu_starts``: int64 tensor of length ``N+1`` giving the
          cumulative start offset of each sequence's checkpoints in
          ``fi_checkpoints``.

        The caller indexes directly with flashinfer-native offsets rather than
        rebuilding the FLA h tensor.
        """
        batched = q.dim() == 4
        q3 = q.squeeze(0) if batched else q
        k3 = k.squeeze(0) if batched else k
        v3 = v.squeeze(0) if batched else v
        g2 = g.squeeze(0) if g.dim() == 3 else g
        beta2 = beta.squeeze(0) if beta.dim() == 3 else beta

        if qk_l2norm:
            q3 = _l2norm(q3)
            k3 = _l2norm(k3)

        head_dim = q3.shape[-1]
        if scale is None:
            scale = head_dim**-0.5

        # initial_state is already K-last [N, H, V, K] (flashinfer-native);
        # only a dtype cast + contiguity are needed, no transpose.
        fi_initial_state = initial_state.float().contiguous()

        state_checkpoints = None
        checkpoint_cu_starts = None
        if output_h:
            per_seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
            # flashinfer-native checkpoint count: only full chunks emit a checkpoint.
            per_seq_h_ckpts = per_seq_lens // CHUNK_SIZE
            total_h_ckpts = int(per_seq_h_ckpts.sum().item())
            H_state = fi_initial_state.shape[1]
            V_state = fi_initial_state.shape[-2]
            K_state = fi_initial_state.shape[-1]
            state_checkpoints = torch.empty(
                total_h_ckpts,
                H_state,
                V_state,
                K_state,
                device=fi_initial_state.device,
                dtype=torch.float32,
            )
            checkpoint_cu_starts = torch.zeros(
                per_seq_h_ckpts.numel() + 1,
                device=fi_initial_state.device,
                dtype=torch.int64,
            )
            checkpoint_cu_starts[1:] = torch.cumsum(per_seq_h_ckpts, dim=0)

        alpha = torch.exp(g2).float().contiguous()
        beta_f = beta2.float().contiguous()

        if _use_raw_kernel():
            # flashinfer's public `chunk_gated_delta_rule` is wrapped in
            # `@flashinfer_api(trace=...)`, which costs ~0.33 ms of Python per
            # call. At 48 GDN layers that is ~16 ms per prefill -- three times
            # the kernel's own host cost, and enough on its own to make the
            # fused kernel look slower than the nine-launch Triton scan.
            # Measured at 1024 tokens, 48 calls, host time, median of 5:
            #
            #     raw kernel, args pre-computed        5.41 ms
            #     + this wrapper's casts               5.97 ms
            #     public API (same work)              21.20 ms
            #
            # The dispatch the public API does for us is a compute-capability
            # switch we have already made in `is_available`, so call the
            # arch-specific kernel directly and keep the conversions here.
            out = torch.empty(
                v3.shape[0],
                v3.shape[1],
                v3.shape[2],
                device=v3.device,
                dtype=torch.bfloat16,
            )
            # `state` is the OUTPUT the kernel writes; `init_state` is the
            # incoming recurrent state. They must be separate tensors -- aliasing
            # them drops the initial state, which is silently correct only for a
            # first chunk and wrong for every continuation.
            final_state = torch.empty_like(fi_initial_state)
            _chunk_gated_delta_rule_raw(
                o=out,
                state=final_state,
                q=q3.contiguous(),
                k=k3.contiguous(),
                v=v3.contiguous(),
                init_state=fi_initial_state,
                alpha=alpha,
                beta=beta_f,
                cu_seqlens=cu_seqlens.to(torch.int64),
                scale=scale,
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=(
                    checkpoint_cu_starts.to(torch.int64)
                    if checkpoint_cu_starts is not None
                    else None
                ),
                checkpoint_every_n_tokens=CHUNK_SIZE if output_h else 0,
            )
        else:
            out, final_state = _chunk_gated_delta_rule(
                q3.contiguous(),
                k3.contiguous(),
                v3.contiguous(),
                g=alpha,
                beta=beta_f,
                scale=scale,
                initial_state=fi_initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=checkpoint_cu_starts,
                checkpoint_every_n_tokens=CHUNK_SIZE if output_h else 0,
            )

        out = out.to(q.dtype)
        if batched:
            out = out.unsqueeze(0)
        # final_state is already K-last [N, H, V, K]; no transpose needed.

        if not output_h:
            return GdnChunkPrefillResult(out=out, final_state=final_state)

        # Return raw flashinfer checkpoints, K-last [total_fi, H, V, K]. The
        # caller indexes directly using flashinfer-native offsets.
        return GdnChunkPrefillResult(
            out=out,
            final_state=final_state,
            h=state_checkpoints,
            h_cu_starts=checkpoint_cu_starts,
            h_layout=GdnCheckpointLayout.FLASHINFER,
        )


# ===-----------------------------------------------------------------------===#
# GDN decode / MTP (K-last, SM90+)
# ===-----------------------------------------------------------------------===#

gdn_decode_step = error_fn
gdn_decode_mtp = error_fn

if is_decode_available():

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="flashinfer_gdn_decode_step",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({SUPPORTED_HEAD_DIM}),
        },
        tags={"hopper", "latency"},
    )
    def gdn_decode_step(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: float | None = None,
        output_state_indices: torch.Tensor | None = None,
        use_qk_l2norm: bool = True,
    ) -> torch.Tensor:
        """Run one single-token (T=1) GDN decode step, K-last pool+indices.

        q, k are [B, 1, H, K]; v is [B, 1, HV, V]; a, b are [B, 1, HV].
        initial_state is the K-last [pool_size, HV, V, K] SSM state pool (same
        layout as gdn_chunk_prefill/gdn_decode_mtp -- no transpose needed at
        this boundary); initial_state_indices ([B]) selects each batch entry's
        read row. ``-1`` marks CUDA-graph padding rows and is handled
        internally by flashinfer (skipped on the float32 path, redirected to a
        sacrificial pool row 0 on the bf16 fast path) -- no caller-side clamp
        needed. The post-step state is written to output_state_indices
        (defaults to initial_state_indices when None; pass distinct output-page
        ids when the grouped cache writes the next state to another page).

        Returns the [B, 1, HV, V] decode output (q.dtype).
        """
        # Normalize decay inputs for FlashInfer's FP32 CuteDSL/DLPack boundary.
        A_log = A_log.detach().float()
        dt_bias = dt_bias.detach().float()
        out, _ = _gated_delta_rule_decode_pretranspose(
            q=q,
            k=k,
            v=v,
            state=None,
            A_log=A_log,
            a=a.to(q.dtype),
            dt_bias=dt_bias,
            b=b.to(q.dtype),
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
        )
        return out

    @register_kernel(
        "attention",
        "gdn_decode_mtp",
        name="flashinfer_gdn_decode_mtp",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({SUPPORTED_HEAD_DIM}),
        },
        tags={"hopper", "latency", "speculative-decoding"},
    )
    def gdn_decode_mtp(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: float | None = None,
        disable_state_update: bool = True,
        use_qk_l2norm: bool = True,
        intermediate_states_buffer: torch.Tensor | None = None,
        output_state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one multi-token (T>1) GDN MTP verify step, K-last pool+indices.

        q, k are [B, T, H, K]; v is [B, T, HV, V]; a, b are [B, T, HV].
        initial_state is the K-last [pool_size, HV, V, K] SSM state pool (same
        layout as gdn_chunk_prefill/gdn_decode_step); initial_state_indices
        ([B]) selects each batch entry's read row. When
        ``output_state_indices`` is None and ``disable_state_update=False``,
        the final state is written back to that same row.

        Padding behavior depends on the state dtype. The standalone FP32 MTP
        kernel skips a batch row when ``initial_state_indices`` is negative;
        its per-token state indices may remain negative for that skipped row.
        The BF16 fast path redirects negative read indices to row 0 and does
        not mask negative per-token scatter indices, so callers must provide
        non-negative destinations (typically by clamping padding to a reserved
        sacrificial row 0).

        intermediate_states_buffer: Optional batch-scoped ``[B, T, HV, V, K]``
        (K-last, same dtype as ``initial_state``) buffer that receives every
        step's post-update state at ``buffer[i_n, step]``.

        output_state_indices: Optional int32 ``[B, T]`` table. Each step's
        post-update state is scattered directly to the corresponding pool row.
        FlashInfer names this argument ``ssm_state_indices``. It is mutually
        exclusive with ``intermediate_states_buffer`` and requires
        ``disable_state_update=False``.

        Dispatches to the bf16-state MTP kernel when ``initial_state`` is
        bfloat16 with K == V == 128 (mirrors gdn_decode_step's own bf16/fp32
        dispatch via gated_delta_rule_decode_pretranspose); otherwise uses the
        float32-only standalone MTP entry point.

        Returns the [B, T, HV, V] decode output (q.dtype).
        """
        # Normalize decay inputs for FlashInfer's FP32 CuteDSL/DLPack boundary.
        A_log = A_log.detach().float()
        dt_bias = dt_bias.detach().float()
        K_dim = q.shape[-1]
        V_dim = v.shape[-1]
        use_bf16_state = (
            _gated_delta_rule_bf16_mtp is not None
            and initial_state.dtype == torch.bfloat16
            and K_dim == SUPPORTED_HEAD_DIM
            and V_dim == SUPPORTED_HEAD_DIM
        )
        if use_bf16_state:
            out = _gated_delta_rule_bf16_mtp(
                A_log=A_log,
                a=a.to(q.dtype),
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b.to(q.dtype),
                initial_state_source=initial_state,
                initial_state_indices=initial_state_indices,
                intermediate_states_buffer=intermediate_states_buffer,
                ssm_state_indices=output_state_indices,
                disable_state_update=disable_state_update,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                scale=scale,
            )
            return out
        out, _ = _gated_delta_rule_mtp(
            q=q,
            k=k,
            v=v,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            A_log=A_log,
            a=a.to(q.dtype),
            dt_bias=dt_bias,
            b=b.to(q.dtype),
            scale=scale,
            intermediate_states_buffer=intermediate_states_buffer,
            ssm_state_indices=output_state_indices,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
        )
        return out
