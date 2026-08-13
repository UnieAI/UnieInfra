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

"""Decode-shaped bf16 MoE: split the reduction, not the output.

At one token a top-k-of-many-experts layer is not a GEMM. On
Qwen3.6-35B-A3B (hidden 2048, 256 experts, top-8, intermediate 512) the
arithmetic is ~42 MFLOP while the weights read are ~50 MB, so the only
number that means anything is achieved bandwidth.

A grouped-GEMM tiling has nothing to parallelise over at that size: with one
token there is no M to tile, and tiling N alone leaves each block walking the
whole reduction serially, so too few loads are ever in flight to saturate HBM.
Measured on one RTX PRO 6000 (sm_120, HBM peak 1792 GB/s), a fresh expert set
per iteration so the 128 MB L2 cannot hold the working set:

    roofline                                0.028 ms   1792 GB/s   100%
    this kernel                             0.060 ms    834 GB/s    47%
    vLLM fused_experts, same harness        0.147 ms    342 GB/s    19%

K is the axis with parallelism left, so it is the one that gets split. The
partials are reduced through a buffer rather than `atomic_add`: atomics make
the sum order depend on which program finishes first, and a forward pass that
is not bit-reproducible cannot be used to validate anything downstream of it.

The partials are small -- 256 KB against 33.5 MB of weights -- so what costs is
the number of launches, not their bytes. Reducing them in torch and applying
the activation as separate ops took six launches per layer and 0.098 ms; the
two reductions below fold the activation and the top-k weighting into their own
epilogues, giving four launches and 0.060 ms.
"""

import torch
import triton
import triton.language as tl

# Above this many (token, expert) pairs the grouped-GEMM path has real work to
# tile and wins; below it, it has nothing to tile over. The crossover was not
# swept -- it is set at the decode batch sizes this path was measured on.
DECODE_MAX_PAIRS = 64

_SPLIT_K = 8
_BLOCK_N = 64
_BLOCK_K = 64
_BLOCK_RED = 128


@triton.jit
def _mv_splitk(
    x_ptr,
    w_ptr,
    ids_ptr,
    part_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    X_PER_PAIR: tl.constexpr,
):
    """``part[k, pair, n] = sum over this split of w[ids[pair], n, :] * x[..., :]``.

    ``X_PER_PAIR`` picks the two call sites: the gate/up matmul reads the
    hidden vector of the pair's token, the down matmul reads the pair's own
    intermediate.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    eid = tl.load(ids_ptr + pid_m)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = pid_m * K if X_PER_PAIR else (pid_m // TOPK) * K
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = k_start + k_per_split
    # Masked because BLOCK_K need not divide k_per_split: with K=512 on the
    # down matmul, SPLIT_K=16 leaves 32 per split against a 64-wide block. The
    # unmasked form did not fault, it returned 1.48e+38 while still timing
    # normally -- only the numerical check caught it.
    for k0 in range(0, k_per_split, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        m = offs_k < k_end
        xv = tl.load(x_ptr + x_base + offs_k, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(
            w_ptr + eid * N * K + offs_n[:, None] * K + offs_k[None, :],
            mask=m[None, :],
            other=0.0,
        )
        acc += tl.sum(wv.to(tl.float32) * xv[None, :], axis=1)

    tl.store(part_ptr + pid_k * (tl.num_programs(0) * N) + pid_m * N + offs_n, acc)


@triton.jit
def _reduce_silu(
    part_ptr,
    out_ptr,
    PAIRS: tl.constexpr,
    N2: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the split-K partials and apply ``silu(gate) * up`` in one pass.

    w13 concatenates gate and up on its output dim, so both live in the same
    partials buffer and neither the reduction nor the activation needs a round
    trip through memory.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    half: tl.constexpr = N2 // 2
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    g = tl.zeros((BLOCK,), dtype=tl.float32)
    u = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(SPLIT_K):
        base = k * (PAIRS * N2) + pid_m * N2
        g += tl.load(part_ptr + base + offs)
        u += tl.load(part_ptr + base + half + offs)
    tl.store(out_ptr + pid_m * half + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16))


@triton.jit
def _reduce_weighted(
    part_ptr,
    gate_ptr,
    out_ptr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the split-K partials AND the token's top-k experts, weighted."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in range(TOPK):
        pair = pid_t * TOPK + j
        gv = tl.load(gate_ptr + pair)
        s = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(SPLIT_K):
            s += tl.load(part_ptr + k * (PAIRS * N) + pair * N + offs)
        acc += gv * s
    tl.store(out_ptr + pid_t * N + offs, acc.to(tl.bfloat16))


def _partials(x, w, ids, *, x_per_pair, topk, pairs):
    n, k = w.shape[1], w.shape[2]
    part = torch.empty((_SPLIT_K, pairs, n), device=x.device, dtype=torch.float32)
    _mv_splitk[(pairs, triton.cdiv(n, _BLOCK_N), _SPLIT_K)](
        x,
        w,
        ids,
        part,
        K=k,
        N=n,
        TOPK=topk,
        SPLIT_K=_SPLIT_K,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        X_PER_PAIR=x_per_pair,
        num_warps=4,
        num_stages=3,
    )
    return part


def decode_moe_body(x, w, topk_weights, topk_ids):
    """bf16 MoE for decode-shaped batches.

    Args:
        x: Hidden states ``[tokens, hidden]``.
        w: Module holding ``w13_weight [E, 2I, H]`` and ``w2_weight [E, H, I]``
            in their checkpoint layout -- this path reads them directly and
            needs none of the transposed copies the ragged matmul wants.
        topk_weights: Router weights ``[tokens, top_k]``.
        topk_ids: Expert ids ``[tokens, top_k]``.

    Returns:
        ``[tokens, hidden]`` in the input dtype.
    """
    tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    pairs = tokens * topk
    inter_dim = w.w2_weight.shape[2]

    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    gate = topk_weights.reshape(-1).to(torch.float32).contiguous()

    p1 = _partials(
        x.reshape(-1), w.w13_weight, ids, x_per_pair=False, topk=topk, pairs=pairs
    )
    inter = torch.empty((pairs, inter_dim), device=x.device, dtype=torch.bfloat16)
    _reduce_silu[(pairs, triton.cdiv(inter_dim, _BLOCK_RED))](
        p1,
        inter,
        PAIRS=pairs,
        N2=w.w13_weight.shape[1],
        SPLIT_K=_SPLIT_K,
        BLOCK=_BLOCK_RED,
        num_warps=4,
    )

    p2 = _partials(
        inter.reshape(-1), w.w2_weight, ids, x_per_pair=True, topk=topk, pairs=pairs
    )
    out = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    _reduce_weighted[(tokens, triton.cdiv(hidden, _BLOCK_RED))](
        p2,
        gate,
        out,
        N=hidden,
        TOPK=topk,
        SPLIT_K=_SPLIT_K,
        PAIRS=pairs,
        BLOCK=_BLOCK_RED,
        num_warps=4,
    )
    return out.to(x.dtype)
