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

import math
import os

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

_MIN_BLOCK_KV = 32

# A high-precision query attending over an fp8 paged KV cache. The kernel loads
# the cache at its storage dtype and converts it up to the query dtype
# (`k.to(q.dtype)`), so the query sets the accumulate precision and quantizing
# it too would only lose accuracy -- and, since the output buffer is
# `empty_like(q)`, would hand the attention output back in e4m3 as well.
#
# Spelled out as mixed-role signatures because `format_signatures` puts one
# dtype on every role. Same shape as the gfx950 gluon MLA decode registration.
_Q_OVER_FP8_KV_SIGNATURES = frozenset(
    format_signature(
        q=dense_tensor_format(q_dtype),
        k_cache=dense_tensor_format(kv_dtype),
        v_cache=dense_tensor_format(kv_dtype),
    )
    for q_dtype in (torch.float16, torch.bfloat16)
    for kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    page_table,
    cache_seqlens,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    # WINDOW_LEFT is exclusive of the current token, so keep WINDOW_LEFT + 1
    # keys (window_left left keys plus the current one).
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_start_offset = cache_len - cur_batch_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    off_q = cur_q * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            token_indices = kv_start_offset + offs_n
            page_indices = token_indices // PAGE_SIZE
            page_offsets = token_indices - page_indices * PAGE_SIZE
            physical_pages = tl.load(
                page_table + cur_batch * page_table_stride_b + page_indices,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # int64 to avoid address overflow for large (>int32 range) KV caches
            kv_loc = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_q * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_q * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    page_table,
    cache_seqlens,
    num_kv_splits,
    max_kv_splits,
    page_table_stride_b,
    page_size,
    max_seqlen_q,
    window_left,
    sm_scale,
    logit_cap,
):
    platform = current_platform()

    BLOCK = 64
    # MI3xx uses a smaller block size to stay within the SGPR limit.
    if platform.is_amd:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    total_q, head_num = q.shape[0], q.shape[1]

    grid = (total_q, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if platform.is_amd:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        page_table,
        cache_seqlens,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        page_table_stride_b,
        page_size,
        MAX_SEQLEN_Q=max_seqlen_q,
        WINDOW_LEFT=window_left,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    page_table,
    cache_seqlens,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    # WINDOW_LEFT is exclusive of the current token, so keep WINDOW_LEFT + 1
    # keys (window_left left keys plus the current one).
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_start_offset = cache_len - cur_batch_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_q * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = cur_q * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            token_indices = kv_start_offset + offs_n
            page_indices = token_indices // PAGE_SIZE
            page_offsets = token_indices - page_indices * PAGE_SIZE
            physical_pages = tl.load(
                page_table + cur_batch * page_table_stride_b + page_indices,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # int64 to avoid address overflow for large (>int32 range) KV caches
            kv_loc = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_q * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_q * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    page_table,
    cache_seqlens,
    num_kv_splits,
    max_kv_splits,
    page_table_stride_b,
    page_size,
    max_seqlen_q,
    window_left,
    sm_scale,
    logit_cap,
):
    platform = current_platform()

    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # MI3xx uses a smaller block size for large heads to stay within shmem limits.
    if platform.is_amd and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    total_q, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        total_q,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if platform.is_amd:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16}
        num_stages = 1
    # Tuning overrides. gemma-4-31B's 10 full-attention layers are head_dim 512,
    # which no specialised kernel in this tree accepts (trtllm XQA caps at 256,
    # FA4 at 128), so this Triton path is the only one -- and on that shape it
    # runs at 23% of HBM bandwidth: 2.02 ms/layer against a 0.47 ms floor.
    # BLOCK_N=32 with BLOCK_DMODEL=512 and num_warps=4 leaves each thread four
    # elements per token. Let the sweep say what the shape wants.
    _nw = int(os.environ.get("TS_TRITON_DECODE_NUM_WARPS", "4"))
    num_stages = int(os.environ.get("TS_TRITON_DECODE_NUM_STAGES", str(num_stages)))
    _bn = os.environ.get("TS_TRITON_DECODE_BLOCK_N")
    if _bn:
        BLOCK = int(_bn)

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        page_table,
        cache_seqlens,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        page_table_stride_b,
        page_size,
        MAX_SEQLEN_Q=max_seqlen_q,
        WINDOW_LEFT=window_left,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=_nw,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    cache_seqlens,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_SEQLEN_Q: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head = tl.program_id(1)

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    # WINDOW_LEFT is exclusive of the current token, so keep WINDOW_LEFT + 1
    # keys (window_left left keys plus the current one).
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_q * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_q * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_q * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_buffer,
    cache_seqlens,
    num_kv_splits,
    max_kv_splits,
    max_seqlen_q,
    window_left,
    sinks=None,
):
    platform = current_platform()
    total_q, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    extra_kargs = {}
    if platform.is_amd:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16}

    grid = (total_q, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        cache_seqlens,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_SEQLEN_Q=max_seqlen_q,
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        WINDOW_LEFT=window_left,
        HAS_SINK=HAS_SINK,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    page_table,
    cache_seqlens,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    max_seqlen_q,
    page_table_stride_b,
    page_size,
    window_left,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        page_table,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        page_table_stride_b,
        page_size,
        max_seqlen_q,
        window_left,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        max_seqlen_q,
        window_left,
        sinks,
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    page_table,
    cache_seqlens,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    max_seqlen_q,
    page_table_stride_b,
    page_size,
    window_left,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        page_table,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        page_table_stride_b,
        page_size,
        max_seqlen_q,
        window_left,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        max_seqlen_q,
        window_left,
        sinks,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    page_table,
    cache_seqlens,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    max_seqlen_q,
    page_table_stride_b,
    page_size,
    window_left,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] == cache_seqlens.shape[0] * max_seqlen_q
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            page_table,
            cache_seqlens,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            max_seqlen_q,
            page_table_stride_b,
            page_size,
            window_left,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            page_table,
            cache_seqlens,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            max_seqlen_q,
            page_table_stride_b,
            page_size,
            window_left,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
        )


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="triton_mha_decode_with_kvcache_cached",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    )
    | _Q_OVER_FP8_KV_SIGNATURES,
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    # Flash-decoding split count. This was `max_kv_splits = 4` with
    # `num_kv_splits` filled with ONES, so every request ran a single split:
    # the effective grid is (total_q, ceil(q_heads/kv_group), 1), which for
    # gemma-4-31B's full-attention decode is 16 blocks on a 188-SM card -- 8.5%
    # of the machine, each block walking all 813 KV blocks of a 26k context in
    # sequence. Measured 2.02 ms/layer against a 0.47 ms bandwidth floor (23%).
    # Splitting the KV axis is what flash-decoding exists to do at batch 1.
    # Size the split count to the machine, not to a constant. Without a split
    # the grid is (total_q, ceil(q_heads/kv_group), 1); at batch 1 that is 16
    # blocks for gemma-4-31B full attention, 8.5% of a 188-SM card, each block
    # walking every KV block of the context in sequence. Measured on that shape
    # (head_dim 512, 26k ctx): 2.007 ms/layer at 1 split, 0.568 at 8 -- 3.54x,
    # and 23% -> 83% of HBM bandwidth. Output matches the 1-split reference to
    # 2.44e-04, one bf16 ULP at this magnitude.
    #
    # The cap matters as much as the floor: at batch 64 the head grid alone
    # already exceeds the SM count, and splitting there only adds stage-2
    # reduction work. 64 splits measured slower than 8 even at batch 1.
    _env_splits = os.environ.get("TS_TRITON_DECODE_KV_SPLITS")
    if _env_splits:
        max_kv_splits = int(_env_splits)
    else:
        _kv_group = max(1, q.shape[1] // k_cache.shape[2])
        _head_blocks = -(-q.shape[1] // min(16, _kv_group))
        _base_blocks = max(1, q.shape[0] * _head_blocks)
        _sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        _kv_blocks = max(1, -(-int(max_seqlen_k) // 32))
        _want = max(1, min(16, -(-_sms // _base_blocks), _kv_blocks))
        max_kv_splits = 1 << (_want.bit_length() - 1)
    attn_logits = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        dtype=torch.float32,
        device=q.device,
    )
    num_kv_splits = torch.full(
        (cache_seqlens.shape[0],), max_kv_splits, dtype=torch.int32, device=q.device
    )
    decode_attention_fwd(
        q,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        out,
        page_table,
        cache_seqlens,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        max_seqlen_q,
        page_table.stride(0),
        k_cache.shape[1],
        window_left,
        sm_scale=softmax_scale,
        logit_cap=logit_cap,
        sinks=sinks,
    )
    return out
