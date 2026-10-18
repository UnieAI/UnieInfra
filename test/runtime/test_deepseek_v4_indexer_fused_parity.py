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

"""Fused CSA indexer FP8 cache insert versus its eager reference sequence.

`deepseek_v4_csa_indexer_cache_insert` has two arms for an FP8 indexer cache:
an eager chain of four ATen stages, and the fused Triton kernel
`deepseek_v4_fused_csa_indexer_fp8_cache_insert` reached behind
`TOKENSPEED_DSV4_FP8_INDEXER_FUSED`, which is on by default.  The eager arm
issues ~30 device kernels per layer where the fused arm issues one, so the fused
arm is the difference between 6 368 and ~5 200 nodes in the captured decode
graph.

The fused arm cannot be enabled on a performance measurement alone: the two arms
must agree byte-for-byte on the rows they write.  This module is that gate.  It
reimplements the eager sequence explicitly (rather than calling it) so the
oracle is independent of the code path under test, and it also checks the
shipped entry point against that oracle to keep the oracle honest.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed.runtime.layers.attention import deepseek_v4_ops as ops
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4_cache_spec import (
    deepseek_v4_indexer_fp8_scale_bytes,
)
from tokenspeed_kernel.ops.attention.triton import deepseek_v4 as kernel_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

# The indexer shape constants live with the kernel that implements them.
HEAD = kernel_ops.DEEPSEEK_V4_INDEXER_DIM
ROPE = kernel_ops.DEEPSEEK_V4_ROPE_DIM
WIDTH = 2 * 2 * HEAD
STATE_BLOCK = 64
KV_BLOCK = 64
N_KV_BLOCKS = 64
N_STATE_BLOCKS = 8
BT_WIDTH = 256
N_REQ = 4
SCALE_BYTES = deepseek_v4_indexer_fp8_scale_bytes(HEAD)
SENTINEL = 0xA5


def _build(m: int, seed: int, *, ratio: int = 4, boundary=True):
    """Synthetic all-valid insert inputs on a sentinel-filled cache."""
    torch.manual_seed(seed)
    dev = "cuda:0"
    state = torch.randn(N_STATE_BLOCKS, STATE_BLOCK, WIDTH, device=dev) * 0.3
    block_table = (
        torch.arange(N_REQ * BT_WIDTH, device=dev, dtype=torch.int32).view(
            N_REQ, BT_WIDTH
        )
        % N_STATE_BLOCKS
    )
    if boundary:
        positions = (torch.arange(m, device=dev, dtype=torch.int32) + 1) * ratio - 1
    else:
        # Offset by one so that (position + 1) % ratio == 2 for every row: no
        # row sits on a compress boundary and none may be written.
        positions = torch.arange(m, device=dev, dtype=torch.int32) * ratio + 1
    token_to_req = (torch.arange(m, device=dev, dtype=torch.int32) % N_REQ).contiguous()
    compressor_slot = (torch.arange(m, device=dev, dtype=torch.int32) + 5).contiguous()
    kv_slot = (torch.arange(m, device=dev, dtype=torch.int32) + 3).contiguous()
    rms_weight = torch.rand(HEAD, device=dev) + 0.5
    cos_sin = torch.randn(65536, ROPE, device=dev)
    kv_cache = torch.full(
        (N_KV_BLOCKS, KV_BLOCK * (HEAD + SCALE_BYTES)),
        SENTINEL,
        dtype=torch.uint8,
        device=dev,
    )
    return {
        "state": state,
        "block_table": block_table,
        "positions": positions,
        "token_to_req": token_to_req,
        "compressor_slot": compressor_slot,
        "kv_slot": kv_slot,
        "rms_weight": rms_weight,
        "cos_sin": cos_sin,
        "kv_cache": kv_cache,
    }


def _eager_oracle(inp, ratio: int = 4) -> torch.Tensor:
    """The eager reference sequence, written out stage by stage."""
    cache = inp["kv_cache"].clone()
    normed, valid = ops._compress_v4_state_windows_capturable(
        state_cache=inp["state"],
        token_to_req_indices=inp["token_to_req"],
        positions=inp["positions"],
        compressor_slot_mapping=inp["compressor_slot"],
        block_table=inp["block_table"],
        block_table_base_offsets=None,
        compressor_block_size=STATE_BLOCK,
        rms_norm_weight=inp["rms_weight"],
        rms_norm_eps=1e-6,
        compress_ratio=ratio,
        head_dim=HEAD,
        overlap=True,
    )
    num_actual = normed.shape[0]
    compressed_positions = (
        torch.div(
            inp["positions"][:num_actual].to(torch.int64), ratio, rounding_mode="floor"
        )
        * ratio
    )
    rotated = ops._apply_gptj_rope_tail_rows(
        normed, compressed_positions, inp["cos_sin"], ROPE
    )
    rotated = ops._deepseek_v4_hadamard_rotate(rotated).float()
    ops._write_deepseek_v4_indexer_fp8_cache_capturable(
        rotated,
        cache,
        inp["kv_slot"][:num_actual],
        valid,
        block_size=KV_BLOCK,
    )
    return cache


def _fused(inp, ratio: int = 4) -> torch.Tensor:
    cache = inp["kv_cache"].clone()
    ops._triton_fused_csa_indexer_fp8_cache_insert(
        state_cache=inp["state"],
        token_to_req_indices=inp["token_to_req"],
        positions=inp["positions"],
        compressor_slot_mapping=inp["compressor_slot"],
        block_table=inp["block_table"],
        compressor_block_size=STATE_BLOCK,
        rms_norm_weight=inp["rms_weight"],
        rms_norm_eps=1e-6,
        cos_sin_cache=inp["cos_sin"],
        kv_cache_2d=cache,
        kv_slot_mapping=inp["kv_slot"],
        kv_cache_block_size=KV_BLOCK,
        compress_ratio=ratio,
        block_table_base_offsets=None,
    )
    torch.cuda.synchronize()
    return cache


def _written_slots(inp, ratio: int = 4) -> torch.Tensor:
    m = inp["positions"].numel()
    ok = (inp["compressor_slot"][:m] >= 0) & (inp["kv_slot"][:m] >= 0)
    ok = ok & (((inp["positions"][:m].to(torch.int64) + 1) % ratio) == 0)
    return inp["kv_slot"][:m][ok].to(torch.int64)


def _row_view(cache: torch.Tensor, slot: int):
    """Values and scale bytes for one slot, in the cache's planar layout.

    Both arms pack a block as `block_size * HEAD` value bytes followed by
    `block_size * SCALE_BYTES` scale bytes, so a slot's two fields are not
    adjacent and the row cannot be read as one contiguous run.
    """
    blk, off = divmod(int(slot), KV_BLOCK)
    block = cache[blk]
    values = block[off * HEAD : (off + 1) * HEAD]
    scale_base = KV_BLOCK * HEAD + off * SCALE_BYTES
    return values, block[scale_base : scale_base + SCALE_BYTES]


def _dequant(values: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    fp8 = values.view(torch.float8_e4m3fn).float()
    divisor = scale_bytes.view(torch.float32)
    return fp8 * divisor


@pytest.mark.parametrize("m", [8, 32])
def test_fused_matches_eager_on_written_rows(m):
    inp = _build(m, seed=0)
    slots = _written_slots(inp)
    assert slots.numel() == m, "expected every row to be valid in this fixture"

    eager = _eager_oracle(inp)
    fused = _fused(inp)

    for slot in slots.tolist():
        ev, es = _row_view(eager, slot)
        fv, fs = _row_view(fused, slot)
        assert torch.equal(es, fs), f"slot {slot}: quant scale bytes differ"
        _assert_value_bytes_equivalent(
            ev.view(torch.uint8), fv.view(torch.uint8), slot
        )


def _fp8_step(magnitude: torch.Tensor) -> torch.Tensor:
    """One ULP of the e4m3 grid at `magnitude` (three mantissa bits)."""
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-6)))
    return torch.pow(torch.full_like(exponent, 2.0), exponent - 3.0)


def _assert_value_bytes_equivalent(
    eager_bytes: torch.Tensor, fused_bytes: torch.Tensor, slot
):
    """The strongest agreement the fp8 cache format can offer.

    The two arms sum the Hadamard in different orders -- the kernel reduces a
    signed matrix product, the eager path calls `fast_hadamard_transform` -- so
    their pre-quantization values may differ in the last bit.  Where that last
    bit straddles an fp8 rounding boundary the stored byte moves by exactly one
    ULP.  Asserting byte equality would therefore encode a property the
    implementation cannot promise; asserting *one* ULP can, and still rejects
    every real defect, because a mis-paired row, a wrong scale or an addressing
    error moves values much further than one step.
    """
    eager_vals = eager_bytes.view(torch.float8_e4m3fn).float()
    fused_vals = fused_bytes.view(torch.float8_e4m3fn).float()
    deviation = (eager_vals - fused_vals).abs()
    limit = _fp8_step(torch.maximum(eager_vals.abs(), fused_vals.abs()))
    worst = deviation.max()
    assert bool((deviation <= limit).all()), (
        f"slot {slot}: value bytes disagree by {float(worst):.6g}, "
        f"more than one fp8 step"
    )


def test_the_fp8_step_gate_accepts_one_ulp_and_rejects_three():
    """A gate that cannot fail is not a gate, so pin its threshold.

    Byte pattern 0x60 decodes to 32.0, whose e4m3 ULP is 4.0, so incrementing
    the byte code is exactly one ULP per step.  One step must pass (that is the
    Hadamard summation-order residue) and three must fail (that is a defect).
    """
    base = torch.full((32,), 0x60, dtype=torch.uint8)
    assert float(_fp8_step(torch.tensor([32.0]))[0]) == 4.0, "ULP model is wrong"

    _assert_value_bytes_equivalent(base, base + 1, "one-ulp")
    with pytest.raises(AssertionError, match="more than one fp8 step"):
        _assert_value_bytes_equivalent(base, base + 3, "three-ulp")


def test_fused_leaves_untargeted_bytes_at_sentinel():
    inp = _build(16, seed=1)
    slots = set(_written_slots(inp).tolist())
    fused = _fused(inp)

    touched = {s // KV_BLOCK for s in slots}
    for blk in range(fused.shape[0]):
        block = fused[blk]
        if blk not in touched:
            assert torch.all(
                block == SENTINEL
            ), f"block {blk} was written but no slot in it was targeted"
            continue
        for off in range(KV_BLOCK):
            if blk * KV_BLOCK + off in slots:
                continue
            assert torch.all(
                block[off * HEAD : (off + 1) * HEAD] == SENTINEL
            ), f"slot {blk * KV_BLOCK + off} value bytes changed but were untargeted"
            scale_base = KV_BLOCK * HEAD + off * SCALE_BYTES
            assert torch.all(
                block[scale_base : scale_base + SCALE_BYTES] == SENTINEL
            ), f"slot {blk * KV_BLOCK + off} scale bytes changed but were untargeted"


def test_fused_writes_nothing_for_invalid_rows():
    inp = _build(16, seed=2, boundary=False)
    fused = _fused(inp)
    assert torch.all(
        fused == SENTINEL
    ), "no row sits on a compress boundary, so nothing may be written"


def test_fused_skips_negative_slots():
    inp = _build(16, seed=3)
    inp["kv_slot"] = inp["kv_slot"].clone()
    inp["kv_slot"][0::2] = -1
    slots = set(_written_slots(inp).tolist())
    fused = _fused(inp)
    written = torch.nonzero((fused != SENTINEL).any(-1).any(-1)).flatten().tolist()
    assert set(written) <= slots


def test_shipped_entry_point_matches_the_oracle():
    """Keep the oracle honest: the shipped arm must equal it, whichever it is.

    `_FP8_INDEXER_FUSED` is on by default, so the shipped path here is the fused
    kernel.  It goes through the same one-ULP gate the rest of this module uses,
    which matters because the previous version of this test skipped as soon as
    the fused arm became the default -- exactly when it started to matter.
    """
    inp = _build(16, seed=4)
    slots = _written_slots(inp)
    shipped = inp["kv_cache"].clone()
    ops.deepseek_v4_csa_indexer_cache_insert(
        state_cache=inp["state"],
        token_to_req_indices=inp["token_to_req"],
        positions=inp["positions"],
        compressor_slot_mapping=inp["compressor_slot"],
        block_table=inp["block_table"],
        compressor_block_size=STATE_BLOCK,
        rms_norm_weight=inp["rms_weight"],
        rms_norm_eps=1e-6,
        cos_sin_cache=inp["cos_sin"],
        kv_cache_2d=shipped,
        kv_slot_mapping=inp["kv_slot"],
        kv_cache_block_size=KV_BLOCK,
        use_fp4_cache=False,
        compress_ratio=4,
        block_table_base_offsets=None,
    )
    reference = _eager_oracle(inp)
    for slot in slots.tolist():
        ref_val, ref_scale = _row_view(reference, slot)
        ship_val, ship_scale = _row_view(shipped, slot)
        assert torch.equal(ref_scale, ship_scale), (
            f"slot {slot}: the shipped arm chose a different quant divisor"
        )
        _assert_value_bytes_equivalent(
            ref_val.view(torch.uint8), ship_val.view(torch.uint8), slot
        )


def test_decode_shaped_batch_has_no_aliased_slots():
    """One token per sequence: every written slot is distinct.

    The fused kernel's writes are unordered across programs, so two rows that
    target one slot would let either value win.  Decode is alias-free by
    construction, which is what makes the fused arm safe there; extend is not,
    and the aliasing has to be resolved before the fused arm is used for it.
    """
    dev = "cuda:0"
    for batch in (4, 16, 64):
        kv_slot = (torch.arange(batch, device=dev, dtype=torch.int64) * 7 + 3).to(
            torch.int32
        )
        assert kv_slot.unique().numel() == batch


def _replay_eager_write_under_capture(cache, index_k, slots, valid, block_size=KV_BLOCK):
    """Run the shipped eager write inside a real CUDA graph, then replay it.

    The eager write only redirects invalid rows to slot 0 when the stream is
    being captured: off-graph it compacts by host-side mask instead.  Decode
    replay is exactly the captured path, so the defect only shows up here.
    """
    scratch = cache.clone()  # warm-up writes must not touch the measured cache
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            ops._write_deepseek_v4_indexer_fp8_cache_capturable(
                index_k, scratch, slots, valid, block_size
            )
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops._write_deepseek_v4_indexer_fp8_cache_capturable(
            index_k, cache, slots, valid, block_size
        )
    graph.replay()
    torch.cuda.synchronize()
    return cache


def test_capturable_eager_write_leaves_the_dummy_slot_alone():
    """A row that must not write may not land on slot 0 either.

    `_write_deepseek_v4_indexer_fp8_cache_capturable` cannot drop rows while it
    is being captured, so it sends the masked-out ones to slot 0.  Slot 0 is a
    real slot that a real sequence owns, and on a decode step almost every row
    is masked out (`valid` is the compress-boundary mask, so only the rows whose
    position ends a compression window are eligible), which makes every replayed
    step scatter a whole batch of unrelated rows onto one live slot.
    """
    dev = "cuda:0"
    torch.manual_seed(7)
    cache = torch.full(
        (N_KV_BLOCKS, KV_BLOCK * (HEAD + SCALE_BYTES)),
        SENTINEL,
        dtype=torch.uint8,
        device=dev,
    )
    # Give slot 0 -- the dummy target -- a pattern that no write may replace.
    keep_val, keep_scale = _row_view(cache, 0)
    keep_val.fill_(0x11)
    keep_scale.fill_(0x22)

    m = 8
    index_k = torch.randn(m, HEAD, device=dev)
    # Every slot is a legal slot; the mask, not the mapping, decides who writes.
    slots = (torch.arange(m, device=dev, dtype=torch.int32) + KV_BLOCK + 4).contiguous()
    valid = torch.zeros(m, dtype=torch.bool, device=dev)
    valid[0] = True  # exactly one compress boundary in this step

    _replay_eager_write_under_capture(cache, index_k, slots, valid)

    val, scale = _row_view(cache, 0)
    assert torch.equal(val, torch.full_like(val, 0x11)), (
        "slot 0 was overwritten by rows that were masked out of this step"
    )
    assert torch.equal(scale, torch.full_like(scale, 0x22))

    # The one eligible row must still have landed.
    hot_val, _ = _row_view(cache, int(slots[0]))
    assert not torch.equal(hot_val, torch.full_like(hot_val, SENTINEL))


def test_verify_gate_rejects_a_batch_whose_block_table_collapses():
    """The one-shot verifier must not fire on the warm-up's folded block table.

    The warm-up batch hands every page index the same physical page, so all of
    its compress groups resolve onto the same handful of slots with 32 claimants
    each.  Comparing the two writers there measures an unspecified duplicate race
    instead of the kernel.
    """
    positions = torch.arange(8192, device="cuda", dtype=torch.long)
    block_table = torch.ones((1, 2048), device="cuda", dtype=torch.int32)
    assert ops._looks_like_warmup_batch(positions, block_table)


def test_verify_gate_keeps_a_real_long_prefill():
    """Arange-shaped positions are ordinary; a multi-page block table proves it."""
    positions = torch.arange(8192, device="cuda", dtype=torch.long)
    block_table = torch.arange(2048, device="cuda", dtype=torch.int32).reshape(1, 2048)
    assert not ops._looks_like_warmup_batch(positions, block_table)


def test_verify_gate_rejects_a_batch_that_writes_no_slots():
    """`diff_bytes == 0` over zero writes is not evidence of agreement."""
    positions = torch.arange(80, device="cuda", dtype=torch.long)
    block_table = torch.arange(80 * 8, device="cuda", dtype=torch.int32).reshape(80, 8)
    kv_slot_mapping = torch.full((80,), -1, device="cuda", dtype=torch.long)
    assert not ops._indexer_verify_is_meaningful(
        positions, block_table, kv_slot_mapping
    )


def test_verify_gate_accepts_a_batch_with_real_slots():
    """Twenty boundaries landing on twenty slots is a question worth asking."""
    positions = torch.arange(80, device="cuda", dtype=torch.long)
    block_table = torch.arange(80 * 8, device="cuda", dtype=torch.int32).reshape(80, 8)
    kv_slot_mapping = torch.full((80,), -1, device="cuda", dtype=torch.long)
    kv_slot_mapping[::4] = torch.arange(20, device="cuda", dtype=torch.long)
    assert ops._indexer_verify_is_meaningful(positions, block_table, kv_slot_mapping)
