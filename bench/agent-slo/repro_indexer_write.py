"""Bisect the live FSVERIFY divergence at unit speed.

The engine's own verifier reports that both arms store, at the slot they claim
to have written, values that do not equal the `rotated` row they computed for
that slot.  Either the row/slot pairing is broken or the two arms rotate
differently; this script separates the two by asking each arm, per written slot,
what it stored and which row in `rotated` it corresponds to.

Run in-container on GPUs 6,7 (see the command in the accompanying notes).
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention import deepseek_v4_ops as ops
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4_cache_spec import (
    deepseek_v4_indexer_fp8_scale_bytes,
)
from tokenspeed_kernel.ops.attention.triton import deepseek_v4 as kernel_ops

HEAD = kernel_ops.DEEPSEEK_V4_INDEXER_DIM
ROPE = kernel_ops.DEEPSEEK_V4_ROPE_DIM
WIDTH = 2 * 2 * HEAD
STATE_BLOCK = 64
KV_BLOCK = 64
SCALE_BYTES = deepseek_v4_indexer_fp8_scale_bytes(HEAD)
SENTINEL = 0xA5


def build(
    m: int,
    *,
    ratio=4,
    degenerate_state=False,
    seed=0,
    n_state_blocks=8,
    state_block=None,
    strided=False,
    arena_pad=256,
    live_positions=False,
    bt_width=256,
    dup_slots=0,
):
    torch.manual_seed(seed)
    dev = "cuda:0"
    sb = STATE_BLOCK if state_block is None else state_block
    if strided:
        # The live engine hands both caches to the kernel as strided views of a
        # larger arena -- the verifier logs state stride (2304, 512, 1) against a
        # last dim that only spans 4*512 = 2048.  A kernel that assumes packed
        # rows reads the wrong elements.
        s_arena = torch.randn(n_state_blocks, sb * WIDTH + arena_pad, device=dev) * 0.3
        state = s_arena[:, : sb * WIDTH].view(n_state_blocks, sb, WIDTH)
    else:
        state = torch.randn(n_state_blocks, sb, WIDTH, device=dev) * 0.3
    n_req = 4
    block_table = (
        torch.arange(n_req * bt_width, device=dev, dtype=torch.int32).view(
            n_req, bt_width
        )
        % n_state_blocks
    )
    if live_positions:
        # The engine's own verifier logs positions=[3, 7, 11, ...]: every row is
        # a compress boundary, so every window is fully valid and spans exactly
        # two aligned state blocks.  A captured decode step looks like this, and
        # it is the geometry the fused kernel was measured on.
        positions = torch.arange(m, device=dev, dtype=torch.int32) * ratio + ratio - 1
    else:
        # Real prefill positions, exactly like a chunked prefill of m tokens.
        positions = torch.arange(m, device=dev, dtype=torch.int32)
    token_to_req = (torch.arange(m, device=dev, dtype=torch.int32) % n_req).contiguous()
    if degenerate_state:
        # The live window dump shows every row reading the same state slot.
        compressor_slot = torch.full((m,), 7, device=dev, dtype=torch.int32)
    else:
        compressor_slot = (torch.arange(m, device=dev, dtype=torch.int32) + 5).contiguous()
    if dup_slots:
        # The engine's verifier logs valid_slots=[64..127] with only 58 distinct
        # slots for 2048 valid rows: many rows target the SAME kv slot.  The
        # eager path is a single scatter, so the last index wins
        # deterministically; the fused kernel launches one program per row, so
        # which writer survives is a race.
        kv_slot = (
            torch.arange(m, device=dev, dtype=torch.int32) % dup_slots + KV_BLOCK
        ).contiguous()
    else:
        kv_slot = (torch.arange(m, device=dev, dtype=torch.int32) + KV_BLOCK).contiguous()
    rms_weight = torch.rand(HEAD, device=dev) + 0.5
    # A real RoPE table, not noise: the live divergence is in the rotation.
    half = ROPE // 2
    inv = 1.0 / (10000.0 ** (torch.arange(half, device=dev).float() / half))
    pos = torch.arange(65536, device=dev).float()[:, None]
    ang = pos * inv[None, :]
    cos_sin = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).contiguous()
    n_blocks = (KV_BLOCK + m + KV_BLOCK - 1) // KV_BLOCK + 2
    kv_width = KV_BLOCK * (HEAD + SCALE_BYTES)
    if strided:
        kv_arena = torch.full(
            (n_blocks, kv_width + arena_pad), SENTINEL, dtype=torch.uint8, device=dev
        )
        kv_cache = kv_arena[:, :kv_width]
    else:
        kv_cache = torch.full((n_blocks, kv_width), SENTINEL, dtype=torch.uint8, device=dev)
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
        "ratio": ratio,
        "m": m,
        "state_block": sb,
    }


def rotate(inp):
    """The shared prologue: compressed windows -> rope -> hadamard."""
    normed, valid = ops._compress_v4_state_windows_capturable(
        state_cache=inp["state"],
        token_to_req_indices=inp["token_to_req"],
        positions=inp["positions"],
        compressor_slot_mapping=inp["compressor_slot"],
        block_table=inp["block_table"],
        block_table_base_offsets=None,
        compressor_block_size=inp["state_block"],
        rms_norm_weight=inp["rms_weight"],
        rms_norm_eps=1e-6,
        compress_ratio=inp["ratio"],
        head_dim=HEAD,
        overlap=True,
    )
    num = normed.shape[0]
    cpos = (
        torch.div(
            inp["positions"][:num].to(torch.int64), inp["ratio"], rounding_mode="floor"
        )
        * inp["ratio"]
    )
    rot = ops._apply_gptj_rope_tail_rows(normed, cpos, inp["cos_sin"], ROPE)
    rot = ops._deepseek_v4_hadamard_rotate(rot).float()
    return rot, valid


def row_fields(cache, slot):
    blk, off = divmod(int(slot), KV_BLOCK)
    base = blk
    vals = cache[base, off * HEAD : (off + 1) * HEAD]
    sb = KV_BLOCK * HEAD + off * SCALE_BYTES
    return vals, cache[base, sb : sb + SCALE_BYTES]


def dequant(vals, sbytes):
    return vals.view(torch.float8_e4m3fn).float() * sbytes.view(torch.float32)


def best_row(rot, quant_row):
    """Which row of `rot` does this dequantized row look like?"""
    err = (rot - quant_row[None, :]).abs().amax(dim=-1)
    k = int(err.argmin().item())
    return k, float(err[k].item())


def run(label, **kw):
    inp = build(**kw)
    ratio = inp["ratio"]
    m = inp["m"]
    rot, valid = rotate(inp)
    num = rot.shape[0]
    slots = inp["kv_slot"][:num].to(torch.int64)
    valid = valid[:num] & (slots >= 0)
    idx = valid.nonzero().flatten()
    print(f"\n=== {label} ===  m={m} valid={int(valid.sum())} rows={num}")
    if idx.numel() == 0:
        print("  no valid rows")
        return

    eager = inp["kv_cache"].clone()
    ops._write_deepseek_v4_indexer_fp8_cache_capturable(
        rot, eager, inp["kv_slot"][:num], valid, block_size=KV_BLOCK
    )
    fused = inp["kv_cache"].clone()
    ops._triton_fused_csa_indexer_fp8_cache_insert(
        state_cache=inp["state"],
        token_to_req_indices=inp["token_to_req"],
        positions=inp["positions"],
        compressor_slot_mapping=inp["compressor_slot"],
        block_table=inp["block_table"],
        compressor_block_size=inp["state_block"],
        rms_norm_weight=inp["rms_weight"],
        rms_norm_eps=1e-6,
        cos_sin_cache=inp["cos_sin"],
        kv_cache_2d=fused,
        kv_slot_mapping=inp["kv_slot"],
        kv_cache_block_size=KV_BLOCK,
        compress_ratio=ratio,
        block_table_base_offsets=None,
    )
    torch.cuda.synchronize()

    for arm_name, cache in (("eager", eager), ("fused", fused)):
        worst = 0.0
        n_bad = 0
        first = None
        for j in range(min(idx.numel(), 512)):
            r = int(idx[j])
            slot = int(slots[r])
            vals, sbytes = row_fields(cache, slot)
            dq = dequant(vals, sbytes)
            err = float((dq - rot[r]).abs().amax().item())
            if err > 1e-3:
                n_bad += 1
                if first is None:
                    k, kerr = best_row(rot, dq)
                    first = (j, r, slot, err, k, kerr)
            worst = max(worst, err)
        tag = f"  {arm_name:5s}: worst |dequant(slot) - rotated[row]| = {worst:.5f}  ({n_bad} of {min(idx.numel(), 512)} slots off)"
        print(tag)
        if first is not None:
            j, r, slot, err, k, kerr = first
            print(
                f"          first off: valid#{j} row={r} slot={slot} err={err:.5f}"
                f" -> best matching row={k} (err {kerr:.5f}, row==r? {k == r})"
            )

    # Direct arm-vs-arm agreement on the written slots.
    n_diff = 0
    worst_ae = 0.0
    for j in range(min(idx.numel(), 512)):
        r = int(idx[j])
        slot = int(slots[r])
        ev, es = row_fields(eager, slot)
        fv, fs = row_fields(fused, slot)
        d = float((dequant(ev, es) - dequant(fv, fs)).abs().amax().item())
        worst_ae = max(worst_ae, d)
        if d > 1e-3 or not torch.equal(es, fs):
            n_diff += 1
    print(f"  eager vs fused: worst |diff| = {worst_ae:.5f}, {n_diff} slots differ")

    # The quantity the engine's verifier reports, so the reproducer speaks the
    # same language: how many rows chose a different power-of-two divisor.
    n_flip = 0
    for j in range(idx.numel()):
        r = int(idx[j])
        slot = int(slots[r])
        _, es = row_fields(eager, slot)
        _, fs = row_fields(fused, slot)
        if not torch.equal(es, fs):
            n_flip += 1
    print(f"  rows with a different divisor: {n_flip} of {int(idx.numel())}")


def eager_write(inp, rot, valid, num):
    cache = inp["kv_cache"].clone()
    ops._write_deepseek_v4_indexer_fp8_cache_capturable(
        rot, cache, inp["kv_slot"][:num], valid, block_size=KV_BLOCK
    )
    torch.cuda.synchronize()
    return cache


def fused_write(inp, ratio, num):
    cache = inp["kv_cache"].clone()
    ops._triton_fused_csa_indexer_fp8_cache_insert(
        state_cache=inp["state"],
        token_to_req_indices=inp["token_to_req"],
        positions=inp["positions"],
        compressor_slot_mapping=inp["compressor_slot"],
        block_table=inp["block_table"],
        compressor_block_size=inp["state_block"],
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


def check_determinism(label, reps=6, **kw):
    """A cache write must be a function of its inputs.

    A kernel whose result changes between identical calls has a race, and
    "the corrupt arm can look faster" applies with full force -- a racy insert
    would move top-k without moving the throughput number that justified it.
    The eager chain is plain torch ops and is the control.
    """
    inp = build(**kw)
    ratio = inp["ratio"]
    rot, valid = rotate(inp)
    num = rot.shape[0]
    slots = inp["kv_slot"][:num].to(torch.int64)
    valid = valid[:num] & (slots >= 0)
    print(f"\n=== {label} === reps={reps}")

    ref = eager_write(inp, rot, valid, num)
    e_bad = sum(
        1 for _ in range(reps) if not torch.equal(eager_write(inp, rot, valid, num), ref)
    )
    print(f"  eager: {e_bad}/{reps} repeats differ from the first")

    f_first = fused_write(inp, ratio, num)
    f_bad = 0
    worst = 0
    for _ in range(reps):
        c = fused_write(inp, ratio, num)
        if not torch.equal(c, f_first):
            f_bad += 1
            worst = max(worst, int((c != f_first).sum().item()))
    print(f"  fused: {f_bad}/{reps} repeats differ from the first (max {worst} bytes)")

    d = (f_first != ref).sum().item()
    print(f"  fused vs eager: {int(d)} bytes differ")

    # Wherever the arms disagree, ask how close the row's magnitude ratio sits
    # to a power of two.  Both arms pick the tightest power-of-two divisor with
    # `ceil(log2(amax / 448))`, so a last-bit difference in `amax` only changes
    # the chosen divisor when that ratio is within a rounding error of 2**k --
    # and when it does, all 128 value bytes of the row change together.
    import math

    h = ops._deepseek_v4_hadamard_rotate(rot).float()
    amax = h.abs().amax(dim=-1)
    n_flip = 0
    shown = 0
    for r in range(num):
        if not bool(valid[r]):
            continue
        slot = int(slots[r])
        _, es = row_fields(ref, slot)
        _, fs = row_fields(f_first, slot)
        e_u32 = int(es.view(torch.uint32).item())
        f_u32 = int(fs.view(torch.uint32).item())
        if e_u32 == f_u32:
            continue
        n_flip += 1
        e_div = float(es.view(torch.float32).item())
        f_div = float(fs.view(torch.float32).item())
        ratio_v = float(amax[r].item()) / 448.0
        lg = math.log2(ratio_v) if ratio_v > 0 else float("nan")
        if shown < 6:
            shown += 1
            print(
                f"    row={r} slot={slot} eager_div={e_div:g} fused_div={f_div:g}"
                f"  log2(amax/448)={lg:.9f}  dist_to_int={abs(lg - round(lg)):.3e}"
                f"  eager_bytes_diff={int((ref[slot // KV_BLOCK] != f_first[slot // KV_BLOCK]).sum().item())}"
            )
    print(f"  rows whose chosen divisor differs: {n_flip}")


if __name__ == "__main__":
    run(
        "DUPLICATE kv slots, live geometry (58 slots for 2048 rows)",
        m=2048,
        state_block=4,
        strided=True,
        degenerate_state=True,
        n_state_blocks=1,
        live_positions=True,
        bt_width=2048,
        dup_slots=58,
    )
    run(
        "LIVE geometry m=2048 sb=4 strided degenerate single-block",
        m=2048,
        state_block=4,
        strided=True,
        degenerate_state=True,
        n_state_blocks=1,
        live_positions=True,
        bt_width=2048,
    )
    check_determinism("F state_block=4 strided=False", state_block=4, m=8192)
    check_determinism("F state_block=64 strided=False", state_block=64, m=8192)
    for sb in (64, 4):
        for strided in (False, True):
            for degen in (False, True):
                run(
                    f"E state_block={sb} strided={strided} degenerate_state={degen}",
                    m=8192,
                    state_block=sb,
                    strided=strided,
                    degenerate_state=degen,
                )
