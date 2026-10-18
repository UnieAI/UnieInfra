#!/usr/bin/env python3
"""Replay both indexer writers on one captured live step.

A synthetic reproducer reported a clean agreement twice on geometry that turned
out not to be the engine's, so the only admissible answer comes from the tensors
the engine actually passed.  `TOKENSPEED_DSV4_FP8_INDEXER_DUMP` saves them from
`_verify_fused_indexer_fp8`; this replays both writers on that exact input.

The byte count alone cannot say *why* a slot differs:

* the kernel wrote the right value to the wrong place, or
* the kernel computed a different value for the right place.

So for every differing slot we dequantize what is there and search every row of
the eager's own `rotated` for the best match.  A match on a different row is a
placement bug; no match anywhere is a values bug, and then the disagreement is
in the compress/rope/hadamard chain rather than in the write.
"""

import sys

import torch

import tokenspeed.runtime.layers.attention.deepseek_v4_ops as ops


def row_fields(cache, slot, head, block_size):
    blk, off = divmod(int(slot), block_size)
    return (
        cache[blk, off * head : (off + 1) * head],
        cache[blk, block_size * head + off * 4 : block_size * head + (off + 1) * 4],
    )


def main(path):
    d = torch.load(path, map_location="cuda:0", weights_only=False)
    head = int(d["rms_norm_weight"].numel())
    block_size = int(d["kv_cache_block_size"])
    ratio = int(d["compress_ratio"])
    state_block = int(d["compressor_block_size"])
    positions = d["positions"]
    t2r = d["token_to_req_indices"]
    cslot = d["compressor_slot_mapping"]
    kslt = d["kv_slot_mapping"]
    num_actual = min(cslot.numel(), positions.numel())

    print("=== captured step ===")
    print(f"  state_cache      {tuple(d['state_cache'].shape)} stride {d['state_cache'].stride()}")
    print(f"  block_table      {tuple(d['block_table'].shape)}")
    base = d["block_table_base_offsets"]
    print(f"  base_offsets     {None if base is None else tuple(base.shape)}")
    print(f"  positions        {positions.numel()}  range=[{int(positions.min())},{int(positions.max())}]")
    print(f"  token_to_req     distinct={int(t2r.unique().numel())}  head={t2r[:8].tolist()}")
    print(
        f"  compressor_slot  distinct={int(cslot.unique().numel())} of {cslot.numel()}"
        f"  head={cslot[:8].tolist()}"
    )
    print(
        f"  kv_slot          distinct={int(kslt.unique().numel())} of {kslt.numel()}"
        f"  head={kslt[:8].tolist()}"
    )
    print(f"  compress_ratio={ratio} state_block={state_block} kv_block={block_size} head={head}")

    # Re-derive the eager chain and confirm it is reproducible from the dump.
    normed, valid = ops._compress_v4_state_windows_capturable(
        state_cache=d["state_cache"],
        token_to_req_indices=t2r,
        positions=positions,
        compressor_slot_mapping=cslot,
        block_table=d["block_table"],
        block_table_base_offsets=base,
        compressor_block_size=state_block,
        rms_norm_weight=d["rms_norm_weight"],
        rms_norm_eps=d["rms_norm_eps"],
        compress_ratio=ratio,
        head_dim=head,
        overlap=True,
    )
    cpos = (
        torch.div(positions[:num_actual].to(torch.int64), ratio, rounding_mode="floor")
        * ratio
    )
    rot = ops._apply_gptj_rope_tail_rows(normed, cpos, d["cos_sin_cache"], int(d["cos_sin_cache"].shape[-1]))
    rot = ops._deepseek_v4_hadamard_rotate(rot).float()

    same_rot = torch.equal(rot, d["rotated"])
    same_norm = torch.equal(normed, d["normed"])
    print(
        f"  recomputed eager: rotated identical={same_rot}"
        f"  normed identical={same_norm}"
        f"  max|drot|={float((rot - d['rotated']).abs().max().item()):.3e}"
    )
    print(f"  valid rows: {int(valid.sum().item())} of {num_actual}")

    # The dump's kv_cache_2d is the FUSED result; write the eager on a copy.
    fused = d["kv_cache_2d"]
    eager = fused.clone()
    ops._write_deepseek_v4_indexer_fp8_cache_capturable(
        rot, eager, kslt[:num_actual], valid, block_size=block_size
    )
    torch.cuda.synchronize()

    diff = fused != eager
    n = int(diff.sum().item())
    print(f"  bytes differing: {n}")

    idx = valid.nonzero().flatten()
    n_slot_diff = 0
    n_row_match_other = 0
    n_no_match = 0
    n_div_only = 0
    shown = 0
    for j in idx.tolist():
        slot = int(kslt[j])
        if slot < 0:
            continue
        ev, es = row_fields(eager, slot, head, block_size)
        fv, fs = row_fields(fused, slot, head, block_size)
        if torch.equal(es, fs) and torch.equal(ev, fv):
            continue
        n_slot_diff += 1
        if not torch.equal(es, fs):
            n_div_only += 1
        deq = fv.view(torch.float8_e4m3fn).float() * fs.view(torch.float32)
        err = (rot - deq[None, :]).abs().amax(dim=-1)
        k = int(err.argmin().item())
        if k != j:
            n_row_match_other += 1
        if float(err[k].item()) > 1e-2:
            n_no_match += 1
        if shown < 12:
            shown += 1
            print(
                f"    row={j} slot={slot} div_eq={torch.equal(es, fs)}"
                f" best_row={k} best_err={float(err[k].item()):.4f}"
                f" -> {'SAME row' if k == j else 'OTHER row'}"
                f" {'NO MATCH' if float(err[k].item()) > 1e-2 else ''}"
            )
    print(
        f"  differing slots: {n_slot_diff} of {int(idx.numel())}"
        f"  (divisor differs on {n_div_only})"
    )
    print(
        f"  fused bytes match a DIFFERENT row: {n_row_match_other}"
        f"   match no row at all: {n_no_match}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/roy/logs/live_indexer_inputs.pt")