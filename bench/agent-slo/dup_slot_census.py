#!/usr/bin/env python3
"""Census the captured live batch's indexer slot targets.

`replay_live_indexer.py` showed the fused write at slot 64 matches some row other
than the one that logically owns it.  Before concluding "duplicate targets", this
counts the claimants per target directly on the dumped tensors, and then, for a
contested slot, asks which of its claimants the fused bytes actually match.
"""

import sys

import torch

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/workspace/roy/logs/live_indexer_inputs.pt"
d = torch.load(DUMP, map_location="cuda:0", weights_only=False)
head = int(d["rms_norm_weight"].numel())
ratio = int(d["compress_ratio"])
bs = int(d["kv_cache_block_size"])
pos = d["positions"]
kslt = d["kv_slot_mapping"]
n = min(kslt.numel(), pos.numel())
valid = ((pos[:n].to(torch.int64) + 1) % ratio) == 0
slots = kslt[:n][valid]

print("valid rows:", int(valid.sum().item()))
print(
    "target slots: distinct =", int(slots.unique().numel()),
    " min =", int(slots.min()), " max =", int(slots.max()),
)
u, c = slots.unique(return_counts=True)
print(
    "claimants per slot: max =", int(c.max()),
    " mean =", round(float(c.float().mean()), 2),
    " slots with >1 claimant =", int((c > 1).sum()), " of", int(u.numel()),
)
print("first slots (target, claimants):", list(zip(u.tolist(), c.tolist()))[:8])
print("last  slots (target, claimants):", list(zip(u.tolist(), c.tolist()))[-4:])

# The slot arithmetic is `page_ids * block + (compressed_pos % block)` with
# `page_ids = block_table[req, compressed_pos // block]`.  Only 64 slots survive,
# so the page lookup itself is the suspect: show it.
bt = d["block_table"]
print("\nblock_table", tuple(bt.shape), "dtype", bt.dtype)
print("  distinct values:", int(bt.unique().numel()))
print("  row 0 head:", bt[0, :40].tolist())
print("  row 0 tail:", bt[0, 2000:2040].tolist())

rot = d["rotated"]
fused = d["kv_cache_2d"]
rows_of = valid.nonzero().flatten()
for target in (int(u[0]), int(u[len(u) // 2]), int(u[-1])):
    rows = rows_of[slots == target]
    blk, off = divmod(target, bs)
    fv = fused[blk, off * head : (off + 1) * head]
    fs = fused[blk, bs * head + off * 4 : bs * head + (off + 1) * 4]
    deq = fv.view(torch.float8_e4m3fn).float() * fs.view(torch.float32)
    err = (rot - deq[None, :]).abs().amax(dim=-1)
    print(f"\nslot {target}: {rows.numel()} claimants, first {rows[:6].tolist()}")
    for k in err.argsort()[:4].tolist():
        tag = "CLAIMANT" if bool((rows == k).any()) else "NOT a claimant"
        print(f"    best row {k:5d} err={float(err[k]):.4f}  {tag}")
