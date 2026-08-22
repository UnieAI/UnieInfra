"""Decode attention at gemma-4-31B's two geometries, against the memory roofline.

Why this exists: the layer ladder says the decode STEP is at parity at batch 1
-- 45.8 ms here against vLLM's 44.1, both ~1.1x the 41.0 ms weight-read floor --
while an earlier RAG measurement at batch 64 read 152 ms a step here against
84 on vLLM. A gap that is absent at batch 1 and 1.8x at batch 64 is a gap in
how the step scales, and for this model the thing that scales is KV traffic:

    sliding layer  32 q x 16 kv x 256, window 1024  ->  16 KiB / token / layer
    global  layer  32 q x  4 kv x 512               ->   8 KiB / token / layer
    50 sliding + 10 global

At batch 64 with a 1k context that is 50 x 1024 x 16 KiB x 64 = 52.4 GB of
sliding KV plus 5.4 GB of global, against 62.5 GB of weights. The KV is the
same order as the weights, so a kernel reading it at half bandwidth doubles
the step -- and is invisible at batch 1, where the weights dominate.

Print achieved GB/s next to every timing. A number near or above the card's
1523 GB/s read bandwidth means the working set fit in the 128 MB L2, not that
the kernel is fast; the per-call working set is printed so that case is
visible rather than inferred.

    python3 unieinfra/bench/kernel_gemma_decode.py
"""

import torch
from tokenspeed_kernel.ops.attention import mha_decode_with_kvcache
from torch.profiler import ProfilerActivity, profile

PAGE = 64
HBM_READ_GBS = 1523.3  # measured read-only stream on this card, not the datasheet
L2_BYTES = 128 << 20

GEOM = {  # name: (num_q_heads, num_kv_heads, head_dim, window, layers)
    "sliding": (32, 16, 256, 1024, 50),
    "global": (32, 4, 512, -1, 10),
}
BATCHES = [1, 4, 16, 32, 64]
CTXS = [1024, 4096, 26000]


def build(nq, nkv, d, batch, ctx, device="cuda"):
    pages_per_seq = (ctx + PAGE - 1) // PAGE
    total_pages = pages_per_seq * batch + 2
    q = torch.randn(batch, nq, d, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(total_pages, PAGE, nkv, d, dtype=torch.bfloat16, device=device)
    page_table = torch.arange(
        pages_per_seq * batch, dtype=torch.int32, device=device
    ).view(batch, pages_per_seq)
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=torch.randn_like(k_cache),
        page_table=page_table,
        cache_seqlens=torch.full((batch,), ctx, dtype=torch.int32, device=device),
        max_seqlen_k=ctx,
    )


def device_us(fn, reps=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (
        sum(
            e.device_time_total
            for e in prof.key_averages()
            if e.device_time_total > 0 and e.count
        )
        / reps
    )


def unwrap(r):
    return r[0] if isinstance(r, tuple) else r


def main():
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name}, {props.multi_processor_count} SM, "
          f"read roofline {HBM_READ_GBS} GB/s, L2 {L2_BYTES >> 20} MB")
    print(f"{'geom':>8} {'batch':>6} {'ctx':>6} {'eff_kv':>7} {'us':>9} "
          f"{'GB/s':>8} {'%peak':>6} {'set MB':>8} {'L2?':>4} "
          f"{'step ms/60L':>12}")

    for name, (nq, nkv, d, win, layers) in GEOM.items():
        for ctx in CTXS:
            eff = ctx if win < 0 else min(ctx, win)
            for batch in BATCHES:
                a = build(nq, nkv, d, batch, ctx)
                try:
                    fn = lambda: unwrap(
                        mha_decode_with_kvcache(
                            **a, window_left=win, solution="triton"
                        )
                    )
                    us = device_us(fn)
                except Exception as exc:  # noqa: BLE001
                    print(f"{name:>8} {batch:6d} {ctx:6d} {eff:7d}  "
                          f"failed: {type(exc).__name__}: {str(exc)[:60]}")
                    continue
                # K and V, bf16, over the effective window only.
                kv_bytes = 2 * eff * nkv * d * 2 * batch
                gbs = kv_bytes / (us * 1e-6) / 1e9
                fits_l2 = kv_bytes < L2_BYTES
                # What one layer's cost implies for a whole step of this model.
                step_ms = us * layers / 1000.0
                print(f"{name:>8} {batch:6d} {ctx:6d} {eff:7d} {us:9.1f} "
                      f"{gbs:8.1f} {100 * gbs / HBM_READ_GBS:5.0f}% "
                      f"{kv_bytes / 1e6:8.1f} {'yes' if fits_l2 else '':>4} "
                      f"{step_ms:12.2f}")
                del a
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
