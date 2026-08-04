#!/usr/bin/env python3
"""Isolation + roofline anchor, taken on the card under test with the server idle.

Three numbers, because the decomposition needs all three and E36 quoted only one:
  * bf16 8192^3 GEMM TFLOP/s   -- co-tenancy check (free-card band 405-427, E31)
  * D2D copy                   -- reported BOTH ways: N/t (the "copy rate") and
                                  2N/t (the bytes the memory system actually
                                  moved). E36's 1462 GB/s is the 2N/t figure and
                                  the distinction decides whether a
                                  weight-reading GEMM at 1299 GB/s is at 89% of
                                  achievable or at 44%.
  * pure read (sum reduction)  -- the right ceiling for a weight-bound GEMM,
                                  which reads weights and writes almost nothing.
"""
import torch, time, json, sys

dev = "cuda:0"
torch.cuda.init()
p = torch.cuda.get_device_properties(0)
out = {"device": p.name, "sm": f"{p.major}.{p.minor}",
       "mem_GB": round(p.total_memory / 1e9, 1)}


def timeit(fn, warm=5, iters=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# --- 1. bf16 GEMM ---
n = 8192
a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
t = timeit(lambda: torch.mm(a, b))
out["gemm_bf16_8192_tflops"] = round(2 * n ** 3 / t / 1e12, 1)
del a, b
torch.cuda.empty_cache()

# --- 2. D2D copy ---
N = 1 << 29  # 512 Mi elements bf16 = 1.0737 GB
src = torch.empty(N, device=dev, dtype=torch.bfloat16)
dst = torch.empty(N, device=dev, dtype=torch.bfloat16)
nbytes = src.numel() * src.element_size()
t = timeit(lambda: dst.copy_(src))
out["d2d_copy_bytes_GB"] = round(nbytes / 1e9, 3)
out["d2d_one_way_GBs"] = round(nbytes / t / 1e9, 1)     # N/t
out["d2d_traffic_GBs"] = round(2 * nbytes / t / 1e9, 1)  # 2N/t = real HBM traffic

# --- 3. pure read (reduction) ---
t = timeit(lambda: torch.sum(src))
out["read_only_GBs"] = round(nbytes / t / 1e9, 1)

# --- 4. read-heavy, GEMM-shaped: a tall-skinny mm reading a big weight ---
# 5120 x 17408 bf16 weight (the MLP shape), batch rows -> the decode GEMM shape
for bs in (1, 32, 64, 96, 126, 128, 256):
    W = torch.randn(17408, 5120, device=dev, dtype=torch.bfloat16)
    X = torch.randn(bs, 5120, device=dev, dtype=torch.bfloat16)
    wb = W.numel() * 2
    t = timeit(lambda: torch.mm(X, W.t()), warm=10, iters=50)
    out[f"mlp_gemm_bs{bs}"] = {
        "us": round(t * 1e6, 1),
        "weight_GBs": round(wb / t / 1e9, 1),
        "tflops": round(2 * bs * 5120 * 17408 / t / 1e12, 1),
    }
    del W, X
    torch.cuda.empty_cache()

print(json.dumps(out, indent=2))
