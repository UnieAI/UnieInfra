"""RCCL all-reduce latency on this node (torchrun --nproc-per-node 4)."""
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
for numel in [96 * 5120, 192 * 5120, 16 * 5120, 6 * 5120, 16384 * 5120]:
    x = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): dist.all_reduce(x)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); t = time.perf_counter()
    iters = 20
    for _ in range(iters): g.replay()
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / (iters * 10) * 1e6
    if r == 0: print(f"rccl all_reduce bf16 {numel*2/1e6:7.2f} MB : {us:8.1f} us (graph)")
dist.destroy_process_group()
