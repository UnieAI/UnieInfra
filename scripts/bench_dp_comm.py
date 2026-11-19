"""Per-layer decode communication on 4 GPUs: TP (2 x all-reduce) vs DP attention (all-gather + reduce-scatter).

rows = total decode rows per step (users x 6 verify rows); hidden 5120 bf16, as in DeepSeek-V4.1.
"""
import socket, time, traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 5120
ROWS = (24, 96, 192, 384)
WORLD = 4


def port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def timed(fn, inner=20, iters=20, graph=True):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / (iters * inner) * 1e6
    t = time.perf_counter()
    for _ in range(iters * 4):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (iters * 4) * 1e6


def worker(rank, p, err):
    try:
        dev = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", init_method=f"tcp://localhost:{p}", rank=rank, world_size=WORLD)
        from tokenspeed_kernel.ops.communication.iris import create_iris_state, iris_all_reduce
        from tokenspeed_kernel.ops.communication.triton import all_gather, create_state, reduce_scatter

        max_rows = max(ROWS)
        iris = create_iris_state(
            group=dist.group.WORLD, rank_in_group=rank, staged_max_numel=max_rows * HIDDEN,
            producer_direct_max_numel=0, dtype=torch.bfloat16, device=dev, heap_size=None)
        rsag = create_state(group=dist.group.WORLD, rank_in_group=rank, max_tokens=max_rows,
                            hidden_size=HIDDEN, device=None)
        out = []
        for rows in ROWS:
            per = rows // WORLD
            tokens = [per] * WORLD
            full = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            local = torch.randn(per, HIDDEN, dtype=torch.bfloat16, device=dev)
            res = {"rows": rows, "MB": rows * HIDDEN * 2 / 1e6}
            def run(name, fn, graph=True):
                try:
                    res[name] = timed(fn, graph=graph)
                except Exception as e:  # report, keep going
                    res[name] = None
                    if rank == 0:
                        print(f"  {name} rows={rows}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            x = full.clone()
            run("iris_AR", lambda: iris_all_reduce(iris, x, safe=False))
            run("rsag_AG", lambda: all_gather(rsag, local, token_list_in_group=tokens))
            run("rsag_RS", lambda: reduce_scatter(rsag, full, token_list_in_group=tokens))
            y = full.clone()
            run("rccl_AR", lambda: dist.all_reduce(y))
            ag_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            rs_out = torch.empty(per, HIDDEN, dtype=torch.bfloat16, device=dev)
            run("rccl_AG", lambda: dist.all_gather_into_tensor(ag_out, local))
            run("rccl_RS", lambda: dist.reduce_scatter_tensor(rs_out, full))
            out.append(res)
        if rank == 0:
            print(f"{'rows':>5} {'MB':>6} | {'2xIris AR':>10} | {'rsag AG+RS':>10} | {'rccl AG+RS':>10} | {'2xRCCL AR':>10}   (us per layer, graph replay)")
            for r in out:
                f = lambda v: f"{v:10.1f}" if v is not None else "       n/a"
                ar2 = 2 * r["iris_AR"] if r.get("iris_AR") else None
                rs = r["rsag_AG"] + r["rsag_RS"] if r.get("rsag_AG") and r.get("rsag_RS") else None
                rc = r["rccl_AG"] + r["rccl_RS"] if r.get("rccl_AG") and r.get("rccl_RS") else None
                rc2 = 2 * r["rccl_AR"] if r.get("rccl_AR") else None
                print(f"{r['rows']:5d} {r['MB']:6.2f} | {f(ar2)} | {f(rs)} | {f(rc)} | {f(rc2)}")
        dist.barrier()
    except Exception:
        err[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    err = mp.Manager().dict()
    mp.spawn(worker, args=(port(), err), nprocs=WORLD, join=True)
    for r, e in err.items():
        print(f"rank {r} failed:\n{e}")
