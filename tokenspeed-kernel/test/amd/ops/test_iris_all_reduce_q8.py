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


"""Correctness of the 8-bit-payload Iris pull two-shot all-reduce (TP4)."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="Iris 8-bit all-reduce targets gfx950"
)

WORLD_SIZE = 4
HIDDEN = 5120
GROUP = 128
# Row counts called back to back so buffers are reused across sizes; 4100
# rows leave a partial last tile.
ROWS = (64, 4100, 1, 512, 4100)


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _gather_inputs(x: torch.Tensor) -> list[torch.Tensor]:
    xs = [torch.empty_like(x.cpu()) for _ in range(WORLD_SIZE)]
    dist.all_gather(xs, x.cpu())
    return [t.to(x.device).float() for t in xs]


def _group_amax(v: torch.Tensor) -> torch.Tensor:
    amax = v.reshape(-1, GROUP).abs().amax(dim=1, keepdim=True)
    return amax.expand(-1, GROUP).reshape(v.shape)


def _int8_error_bound(xs: list[torch.Tensor]) -> torch.Tensor:
    """Per-element INT8 bound: half a step per input and for the reduced sum."""
    total = sum(xs)
    inputs = sum(_group_amax(v) for v in xs) / 254
    # The requantized sum is within `inputs` of total.
    requant = (_group_amax(total) + _group_amax(inputs)) / 254
    bound = inputs + requant
    # BF16 output rounding and FP32 arithmetic.
    return bound + (total.abs() + bound) * 2**-8 + 1e-5


def _worker(rank: int, port: int, error_dict) -> None:
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
        )
        from tokenspeed_kernel.ops.communication._iris import (
            IrisGroup,
            configure_iris,
            get_or_create_context,
        )
        from tokenspeed_kernel.ops.communication._iris.all_reduce_q8 import (
            IrisQ8AllReduce,
            iris_pull_twoshot_q8,
            iris_q8_quantize,
        )

        configure_iris(heap_size=None, allocator_type=None)
        max_numel = max(ROWS) * HIDDEN
        peers = IrisGroup(
            get_or_create_context(heap_size=1 << 30), dist.group.WORLD, rank, device
        )
        ops = {
            fp8: IrisQ8AllReduce(peers, max_numel, fp8=fp8, group=GROUP)
            for fp8 in (False, True)
        }
        peers.finish_prepare()
        torch.manual_seed(rank)
        for fp8, op in ops.items():
            for i, rows in enumerate(ROWS):
                x = (torch.randn(rows, HIDDEN, device=device) * (i + 1)).to(
                    torch.bfloat16
                )
                assert op.can_run(x)
                xs = _gather_inputs(x)
                ref = sum(xs)
                out = torch.empty_like(x)
                if i == 0:
                    op.run(x, out)  # compile both kernels before the guard
                with assert_no_triton_compile(iris_q8_quantize, iris_pull_twoshot_q8):
                    op.run(x, out)
                    inplace = x.clone()
                    op.run(inplace, inplace)
                torch.cuda.synchronize()
                torch.testing.assert_close(inplace, out, atol=0, rtol=0)
                rel = ((out.float() - ref).norm() / ref.norm()).item()
                if fp8:
                    # E4M3 keeps 3 mantissa bits: ~2-4% RMS error after two roundings.
                    assert rel < 4.5e-2, f"fp8 rows={rows} rel={rel}"
                else:
                    assert rel < 1.2e-2, f"int8 rows={rows} rel={rel}"
                    excess = (out.float() - ref).abs() - _int8_error_bound(xs)
                    assert excess.max().item() <= 0, f"int8 rows={rows}"

        # CUDA graph replay with skewed ranks reuses the same buffers.
        op = ops[False]
        x = torch.empty(512, HIDDEN, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(x)
        x.fill_(1.0)
        op.run(x, out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            op.run(x, out)
        for replay, value in enumerate((0.5, -3.0, 2.0)):
            # Exact in INT8: each group is constant, so q = +-127 with scale |value|.
            x.fill_(value * (rank + 1))
            dist.barrier()
            if rank == replay:
                torch.cuda._sleep(1_000_000)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full_like(x, value * 10)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        error_dict[rank] = traceback.format_exc()


def test_iris_all_reduce_q8_world4():
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs")
    error_dict = mp.Manager().dict()
    mp.spawn(_worker, args=(_open_port(), error_dict), nprocs=WORLD_SIZE, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
