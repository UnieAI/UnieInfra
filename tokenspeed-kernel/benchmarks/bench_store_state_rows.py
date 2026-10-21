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

"""Time store_state_rows against the pre-change kernel and a torch floor."""

import torch
import triton
from tokenspeed_kernel._triton import tl
from tokenspeed_kernel._triton import triton as ts_triton
from tokenspeed_kernel.ops.kvcache.triton import store_state_rows


@ts_triton.jit(do_not_specialize=["num_rows", "state_width"])
def old_kernel(
    states,
    slab,
    blocks,
    num_rows,
    state_width,
    state_dim_2,
    state_dim_3,
    state_stride_0,
    state_stride_1,
    state_stride_2,
    state_stride_3,
    slab_stride_0,
    slab_stride_1,
    slab_stride_2,
    slab_stride_3,
    BLOCK: tl.constexpr,
):
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    row = offsets // state_width
    feature = offsets % state_width
    live = row < num_rows
    destination = tl.load(blocks + row, mask=live, other=-1).to(tl.int64)
    live = live & (destination >= 0)
    dim_1_index = feature // (state_dim_2 * state_dim_3)
    remainder = feature % (state_dim_2 * state_dim_3)
    dim_2_index = remainder // state_dim_3
    dim_3_index = remainder % state_dim_3
    state_offset = (
        row * state_stride_0
        + dim_1_index * state_stride_1
        + dim_2_index * state_stride_2
        + dim_3_index * state_stride_3
    )
    slab_offset = (
        destination * slab_stride_0
        + dim_1_index * slab_stride_1
        + dim_2_index * slab_stride_2
        + dim_3_index * slab_stride_3
    )
    value = tl.load(states + state_offset, mask=live)
    tl.store(slab + slab_offset, value.to(slab.dtype.element_ty), mask=live)


def old_store(states, slab, blocks):
    state_width = states[0].numel()
    old_kernel[(triton.cdiv(blocks.numel() * state_width, 256),)](
        states,
        slab,
        blocks,
        blocks.numel(),
        state_width,
        states.shape[2],
        states.shape[3],
        *states.stride(),
        *slab.stride(),
        BLOCK=256,
    )


def time(fn, *args, iters=50):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def main() -> None:
    heads, key, value = 48, 128, 128
    blocks = 778
    slab = torch.zeros((blocks, heads, key, value), dtype=torch.float32, device="cuda")
    # Page-interleaved arena view, as the pool hands out recurrent state.
    interleaved = slab.as_strided(
        (blocks // 2, heads, key, value),
        (2 * heads * key * value, key * value, value, 1),
    )
    # Capacity metadata pads the request dimension with -1 blocks, so in a mixed
    # step most rows carry no destination: their programs are still scheduled and
    # run their address arithmetic, but move no bytes.
    sparse = (-torch.ones(64, dtype=torch.int32, device="cuda")).scatter(
        0,
        torch.arange(9, device="cuda"),
        torch.arange(9, dtype=torch.int32, device="cuda"),
    )
    padded = torch.randn((64, heads, value, key), dtype=torch.float32, device="cuda")
    padded = padded.transpose(-1, -2)
    for label, fn in (("old", old_store), ("new", store_state_rows)):
        best = min(time(fn, padded, slab, sparse) for _ in range(3))
        print(f"rows= 64 of which 9 live  {label:<16}{best:9.1f} us")
    print()
    for rows in (4, 9, 16, 64):
        index = torch.arange(rows, dtype=torch.int32, device="cuda")
        # Scan output is value-major: the copy is a transposed write.
        scan = torch.randn(
            (rows, heads, value, key), dtype=torch.float32, device="cuda"
        ).transpose(-1, -2)
        contiguous = scan.contiguous()
        moved = 2 * rows * heads * key * value
        for name, target in (("interleaved", interleaved), ("contiguous", slab)):
            for label, fn, source in (
                ("old", old_store, scan),
                ("new", store_state_rows, scan),
                (
                    "torch index_copy",
                    lambda s, t, b: t.index_copy_(0, b.long(), s),
                    scan,
                ),
                ("old/contiguous", old_store, contiguous),
                ("new/contiguous", store_state_rows, contiguous),
            ):
                best = min(time(fn, source, target, index) for _ in range(3))
                print(
                    f"rows={rows:>3} slab={name:<12} {label:<16}"
                    f"{best:9.1f} us {moved / best / 1e3:8.1f} GB/s"
                )
        print()


if __name__ == "__main__":
    main()
