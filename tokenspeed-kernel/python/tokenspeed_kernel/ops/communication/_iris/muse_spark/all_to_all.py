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


<<<<<<< HEAD
<<<<<<< HEAD
"""Muse TP4/TP8 QKVG all-to-all from token owners to attention-head owners.

With P ranks, each source row packs P Q/K/V/gate bundles of widths
8192/P, 1024/P, 1024/P, 8192/P. Exchange [ceil(M/P), 18432] BF16 inputs
=======
"""Muse TP4 QKVG all-to-all from token owners to attention-head owners.

Each source row packs four destination bundles of Q/K/V/gate widths
2048/256/256/2048. Both kernels exchange [ceil(M/4), 18432] BF16 inputs
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
"""Muse TP4/TP8 QKVG all-to-all from token owners to attention-head owners.

With P ranks, each source row packs P Q/K/V/gate bundles of widths
8192/P, 1024/P, 1024/P, 8192/P. Exchange [ceil(M/P), 18432] BF16 inputs
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
for this rank's head features over all M rows, masking the last owner's tail.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_drain_subgroup_vmem,
<<<<<<< HEAD
<<<<<<< HEAD
    _iris_peer_pointers,
    _iris_rendezvous,
=======
    _iris_tp4_peer_pointers,
    _tp4_rendezvous,
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
    _iris_peer_pointers,
    _iris_rendezvous,
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
)


@gluon.jit(do_not_specialize=["rows"])
def iris_a2a_transpose_qkvg(
    source,
    output,
    ready_flags,
    heaps,
    rows,
    RANK: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Exchange complete owned Q/K/V/gate rows for this rank's attention heads.

<<<<<<< HEAD
<<<<<<< HEAD
    source contains ceil(rows / WORLD_SIZE) BF16 rows, with one head bundle
    per destination rank. Pull this rank's bundle from every row owner into
    output [rows, 18432 / WORLD_SIZE], masking padding. ready_flags publish
    source writes and wait for peer reads before the next projection reuses source.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    BUNDLE_WIDTH: gl.constexpr = 18432 // WORLD_SIZE
    program = gl.program_id(0)
    peers = _iris_peer_pointers(source, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    # hipBLAS writes the symmetric source through ordinary write-back caches.
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    local_elements = gl.cdiv(rows, WORLD_SIZE) * BUNDLE_WIDTH
    lanes = gl.arange(0, BLOCK, layout=gl.BlockedLayout([8], [64], [NUM_WARPS], [0]))
    for tile in range(program, gl.cdiv(local_elements, BLOCK), gl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        source_offsets = (
            (offsets // BUNDLE_WIDTH) * 18432
            + RANK * BUNDLE_WIDTH
            + offsets % BUNDLE_WIDTH
        )
        for step in gl.static_range(WORLD_SIZE):
            destination = ((RANK + step) % WORLD_SIZE) * local_elements + offsets
            mask = (offsets < local_elements) & (destination < rows * BUNDLE_WIDTH)
            values = gl.amd.cdna4.buffer_load(
                peers[(RANK + step) % WORLD_SIZE], source_offsets, mask, 0, cache=".cg"
=======
    source contains ceil(rows / 4) BF16 rows, each with four 4608-feature head
    bundles. Pull this rank's bundle from every row owner into output with shape
    [rows, 4608] in token order, masking the padded tail. ready_flags publish
=======
    source contains ceil(rows / WORLD_SIZE) BF16 rows, with one head bundle
    per destination rank. Pull this rank's bundle from every row owner into
    output [rows, 18432 / WORLD_SIZE], masking padding. ready_flags publish
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    source writes and wait for peer reads before the next projection reuses source.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    BUNDLE_WIDTH: gl.constexpr = 18432 // WORLD_SIZE
    program = gl.program_id(0)
    peers = _iris_peer_pointers(source, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    # hipBLAS writes the symmetric source through ordinary write-back caches.
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    local_elements = gl.cdiv(rows, WORLD_SIZE) * BUNDLE_WIDTH
    lanes = gl.arange(0, BLOCK, layout=gl.BlockedLayout([8], [64], [NUM_WARPS], [0]))
    for tile in range(program, gl.cdiv(local_elements, BLOCK), gl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        source_offsets = (
            (offsets // BUNDLE_WIDTH) * 18432
            + RANK * BUNDLE_WIDTH
            + offsets % BUNDLE_WIDTH
        )
        for step in gl.static_range(WORLD_SIZE):
            destination = ((RANK + step) % WORLD_SIZE) * local_elements + offsets
            mask = (offsets < local_elements) & (destination < rows * BUNDLE_WIDTH)
            values = gl.amd.cdna4.buffer_load(
<<<<<<< HEAD
                peers[(RANK + step) % 4], source_offsets, mask, 0, cache=".cg"
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
                peers[(RANK + step) % WORLD_SIZE], source_offsets, mask, 0, cache=".cg"
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            )
            gl.amd.cdna4.buffer_store(values, output, destination, mask)
    # The output projection may immediately overwrite this buffer. Every reader
    # must finish before any rank returns to that producer on the main stream.
    _iris_drain_subgroup_vmem()
    gl.barrier()
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    _iris_rendezvous(
        ready_flags + FLAG_ROWS * WORLD_SIZE,
        heaps,
        program,
        epoch,
        RANK,
        NUM_WARPS,
        False,
<<<<<<< HEAD
=======
    _tp4_rendezvous(
        ready_flags + FLAG_ROWS * 4, heaps, program, epoch, RANK, NUM_WARPS, False
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    )


@gluon.jit(do_not_specialize=["rows"])
def iris_fused_a2a_transpose_qkvg_norm_rope(
    source,
    query,
    key,
    value,
    gate,
    positions,
    cos_sin_cache,
    ready_flags,
    heaps,
    rows,
    RANK: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    ROPE: gl.constexpr,
    EPS: gl.constexpr,
    Q_SCALE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Exchange owned QKVG rows and normalize/rotate the received Q and K heads.

    Read this rank's heads from each source row. Apply head-width-64 RMSNorm,
    Q_SCALE to queries, and optional RoPE using positions and cos_sin_cache;
    compute in FP32 and round once to BF16. Write query/key/value/gate outputs,
    copying V and gate unchanged. ready_flags protect source reuse. The attention
    prologue still owns the KV-cache write.
    """
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    WORLD_SIZE: gl.constexpr = len(heaps)
    BUNDLE_WIDTH: gl.constexpr = 18432 // WORLD_SIZE
    Q_WIDTH: gl.constexpr = 8192 // WORLD_SIZE
    KV_WIDTH: gl.constexpr = 1024 // WORLD_SIZE
    Q_HEADS: gl.constexpr = 128 // WORLD_SIZE
    QK_HEADS: gl.constexpr = 144 // WORLD_SIZE
<<<<<<< HEAD
    program = gl.program_id(0)
    peers = _iris_peer_pointers(source, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    # The completed hipBLAS producer uses ordinary write-back stores.
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    local_rows = gl.cdiv(rows, WORLD_SIZE)
=======
=======
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    program = gl.program_id(0)
    peers = _iris_peer_pointers(source, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    # The completed hipBLAS producer uses ordinary write-back stores.
<<<<<<< HEAD
    _tp4_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    local_rows = gl.cdiv(rows, 4)
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    local_rows = gl.cdiv(rows, WORLD_SIZE)
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    # Match the portable head64 prologue's per-head reduction association.
    head_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NUM_WARPS, 1], [1, 0])
    heads = gl.arange(0, 64, layout=gl.SliceLayout(1, head_layout))
    channels = gl.arange(0, 64, layout=gl.SliceLayout(0, head_layout))
    copy_lanes = gl.arange(
        0, 4096, layout=gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    )
<<<<<<< HEAD
<<<<<<< HEAD
    for job in range(program, local_rows * WORLD_SIZE, gl.num_programs(0)):
        owner = job % WORLD_SIZE
        local_row = job // WORLD_SIZE
        row = owner * local_rows + local_row
        if row < rows:
            peer = peers[0]
            for candidate in gl.static_range(1, WORLD_SIZE):
                peer = gl.where(owner == candidate, peers[candidate], peer)
            source_row = local_row * 18432 + RANK * BUNDLE_WIDTH
            offsets = source_row + heads[:, None] * 64 + channels[None, :]
            raw = gl.amd.cdna4.buffer_load(
                peer, offsets, heads[:, None] < QK_HEADS, 0, cache=".cg"
=======
    for job in range(program, local_rows * 4, gl.num_programs(0)):
        owner = job % 4
        local_row = job // 4
=======
    for job in range(program, local_rows * WORLD_SIZE, gl.num_programs(0)):
        owner = job % WORLD_SIZE
        local_row = job // WORLD_SIZE
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
        row = owner * local_rows + local_row
        if row < rows:
            peer = peers[0]
            for candidate in gl.static_range(1, WORLD_SIZE):
                peer = gl.where(owner == candidate, peers[candidate], peer)
            source_row = local_row * 18432 + RANK * BUNDLE_WIDTH
            offsets = source_row + heads[:, None] * 64 + channels[None, :]
            raw = gl.amd.cdna4.buffer_load(
<<<<<<< HEAD
                peer, offsets, heads[:, None] < 36, 0, cache=".cg"
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
                peer, offsets, heads[:, None] < QK_HEADS, 0, cache=".cg"
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            )
            fp32 = raw.to(gl.float32)
            inverse = gl.rsqrt(gl.sum(fp32 * fp32, 1) / 64 + EPS)
            normalized = fp32 * inverse[:, None]
<<<<<<< HEAD
<<<<<<< HEAD
            normalized = normalized * gl.where(heads < Q_HEADS, Q_SCALE, 1.0)[:, None]
=======
            normalized = normalized * gl.where(heads < 32, Q_SCALE, 1.0)[:, None]
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
            normalized = normalized * gl.where(heads < Q_HEADS, Q_SCALE, 1.0)[:, None]
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            if ROPE:
                position = gl.load(positions + row).to(gl.int64)
                cos = gl.load(cos_sin_cache + position * 64 + channels // 2)
                sin = gl.load(cos_sin_cache + position * 64 + 32 + channels // 2)
                even_indices = heads[:, None] * 0 + (channels & ~1)[None, :]
                odd_indices = heads[:, None] * 0 + (channels | 1)[None, :]
                x1 = gl.gather(normalized, even_indices, 1)
                x2 = gl.gather(normalized, odd_indices, 1)
                # Match the attention prologue's explicit rotation association.
                first = gl.fma(x1, cos[None, :], -x2 * sin[None, :])
                second = gl.fma(x2, cos[None, :], x1 * sin[None, :])
                normalized = gl.where((channels & 1)[None, :] == 0, first, second)
            result = normalized.to(gl.bfloat16)
            gl.amd.cdna4.buffer_store(
                result,
                query,
<<<<<<< HEAD
<<<<<<< HEAD
                row * Q_WIDTH + heads[:, None] * 64 + channels[None, :],
                heads[:, None] < Q_HEADS,
=======
                row * 2048 + heads[:, None] * 64 + channels[None, :],
                heads[:, None] < 32,
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
                row * Q_WIDTH + heads[:, None] * 64 + channels[None, :],
                heads[:, None] < Q_HEADS,
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            )
            gl.amd.cdna4.buffer_store(
                result,
                key,
<<<<<<< HEAD
<<<<<<< HEAD
                row * KV_WIDTH + (heads[:, None] - Q_HEADS) * 64 + channels[None, :],
                (heads[:, None] >= Q_HEADS) & (heads[:, None] < QK_HEADS),
            )
            tail = gl.amd.cdna4.buffer_load(
                peer,
                source_row + (Q_WIDTH + KV_WIDTH) + copy_lanes,
                copy_lanes < (Q_WIDTH + KV_WIDTH),
                0,
                cache=".cg",
            )
            gl.amd.cdna4.buffer_store(
                tail, value, row * KV_WIDTH + copy_lanes, copy_lanes < KV_WIDTH
=======
                row * 256 + (heads[:, None] - 32) * 64 + channels[None, :],
                (heads[:, None] >= 32) & (heads[:, None] < 36),
=======
                row * KV_WIDTH + (heads[:, None] - Q_HEADS) * 64 + channels[None, :],
                (heads[:, None] >= Q_HEADS) & (heads[:, None] < QK_HEADS),
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            )
            tail = gl.amd.cdna4.buffer_load(
                peer,
                source_row + (Q_WIDTH + KV_WIDTH) + copy_lanes,
                copy_lanes < (Q_WIDTH + KV_WIDTH),
                0,
                cache=".cg",
            )
            gl.amd.cdna4.buffer_store(
<<<<<<< HEAD
                tail, value, row * 256 + copy_lanes, copy_lanes < 256
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
                tail, value, row * KV_WIDTH + copy_lanes, copy_lanes < KV_WIDTH
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            )
            gl.amd.cdna4.buffer_store(
                tail,
                gate,
<<<<<<< HEAD
<<<<<<< HEAD
                row * Q_WIDTH + copy_lanes - KV_WIDTH,
                (copy_lanes >= KV_WIDTH) & (copy_lanes < (Q_WIDTH + KV_WIDTH)),
=======
                row * 2048 + copy_lanes - 256,
                (copy_lanes >= 256) & (copy_lanes < 2304),
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
                row * Q_WIDTH + copy_lanes - KV_WIDTH,
                (copy_lanes >= KV_WIDTH) & (copy_lanes < (Q_WIDTH + KV_WIDTH)),
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
            )
    # All QKVG reads finish before the output projection reuses the producer buffer.
    _iris_drain_subgroup_vmem()
    gl.barrier()
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    _iris_rendezvous(
        ready_flags + FLAG_ROWS * WORLD_SIZE,
        heaps,
        program,
        epoch,
        RANK,
        NUM_WARPS,
        False,
<<<<<<< HEAD
=======
    _tp4_rendezvous(
        ready_flags + FLAG_ROWS * 4, heaps, program, epoch, RANK, NUM_WARPS, False
>>>>>>> 3d44d6e9 (feat(models): add Muse Spark Open support (#2388))
=======
>>>>>>> 050ca725 (feat(amd): Enable Muse Spark TP8/EP8 optimizations (#2449))
    )
