# KV cache kernels

## Range clearing

`kvcache.zero_byte_ranges` dispatches by physical placement. Device allocations
use the original Triton helper; pinned Host allocations use
`cute_dsl_zero_host_byte_ranges` on the execution device's current stream. CPU
arenas clear synchronously. Host callers retain backing storage until stream
completion, including before a CPU read. Offsets are relative to the tensor view;
the CuTe kernel uses int64 byte addressing and a bounded grid that strides each
range, preserving unaligned tails and neighboring bytes. Range counts, lengths,
and pointers are dynamic arguments and do not trigger new compilations.

This scheduler/lifecycle operation consumes a CPU range list outside CUDA graph
capture. Capturing it fails explicitly rather than retaining a temporary pinned
metadata pointer in a graph. Model forward graph replay is unchanged.

## Recurrent state rows

`store_state_rows` scatters one request's recurrent state per row into its
slab block, honouring both tensors' strides, so a transposed scan output and a
page-interleaved arena view need no staging copy. The grid is
(plane tiles, rows, heads) and the tile is two-dimensional over the last two
dimensions: an element's coordinates come from `program_id` and two `arange`s
instead of a flat index divided by a runtime width. Both matter because the copy
is a transpose in serving -- the scan returns a value-major state while the slab
is key-major -- and because capacity-shaped prefill metadata pads the request
dimension with `-1` blocks, so most programs of a mixed step move nothing and
must not cost a division per element. A negative block skips its row.

The destination block per row is device-resident: the host never reads it, so
the launch grid covers the padded request count and the masked rows retire
immediately. `copy_state_rows` is the multi-layer sibling used when one
call carries several layers through per-layer address tables.

## Sparse KV offloading

`offload.py` is the runtime boundary. NVIDIA uses the registered
`cute_dsl_offload_materialize` implementation and CuTe DSL row helpers.
The private `_cute_dsl` Python package ships in both wheels and source archives.
NVIDIA tests check the GPU vendor before importing CuTe; ROCm also exposes
`torch.cuda`, so GPU availability alone does not identify this backend.
It accepts physical history IDs (positive int32; zero/negative are invalid),
request IDs (int32 or scheduler int64), persistent hot tags/LRU and preallocated
scratch. An invocation resolves the whole request's Q×K selection union.
Original order, duplicate references and invalid masks survive in the output.

One CTA per request constructs an open-addressed selection hash. Atomic minimum
of original indices makes duplicate ownership deterministic. After a CTA barrier,
resident scanning selects the minimum physical slot for duplicate tags, protects
selected/current ordinary rows and stably partitions the old LRU. Unique misses
use first-occurrence order and the oldest unprotected slots. Before publishing
tags, an unconditional device trap rejects capacity overflow; probes are bounded.
The new LRU is remaining old slots, installed misses, then protected old slots.
Reserved/ring rows participate in lookup but cannot be victims.

The hash uses two int32 arrays at a load factor ≤0.5. `hash_geometry` selects
shared tables up to 128 KiB, and budgeted global tables above that threshold.
Global `entry_dest` separates canonical destinations from the final output.
Miss pairs occupy their canonical input positions, with other entries masked;
the static copy grid therefore requires no CPU read of the miss count.

A separate row-parallel CuTe kernel copies mapped pinned Host bytes. Integer
32-bit transfers are used for aligned rows, with a byte path for arbitrary tails.
Current/seed/accepted/reset helpers also use CuTe. The caller records ready only
after both resolver and payload copy, and joins write-through before reusing
current slots. Shared tables are rebuilt on every replay; padding rows never
write persistent tags or LRU.

Compiled executors are cached by device, static geometry and pointer element
types. Batch size, pointers, table width/stride and sequence/accept counts remain
runtime values. Warmup precedes graph capture; an unwarmed capture fails explicitly.
The runtime also warms admission's int64 request reset before serving.
The residency path uses only CuTe DSL. The former resident/selection sorting,
binary-search lookup and adjacent-entry miss deduplication have been removed.
Hash ownership merges duplicate loads while preserving all attention references;
upstream top-k ordering remains part of the attention numerical contract.
