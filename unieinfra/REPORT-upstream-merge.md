# Upstream merge, and what it cost

Merging 184 upstream `tokenspeed` commits (`e7f65f58`) into `unieai`, and the
work to get the tree serving again afterwards.

## What the merge actually was

Not a fold-in. Upstream deleted the radix prefix-cache subsystem this fork had
been building on — `csrc/resource/{radix_tree,kv_prefix_cache,hybrid_prefix_cache}`,
17 files — and replaced it with `csrc/cache/{core,manager,coordinator,tier}`,
plus a guard test that fails the build if the old symbols return. 155 files are
deleted by the merge; only the 11 we had also modified surfaced as conflicts.

The whole of `tokenspeed-scheduler/` was taken from upstream, because eight of
fifteen conflicts referenced deleted types on our side and the files are too
coupled to mix. That dropped four fork commits outright: 0830a199, b10ef2be,
5f001714, 46b6614b.

**Two of the four were then retired by measurement rather than argument:**

- `5f001714` (C>32 crash) — upstream's pool covers it. GATE_c64 passes with
  zero occurrences of `failed to allocate Mamba checkpoint slot`.
- `0830a199` (hybrid prefix cache) — upstream's block-pool cache reuses
  *better* than our radix version in every arm measured: 68.2% against 66.5%
  at C=128, 48.7% against 40.8% at C=32.

## The three defects found afterwards, and how

### 1. `--enable-mixed-batch` accepted and unserveable

Upstream's MHA asserts against MIXED while inheriting `supports_mixed_batch =
True`. The startup guard that existed to catch exactly this (c81fa7fa) had lost
its enforcement site in the merge, so the failure landed as 128 of 128 requests
returning Internal Server Error instead of a refusal at startup.

Guard restored (a8c0c872), then the implementation re-ported (5382c99e).

### 2. Prefill chunks capped below one page

`AlignPrefillChunk` returned only `page_size - page_offset` for a misaligned
request, so anything starting off the page grid — on a high-reuse shape, almost
everything, because a prefix hit rarely ends on a page boundary — advanced
under 128 tokens per prefill step against prompts of 606-1015.

Found by looking at exact chunk sizes on single-sequence prefill steps: they
clustered at 96-127. Fixed in ba69ea89 (+75.7% throughput, -44% TTFT).

### 3. The two L2 transfer streams race each other

**Caveat found late, and then pinned down.** Ordering the streams fixes the
SERVING path -- arms complete 256/256 where they previously lost 146 requests
-- but a `kernel`-backend arm still logged 2 `illegal memory access` at
teardown, after every request had completed and the gate had passed. Checking
every completed arm separates it cleanly:

| backend | arms | failed requests | illegal-mem |
|---|---|---|---|
| `direct` | 5 | 0 | **0** |
| `kernel` | 1 | 0 | **2** |

So the residual is specific to the mapped-host Triton transfer, not to the L2
tier as such, and it is confined to shutdown -- serving is clean on both. The
recommended configuration uses `--kvstore-io-backend direct`, which is clean
across every arm measured. Whoever picks this up should treat the mapped-host
path's shutdown as unexamined rather than the fix as incomplete for serving.


`write_stream` and `load_stream` are never ordered against one another while
both address the same device blocks and the same pinned Host buffer, so a block
can be loaded into while its writeback is still reading it. Fixed in 26470983.

This one took a day, and the reason is worth recording: **three separate things
made the fault disappear and none of them localised it.**
`CUDA_LAUNCH_BLOCKING=1`, `--kvstore-io-backend direct`, and per-transfer
synchronisation all work by removing overlap globally, so all three were the
same control wearing different clothes. The fix that finally named it removes
overlap between exactly two things and leaves both overlapping compute.

## Numbers

RAG shape, C=128, GMU 0.80, 256 requests, all gate-passing with 0 failed:

| build | tok/s | mean TTFT | reuse |
|---|---|---|---|
| merged, direct backend | 231.01 | 14590.9 | 75.9% |
| + prefill alignment fix | 405.74 | 8115.6 | 79.7% |
| + mixed batching | 500.71 | 7014.3 | 76.7% |
| + paged-extend plan restored | 511.86 | 6979.4 | 77.6% |
| + flashinfer GDN, prefill graph off | **592.40** | 5840.5 | — |
| default backend + stream ordering | 468.89 | 7505.7 | — |
| | | | |
| f162d993 nomix target | 754.0 | 3905.8 | 66.5% |
| f162d993 mix target (needs a deleted flag) | 1086.8 | 2227.6 | 85.8% |

231 -> 592 over the session, and the default configuration serves again. Still
79% of the nomix target. The mix target was measured with
`--mixed-prefill-decode-min-load 0`, which the merge deleted and upstream has no
equivalent for, so it is not a configuration this tree can reproduce.

### 4. The prefill CUDA graph silently corrupts the flashinfer GDN kernel

This one was wrong twice before it was right, and the sequence is the point.

`TOKENSPEED_ENABLE_FI_GDN_SM120=1` first measured **579.97 tok/s, +13.3%, 0
failed** — with every completion `"!!!!!!!!!!!!!!!!!!!!"`. Nothing caught that
except the harness `.sample` file: not the request-failure gate, not the
latency numbers, not reuse. The default had already been flipped on and the
commit written. I then concluded the kernel was numerically broken and kept it
off for correctness, which was also wrong.

Four arms settle it:

| | tok/s | mean TTFT | output |
|---|---|---|---|
| triton, prefill graph | 511.86 | 6979 | normal |
| triton, no graph | 472.02 | 7452 | normal |
| flashinfer, prefill graph | 579.97 | 5975 | `!!!!!!` |
| flashinfer, no graph | 574.94 | 6042 | normal |

Same kernel, same build, one flag between coherent text and exclamation marks.
It plans from host-visible batch data, so a capture bakes the dummy batch's
plan and every replay runs it against the real one. A direct A/B on
engine-shaped inputs — multi-sequence `cu_seqlens`, live non-zero
`initial_state`, off-chunk lengths — agrees with the Triton path to 4.9e-4 on
out and ~5e-3 on final_state. The numerics were never the problem.

Net: the prefill graph is worth +8.4% on Triton, the kernel is worth +21.8%
over Triton without it, and the combination measures **592.40 tok/s, mean TTFT
5840 ms, mean TPOT 96.26 ms** with coherent output. The wrapper now raises
under `torch.cuda.is_current_stream_capturing()` rather than letting a graph
corrupt every reply (8eb78a11).

Left open, and narrowed by elimination rather than solved. The GDN mixer
already carries `@break_point`, so the kernel runs as an eager break — outside
the capture — and this should not be possible. Three explanations were checked
and killed: a stale identity-keyed plan (flashinfer's `gdn_kernels` has no
caching of any kind), a rank difference making `_land_in` narrow the batch axis
instead of the token axis (both kernels return `(1, T, H, V)`), and the
numerics themselves (4.9e-4 against Triton on multi-sequence, live-state,
off-chunk inputs).

Whoever takes it needs instrumentation rather than reading: log what the kernel
receives at capture versus at replay, behind a debug bypass of the startup
guard. The payoff is the graph's +8.4% stacked on the kernel's +21.8%.

## The largest lever found, measured, and NOT shipped

Concurrency is the gap. On a clean arm at the new defaults the engine holds
**69 in-flight against a client driving 128**, while f162d993 held 128. KV is
not the constraint -- 2964 blocks free -- the recurrent state pool is: 247
blocks per linear group, three groups, and roughly 3.5 blocks charged per
request.

`MambaStateManager` is built as `SwaManager(..., sliding_window=2, ...)`, and
`CachedPagesBehindBoundary` computes `(window - 1 + block_tokens - 1) /
block_tokens`, so 2 pins one page BEHIND every boundary and 1 pins none. Its
own comment says "retain only the request's live state page", which is window
1. A recurrent state summarises everything before it, so on the face of it the
page behind carries nothing the state does not already hold.

Changed to 1 and measured, RAG C=128, 256 requests, 0 failed:

|  | window 2 | window 1 |
|---|---|---|
| throughput | 596.63 | **703.50** tok/s (+17.9%) |
| mean TTFT | 5775.6 | 4805.9 ms |
| mean TPOT | 97.14 | 85.30 ms |
| prefix reuse | 77.0% | **80.6%** |
| peak in-flight | 69 | 77 |
| output | coherent | coherent |

703.50 is 93% of the 754.0 reference, from 231 at the start.

**It is reverted, because it breaks 8 scheduler tests.** The window is not only
a capacity knob -- it changes state prefix MATCHING.
`MambaStateKindTest.FactoryMapsStateKindToAlignSemantics` pins "state resumes
off snapshot @2" with a specific hit-block count and block vector, and five
`CoordinatorPromotionTest` cases plus two admission tests fail with it. One
benchmark arm with correct output is not enough to overrule eight tests that
encode the intended semantics.

### And then the +17.9% turned out to be a correctness break

The obvious refinement was to split the window in two: keep 2 for matching
(which is what the promotion tests pin) and use 1 only for release. That took
the failures from 8 down to 1, and the survivor named the mistake --
`MambaStateKindTest.StateGroupRetentionKeepsOnlyLastPage`, with 4 pages and
`num_computed=16`:

    window 2:  skipped = 16 - 2 + 1 = 15  ->  15/4 = 3 pages freed, page 3 KEPT
    window 1:  skipped = 16 - 1 + 1 = 16  ->  16/4 = 4 pages freed, page 3 GONE

The test's own comment says it: *"skipped = n-1: only the live state page"*.
Window 2 already retains exactly the live page. The page I thought was surplus
IS the request's live recurrent state, and window 1 frees it.

So the 703.50 tok/s came from throwing away live state. Requests kept serving,
the output sample read as coherent English, and prefix reuse went UP to 80.6%
-- none of which was evidence of anything. **Reverted in full; 402/402 pass.**

That is the fourth time in this session that a large, clean-looking gain came
out of a broken configuration, and the second where the generated text looked
fine anyway. The pattern is worth naming: on this engine, a change that frees
cache capacity will always look like a throughput win, because throughput is
what capacity buys -- the question to ask first is what stopped being stored.

### The ceiling is arena fragmentation, measured

`TOKENSPEED_LOG_ADMISSION=1` names the refusal. On a RAG C=128 arm the
scheduler rejected admission 6392 times, and **6391 of them report
`empty_parents=0`**: the shared LCM arena had no fully-empty parent block left.

KV had thousands of individual slots free at the same time. Both are true
because a parent is allocatable only when ALL of its slots are free -- on this
model one parent carries 12 full-attention blocks plus one block for each of
three linear-attention groups. Cached blocks from a high-reuse workload land
scattered across parents, so nearly every parent holds something and almost
none is empty, while the free slots are spread thin across all of them.

So the ceiling is **fragmentation of the shared arena**, not exhaustion of any
pool. That is the structural cost of coupling KV and recurrent state into one
allocation unit.

**Correction, from the peer session's committed f162d993 data.** I wrote that
this "is exactly what the fork's separate mamba pool could not suffer from".
That is wrong. The fork hit the same ceiling, in its own allocator, and the
ladder is committed:

| f162d993 pool | in-flight, client driving 128 |
|---|---|
| 165 chunks (`--mamba-full-memory-ratio 1.0`) | 74 |
| 221 chunks (GMU 0.80, the baseline) | 80 |
| 221 chunks, prefix cache off | 93 |
| 264 chunks (`--mamba-full-memory-ratio 4.0`) | 94 |
| 306 chunks (GMU 0.90) | 128, saturated |

A near-constant 0.36 in-flight requests per chunk across five pool sizes. So
the 72 measured here is the same ceiling wearing new clothes, not a regression
the merge introduced, and the shared arena is not what causes it. The merged
arena is in fact the larger of the two -- 377,856 token capacity in one pool
against the fork's 132,432 KV tokens plus 221 mamba chunks.

The one lever that ever moved it was giving the pool more memory: on the fork,
GMU 0.80 to 0.90 took the pool 221 to 306 chunks, saturated in-flight at 128,
and lifted throughput 314.8 to 394.2 tok/s (+25.2%). GMU 0.94 OOMed, so 0.90
was that card's ceiling.

That matters here because **the benchmark harness hardcodes
`--gpu-memory-utilization 0.80` for both engines** (`.cq_nomb_rag.sh:61`), well
under this build's own resolved default of 0.88 for a single GPU
(`server_args.py:485`). Every number in this report, the 596.63 included, was
measured with that fraction pinned low.

### The memory ladder, measured here

Clean A/B on the merged tree, GPU 3, RAG shape, `CONC=128 NREQ=128`, all arms
back to back in one sitting, 128/128 successful and 0 failed everywhere, output
sampled and coherent:

| `--gpu-memory-utilization` | parents | token capacity | peak in-flight | tok/s |
|---|---|---|---|---|
| 0.80 (harness default) | 247 | 379,392 | 73 | 534.68 |
| 0.92 | 366 | 562,176 | 107 | 614.33 |
| 0.95 | 396 | 608,256 | 116 | 622.15 |

**+14.9% for one flag** at 0.92, and the in-flight column moves with the pool
exactly as the fork's ladder predicted: parents +49% gave in-flight +49%
(73 to 107). The peer's 0.36-in-flight-per-chunk relation was measured on the
fork's separate mamba pool and reproduces here on upstream's shared LCM arena,
across two allocators and six pool sizes -- which makes it a property of the
engine's *admission*, not of either allocator.

Returns are flattening (0.92 to 0.95 buys 9 in-flight and 1.3%), and in-flight
still has not saturated the client's 128, so pool size is not the only term.

**0.95 is not a safe setting, and the 128-request ladder above cannot see it.**
Re-run at `NREQ=256` -- the shape the 754.0 baseline was actually captured on --
0.95 dies with `torch.OutOfMemoryError` mid-run, taking the engine process with
it; the harness marked the result `.INVALID` and the client saw
`tcp connect error`. The same fraction completed 128/128 cleanly minutes
earlier. A longer run holds more concurrent decode state, so the peak arrives
later than a 128-request wave ever reaches. This is the shape-dependence
warning in `AGENTS.local.md` landing on my own arm: I had already written
"nothing OOMed at 0.95" off the short run, and it was wrong within the hour.
Treat every row of the ladder above as valid only at `NREQ=128`.

**A correction this forces on the rest of this report.** I have described
596.63 tok/s as the figure "at pure defaults, no flags, no env". That is wrong.
Every harness in `/workspace/roy/.cq*.sh` pins `--gpu-memory-utilization 0.80`
(only `.cq90.sh` and `.cq_uniq90.sh` use 0.90), so every number here was taken
with a non-default flag forced, on a pool about 10% smaller than this build's
own resolved default of 0.88. The comparison against 754.0 is still fair --
the peer's f162d993 baseline was captured through the same harness at the same
0.80 -- but "at defaults" was never true of either side, and a real deployment
that never touches the flag already gets a bigger pool than anything measured
here.

**Correction.** An earlier revision of this section blamed victim ordering --
"it evicts by age, not by how close a parent is to empty". That is wrong as a
cause of rejections, and the code proves it. `AdmissionPlanner::Plan` has
exactly one failure exit, `if (victim_candidates_.empty()) return false`, so a
rejection means every evictable block was already evicted and the demand still
did not fit. `fits()` is monotone under eviction -- emptying a parent trades
`slots - 1` local free slots for one empty parent, which never lowers the
number of admissible parents -- so no ordering can reach a fitting state that
age order fails to reach. **Eviction order cannot change accept/reject at all;
it only decides which reuse is sacrificed on the admissions that succeed.**

What a rejection actually reports is that reclaimable capacity is genuinely
exhausted: the blocks that remain are protected prefix hits or live state of
the 72 in-flight requests. Free slots exist but sit in parents bound to a group
that is not the one asking, and a parent only rebinds once every slot in it is
free. That is a real fragmentation cost, but it is a *binding* cost, not an
ordering one, and it is not addressable from the comparator.

I called that a bounded piece of work -- make victim selection parent-aware --
and then tried it. It is not bounded, and the reason is worth recording so the
next person does not spend the same afternoon.

The obvious shape is to reorder `AdmissionPlanner::evictionKey`, which today is
`(epoch, tier, position_rank, group_id, lcm_block_id, slot_index)`. Two
orderings were built and run against the 402-test C++ suite:

| ordering | result |
|---|---|
| parent occupancy ascending, ahead of age | 1 failure: `UsesLocalSlotAfterEvictingAnotherWholeParent` |
| group by parent, parents ranked by oldest member | 8 failures, incl. `RestoresOlderSiblingWhenMiddleCandidateFreesRequiredParent`, `MixedGroupTieEvictsNonClosedBeforeFullHistory`, `OrdinarySwaKeepsLongerUnhitFrontier`, `ActualHitPromotesChunkBoundaryToNormalValue` |

The first failure killed my justifying argument. I had reasoned that draining
the least-occupied parent empties one for *fewer* evictions than age order
needs, so it spends less reuse rather than more. The test shows the case where
that is false: a parent that still has free slots, bound to the group that is
asking, is already contributing usable capacity, so emptying it buys nothing
and costs an eviction. Cheapest-to-empty is the wrong metric; the free slots
have to be netted out first.

The second is the real obstacle. Within one epoch the existing order encodes
*which* boundary to reclaim, that a closed prefix is reclaimed from its suffix,
and which SWA frontier survives -- and those decisions range across parents.
Grouping victims by parent permutes all of them at once. Each of those eight
tests is a deliberate policy someone wrote down, not an incidental assertion.

So parent-awareness cannot be a reordering of this comparator. It would have to
be a second pass: keep the existing order, and when the eviction loop has freed
enough slots but still no whole parent, go back and consolidate. That is a
design change in a subsystem whose invariants are denser than they look from
the outside -- eight tests' worth denser than I assumed. Both attempts are
reverted; the suite is green at 402/402.

### What the gap actually is: concurrency traded for reuse

The ceiling is real -- 69 in-flight against a client driving 128, with KV
uncontended -- and the arithmetic names the trade rather than a bug:

|  | f162d993 | merged |
|---|---|---|
| recurrent state pool | 221 chunks, SEPARATE | 247 blocks, shared LCM arena |
| charged per in-flight request | ~1.73 | ~3.58 |
| concurrency reached | 128 | 69 |
| prefix reuse | 66.5% | 77-80% |

The pools are comparable. What differs is the per-request charge, and it is
roughly double. Upstream's boundary-and-promotion machinery retains more state
per request, which is exactly why prefix reuse is ten points HIGHER than the
baseline's -- and exactly why concurrency is half.

So the merged engine is not less efficient at the same operating point; it sits
at a different one. It recomputes less and holds fewer requests. On a workload
whose bottleneck is concurrency, that reads as a 21% throughput deficit; on one
bottlenecked by prefill compute it would read as a win.

Two consequences worth stating plainly:

- **This is architectural, not a defect.** Upstream's shared arena couples KV
  and recurrent-state capacity; the fork's separate mamba pool did not. Moving
  the operating point means changing the packing ratio or the memory budget,
  both of which cost something else.
- **754.0 was measured at the other operating point**, on a build whose cache
  and admission this tree no longer contains, and which will not start here
  (see above). Treating it as a target this tree has missed by 21% assumes the
  two are comparable, and nothing measured tonight supports that.

## What is still unexplained

**It is not prefill scheduling, and that took two experiments to establish.**

Tokens admitted per prefill step is ~456 against f162d993's 1994, which looks
damning until the reuse is folded in: at 76.7% reuse a 606-1015 token prompt
needs ~194 new tokens, and 456 tokens over 2.35 sequences per step is exactly
that. 113 prefill steps for 256 requests is about one step per request. The low
tokens/step is what high reuse looks like, not a defect.

Two attempts to buy throughput by rescheduling prefill, both measured and both
reverted:

- **Span the promotion boundary** instead of stopping on it. Tokens/step went
  to 2768, above the baseline, and prefix reuse collapsed 76.7% -> 1.4%. The
  split is load-bearing for reuse; `consumeCompletedBoundaryKind` only registers
  a promotion when the computed count lands on the boundary exactly.
- **Pack more requests per prefill step** by dropping the head-of-line break.
  It worked mechanically — prefill steps 113 -> 63, sequences per step
  2.35 -> 2.89, reuse essentially unchanged at 73.5% — and bought nothing:
  492.24 tok/s against 500.71. Halving the prefill forwards did not move
  throughput, so their count was never the bottleneck.

That leaves the gap somewhere in per-token compute rather than in scheduling,
which is consistent with what the merge changed underneath: upstream's bf16 MoE
kernel replaced ours (losing the sm_12x shared-memory guard), `cutlass_unquant`
took upstream's form, and kernel selection moved with them. None of that has
been measured. **The next attribution should be kernel-level and top-down per
`AGENTS.local.md` step 0, not another scheduling hypothesis** — three have now
died in a row.

Also unexplained, and the largest single unknown: the alignment fix's win is
1.76x while the work it saves is only 1.19x, leaving ~1.48x unattributed. A
wall-clock proxy suggests the same small steps simply run faster afterwards,
but this repository's own notes say host-side timings run ~3.6x high here, so
that instrument is the wrong one. `TOKENSPEED_LOG_MM_TIMING=1` on both builds
would separate "less overhead per step" from "better kernel shape".

## Two more capabilities the merge took, found while looking for headroom

- **fp8 KV is gone.** `--kv-cache-dtype fp8_e4m3` fails at startup with
  `NoKernelFoundError: No kernel found for attention.mha_extend_with_kvcache
  (k_cache=dense[storage=torch.float8_e4m3fn], ...)`, and
  `--attention-backend triton` does not help. The pool sizing works — logged
  `token_capacity` doubles from 379392 to 758784 — so only the extend kernel is
  missing. Our own notes record fp8 KV as worth 1.92x capacity and +19% here,
  so this is real headroom that the merge removed rather than a setting nobody
  tried.
- **Speculative decoding is unavailable for this model.** The Qwen3.5-27B
  checkpoint carries no MTP layers and no draft shards, and layer-truncation
  self-speculation was ruled out for Qwen3 previously. Not a merge loss, but
  it closes the other obvious route to the target.

## Hypotheses killed, so nobody re-runs them

| hypothesis | how it died |
|---|---|
| ceiling = token_capacity / max_model_len | token_capacity barely moves with max-model-len; running_max went 47 -> 50, not ~93 |
| the head-of-line prefill break is the admission limit | built and measured: 829 -> 728 tokens/step. Also disproved by seqs/step being 3.36 |
| `--max-model-len` causes the L2 fault | 8192 faults as readily as 4096 |
| `chunked_prefill_size > max_model_len` causes it | matching them still faults |
| the prefill CUDA graph is the vehicle | `--disable-prefill-graph` still faults |
| the alignment fix would reduce the fault | it did not: 2 hits, 146 failed |
| per-block ordering beats blunt stream ordering | 464.97 against 468.89 — inside noise |
| spanning the promotion boundary | reuse 76.7% -> 1.4% |
| prefill forward count is the bottleneck | packing halved the steps (113 -> 63) and throughput went 500.71 -> 492.24 |
| the paged-extend plan's 64 host syncs are the prefill gap | restoring it was +2.2%, not the predicted ~2x |
| the flashinfer sm120 GDN kernel is a 13% win | it produces `!!!!!!` instead of text |
| the GDN kernel is numerically broken | correct to 4.9e-4 on engine-shaped inputs; the prefill graph was breaking it |
| a stale identity-keyed plan explains the graph corruption | flashinfer's gdn_kernels has no caching at all: 0 lru_cache, 0 globals, 0 module dicts |
| the two kernels return different ranks, so `_land_in` narrows the wrong axis | both return `(1, T, H, V)` |

## Method notes that cost real time

- **A zero from a log that is still being written means nothing.** Grepping a
  live log for `illegal memory access` returned 0 on a run that had 2, and I
  briefly believed a hypothesis on it. Wait for the run to finish.
- **`Failed requests: 0` belongs in the harness as a gate, not in the reader's
  head.** Every merged-tree RAG number produced before that gate existed was
  invalid, including a 225.8 tok/s figure that had a whole concurrency-ceiling
  story built on it. The gate now renames the file to `.bench.INVALID`.
- **A clean numeric fit is a reason to test harder, not evidence.**
  379392/8192 = 46.3 against a measured 47 was a coincidence, and it survived a
  round of reasoning because it fit so well.
- **Kill leftover servers by PID and verify the GPU is empty before starting.**
  Three arms were lost to a stale harness that kept restarting a server on the
  card, one of them failing as an OOM at startup that looked like a capacity
  problem.
- **`docker exec ... | grep -E "tokenspeed" | kill -9` is `pkill -f` wearing a
  costume, and this file already warns about `pkill -f`.** Clearing leftovers
  that way inside the shared `unieinfra-dev` container killed another session's
  arm twice — a SIGKILL during startup with no traceback and a 1 KB log, which
  reads like an engine fault rather than someone else's cleanup. Being on a
  different GPU is not isolation; the container is shared. Record the PIDs your
  own launch produced and kill only those.
- **A throughput number cannot tell a working engine from a broken one, and
  neither can a request-failure gate.** The GDN arm scored 0 failed requests,
  passed the harness gate, and returned exclamation marks for every completion.
  Read the `.sample`.
- **When an old note says a measurement was bad, do not go looking for reasons
  it is stale because the new number is attractive.** That is exactly the move
  that nearly landed the broken kernel as a default.
- **The root filesystem being full reads as a compiler bug.**
  `error writing to /tmp/ccXXXX.s: No space left on device` — `/workspace` is a
  different device with room. Set `TMPDIR` under `/workspace` for builds.

## The A/B that was built and not run

Every number in this report is absolute. None is a same-harness differential
against f162d993, so "78% of 754.0" compares across two builds, two
architectures and several hours. The baseline's configuration cannot even be
expressed on this tree: `--mixed-prefill-decode-min-load` was deleted by the
merge and upstream has no equivalent, and the radix cache and fork admission it
ran on are gone.

The apparatus to settle it is built and verified:

- `/workspace/roy/.f162` — worktree at f162d993, scheduler ext built
  (md5 `fbbf22be77cd`).
- `/workspace/roy/.f162venv` — a venv with `--system-site-packages` holding the
  baseline `tokenspeed_scheduler`. Getting it to load took one trick worth
  recording: the scikit-build editable `.pth` installs a meta-path finder that
  pins the package to the HEAD source tree and the extension to dist-packages,
  and **no PYTHONPATH can override it**. Shadowing that finder's module name
  with an empty file inside the venv leaves normal `sys.path` resolution in
  charge, and the venv's copy wins. Verified: both the package and the ext
  resolve inside the venv.
- `/workspace/roy/.cq_f162.sh` — `.cq_nomb_rag.sh` with `W` pointed at the
  worktree, `W` threaded through to `.provenance.sh` so the label follows the
  build, and the server launched from the venv's interpreter.

Nothing in this touches the shared install, so it needs no coordination and has
nothing to restore — unlike the earlier plan, which swapped the dist-packages
`.so` and required an eight-minute window with the other session stood down.

Run it as:

    GPU=3 CONC=128 NREQ=256 ./.cq_f162.sh ours F162AB

and check `head:` in the log reads `f162d993` before trusting the number.

### It was run, and it cannot start

The A/B was executed. f162d993 does not come up on today's environment, and the
reason is not the merge or the permissions:

    AttributeError: 'ServerArgs' object has no attribute
    'enable_int8_mamba_checkpoint'

`f162d993/python/.../attention/registry.py:618` reads that attribute
unguarded, while f162d993's own `ServerArgs` never declares it -- zero dataclass
fields, and the only other reference is a defensive
`getattr(args, "enable_int8_mamba_checkpoint", False)` in the arg-normalisation
path. So the attribute existed on the args object only because the CLI and
gateway of that era put it there. Today's installed `smg_grpc_servicer` lives
in the system dist-packages, has moved since, and does not.

**So 754.0 is not re-measurable on this machine as it stands.** Reproducing it
needs the era-matching gateway reverted alongside the worktree, which is a
system-package change rather than anything inside this repository.

That does not make 754.0 wrong -- it was measured, and the ladder is committed.
It makes it **unverifiable from here**, which is the honest status of the
denominator in "78% of 754.0". Any future plan that budgets against a 22% gap
should first establish that the gap is measurable, because right now it is not.

The isolation itself works and is worth keeping: with the three editable
finders shadowed inside the venv, `tokenspeed`, `tokenspeed_kernel` and
`tokenspeed_scheduler` all resolve to the worktree, confirmed from the engine
log's own paths. What it cannot isolate is the installed gateway that spawns
the engine.

## Handover: the GDN prefill-graph corruption

The FlashInfer GDN kernel is worth +21.8% but currently requires
`--disable-prefill-graph`, which itself costs 8.4%; with the graph on, every
completion degenerates to `!!!!!!`. If the two are made to compose they stack.
This section records how far the diagnosis got, including one line of reasoning
that reads plausible and is not supported.

**Established.** `_record_break` (`breakable_cuda_graph.py:304`) binds tensor
arguments once, at capture, and its docstring states the assumption: tensor
args "alias persistent storage ... so they carry live values at replay". The
GDN metadata builder violates that assumption -- every branch allocates a fresh
`query_start_loc` per forward (`hybrid_linear_attn.py:811` `arange`, `:819`
`arange`, `:842` `zeros`, `:857` `empty`), none of them writing into a
persistent buffer, while the class does maintain `query_start_loc_list` (`:342`)
and `cached_cuda_graph_decode_query_start_loc` (`:343`), so the machinery
exists and this path bypasses it.

**Ruled out.** That the prefill graph is not a breakable capture. It is:
`prefill_graph.py:427` constructs `BreakableCapture` and captures `_run_inner`
inside `with cap:`, so `is_breakable_capture_active()` is True throughout the
GDN forward. Note that this does *not* conflict with
`torch.cuda.is_current_stream_capturing()` reading False inside the break --
those are different predicates, and running the fn off the capturing stream is
exactly what a break does. Reading the second as evidence of "no break" is a
trap; it cost one session an afternoon of the wrong hypothesis.

**Not established, and it is the load-bearing step.** That the *bound tensor
argument* is what goes stale. The GDN call at `:1747` sits in `forward_extend`
(`:1438`), which is reached from `forward` (`:1938`) -- and `forward` is the
function carrying `@break_point`. So the whole GDN forward, planner included,
runs eagerly at replay, and `query_start_loc` is read from
`self.forward_metadata` at replay time, which should make it fresh rather than
stale. For the staleness story to hold, what must be stale is the *metadata
object itself* -- i.e. the prefill graph path replays without re-running
`init_forward_metadata`, leaving `self.forward_metadata` holding the dummy
capture batch. That is a different bug from a frozen argument binding, and it
implies a different fix.

**Decided, by running it.** With `TOKENSPEED_FI_GDN_ALLOW_PREFILL_GRAPH=1`
(a diagnostic escape hatch added for this, which logs an error and guarantees
garbage output) and `TOKENSPEED_TRACE_GDN=1`, every GDN call in a `CONC=8`
serving run reports the same thing:

    call capturing=False q_rows=8192 cu_last=8192 cu_n=2 state_id=<varies> cu_id=138186089069488

Three facts, and together they settle it:

- `cu_n=2` with `cu_last=8192` is the *dummy capture batch* -- a single
  sequence spanning the whole padded bucket. A real 8-way concurrent extend
  batch cannot produce that shape on every call. So the kernel is being handed
  the capture-time `cu_seqlens` at replay.
- `cu_id` is byte-identical on every call while `state_id` changes on every
  call. The recurrent state is rebound per forward; `query_start_loc` is not.
  So it is the tensor binding, not a planner-side host plan cache.
- `capturing=False` throughout, which is the eager break working as designed --
  the predicate trap described above, seen directly.

This is the peer session's hypothesis confirmed, and it lines up with a second
reading of the graph itself: `prefill_graph.py` calls `init_forward_metadata`
only on the capture path (`:620`), and `replay()` (`:671`) lands input embeds
and replays without ever rebuilding it. The metadata object the eager break
reads at replay is the one built for the dummy batch. Meanwhile
`_replay_bucket`'s own docstring states the intended contract -- "the attention
break reads the LIVE ambient ctx" -- so this is a violation of the design, not
a limit of it.

**The fix, for whoever takes it**, is to make the live batch's metadata reach
the eager break at replay rather than the capture-time object. The captured
segments are unaffected: their kernels are already baked into the graph and
read the static buffers that `_land_in` refreshes. Acceptance is unchanged and
free -- GDN enabled *without* `--disable-prefill-graph`, `.sample` must be
coherent, and the degeneracy gate rejects the exclamation-mark failure
automatically. The payoff is the graph's +8.4% stacked on the kernel's +21.8%.

**The experiment that decided it**, for whoever takes this: inside the GDN
mixer at replay, log `data_ptr()` and the contents of `query_start_loc`
alongside the live metadata's. Fresh contents mean the argument binding is
innocent and the fault is in the planner's own host-side plan cache; dummy-batch
contents mean the metadata is not being rebuilt per replay. One arm decides
which fix to write. Acceptance is free either way: run GDN enabled *without*
`--disable-prefill-graph` and require the `.sample` to be coherent -- the
degeneracy gate rejects the exclamation-mark failure automatically, which is
the check that caught this being reported as a +13.3% win in the first place.

## The comparison against 754.0 was mismatched, and the real figure is ~60%

Every "% of baseline" in this report compared **our mixing-on number against the
baseline's mixing-off number**. Once mixing could be switched off -- it could
not be, until `1eb1d46f` added `--disable-mixed-batch` -- the mismatch was
measurable, and it is large.

RAG shape, `CONC=128 NREQ=256`, GMU 0.92, 256/256 successful and 0 failed on
both arms:

| | tok/s | mean TTFT | peak in-flight |
|---|---|---|---|
| mixing on (the default) | 645.49 | 3679 | 110 |
| mixing off (`--disable-mixed-batch`) | 461.25 | 4379 | 103 |

**Mixed batching is worth +39.9% here**, and it engages by upstream's own policy
with no threshold knob. The f162d993 arms differ on exactly that axis: 754.0 was
captured with `--enable-mixed-batch` at its default threshold, where mixing
seldom fires, and 1086.8 with `--mixed-prefill-decode-min-load 0`, which forces
it. So the matched pairs are:

| pair | ours | f162d993 | ratio |
|---|---|---|---|
| mixing engaging | 645.49 | 1086.8 | **59.4%** |
| mixing not engaging | 461.25 | 754.0 | **61.2%** |

Both pairs land at roughly 60%, which is worse than the 79% this report has
been carrying and much worse than the 85.6% the memory ladder briefly seemed to
buy. The two independent pairs agreeing is a reason to trust the worse number,
not to discount it.

Restated honestly: the memory fraction is a real +10.4% on the matched shape
and the merge's own fixes took the tree from 231 to 645, but **the tree is at
about 60% of the fork on both matched configurations**, and the earlier
"79% of the nomix target" understated the gap by comparing across the mixing
axis. The 1086.8 arm is reachable in principle now -- mixing engages by default
-- so it is the right target, and it is 1.68x away.

## Where the remaining gap lives: decode does not batch

With the mixing axis matched and the memory lever taken, the gap is not
admission and not prefill. Arithmetic on the matched arms, mixing off on both
sides:

| | peak in-flight | tok/s | per request |
|---|---|---|---|
| ours, GMU 0.92 | 103 | 461.25 | **4.48 tok/s** |
| f162d993, GMU 0.80 | 128 | 753.97 | **5.89 tok/s** |

**Corrected from the peer session.** An earlier revision of this table put the
fork at 80 in-flight and 9.43 tok/s per request, and concluded we were 2.1x
slower per request while holding *more* requests. That was a shape mislabel:
80-94 in-flight is the fork's **unique-prompt** arm (chat corpus, no shared
prefix), while on the RAG shape it saturated at 128 of 128. Matched, the
per-request gap is **1.31x**, not 2.1x -- still real, half the size, and we
hold fewer requests rather than more.

Prefill is not the problem -- mean TTFT is now 3679 ms against the
baseline's 3906 -- and the layer-by-layer probe recorded earlier in this project
has per-decode-token cost at parity with vLLM at concurrency 1 (36.4 vs 36.9
ms). Parity at C=1 and half the rate at C≈100 is a scaling statement.

Our own concurrency sweep, GMU 0.92, mixing on, 256 requests, 0 failed:

| client concurrency | peak in-flight | tok/s | per request | mean TPOT |
|---|---|---|---|---|
| 32 | 32 (client-limited) | 502.11 | 15.69 | 54.5 ms |
| 128 | 110 | 645.49 | 5.87 | 123.9 ms |

**3.4x the in-flight requests buys 1.28x the throughput.** For a 27B decode
step, which is bandwidth-bound on weights, adding requests to a batch should be
nearly free and TPOT should stay close to flat; instead TPOT grows about
linearly with batch size. Something in the decode step is O(batch) with a large
constant.

It is not a per-request loop in the hybrid backend's decode path --
`forward_decode` (`hybrid_linear_attn.py:1278-1438`) contains no per-request
iteration, no `.item()`, and no `.cpu()`. That was the first candidate and it is
excluded.

**Host-versus-device, measured: not host-starved.** Sampling GPU 3 once a
second across a C=128 arm, over the serving window (41 samples with the device
non-idle): median 99%, mean 77.2%, 68.3% of samples at or above 90%. A
scheduler that could not feed the device would show the opposite.

The instrument has a limit worth stating rather than glossing:
`utilization.gpu` reports the fraction of time at least one kernel was
resident, so a stream of small inefficient kernels reads 100% exactly like a
saturated one. This rules out gross host starvation. It does **not** establish
that the decode step's device work is efficient, and it cannot -- that needs
per-kernel timing, not a sampler.

Also worth recording for anyone reading deltas off this report: two runs of the
identical configuration (C=128, GMU 0.92, mixing on) measured 645.49 and 684.72
tok/s, a 6.1% spread. Nothing smaller than that should be read as an effect.

### The fork's own curve, and the test that decides the branch

The peer session's committed f162d993 ladder, RAG shape, mixing off, GMU 0.80:

| C | tok/s | peak in-flight | per request | reuse |
|---|---|---|---|---|
| 1 | 26.64 | 1 | 26.64 | 72.1% |
| 16 | 270.33 | 16 | 16.90 | 47.7% |
| 32 | 384.61 | 32 | 12.02 | 40.8% |
| 64 | 560.77 | 64 | 8.76 | 50.1% |
| 128 | 753.97 | 128 | 5.89 | 66.5% |

This kills a weaker version of the argument above. Per-request rate falling
steeply with batch is what continuous batching does on *both* engines -- the
fork drops 4.5x from C=1 to C=128 -- so "per-request rate collapses with
concurrency" on its own distinguishes nothing.

What does distinguish: whether our curve sits a constant factor below the
fork's everywhere, or only diverges at large batch.

| | ours | f162d993 | ratio |
|---|---|---|---|
| C=32 | 502.11 (mixing ON) | 384.61 | not comparable |
| C=128 | 461.25 (mixing off) | 753.97 | 0.61x |

Our C=32 arm has mixing on and the fork's ladder does not, so the one row that
would decide it is not yet matched. Deflating by the 39.9% mixing is worth at
C=128 gives roughly 359 against 384.61 -- near parity -- which would mean the
curves coincide at moderate batch and diverge only at large, i.e. decode
batching is the target. But that deflation is an estimate: the 39.9% was
measured at C=128 and there is no reason it carries to C=32. The matched arm
(C=32, `--disable-mixed-batch`, GMU 0.80) is the one that decides it, and a
result near 359 rather than near 384 flips the answer to a flat per-step
deficit instead.
