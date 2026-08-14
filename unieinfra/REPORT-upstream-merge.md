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
