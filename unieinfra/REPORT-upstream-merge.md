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
| + mixed batching | **500.71** | 7014.3 | 76.7% |
| default backend + stream ordering | 468.89 | 7505.7 | — |
| | | | |
| f162d993 nomix target | 754.0 | 3905.8 | 66.5% |
| f162d993 mix target | 1086.8 | 2227.6 | 85.8% |

231 -> 501 over the session, and the default configuration serves again. Still
66% of the nomix target and 46% of the mix target.

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
- **The root filesystem being full reads as a compiler bug.**
  `error writing to /tmp/ccXXXX.s: No space left on device` — `/workspace` is a
  different device with room. Set `TMPDIR` under `/workspace` for builds.
