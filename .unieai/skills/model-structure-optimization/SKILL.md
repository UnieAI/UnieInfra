---
name: model-structure-optimization
description: "Read this skill when a model already runs correctly in the engine and the goal is a large speedup (1.5x-2x, '快2x', '要贏 vLLM/sglang 一大截', 'prefill 太慢', 'TTFT 太高', 'long context 很慢') rather than a few percent. It finds work the model's own structure makes unnecessary and removes it before any kernel tuning: rows no later layer consumes (row narrowing / CED decoder on the prompt tail), chunks that emit nothing, per-layer metadata identical across layers, scoring over capacity instead of live length, full-history scoring where a candidate pool exists, cache retained beyond what kernels read, windowed state that should be replayed instead of prefix-cached. It then makes the remaining forward graph-capturable and sync-free (prefill graphs split around a row narrowing, no .item(), no serving-time JIT). Distilled from upstream tokenspeed's DeepSeek-V4/V4.1 history. Not for porting a new model (model-bring-up) or step-level decode kernel accounting (engine-decode-parity)."
---

# Model structure optimization

A 2x does not come from kernels. On upstream tokenspeed's DeepSeek-V4.1 work
the end-to-end gains from per-kernel fusion on NVIDIA were 2–8% each
(a24104a7, f774d2b9); the large ones came from not doing work at all, or from
fixing a change that made the engine redo work (366d2a00, −30% decode). Do this
pass first, on a correct model, and only then go to `engine-decode-parity`.

The single question behind every item below: **who consumes this value?** If
the answer is "nobody on this step", the work is removable. If it is "every
layer, identically", it is hoistable. If it is "only the live part", it is
boundable.

## 0. Prerequisites

- The model is correct: temperature-0 token agreement with a reference engine
  and an accuracy gate (GSM8K ≥ 0.90 for DeepSeek-class) pass. Every change
  below is re-gated against them.
- You can attribute GPU time per layer and per op on prefill *and* decode
  (nvtx ranges per sublayer; an eager trace for attribution, a graphed run for
  timing). Know the split between prefill and decode time for the target
  workload before choosing where to look: an agent workload with 30k-token
  prompts and 500-token answers is a prefill problem.

## 1. Read the model as a dataflow graph of cache writes

Write down, per layer: what it reads from cache, what it writes to cache, which
rows of its output the next layer consumes. Use the reference implementation,
not the engine code, so the engine's current choices do not anchor you.

DeepSeek-V4.1 read this way:

| layers | writes | reads |
|---|---|---|
| 0–1 | SWA window KV | own window |
| 2–19 (ratio 2) | SWA; sources 2/8/14 also compressed KV + index K | own window + source's compressed KV, source's top-k |
| 20 | SWA, compressed KV (ratio 1), index K, candidate blocks | — |
| 21–39 | SWA only | own window + layer 20's compressed KV; top-k from index sources 24/28/32/36 |

The last layer that writes *global* (non-window) state is 20. Everything after
it writes only a 128-token window. That one fact is worth roughly half of
prefill.

## 2. The removal checklist

Go through every item; for each, write down the quantity it saves on the target
workload before implementing it.

1. **Row narrowing after the last global-state writer.** If no layer after L
   writes global cache state, then for a prefill chunk only the rows feeding
   (a) the next token's logits and (b) the windows later layers keep need to
   pass L. DeepSeek-V4.1's decoder (layers 20–39) runs only on each completing
   prompt's last 128 rows (4986d149, `_ced_decoder_start`). The narrowing layer
   itself takes all rows in, emits the view's rows out.
2. **Zero-output chunks.** A prefill chunk that does not complete its prompt
   emits no token: skip everything after the last cache producer — decoder
   layers, final norm, LM head, sampler, draft writes (cb127fde). Keep the
   cache producers running on all rows.
3. **Hoist identical per-layer work.** Slot mappings, decode schedules,
   compressed-slot tables depend on (step, length tensors), not on the layer.
   Compute one per distinct key per forward and share (3cfd377d: one decode
   schedule per (SWA length, global length) pair). Cross-layer *model* sharing
   is the same idea one level up: V4.1 shares compressed KV and top-k across
   layers by design, so the engine must produce them once and alias the
   consumers to the source's buffers, never recompute.
4. **Bound by live length, not capacity.** Any score/reduce/select over a row
   sized by the page table or `max_model_len` (524288 columns for a 1M pool)
   must visit only live tiles: fixed launch grid for graph replay, early exit
   on device lengths (339edcd2; eda73219 −43–47% score time; 4c4de244 up to
   30x).
5. **Score only the candidate pool.** When a coarse stage already chose
   candidates (V4.1 layer 20 picks 2048 blocks of 8), later fine stages gather
   and score the compacted pool instead of masking a full-history score row
   (97c0066e, 9263c4a3). Masking after computing everything saves nothing —
   check whether your engine's "candidate filter" actually reduces FLOPs.
6. **Retain exactly what kernels read.** Size each cache group's retention to
   the window the kernel reads, not a round number (fff486b0, a98c3fc0). Every
   extra retained row is admission capacity lost.
7. **Replay instead of caching approximate windowed state.** SWA rows and
   compressor tails are cheap to regenerate and awkward to prefix-cache. Mark
   them replayable: on a prefix hit, re-feed the last window's tokens and mask
   the global writes for those replayed rows so shared rows stay exactly what
   the first computation produced (4986d149, scheduler §1.3).
8. **Hoist position-independent work out of sequential loops.** A draft block's
   projection is one GEMM for all positions; only the bias depends on the
   previous token (b063800e).
9. **Data format to fit the hardware, not a forced kernel.** If the target
   lacks the units for the reference format (sm90 has no FP4 tensor cores),
   store a wider format with a native reader and record the capacity cost
   (7c70d580). Measure first whether a requantization is actually lossy — on
   V4.1 folding 32×32 ue8m0 FP8 weight blocks into 128×128 changed ≤0.01% of
   elements (rel. err ~1e-7), so the existing 128-block FP8 GEMM path applies.

## 3. Make the remaining forward graph-capturable and sync-free

1. **No host syncs on the prefill path.** Every `.item()`, every row-by-row
   read of a CUDA buffer, every length reduction done on device then read on
   host serialises the chunk. Use the scheduler's host mirrors (18e5fd4b,
   ceaf86ae).
2. **Split the graph around a row narrowing** instead of opting out of prefill
   graphs: encoder graphs per token bucket, the narrowing layer eager, decoder
   graphs per decoder-row bucket shared across token buckets, capped at
   max_num_seqs × window (13657d1d, `NarrowingPrefillModel`). Eager and graphed
   must be one composition so they cannot drift. Under attention-DP row counts
   are rank-local — keep it off there.
3. **Keep each eager break to the ops that read per-request metadata.** A
   `@break_point` on the whole attention module also re-runs its projections,
   norms, RoPE, o-proj and the TP all-reduce from Python on every replay. On
   V4.1 H200 TP4 that was ~3 ms of host time per layer and prefill steps ran
   ~145 ms wall for ~45 ms of GPU work (b647604c splits it: graphed
   projections → break holding cache writes, selection, attention → graphed
   o-proj + all-reduce). Symptom in a trace: per-rank idle and all-reduce
   wait trade off between ranks while their sum stays constant — the step is
   bounded by the slowest rank's host. Confirm with py-spy on the forward
   thread (`docker exec --privileged … py-spy record --threads -f raw`), not
   the torch profiler, which records no Python ops on that thread.
4. **Separate graph memory pools** for draft and target; reserve graph pools
   inside the KV budget; warm on the capture stream (fcbfd646, b121eb08).
5. **Capture failure is fatal**, never a silent eager fallback.
6. **No JIT on the serving path**: constexpr only for startup-fixed values; a
   no-recompile sweep test per batch-shaped kernel.

## 4. Numerics rules for structural changes

- Narrow *after* the split-K / quantized GEMMs of the narrowing layer, not
  before: gathering first changes the reduction shape of the rows you keep.
- When an incomplete prefill shares a batch with emitting work, keep one
  decoder row for it: removing it changes quantized attention/MoE batch shapes
  and can change other requests' logits (cb127fde).
- Every removal is gated on temperature-0 token agreement with the unoptimized
  path on a mixed batch (prefill + decode + prefix hit + chunked prompt), not
  just a single request.

## 5. Order and reporting

1. Profile prefill and decode separately on the target workload; rank items
   from §2 by measured time they would remove.
2. Implement the largest first; gate; re-measure the whole curve.
3. §3 next, then `engine-decode-parity` for what remains.
4. Write each change up with: the quantity removed, the workload, hardware,
   TP/EP, baseline sha, the numbers, and what was not measured. Update the
   design doc in the same change.
