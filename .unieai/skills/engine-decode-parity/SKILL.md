---
name: engine-decode-parity
description: "Read this skill when an inference engine (or a fork of one) has to match or beat a reference engine on decode — phrasings like '追平 vLLM', '輸 sglang', 'total tps 差多少', 'TTFT tail 爆炸', 'throughput 飽和', 'why is our decode slower', or a request to port an upstream engine's optimizations. It chains step-level measurement (device-busy ms/step, launches/step, family attribution) to an arithmetic floor, then applies the strategy catalogue distilled from Tokenspeed/vLLM upstream commits: fold epilogues, one launch per seam, recompute instead of materialize, select instead of sort, compute over the visible set, produce in the consumer's dtype, hoist per-step metadata, prepare weights once at load, pick kernels per token bucket, overlap independent branches, collapse collective chains. It covers how to port 1:1 without replacing the existing path, how to declare and gate kernel capabilities, and how to verify with ULP-relative bounds. It is not for prefill-only tuning, single-kernel micro-benchmarks, or flag/scheduler-parameter sweeps. Before kernel work on a new model, run model-structure-optimization first (work the model never needed is the bigger lever); for bringing a model up from scratch use model-bring-up."
---

> Companion skills: **model-bring-up** (new model → correct engine path),
> **model-structure-optimization** (remove work the model's shape makes
> unnecessary: row narrowing, cross-layer sharing, candidate-pool scoring,
> zero-output chunks, prefill graphs). Upstream's DeepSeek-V4.1 history shows
> the order that pays: structure first, graphs/syncs second, kernels third —
> the structural changes and one structural regression fix (−30% decode) were
> the largest NVIDIA wins; per-kernel fusion landed at 2–8% end to end.

# Engine decode parity

Matching a reference engine on decode is an accounting problem before it is a
kernel problem. Almost every real gap is one of four things: too many launches,
a buffer sized by capacity instead of by work, work repeated per layer that
belongs once per step, or an epilogue that round-trips through memory because it
lives in its own kernel. This skill is the order to check them in, and the
evidence each check requires.

## 0. Define parity before measuring

Fix the corpus, the concurrency, the TP degree and the token budget. Report the
whole curve, never one point:

- `total_tps`, `requests/min`, `tokens/req`
- TTFT p50 / p90 / p95 / p99, TPOT p50

**Parity means same order of magnitude and within ~5–10%.** Do not demand
bit-equality of throughput, and do not confuse the two bars: `+10%` is the bar
for an *optimization you invented*; a *port of the reference's own technique*
counts as parity when it lands in the same band and the numeric output is
equivalent.

Sanity-check the comparison before believing it. If the token counts differ
(`tokens/req` 41 vs 127), the arms are not measuring the same work and the
faster one may simply be generating less — a corpus with no chat template will
silently do that. An engine that fails to load the corpus and one that loads it
badly look identical in a tps column.

## 1. Measure ms/step, and prove the bottleneck is the GPU

The decisive number is **device-busy milliseconds per scheduler iteration** at
the target batch size, not the throughput. If device-busy / iteration is ~0.94,
the engine is GPU-bound and no amount of host-side work (commit, dispatch,
scheduler) can help; if it is 0.6, the GPU is starving and you are looking at a
host problem wearing a kernel costume.

Then get **launches per step and mean µs per launch**. An empty Triton launch
costs 7.7–13.9 µs on Hopper, so `kernels_per_step × 10 µs` is the floor that
fusing has to beat. A step with 4,271 kernels at 5.5 µs is spending ~40 ms of
its life in launch overhead alone.

Attribute by **kernel family** (join the trace's `External id` to a kernel name,
then to the issuing frame — do not guess from names). Two traps that produce
confident wrong answers:

- **CUDA-graph replay erases `cpu_op`.** In a captured decode step, ~90% of
  kernels carry no attribution. Take an **eager** trace to attribute — and then
  remember that an eager *prefill* pass is not the decode path, so a frame's
  kernel count from a prefill trace does not describe decode.
- **The warm-up pass looks like a hot kernel.** Check which forward mode
  produced the frame before acting on it.

Also: count tokens from `usage.completion_tokens`, never by counting SSE chunks;
and never quote torch-profiler `cpu_op` totals as GPU time (inflated ~3.6×).

## 2. Turn the gap into a work list by subtraction

Convert the whole difference into **per-step milliseconds** (a c16 gap from
23.59 to 16.66 ms/step is 6.93 ms), then subtract the floors you cannot beat:

- **Bandwidth floor.** MoE weights streamed per step: `experts × layers ×
  bytes / HBM_BW`. If the reference pays the same GEMM count at the same size
  (they do — 86 GEMMs either way), that time is not the gap; stop looking there.
- **Launch floor.** `kernels/step × 10 µs`.
- **Elementwise floor.** Rank families by **time**, not by node count.
  Elementwise was 44% of our nodes and 12.2% of GPU time (2.88 ms vs the
  reference's 0.14) — the node count would have sent us to the wrong place, and
  the reference's 0.14 says the whole family is avoidable.

What remains, ranked by measured ms, is the work list. Keep it ranked; never
work on the second item because it was easier to read.

## 3. The strategy catalogue

Every entry is a quantity attacked. The rename is deliberate: find the quantity
in your engine, not the commit in theirs.

**A. Fold the epilogue into the producer.** The reference's mHC runs one CUDA
launch (`mhc_big_fuse`) where a naive layer runs a split-K reduction, a mix
epilogue, a residual apply and a norm — `projection + square_sum → layer_input`
with the normalisation inside. Look for a kernel that writes a buffer only so a
*second* kernel can read it and write again; that buffer is the target.

**B. One launch per seam, counted per layer.** Count kernels per layer, not per
module: a seam that runs twice per layer costs double. 4 kernels/seam × 2 seams
= 8/layer, against the reference's 2/seam = 4/layer, is the difference between
2.13 ms and 0.96 ms. When a fusion looks impossible, check whether the *grid
shape* is the blocker rather than the math — a row-wide reduction cannot live in
a grid split over the row (`hidden/1024` programs), but one program per token
can own the row and reduce it.

**C. Recompute instead of materialize when a grid barrier would be needed.**
Fusing A into B normally needs every program of B to see all of A. Instead of a
second launch or a cooperative barrier, let each program recompute the small
producer value itself. A 6-float gate (4 reduced GEMM columns + one 4×4
Sinkhorn, 20 iterations) is free next to the `hidden`-wide collapse the program
was going to do anyway. Recompute is cheap when the producer's output is tiny
and the consumer's grid is huge.

**D. Compute over the visible set, not the capacity-sized buffer.** A decode
logits row is page-table-capacity wide (524288 columns with a 1M-token pool)
while a few thousand rows are actually visible. Anything that reduces over that
row — block maxima, top-k, sorting — should iterate the visible blocks and skip
the rest.

**E. Select instead of sort.** "Successive winners without fully sorting each
lane's candidates", an MSB-first radix select (four 8-bit histogram passes over
order-preserving uint32 keys) that emits ids in index order. If the consumer
only needs the top *k* ids, a sort is spend with no purchase. Cheap wins exist
inside this: fold two constant-folding ops into one expression, use
`masked_fill` instead of `full_like` + `where`, replace 64-bit shuffle
reductions with two 32-bit ones.

**F. Produce in the consumer's dtype.** If the expert backend wants bf16
weights, score/normalise/scale in fp32 and store bf16 directly — one store, no
separate scale-and-convert pass. Same for quantised layouts: emit the packed
block layout the next kernel reads instead of a canonical layout it must
repack.

**G. Hoist per-step work out of the layer loop.** Work that depends only on the
step, not the layer, must run once per step: sanitising a slot mapping, building
a decode schedule, resolving a compressed-slot table. Upstream's commits are
literally "sanitize SWA slot mapping **once per step**" and "share the native
decode schedule **across layers** in a forward". A per-layer memo that is
inert (because its guard never fires) reads as if it were already hoisted — check
that the guard is true on the path you think you fixed.

**H. Reuse an existing kernel before writing one.** The reference routes new
projections through the router GEMM it already had, and extends that kernel's
supported hidden size rather than adding a second kernel. Hunt for the
already-tuned kernel before opening an editor.

**I. Skip work that cannot produce output.** Incomplete prefill chunks, zero
tokens, `x.shape[0] == 0`, a bucket that will not be used. Early-return, and
make it cheap so the branch does not become its own cost.

**J. Specialise on measured shapes — with the gate declared in the function.**
The convention worth copying: the fused entry point carries its own
capability declaration next to the code,

```python
mhc_big_fuse.supported_n_splits      = frozenset({1, 2, 4, 8, 16, 32, 64})
mhc_big_fuse.supported_hc_mults      = frozenset({4})
mhc_big_fuse.hidden_size_multiple    = 8
mhc_big_fuse.supports_fused_norm     = True   # norm_weight may be passed
```

so the caller can ask *the kernel* what it supports and fall back to the
general Triton path outside that set, instead of a global flag or a silent
assumption. A capability that is declared and checked cannot degrade silently; a
capability expressed as a command-line flag can, and will.

**K. Prepare weights once, at load — and re-check after every refactor.**
Pre-swizzled scales, transposed layouts, requantized blocks belong in
`process_weights_after_loading`, never in `forward`. Upstream lost ~30% of
DeepSeek-V4.1 decode (B300 TP4, BS1 8.38→5.88 ms when fixed, 366d2a00) because
a refactor moved MXFP8 scale swizzling back into every forward. Symptom in a
trace: a small, identical kernel per linear per step whose inputs never
change. A persistent buffer whose address a graph captured must be updated in
place after a weight update, not reallocated.

**L. Pick the kernel per token bucket from a measured table, not per model.**
Build a µs table per candidate at the real per-rank shapes — decode = batch ×
verify width (one DSpark verify is 6 rows; 16 requests are 96), a prefill
chunk — under graph replay, cold cache, paired alternating samples. Kernels
that dequantize in registers scale with tokens; bandwidth-bound ones do not
(Hopper MXFP4 MoE per layer: CUTLASS 621 vs Marlin 1552 µs at 192 tokens, 671
vs 2855 at 576, f9314560). Admit a specialised kernel for a bucket only if it
wins by ≥5% at every measured row of that bucket in every repeat (4101fed3).
The answer flips with concurrency (MegaMoE −1–4% at c16–128, +7–10% at
256–512, c8225898) — select per bucket, not once.

**M. Overlap independent branches on an aux stream.** In attention, the
compressor/indexer branch and the Q/RoPE branch share only their input; submit
the longer branch first on the aux stream and join before the consumer. Same
for small side computations (hyper-connection mixes). Verify with a trace that
the branches actually overlap — a hidden sync (an `.item()`, a host read of a
device length) silently serialises them.

**N. One collective per step, carrying packed results.** A per-step chain of
masked lookups, GEMMs, reductions and two all-gathers became one Triton kernel
plus one all-gather of packed int64 (logit, token) candidates per draft
position in upstream's DSpark sampler (e87f7267). Count collectives per step
like launches; each is a few µs of latency plus a sync point.

**O. No compilation on the serving path.** A Triton `constexpr` is a JIT cache
key: a constexpr derived from batch size, chunk length or page-table width
recompiles mid-serving and stalls 100+ ms. Constexpr only for values fixed at
startup; batch-varying values go to runtime args or fixed buckets. Gate with a
no-recompile sweep test and serve CI with the compile check set to error.

## 4. Laws that keep the measurement honest

- **A parameter sweep is a diagnostic, never a deliverable.** If a flag changes
  the number, the flag is telling you where the defect is.
- **A disable flag is a bug report.** `--max-cudagraph-capture-size` below
  `--max-num-seqs` did not "save memory", it removed the decode graph and ran
  steps eager with nothing reporting it.
- **A silent trim is a correctness bug.**
- **The corrupt arm can look faster.** A 1.217× A/B once traced to a numerics
  bug, not speed.
- **Know your noise floor.** Re-run the control: ours was 1.008×. Anything under
  ~2% is nothing.
- **Ask what quantity the code computes before asking what value to give it.**

## 5. Port it 1:1, then improve it

Order matters: numeric equivalence, then no perf regression, then win.

- Register the ported implementation as an **additional** solution
  (`register_kernel`, one more entry in the family), never as a replacement.
  Both arms stay runnable, so the old path is the control.
- Keep the third-party code in `thirdparty/`, keep the Triton fallback for the
  shapes the fast path does not claim.
- **If the reference engine is also your upstream, check whether it already
  implemented what you are about to write.** `git log --grep=perf` over the
  fetch. We hand-wrote a Triton fusion for a seam; upstream had shipped
  `mhc_big_fuse.cu` for the same seam, with a `supports_fused_norm` gate. Their
  version is better tested and it is already a build target.

## 6. Verification ladder

1. **Unit-test the fused op against the unfused path.** Bound the error in
   **ULPs, not absolutes**: bf16 keeps 8 mantissa bits, so one ULP is 0.008 near
   1.0 and 0.031 near 4.0. An absolute tolerance of 0.016 fails on a value of
   4.06 that is off by exactly one ULP, and passes a genuine 2% error near 0.1.
2. **Demand bitwise equality only where the value feeds the next stage** (a mix
   vector handed to the next seam). Elsewhere, allow the rounding the fusion
   legitimately changes — and say so, because folding away an intermediate
   buffer usually *removes* a bf16 round-trip and makes the fused path the more
   accurate one. Measured example: post/comb mixes bitwise equal, collapsed row
   within 2 ULP and 2.5e-3 relative L2, entirely explained by one removed round.
3. **End-to-end with the same corpus, a fresh cell, an empty prefix cache** —
   and report the full curve. A point win with a curve that lost its tail is a
   regression.
4. **Watch for the inversion**: the fused arm can lose at low concurrency and
   win at high, or vice versa, because it changes the number of blocks. Measure
   the whole sweep.
5. **Bit-exact whenever the output makes a discrete choice.** A fused quantizer
   whose payload decides which rows the indexer scores, a hash, a top-k set, a
   sampler: reproduce the eager chain bit for bit (upstream reproduces
   PyTorch's `amax * (1/448)` exactly, 927bf1a8). ULP bounds are for values that
   only flow into arithmetic.
6. **Do not change another request's arithmetic.** Moving a gather in front of
   a split-K or quantized GEMM, or dropping a row from a batch, changes the
   reduction shape and can alter *other* requests' logits. Keep the GEMM shape,
   select afterwards.
7. **Graph-replay tests with poisoned tails.** Fill unwritten buffer tails with
   garbage before replay to prove a length-bounded kernel obeys device lengths;
   replay with changed inputs; PDL on and off; one all-padding batch.
8. **Accuracy gate on the served model** (GSM8K >= 0.90 on DeepSeek-class
   models, full set when the checkpoint changes) in addition to temperature-0
   token agreement against a reference engine. Lossy speedups (FP8 activations
   for FP4 experts) ship opt-in, behind a flag, after this gate.
9. **State what was not measured** in the write-up ("kernel microbenchmark
   only; end-to-end not measured"). An unmeasured claim is a follow-up, not a
   result.
