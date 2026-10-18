---
name: engine-decode-parity
description: "Read this skill when an inference engine (or a fork of one) has to match or beat a reference engine on decode — phrasings like '追平 vLLM', '輸 sglang', 'total tps 差多少', 'TTFT tail 爆炸', 'throughput 飽和', 'why is our decode slower', 'agent workload 比 vLLM 慢', 'GPU 沒吃滿', or a request to port an upstream engine's optimizations. It chains step-level measurement (device-busy ms/step, launches/step, family attribution, per-rank idle vs collective wait, py-spy on the forward thread when the step is host-bound, e2e = TTFT + per-token decode under load) to an arithmetic floor, then applies the strategy catalogue distilled from Tokenspeed/vLLM upstream commits: fold epilogues, one launch per seam, recompute instead of materialize, select instead of sort, compute over the visible set, produce in the consumer's dtype, hoist per-step metadata, prepare weights once at load, pick kernels per token bucket, overlap independent branches, collapse collective chains, keep eager graph breaks to the metadata-reading ops, cover the batched prefill size with a captured bucket, re-profile after every fix. It covers how to port 1:1 without replacing the existing path, how to declare and gate kernel capabilities, and how to verify with ULP-relative bounds. It is not for single-request prefill-only tuning, single-kernel micro-benchmarks, or flag/scheduler-parameter sweeps. Before kernel work on a new model, run model-structure-optimization first (work the model never needed is the bigger lever); for bringing a model up from scratch use model-bring-up."
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

**A collective's kernel time is mostly waiting until you prove otherwise.** On
DeepSeek-V4.1 at H200 TP4 (AgentX, c=16) the trace said all-reduce was 52% of
GPU time. Per rank it was 10.3 / 9.9 / 6.0 / 5.0 s of a 20.4 s window while every
rank's compute was identical (MoE 4.6-4.7 s, rest ~4.7 s). Equal compute with
unequal collective time means the collective is spinning for the last rank to
arrive. The ranks that wait least are the stragglers; look at *their* GPU idle
gaps (here 5.6-6.6 s, almost all gaps < 1 ms between small eager kernels): the
step was host-launch-bound, not NVLink-bound. Compare per-rank compute, per-rank
collective time and per-rank idle before touching communication.

**Attribute launches to source lines, not kernel names.** Profile a few dozen
steps with `with_stack=True`, then map each `cudaLaunchKernel` /
`cudaLaunchKernelExC` / `cudaMemcpyAsync` runtime event to the innermost
`python_function` frame from your package that contains it (same tid, smallest
enclosing interval). On V4.1 this named one function (the prefill attention
loop in the backend's `forward_v41`) as the source of 27% of all launches, plus
the `torch.where` / `cat` / `copy_` / `zeros_like` it issued per request per
layer — a per-request Python loop inside a per-layer call is O(requests ×
layers) launches per step.

**Rank idle + collective wait is constant → the step is bounded by host time.**
After the per-request loop was batched, V4.1 prefill steps still ran ~145 ms
wall for ~45 ms of GPU work. Per step, one rank showed 10-150 ms idle and
another 5-120 ms of all-reduce, but the *sum* was ~100 ms on every rank: the
slowest rank's host decides, and which rank that is changes step to step. Then
find where the host goes:

- **The torch profiler cannot see it.** tokenspeed runs the model on a
  `tokenspeed::forward` thread whose Python ops the profiler does not record;
  you only see `cudaLaunch*` runtime calls with 100+ µs of nothing between them
  (119 of 137 ms per step on V4.1). Use py-spy on the live server:
  `docker run --cap-add SYS_PTRACE …`, then
  `docker exec --privileged <c> py-spy record -d 20 -r 200 -p <pid> --threads --idle -f raw`
  (py-spy lives in the image, not on the host). Never sample while the torch
  profiler is exporting — every sample lands in `stop_trace`.
- **Read the forward thread and the main thread together.** Forward thread
  busy 64% + main thread 73% in `pending.result()` means the launches are the
  bottleneck and the overlap schedule is already doing its job; the 36% forward
  idle is the main thread's stream sync, cache polling and gloo collectives.
- **Group samples by the deepest model frame.** On V4.1 41% of forward-thread
  time sat under one `replay_fn` (the eager break of a breakable prefill graph),
  and its children were the q/kv projections, norms, RoPE, o-proj and
  all-reduce — none of which needed to be eager (strategy P).

**Decompose end-to-end per request before choosing.** e2e = TTFT +
(out − 1) × per-token decode. At c16 V4.1 was 2.81 s = 0.43 + 127 × 18.6 ms
while pure decode at the same batch was 7.9 ms/token: the gap was prefill steps
interleaved with decode, not decode kernels. Probe single-request TTFT at sizes
that straddle every graph bucket boundary (128…8192 new tokens, cold and with a
cached prefix) — a flat floor (130 ms for 128 to 1024 tokens) is per-step host
cost, a jump at a size is a capture bucket ending.

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

**O. No compilation on the serving path.** Read the engine's own compile
log first: tokenspeed prints `Triton JIT compiled <kernel> while serving in N ms
(<which key changed>)`. Sum it per kernel over a measured cell. On V4.1 one
kernel (`_finish_topk_kernel`, `BLOCK = next_power_of_2(width)` plus stride
divisibility) recompiled 16 times for 23 s of stalls inside an 87 s cell; fixing
it cut that cell's duration 19% and TTFT p99 from 8.8 s to 3.3 s.

A Triton `constexpr` is a JIT cache
key: a constexpr derived from batch size, chunk length or page-table width
recompiles mid-serving and stalls 100+ ms. Constexpr only for values fixed at
startup; batch-varying values go to runtime args or fixed buckets. Gate with a
no-recompile sweep test and serve CI with the compile check set to error.

**P. An eager break holds only the ops that read per-request metadata.**
Breakable prefill graphs replay captured segments and re-run each break's
Python. A break on the whole attention module re-runs the token-local work too.
Split it: projections/norms/RoPE/indexer projections in the graph on the
bucket's padded rows → a narrow break with cache writes, selection, attention →
o-proj and all-reduce back in the graph. The split only matters at capture
(replay never runs the dispatcher), so the eager path stays byte-identical; a
mixed extend+decode batch replays the same graph, so check that the token-local
part is identical across those modes. V4.1 H200 TP4 (b647604c): AgentX
66.7/45.3/37.9 → 52.2/36.6/33.0 s at c8/16/32, short-prefill TTFT 130 → 70 ms,
GPU busy 77 → 91%. Padded rows must be safe for every graphed op (positions
zeroed, RoPE clamps negatives).

**Q. Cover the batched prefill size with a captured bucket.** Above the largest
prefill-graph bucket the whole forward runs eager (`graphs=0` in a per-step
count of `cudaGraphLaunch`). The default cap was min(2048, chunk); an agent
workload batches several 1-2k-token prompts per step, so many steps fell off the
cliff (2083 tokens: 216 ms; with a 4096 cap: 91 ms). `--prefill-graph-max-tokens
4096` took c32 from 33.0 to 27.5 s. 8192 OOMed during decoder capture on
141 GB H200 (the reserve estimate undercounted) — size it to memory, and log
the decision.

**R. Re-profile after every fix; the bottleneck moves.** After P and Q the same
trace said MoE 36% and all-reduce 21% with 10 ms idle per prefill step: the
host problem was gone and the next work is GPU kernels. Doing kernel work first
would have bought nothing, because the GPU was idle 40% of the step.

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
   for FP4 experts) ship opt-in, behind a flag, after this gate. Measured:
   V4.1 `--moe-mxfp4-fp8-activation` GSM8K 0.9287 vs 0.928 BF16; AgentX
   49.0/33.5/27.5 → 43.3/32.8/27.7 s, bs16 TPOT 7.33 → 6.64 ms — a real win at
   c8 and decode, noise at c16/c32 until repeated. Run the gate from the repo's
   CI config (`test/ci/eval/*gsm8k*.yaml`) so the numbers compare to CI.
9. **State what was not measured** in the write-up ("kernel microbenchmark
   only; end-to-end not measured"). An unmeasured claim is a follow-up, not a
   result.
