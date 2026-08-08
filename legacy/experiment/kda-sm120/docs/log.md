# Experiment Log — sm_120 serving performance

Hardware: 4x RTX PRO 6000 Blackwell Server Edition (sm_120), 97,887 MiB each,
PCIe only, no NVLink. P2P available only within the PIX pairs {0,1} and {2,3}.

Each entry is **objective -> path -> result -> verdict**. Rejected entries are
kept with their reasons, per the contract.

---

## E1 — Split mixed prefill+decode attention batches

**Objective.** In a mixed batch, `max_seqlen_q` is set by the longest prefill
row, so q_len=1 decode rows are tiled as if they were prefill. Issue two
FlashAttention calls instead of one so each half gets tiling for its own query
length.

**Path.** `unieinfra-vllm/vllm/v1/attention/backends/flash_attn.py`. The batch
ordering invariant from `split_decodes_and_prefills` (decode -> short_extend ->
long_extend -> prefill) makes the split a zero-copy slice. Guarded narrowly:
FA2 only (`scheduler_metadata is None`, since `aot_schedule` is FA3-only),
causal, no sinks, no mask/aux modifiers. Gated on decode-token count, not
concurrency, because the layer cannot see concurrency.

**Result.** Qwen3-8B, in1024/out256, ON vs OFF:

| C | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| effect | -0.24% | +0.41% | **+2.60%** | **+9.55%** | **+60.45%** |

At 30B scale (period-blocked, both arms settled):

| model | shape | effect |
|---|---|---|
| Qwen3-32B @C=64 | in1024/out256 | **+4.74%** (32 sigma) |
| Qwen3.6-35B-A3B @C=256 | in1024/out256 | **+6.49%** (3 sigma) |
| Qwen3-8B @C=64 | in3072/out128 | **+5.73%** |

Also damps the throughput limit cycle: swing 49.8% -> 3.9% at Qwen3-32B C=64.

**Verdict. PROMOTED.** Shipped on by default above 32 decode tokens
(`VLLM_SPLIT_MIXED_ATTN`, `VLLM_SPLIT_MIXED_ATTN_MIN_DECODE_TOKENS`). Threshold
placed at the first concurrency where the arms' ranges separate. Defaults
validated falsifiably: with no env vars set, C=16 landed on the old OFF number
and C=128 on the old ON number.

**Caveat found later.** On 35B-A3B attention is only 1.9% of GPU time, so the
+6.49% there cannot be attention time saved. The mechanism is more likely the
downstream effect of a more uniform per-step batch shape. Not yet explained.

---

## E2 — Does TP > 1 help 8B on this box?

**Objective.** Per-user token rate at C=128 was 5.4 tok/s. Ask whether tensor
parallelism recovers it.

**Path.** TP=2 on GPUs 1+3, in3072/out128, C=16/64/128, against TP=1.

**Result.** +21.3% / +11.4% / +12.7% total throughput, but **-39% to -44%
per-GPU**. Two independent TP=1 replicas beat TP=2 outright: 1314 tok/s serving
128 users at 10.27 tok/s each, versus 732 tok/s serving 64 at 11.44.

**Verdict. REJECTED for 8B.** TP's only real win here is 2.24x KV capacity,
which matters only when context will not fit on one card.

---

## E3 — Was the TP result confounded by missing P2P?

**Objective.** E2 ran on GPUs 1+3, the only free pair, which is NODE (across
the host bridge). Its log showed every fast all-reduce backend falling away to
PYNCCL. Determine whether the result measured TP or measured the absence of P2P.

**Path.** `nvidia-smi topo -p2p r` showed P2P is OK only within PIX pairs. GPU 2
became free, so 2+3 (PIX) was available. Re-ran with an assertion that each arm
abort unless the log shows `CUSTOM` in the all-reduce dispatch list.

**Result.** CUSTOM engaged. Gains over the NODE pairing: **+2.62% / +0.97% /
+0.63%** at C=16/64/128, all inside noise.

**Verdict. HYPOTHESIS REJECTED.** The E2 numbers were real. Mechanism: custom
all-reduce caps at 8 MiB, and this shape's prefill all-reduce is 64 MiB, so
custom only ever handled the decode steps. The cap is a deliberate design
choice, not a knob.

---

## E4 — Async TP / sequence parallelism

**Objective.** Overlap the all-reduce with the GEMM instead of making it faster.

**Path.** Three arms with `--compilation-config` enabling `enable_sp` and
`fuse_gemm_comms`.

**Result.** All three arms produced identical numbers (739.4 / 738.1 / 738.0).
The engine's resolved config showed `enable_sp: False` — the CLI setting was
discarded. Cause: `SP_MIN_HIDDEN_SIZE` is `{90: 8192, 100: 8192}` with no entry
for 120, so `get_sequence_parallelism_threshold` returns None on the table miss.
The caller then logs "Model hidden_size too small", which misattributes it.

**Verdict. UNREACHABLE, and not worth reaching.** Even with a `120` entry the
threshold is hidden_size >= 8192; the largest target model is gemma-4-31B at
5376. Also `IS_DENSE` is hardcoded `False`, disabling SP for everyone.

**Byproduct.** Three arms of the same config, three independent server restarts:
739.4 / 738.1 / 738.0, spread 0.19%. That is the run-to-run reproducibility
figure, and it retroactively confirmed the E3 "inside noise" call.

---

## E5 — What is UnieInfra worth over baseline at 30B?

**Objective.** Every rigorous number to date was split ON/OFF *inside* the fork,
on Qwen3-8B. Measure the fork's own layers.

**Path.** Additive 4-rung ladder so each layer is separable:
L0 `UNIEINFRA_GEMM=0` + `UNIEINFRA_QUANT=0`; L1 + kernel selection layers;
L2 + `--quantization unieinfra`; L3 + the split.

**Result.** Qwen3-32B @C=32, all rungs settled: 402.5 / 402.4 / 404.2 / 404.5.
End to end **+0.50%**, inside noise. L0 and L1 track each other window for
window. Qwen3.6-35B-A3B @C=256: net -1.25%.

**Verdict.** The kernel-selection layers are **inert at 30B**, not harmful.
`unieinfra_gemm.py`'s own docstring says it was qualified on *batch-1 Qwen3-8B
decode* (+4.2% there) and that "a shape the registry declines falls back to" the
upstream path. The gap is unwritten work, not a regression.

**Scope correction.** All of this ran under `vllm serve`, and
`defaults_enabled()` gates the entire UnieInfra defaults layer on the console
script name — deliberately, to keep the `vllm` alias identical to upstream. So
prefix-caching tuning, the MTP ladder, fp8 KV and KV offload were **all inert**.
This verdict covers the kernel layers only. `UNIEINFRA_DEFAULTS=on` enables the
rest.

---

## E6 — Retracted: "the kernel layers cost 5.19%"

**Objective.** (as E5, first attempt, at C=64.)

**Result at the time.** L0 559.0 -> L1 530.0, **-5.19%, 13 sigma**.

**Verdict. RETRACTED.** The C=32 rerun gave -0.03%. The C=64 L1 arm had not
converged (period means 539 -> 525, still falling) while L0 and L2 had. A
non-converged midpoint distorts the segments on both sides in opposite
directions, which is exactly why the ladder appeared to drop then partially
recover.

**Lesson, now rule 1 of the contract.** High sigma proves two numbers differ,
not that they are comparable. `stationarity.py` was written in response and
gates every comparison since.

---

## E7 — Retracted: "gemma-4 has no working attention backend"

**Objective.** gemma-4-31B returned `" France is France is France is ..."`.
Find a backend that works.

**Path taken at the time.** Traced FLASH_ATTN's rejection (multimodal PrefixLM
mask; 512-dim global layers need FA4, unavailable here), found FLEX_ATTENTION
dies on a hardcoded `recompile_limit = 16`, found TokenSpeed's flashinfer path
asserting on `window_left`. Concluded all backends were broken by gemma-4's
three-way layer heterogeneity (head_dim 256/512, kv heads 4/16, window
1024/none).

**Result.** Plain HF transformers with SDPA returns **the same degenerate
text** for that prompt, and `" Paris."` for a long one. vLLM's TRITON_ATTN
matches HF on both, and answers `"Paris"` through the chat template.

**Verdict. RETRACTED — gemma-4 works on vLLM.** It is an instruction-tuned
checkpoint given a bare completion prompt. I judged correctness against my
expectation instead of against a reference.

**Survivors.** `VLLM_FLEX_RECOMPILE_LIMIT` added (default unchanged);
TokenSpeed's `gemma.py` CUDA-graph H2D copy fixed and verified; TokenSpeed's
`fa2.py` wrapper keying changed but **verified ineffective** — the cudagraph
wrapper lives in `mha.py`, keyed by batch size, and the per-group machinery that
would fix it (`_fa2_flat_decode_wrappers[(bs, gid)]`) exists but gemma-4 does
not reach it.

---

## E8 — Retracted: "speculative decoding loses everywhere"

**Objective.** Establish whether speculation helps, given the user's throughput
priority.

**Path taken at the time.** Measured ngram, a Qwen3-0.6B draft model, and the
checkpoint's native MTP heads across C=1..128, using `steady_bench`. All arms
settled, all period-blocked, all with large sigma. Reported -29% to -66%.
Built a whole causal story: acceptance rate cannot save it, the KV cost is
severe, UnieInfra's `_DEFAULT_SPEC_SCHEDULE` is wrong, the honest default is
k=0 everywhere.

**Result.** All of it was a token-counting bug in my own harness.
`steady_bench` counted **one token per SSE chunk**. Speculation emits every
token accepted in a step as one chunk. Recomputed from `server_tok_s`, same
runs:

| case | reported | correct |
|---|---|---|
| 35B-A3B MTP @C=32 | -42.4% | **+45.2%** |
| 35B-A3B ladder @C=32 | -42.7% | **+44.6%** |
| 35B-A3B static @C=1 | -49.4% | **+27.3%** |
| Qwen3-32B draft @C=32 | -58.1% | **+24.3%** |
| Qwen3-32B ngram @C=32 | -29.0% | **+6.8%** |
| 35B-A3B ladder @C=128 | -66.5% | -15.5% |

**Verdict. RETRACTED. Speculation is a large win.** UnieInfra's default ladder
is not broken; at C=32 it delivers +44.6%. Only C=128 is negative, and that
diagnosis holds: MTP sizes the KV cache down at load (430,665 -> 135,483) while
C=128 needs 163,840, so the schedule's k=0 rung cannot recover memory already
given up.

**The harness had already flagged it.** `client_captures_frac` read 0.34-0.69 on
every speculative arm against ~1.00 on every baseline. It was computed for
exactly this purpose, written into every result file, and never read.

**Fixed.** `steady_bench.py` counts `choices[0].token_ids`; `ladder.py` reads
`server_tok_s` and prints `[client counted N% of server tokens]` when the ratio
leaves 1.0 +/- 2%.

**Blast radius.** The bug can only distort experiments where tokens-emitted-
per-step differs between arms. E1's results were re-verified unchanged.

---

## E9 — Self-speculation by truncating layers

**Objective.** Every speculative method costs KV (ngram -14.8%, draft -45.1%,
MTP -69.7%). Self-speculation is the one variant that cannot, since the draft is
a prefix of the same network sharing the same cache. Ask whether the model's
early layers can draft.

**Path.** Logit lens: push each intermediate hidden state through the final norm
and lm_head, find the earliest layer after which the argmax stops changing.

**Result.**

| depth | Qwen3-8B (36L) | Qwen3-32B (64L) |
|---|---|---|
| ~67-75% | 5.3% settled | 0.1% |
| ~88% | 14.7% | 9.8% |
| ~94% | 58.4% | 31.2% |

Median settle depth on 8B is layer 34 of 36. **32B is worse than 8B.**

**Verdict. REJECTED.** A draft at 89% depth would be accepted ~35% of the time
while costing 89% of full compute. Logit lens understates a *trained* early
exit, so this rules out the untrained variant only — but that is the variant
available without a training project.

**Bug found and fixed mid-experiment.** HF returns the last hidden state
already post-norm while intermediates are pre-norm. Applying the final norm to
it twice made it disagree with the model's own logits (argmax match 0.545
instead of 1.000) and corrupted the settle-depth walk.

---

## E10 — Where does time actually go on 35B-A3B?

**Objective.** Choose a fusion target from evidence rather than from which
explanation sounds best.

**Path.** Torch profiler, offline, batch 1, 32 tokens, spec on and off.

**Result.** 241.27 ms of GPU time, base arm:

| block | ms | % |
|---|---|---|
| `_decode_gemv_kernel` (UnieInfra's) | 75.5 | 31.3 |
| **MoE path total** | **95.7** | **~40** |
| — `fused_moe::run_global` | 50.3 | 20.9 |
| — `cutlass MoeFCGemm` x2 | 26.9 | 11.1 |
| — `topkGating` | 6.0 | 2.5 |
| — finalize / fusedBuild / expandInput | 12.6 | 5.2 |
| linear attention | 5.1 | 2.1 |
| **attention** | **4.7** | **1.9** |

Reproducibility across independent runs: 241.270 vs 241.266 ms.

**Verdict. This is the target list.** MoE dominates; attention is 1.9% and not
worth optimising on this model. Note UnieInfra's decode GEMV *is* active at
batch 1 (31.3%) even though E5 found the layer inert at C=32 — consistent with
it being qualified for batch-1 decode.

**Open.** ncu on the MoE kernels, to separate bandwidth-bound from
occupancy-bound before choosing what to fuse. In progress.

---

## Cross-cutting method notes

These cost more time than any single experiment and are the reusable part.

1. **Period-blocked means.** Throughput settles into a period-2 limit cycle
   (prefill burst, decode burst) whose swing reached 50% of the mean at 90% KV
   utilisation. The same run reads `547.3 +/- 203` by window mean and
   `547.3 +/- 0.4` by period mean — same level, ~500x tighter, because the
   removed variance was structure, not noise.
2. **Stationarity before comparison.** See E6.
3. **Reference paths for correctness.** See E7.
4. **Read your own instrumentation.** See E8.
5. **Verify the switch engaged.** Three separate flags were silently ignored
   this session: `VLLM_ATTENTION_BACKEND` (removed from this vLLM),
   `--compilation-config pass_config` (overwritten by the optimization-level
   table), and `UNIEINFRA_GEMM` alone (a second entry point,
   `UNIEINFRA_QUANT`, stayed on). Grep the run's own log for evidence the
   layer is active.
6. **One vLLM experiment at a time.** The harnesses use container-wide
   `pkill -f "VLLM::"`. A concurrent run on another GPU in the same container
   gets killed mid-measurement and reports as a startup failure.
7. **`pkill -f PATTERN` from a shell whose argv contains PATTERN kills the
   shell.** Hit three times. Put the pattern in a script file.

---

## E11 — Re-qualify the decode GEMV on 30B shapes

**Objective.** `_decode_gemv_kernel` holds 31.3% of GPU time at batch 1 and is
UnieInfra's own. Its tuned table has three entries, all Qwen3-8B. Ask whether
the 30B targets are being served well.

**Path.** Added the 30B projection shapes to `bench_decode_gemv.py` (its shape
list was Qwen3-8B only too), ran `--mode compare` and `--mode sweep`.

**Result.** Three hypotheses died and one held.

- *"Untuned shapes will underperform."* **No.** At M=1 the heuristic reaches
  91-102% of measured DRAM peak (1.460 TB/s) on every 30B shape.
- *"Tuning recovers the 7-10 point gap vs the tuned 8B entries."* **No.** M<=2
  is already at the ceiling; a bandwidth-bound kernel at peak has no room.
- *"The M=8 collapse to 49-70% of peak is a finding."* **No.** Already known,
  documented in `_is_profitable`'s docstring with an E33 table my run
  reproduced to three decimals (qkv 1.16/1.04/0.98/0.62 vs my
  1.159/1.032/0.973/0.608), and already handled by the gate.
- *"`_is_profitable` is shape-blind."* **Yes.** It was `del N, K` then
  `return M <= 2` -- a rule induced from three Qwen3-8B projections and applied
  to every shape. Qwen3.6-35B-A3B's o_proj (N=2048, K=4096) beats cuBLAS at
  **every** M in 1..8 (1.16/1.86/1.81/1.40 swept, 1.15/1.95/1.77/1.10 warm --
  both methodologies agree, which is the standard that docstring itself sets).
  Its sibling qkv does not (1.005 at M=4, 0.837 at M=8), so it cannot even be a
  per-model rule.

**Change.** Six `_TUNED_CONFIGS` entries for this model, plus
`_PROFITABLE_TO_M8` so the gate is shape-aware.

**End-to-end, batch 8, 3 repeats, deployment config:**

| | run 1 | run 2 | run 3 | mean | sd |
|---|---|---|---|---|---|
| old | 788.7 | 806.0 | 796.8 | 797.2 | 7.1 |
| new | 807.6 | 806.4 | 798.4 | 804.1 | 4.1 |

**+0.87%, 1.48 sigma. Not measurable.** And the arithmetic says it should not
be: o_proj runs once per layer x 40 layers, saving ~2.4 us each under warm-cache
timing = ~95 us against a ~9.9 ms step. Predicted 1.0%, measured 0.87%,
run-to-run spread 2.2%.

**Verdict. KEPT, but not as a speedup.** It fixes a real logic defect, cannot
regress (it only widens shapes measured profitable; 8B and m35_qkv verdicts are
unchanged), and the kernel-level gain is real at 1.1-1.8x. The honest statement
is "kernel-level 1.1-1.8x, end-to-end below noise" -- not "+0.87%".

**The larger conclusion: this path is closed.** The kernel runs at 87-102% of
DRAM peak in its operating range. Decode weight projection is a streaming
problem at ~2 FLOP/byte, so the only lever left is moving fewer bytes (fp8 /
nvfp4 variants already exist in-tree) or amortizing them over more tokens
(higher batch, or speculative decoding, which measures +45%). Writing a faster
kernel cannot help.

---

## E12 — Where does PREFILL time go?

**Objective.** Every kernel measurement above is decode. For a RAG/agent
workload the prompt is long and the completion short, so prefill dominates wall
time, and the two are different problems: decode is GEMV and bandwidth-bound at
M = batch size; prefill is GEMM and compute-bound at M = prompt length.
`_decode_gemv_kernel` caps at MAX_DECODE_M=8 and never runs during a 3072-token
prefill at all.

**Path.** `scratchpad/prof_prefill.py`, 3070-token prompt + 8 tokens out,
Qwen3.6-35B-A3B, CUDA graphs on, `quantization="unieinfra"` -- the deployment
configuration, unlike the invalidated E10 ncu run. Prompt prefix varied per call
so prefix caching does not silently remove the thing being measured.

**Result.** 146.3 ms Self CUDA. Kernel-level (operator annotations such as
`vllm::moe_forward_share` excluded, since they wrap the kernels below them):

| block | ms | % |
|---|---|---|
| **MoE total** | **60.1** | **41** |
| — `fused_moe::run_global` x2 | 34.9 | 23.9 |
| — `MoeFCGemm` x2 | 18.4 | 12.6 |
| — finalize + expandInput | 6.8 | 4.6 |
| **dense GEMM** `cutlass_80_tensorop_bf16` x3 | **26.7** | **18** |
| `_decode_gemv_kernel` (the 8 output tokens) | 17.1 | 12 |
| linear attention (`chunk_fwd`, `gated_delta_rule`) | 7.0 | 5 |
| **attention** (`flash_fwd_splitkv`) | **3.6** | **2.4** |

**Attention is 2.4%, essentially the same as decode's 1.9%.** Predicted before
the run: 30 of this model's 40 layers are linear attention (O(n)), so a 3072-
token context produces no quadratic blow-up. The architecture designs the
long-context attention cost away, which caps any attention-side optimisation --
including this session's split -- at ~2.4% on this model.

**MoE is again the largest block at 41%**, and unlike E10 this measurement is in
the deployment configuration, so it stands.

**New lead.** The prefill GEMMs run on `cutlass_80_tensorop_bf16` -- an
Ampere-generation tensorop kernel on an sm_120 part, holding 18% of the time.
Not necessarily wrong (cutlass reuses kernels across architectures when they
remain optimal) but untested, and Blackwell has tensor-core capabilities an
sm_80 kernel cannot reach. Testable the same way E11 was: benchmark the actual
prefill shapes (M=3070) against alternative backends.

## E13 — Retracted: "prefix caching changes the output, do not enable"

**Objective.** E12's neighbour finding was that `--enable-prefix-caching` is
silently off on Qwen3.6-35B-A3B and worth +31.5% throughput on a RAG shape. That
is only recommendable if cache hits do not change what the model says — 30 of
its 40 layers are linear attention, whose recurrent state cannot be sliced at an
arbitrary prefix boundary the way a KV cache can.

**Path.** Greedy decode, fixed seed, same prompts with caching off then on,
compared token for token. First version: 3 prompts differing in a short suffix,
32 tokens. All three produced identical text *with caching off*, so the test
could not discriminate and proved nothing. Second version: substantively
different questions, 128 tokens, interleaved forward/reverse so the second pass
is served from a cache seeded by a different request. That version reported
`distinct answers 3/3`, `fwd/rev self-consistency 3/3`, and **2/6 generations
identical between on and off** — read as evidence the cache corrupts output.

**Result — the verdict was wrong, and the control is why.** Running the
identical configuration twice, caching off *both* times, gives **4/6 identical**.
This model is not bit-deterministic across processes: batch composition changes
floating-point reduction order, and divergence appears in the late tokens (the
first ~70 characters match in every pair). 2/6 sits inside that noise floor.

**Verdict.** Retracted. It neither convicts nor clears the cache. A comparison
without a same-config control measures the harness, not the feature — the same
error class as E6, where high sigma was mistaken for comparability. Bit-identity
is simply the wrong instrument for a model that is not bit-deterministic.

## E14 — Prefix caching, measured with an instrument that works

**Objective.** Settle E13's question using a metric immune to reduction-order
noise.

**Path.** Ten probes answerable only from a shared 2100-token document, scored
against checkable answers rather than against reference text — floating point
can reorder a sentence, it cannot turn 1974 into 1975. Three arms: `off1`,
`off2`, `on`. off1-vs-off2 is the noise floor, carried in the same run so the
`on` arm is judged against measured baseline disagreement rather than against
an assumption of determinism.

**Result.**

| arm | scored | cached tokens | text agreement vs off1 |
|---|---|---|---|
| off1 | 8/10 | 0 | — |
| off2 | 8/10 | 0 | **10/10** (noise floor) |
| on | 9/10 | 1900 | **9/10** |

**The noise floor here is zero**, unlike E13's 4/6. The variable is generation
length: E13 generated 128 tokens, this generates 48. Cross-process
nondeterminism accumulates with length, so short generations discriminate and
long ones drown the signal — the opposite of the intuition that drove E13's
design, and the reason its verdict was unreadable.

**The 9-vs-8 is a scoring artifact, not an improvement.** Reading the text
rather than trusting the score: the single divergent probe is `serial`, where
off1 emits `'Q: What is its serial prefix?\nA:\n\n\nQ: What is its serial
prefix?...'` (repeats the question) and `on` emits `'FIELD MANUAL, SECTION 7.
The Kestrel relay operates at...'` (recites the document). Neither answers. `on`
scored OK only because the recited document contains the substring `QX-88`.
Substring matching cannot distinguish answering from reciting the source.
Corrected, **both arms are 8/10**.

**Verdict.** Prefix caching **does** change output — 1/10 divergence against a
measured zero noise floor is signal, not noise — **with no measured accuracy
cost**. That is the expected signature of `mamba_cache_mode="align"`: reuse
boundaries are restricted, the recurrent state is not reconstructed
bit-identically, but the reconstruction is not wrong either.

**Recommendation.** A genuine trade-off, not a clean pass: +31.5% throughput and
-24% p50 for output that differs on roughly 10% of prompts at no measured
quality cost. Worth taking for RAG and agent workloads; not for anything
requiring reproducible output. Ten probes is a small sample and the claim is
scoped to that — "the divergence measured here carries no quality cost", not
"caching is safe".

## E15 — fp8 KV cache is available on sm_120

**Objective.** `probe_fp8_kv_capability` declines fp8 KV because FA3/FA4 do not
exist on sm_120. Is that a hardware limit or a backend-selection default?

**Path.** Query each backend's own gate, then measure. Correctness is scored on
checkable answers, not token identity: fp8 KV is a lossy quantization, so
identical tokens are not what success looks like and demanding them would fail a
feature working as designed.

**Result.** The limit is FlashAttention's alone:

| backend | `supports_kv_cache_dtype("fp8")` |
|---|---|
| FLASH_ATTN | False |
| TRITON_ATTN | **True** |
| FLASHINFER | **True** |

Qwen3-32B, `--kv-cache-dtype fp8` with the backend pinned:

| arm | KV capacity | accuracy |
|---|---|---|
| bf16 | 90,688 | 6/6 |
| fp8 + TRITON_ATTN | **179,328** (1.98x) | **6/6** |
| fp8 + FLASHINFER | **181,376** (2.00x) | **6/6** |

`kv_cache_dtype=fp8` is confirmed in the engine config, so the flag engaged
rather than being silently dropped. The only text difference is phrasing
("Mercury is the planet closest to the Sun" vs "the closest planet to the Sun").

**Why it matters beyond capacity.** E8 found speculative decoding worth
+27~45% everywhere except C=128, where it turned **-15.5%** because MTP cut the
KV pool from 430,665 to 135,483 while C=128 needs 163,840. **179,328 > 163,840**:
fp8 KV removes the one condition under which this session's largest win reverses.

**Native or dequantizing?** Capacity and correctness are identical whether a
backend reads fp8 natively or dequantizes it to bf16 first. `probe_fp8_kv_
capability`'s docstring predicts the latter for FlashInfer when the query is
bf16 — that path pays quantization error and dequant traffic while delivering no
bandwidth saving, which is worse than not quantizing. Only throughput separates
them, so throughput was measured: decode-heavy 512 in / 1024 out, C=32,
period-blocked over whole limit-cycle periods with `gate_stationary` applied to
the raw windows.

| arm | tok/s | vs bf16 | stationary |
|---|---|---|---|
| bf16 | 561.0 ± 4.4 | — | yes |
| fp8 + TRITON_ATTN | **590.6 ± 8.5** | **+5.28%** (3.1σ) | yes |
| fp8 + FLASHINFER | **588.7 ± 8.4** | **+4.94%** (2.9σ) | yes |

**Both are faster, so neither dequantizes.** The hazard the probe named did not
materialise on this hardware. Note the arms reported `converged: False` with a
16% tail spread — that is the period-2 limit cycle, not drift: period-blocked,
the same data is stationary with a ±0.8% standard error. A fixed-tolerance
convergence test cannot pass on a lockstep oscillation, and reading its verdict
as "the run failed" would have discarded a valid measurement.

**Verdict.** fp8 KV on sm_120 is a clean win: **2x capacity, +5% throughput,
6/6 accuracy**, and it removes the sole condition under which E8's speculative
decoding reverses. The probe was never wrong — it deferred pending a per-model
measurement, and this is that measurement.

**Scope.** Qwen3-32B, decode-heavy, C=32. The probe defers *per model*; the
hybrid 35B-A3B has only 10 of 40 layers holding a KV cache and has not been
tested here.

## E16 — Confirming the prefill profile, and what is actually running

**Objective.** E12 put MoE at 41% of prefill. Before hand-writing a kernel on
that basis, reproduce it independently and identify the specific kernels.

**Path.** vLLM's built-in torch profiler rather than ncu, which needs neither of
the concessions that invalidated E10: CUDA graphs stayed enabled
(`enforce_eager=False`) and `quantization=unieinfra` stayed in, both verified
from the engine log. 6 prefill-heavy requests (3072 in / 8 out) after warm-up.

**Result.** 259.70 ms of GPU kernel time over 74 kernels:

| family | % |
|---|---|
| **MoE** | **43.9** |
| dense GEMM | 31.6 |
| decode GEMV | 7.5 |
| linear attn | 6.6 |
| unclassified | 6.1 |
| **attention** | **0.2** |

E12's 41% reproduces at 43.9%. Attention is 0.2%, below even E12's 2.4%.

**The finding.** The three largest kernels are all Ampere-generation:

| % | kernel |
|---|---|
| 21.1 | `fused_moe::Fused_Moe_Kernel_sm80<bfloat16_t, ...>` |
| 21.0 | `cutlass_80_tensorop_bf16_s16816gemm_relu_256x128_32x3` |
| 11.0 | `MoeFCGemm<MmaMultistage<...>>` |

**53.1% of prefill runs on sm_80 kernels** (`s16816` is `mma.sync.m16n8k16`).
Meanwhile ptxas confirms sm_120 accepts TMA, `mbarrier`, `setmaxnreg`,
`elect.sync`, `stmatrix`, fp8 MMA and block-scaled MMA; only `wgmma` and
`tcgen05.mma` are rejected. The kernels holding half of prefill use none of the
available machinery.

**Cause.** Not the closed-allowlist pattern of
[[sm120-closed-allowlist-tables]]. The vendored TensorRT-LLM
`cutlass_extensions` contains **only** `Fused_Moe_Kernel_sm80`, with the MMA atom
hard-coded to `SM80_16x8x16_F32BF16BF16F32_TN`. There is no post-Ampere
implementation to dispatch to. Note it does adapt its M tile (`ROUTINE_PATH`
branches at kMaxTileM >=128 / 64 / 32 / 16), so small-M tile quantization is
already handled — that is not the gap.

**The gap is bandwidth.** 35B-A3B is hidden=2048, moe_intermediate=512,
256 experts, top_k=8. At M=3072 that is 24,576 token-expert pairs over 256
experts, **~96 rows per expert**, so per-expert gate_up is
`[96 x 2048] @ [2048 x 1024]`: 402 MFLOP against 4.2 MB of weights,
**AI ~96 FLOP/byte** against a ridge point near 139 (≈250 TFLOP/s bf16 over
≈1.8 TB/s). MoE prefill is bandwidth-bound on weight reads.

**Consequence.** nvfp4 weights take AI to ~384 FLOP/byte — past the ridge, with
a quarter of the weight traffic — and `MmaSM120BlockScaledOp` (mxfp4 / nvfp4 /
mxfp8) exists in CUTLASS 4.5.2's CuteDSL for exactly this. A
`/workspace/models/nvidia/Qwen3-8B-NVFP4` checkpoint is already on the box to
validate the data path. This, not attention, is where a hand-written sm_120
kernel pays: attention is 0.2%.


## E17 — The SM12x NVFP4 kernels are present but not auto-selected

**Objective.** E16 pointed at MoE (43.9%, bandwidth-bound at ~96 FLOP/byte) and
concluded a purpose-built sm_120 kernel was the lever. Before writing one, check
whether one already exists.

**Result.** It does, and it is named after this exact GPU. `vllm/config/kernel.py`
documents `moe_backend="flashinfer_b12x"` as *"FlashInfer CuteDSL fused MoE for
SM12x (RTX Pro 6000 / DGX Spark)"* and `linear_backend="flashinfer_b12x"` as
*"FlashInfer b12x CuteDSL NVFP4 GEMM (SM120+)"*. Both are excluded from
auto-selection by an explicit comment in
`fused_moe/oracle/nvfp4.py`: *"FLASHINFER_B12X is intentionally excluded from
auto-selection until the upstream CUTLASS SM121 MMA op guard is resolved"*.

Availability on this box: `has_flashinfer_b12x_moe()` **True**,
`flashinfer.fused_moe.b12x_fused_moe` and `B12xMoEWrapper` present,
`cutedsl_grouped_gemm_nt_masked` **True**, capability (12,0),
`is_device_capability_family(120)` **True**.

**Correction.** Those checks were read as establishing both the MoE *and* the
GEMM path. They do not — every one of them is about the MoE kernel. Tested
directly on Qwen3-8B-NVFP4, the linear path refuses outright:
`ValueError: --linear-backend=flashinfer_b12x was requested but no
'flashinfer_b12x' kernel exists for NVFP4 layers`. Only the MoE kernel exists,
and exercising it needs an NVFP4 **MoE** checkpoint, which the dense 8B is not.

**Consequence.** `nvidia/Qwen3.6-35B-A3B-NVFP4` (23.5 GB) fetched — the same
model under optimisation, not a stand-in. Its `config_groups`: group_0 is
8-bit W8A8 over 90 linear_attn and 40 self_attn modules; **group_1 is 4-bit
float, group_size 16, W4A4, over 40 expert and 120 shared_expert modules** —
precisely the block E16 measured at 43.9%. `mtp*` is in `ignore`, so the
unquantized MTP head keeps E8's +27~45% speculative decoding available.

**Next.** Three arms, not two: bf16 deployment / nvfp4 auto-backend / nvfp4 with
`moe_backend=flashinfer_b12x`. RedHatAI ships this checkpoint with
`--moe_backend flashinfer_cutlass`, so NVFP4 MoE already has an auto-selected
path; running only the b12x arm would leave any gain unattributable between the
quantization and the kernel. Known confound, stated: the bf16 arm carries
`--quantization unieinfra` and the NVFP4 arms cannot, so the comparison also
removes the UnieInfra kernel layer — bounded by E5's +0.50% at 30B scale.

## E18 — NVFP4 on the 43.9%: +36.4%, from the fallback backend

**Objective.** E16 measured MoE at 43.9% of prefill, bandwidth-bound on weight
reads (per-expert `[96 x 2048] @ [2048 x 1024]`, ~96 FLOP/byte against a ridge
near 139). NVFP4 quarters the weight traffic. Is the predicted gain real?

**Path.** Three arms on one GPU, sequentially, prefill-heavy 3072 in / 128 out,
C=16. `nvidia/Qwen3.6-35B-A3B-NVFP4` — the same model, not a stand-in. Three
arms rather than two because RedHatAI ships this checkpoint with
`--moe_backend flashinfer_cutlass`, so NVFP4 MoE already has an auto-selected
path; running only `b12x` would leave any gain unattributable between the
quantization and the kernel.

**Result.**

| arm | tok/s | vs bf16 | accuracy | KV capacity |
|---|---|---|---|---|
| bf16 (deployment) | 586.5 | — | 6/6 | 430,665 |
| **nvfp4, auto backend** | **800.2** | **+36.4%** | 6/6 | **1,732,608** |
| nvfp4 + `moe_backend=flashinfer_b12x` | — | refused | — | — |

**+36.4%, and the auto-selected backend was MARLIN** — sixth of eight in the
preference list, with every FlashInfer backend skipped. Marlin is W4A16: it
reads 4-bit weights from HBM but dequantizes in-register and computes in bf16,
so **the 4-bit tensor cores were never touched**. The whole gain is weight
bandwidth, exactly the term the arithmetic-intensity analysis predicted.
Accuracy is unchanged at 6/6, so nothing was traded for it.

**Why b12x refused — and it is not the reason the source comment gives.**
`fused_moe/oracle/nvfp4.py` excludes FLASHINFER_B12X from auto-selection "until
the upstream CUTLASS SM121 MMA op guard is resolved". The actual failure is
different and much more tractable:

```
ValueError: b12x fused MoE requires CUDA 13 or later. Current CUDA version: 12.9.
```

A toolchain version, not a capability limit and not the documented guard. This
gives the pending 0.26.1rc0 bump a concrete payoff beyond keeping current: a
CUDA 13 build unlocks the SM12x CuteDSL kernel that would put the *compute* on
4-bit tensor cores on top of the +36.4% already banked.

**Secondary, and larger than expected.** The model drops 67 GB -> 22 GB, and the
45 GB freed becomes KV: **430,665 -> 1,732,608 tokens, 4x**. Unmeasured here —
at C=16 the workload needs ~51,200 and neither arm is constrained. Same blind
spot as E19's first pass, seen before the run this time rather than after.

**Confound, restated.** The bf16 arm carries `--quantization unieinfra`, which
the NVFP4 arms cannot. E5 measured that layer at +0.50% end-to-end at 30B scale,
so it cannot account for +36.4%, but it is not zero.

## E19 — What 2x KV capacity actually buys, and it is not throughput

**Objective.** E15 measured fp8 KV at +5.28% and attributed it to fp8. Correct
but incomplete: at C=32 the workload needs 32 x ~1024 = ~32,700 KV tokens and
bf16 already provides 92,032. `batch_running` held at 31.9/32 with zero
preemptions — the pool was never binding, so doubling it could not contribute.
The +5.28% is bandwidth alone.

**Path.** Sweep C = 32/64/128/256 for bf16 and fp8, one server per config,
`--max-num-seqs 512` pinned so the sequence limit never substitutes for the KV
limit. Preemption counts read from the server log, not inferred from a
throughput shortfall.

**Result (bf16 arm).**

| C | batch_running | est. KV in use | tok/s | preemptions |
|---|---|---|---|---|
| 32 | 31.9/32 FULL | 32,666 | 570.8 | 0 |
| 64 | 63.7/64 FULL | 65,229 | 880.3 | 0 |
| 128 | **90.8/128 (71%)** | **92,979** | 871.1 | 0 |

**The mechanism is admission throttling, not preemption.** At C=128 the
scheduler admits 90.8 concurrent requests, holding estimated KV in use at
92,979 against a 92,032 ceiling, and queues the rest. vLLM does not over-admit
and then evict.

**Two corrections this produced.**

1. *"SM is at 100%, so the GPU is saturated at C=32"* — wrong. Throughput rose
   54% from C=32 to C=64. SM utilisation counts resident warps, not saturation.
2. *The KV wall arithmetic* used peak sequence length (1536), predicting C=64
   would exceed capacity. In steady state with staggered arrivals the mean
   length is ~1024, so the wall is at 92,032/1024 ≈ 90 concurrent —
   `batch_running` landed at 90.8.

**The ceiling, confirmed on both arms.** Admitted concurrency is set by
capacity divided by mean sequence length (~1024 in steady state, not the 1536
peak), and both arms land within 1% of that:

| arm | KV capacity | predicted ceiling | measured `batch_running` |
|---|---|---|---|
| bf16 | 92,032 | ~90 | 90.8 (C=128), 92.7 (C=256) |
| fp8 | 184,064 | ~180 | 127.4 (C=128), **181.2** (C=256) |

At C=256 fp8 holds 185,549 KV tokens in use against a 184,064 ceiling. **2x
capacity converts to 2x admitted concurrency: 91 -> 181.** This is structural,
read from the scheduler, and independent of throughput noise.

**Throughput above the wall is still unmeasured.** Every 300s point above it
failed `gate_stationary`, and the three failures had *different* modes — which
is the argument for testing drift, swing and envelope separately instead of
using one dispersion number:

| arm | tail spread | drift | verdict |
|---|---|---|---|
| bf16 C=128 | 28.7% | **-11.5%** | FAIL — small swing, monotonic decline |
| fp8 C=128 | 57.2% | -0.4% | **PASS** — large swing, no trend |
| fp8 C=256 | 153% | -4.0% | FAIL — envelope +144%, one window at 126 tok/s |

Read by tail spread alone the conclusion inverts: the widest-spread arm is the
usable one, the narrowest is not. fp8 C=128 measured 1479.7 against bf16's
941.9 (+57%), but bf16 had not converged, so the delta is not quotable — E6's
situation exactly. Reruns at 900s per point are in flight.

**Below the wall the advantage is bandwidth, and it grows with batch size:**
+4.67% at C=32, +10.77% at C=64, both arms FULL with zero preemptions and
neither capacity-limited. A larger batch reads more KV per decode step, so
halving KV reads matters more. An earlier version of the verdict script split at
C=64 and read that growth as evidence for the capacity story; the split belongs
at the measured wall (~91), which puts C=32 and C=64 both below it.

## E20 — NVFP4 lifts the concurrency ceiling 181 -> 673, and the "idle KV" was my arithmetic

**Objective.** E18 measured NVFP4 at +36.4% on a prefill-heavy shape but could
not exercise its 4x KV capacity, since C=16 needed ~51,200 tokens and neither arm
was constrained. Sweep a decode-heavy shape to C=256 and find the ceiling.

**Result.** 900s per point, 512 in / 1024 out, `--max-num-seqs 512`:

| arm | C | batch_running | tok/s | p5 | CV | stationary |
|---|---|---|---|---|---|---|
| bf16 | 64 | 63.8/64 FULL | 2224.5 ± 19.1 | 2104 | 4.4% | yes |
| nvfp4 | 64 | 63.3/64 FULL | **3499.2 ± 44.3** | 3112 | 7.6% | yes |
| bf16 | 256 | **181.4/256 (71%)** | 3330.9 ± 55.7 | 2618 | 11.5% | yes |
| nvfp4 | 256 | **256.0/256 FULL** | **5858.4** | 4685 | 13.2% | yes |

**+57.3% at C=64 (26.4σ) and +75.9% at C=256.** At C=64 both arms are FULL and
neither is capacity-bound, so that gain is compute and bandwidth alone — larger
than the +36.4% measured on the prefill-heavy shape, so NVFP4's value is
shape-dependent and was understated by E18.

**Correction: the "61% idle KV" reported earlier does not exist.** Estimating KV
in use as `batch_running x 1024` gave 39% occupancy for bf16 at its throttle
point, which looked like the scheduler refusing to admit into free space. That
estimate is wrong for this model. The engine log says why:

```
Setting attention block size to 1056 tokens to ensure that attention page size
  is >= mamba page size.
Padding mamba page size by 0.76% to ensure that mamba page size and attention
  page size are exactly equal.
```

Each sequence costs **one mamba page (1056) plus on average 1.5 attention blocks
(1584) = 2640 tokens**, not 1024. Then:

- bf16: 478,646 / 2640 = **181.3** — measured **181.4**
- nvfp4: 1,775,908 / 2640 = **672.7**, far above 256, hence FULL

bf16 was exactly at its ceiling with nothing spare. The apparent 39% is just
1024/2640. **NVFP4 raises the ceiling 181 -> ~673** because it frees 45 GB of
weights into the same pool.

**Consequence for the "admit more when not full" question.** There is no idle
capacity to exploit on either model — the scheduler admits precisely to the
limit. The two real levers are (a) enlarge the pool, which NVFP4 does, and
(b) stop *before* the limit, which the Qwen3-32B data suggests but this model
does not: 35B-A3B at its ceiling is stationary with CV 11.5%, whereas Qwen3-32B
at 99% occupancy is non-stationary with CV 32% and a halved p5. Fine-grained
16-token blocks appear to thrash near the limit in a way 1056-token blocks do
not, so any occupancy cap must be per-model, not a global constant.

## E21 — TokenSpeed does not build for sm_120, and the reason is TMEM

**Objective.** The vLLM-side work has a ceiling set by vLLM's architecture. Test
whether TokenSpeed — a newer engine whose kernel layer is a capability-gated
registry rather than a fixed dispatch — can serve Qwen3.x and gemma-4 on this
hardware at all. Nothing downstream matters until it compiles and runs.

**Path.** Built the upstream image, `tokenspeed @ 3f88dcc2`, from
`docker/Dockerfile.nvidia` (base `lightseekorg/tokenspeed-runner:cu130-torch-
2.11.0-flashinfer-0.6.15.post1`) with `CUDA_ARCH_LIST="12.0a"`. Upstream's
default is `9.0a 10.0a` and `tokenspeed-kernel/python/setup.py:52` defaults to
`("100a", "103a")`, so sm_120 is a combination upstream has never compiled.

**Result.** One of 35 `.cu` files fails, and it fails on hardware, not flags:

```
ptxas .../attn_res_fwd_tma.ptx, line 551; error: Feature '.32x32b' not supported on .target 'sm_120a'
ptxas .../attn_res_fwd_tma.ptx, line 551; error: Instruction 'tcgen05.st' not supported on .target 'sm_120a'
```

`tcgen05` is the 5th-gen tensor core tensor-memory (TMEM) instruction family.
Datacenter Blackwell (sm_100/sm_103) has TMEM; sm_120 does not. There is no
compiler flag for this and no workaround short of rewriting the kernel against
the ordinary MMA path. `attn_res` is Kimi-K3-specific, so it costs nothing for
the Qwen3.x and gemma-4 targets.

**Path taken.** Added `GROUP_REQUIRED_ARCHS` to `setup.py` and filtered kernel
groups against the target archs in `CudaKernelBuilder.run()`, printing each
dropped group rather than dropping it silently. The Python dispatcher in
`ops/attn_res/__init__.py` already falls back to a portable torch path when the
module is absent, so the wheel stays correct; it loses one kernel instead of
failing to build.

**Verdict.** Not a promotion — this is bring-up, and no performance claim
attaches to it. Recorded because it establishes the shape of the sm_120 gap:
the blockers are TMEM-era instructions in narrowly-scoped kernels, not the
engine's core paths. 1 of 35 sources is a far smaller hole than the 42 kernels
registered at `min_arch=(10,0)` suggested.

**Open, and the more dangerous half.** `platform.py:382` computes SM features
monotonically (`arch_version >= ArchVersion(10, 0)` grants `tensor_core:f4`),
so sm_120 satisfies every capability gate written for sm_100. Those 42 kernels
will therefore be *selected* on sm_120, not filtered out. Where the selected
kernel has no sm_120 cubin the failure is at runtime, which is the harder
failure to attribute. This is the first thing to probe once the image builds —
`explain_selection` over the registry on this GPU, before any serving run.

## E22 — `sm%` is stuck at 100 on two of the four GPUs, and the harness reads it

**Objective.** Pick a GPU for the TokenSpeed bring-up. GPU 0 holds another
tenant's vLLM (83 GB) and GPU 1 was running a `tokenspeed::scheduler_0` (92 GB)
belonging to another session, leaving 2 and 3 — both of which reported
`utilization.gpu = 100%`.

**Result.** They are idle, and the counter is wrong.

| | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| processes (`pmon`) | VLLM::EngineCore | tokenspeed::sch | none | none |
| `utilization.gpu` | **0%** | 98% | **100%** | **100%** |
| `utilization.memory` | 0% | 60% | 0% | 0% |
| power | 87 W | 518 W | 114 W | 121 W |

The two readings that matter are the diagonal: GPU 0 has a live 83 GB process
and reports 0%, GPUs 2 and 3 have no process at all and report 100%. `pmon`
shows no SM activity on 2 and 3, `utilization.memory` is 0, and power is near
GPU 0's idle draw rather than GPU 1's 518 W.

**Calibration.** Utilisation could not settle it, so measure work instead. bf16
8192³ GEMM and a 1 GiB device-to-device clone, in the sm_120 image:

| | GPU 2 | GPU 3 |
|---|---|---|
| bf16 GEMM peak | 426.8 TFLOP/s | 427.1 TFLOP/s |
| bf16 GEMM steady | ~405 | ~406 |
| clone bandwidth | 1452 GB/s | 1451 GB/s |

Two cards agreeing to 0.1% at a high absolute figure is not what contention
looks like. Both are free.

**Consequence for the harness.** `eval/steady_bench.py:251` samples
`nvidia-smi dmon -i $GPU_IDX -s u`, and `dmon` reproduces the fault: it prints
`sm 100` for GPUs 2 and 3 while nothing runs on them. Any run on those cards
that reports GPU utilisation is reporting a constant. Throughput and the `mem%`
column are unaffected.

**Verdict.** Use GPUs 2 and 3; do not quote `sm%` from them, and do not let a
future occupancy or saturation argument rest on it. The instrument, not the
card, is the thing that is broken — which is the same failure shape as the three
retractions in the README, caught this time before it produced a conclusion.

**Not done.** The stuck counter's cause is unknown; a GPU reset would likely
clear it but would disturb other tenants, so it has been left alone.

## E23 — TokenSpeed loses to stock vLLM 0.26 on sm_120, and the gap grows with load

**Objective.** The Phase 0 question: with TokenSpeed built and serving on
sm_120, is it faster than vLLM on Qwen3.5-27B? Everything before this was
bring-up.

**Setup.** Qwen3.5-27B bf16, one GPU per arm, GPU 2 (TokenSpeed) and GPU 3
(vLLM 0.26.0), calibrated equal at 427 TFLOP/s within 0.1% (E22). Both at
defaults: `gpu_memory_utilization 0.88`, `max_num_seqs 160`, prefix caching on.
1024 in / 256 out, unique prompt prefixes, 600 s cap per point.

**Result.** Server-side token counts, blocked at each arm's measured period:

| C | TokenSpeed | vLLM 0.26 | delta |
|---|---|---|---|
| 8 | 175.5 | 175.1 | ~0 |
| 32 | 459.3 ± 1.4 | 467.9 ± 2.3 | vLLM +1.9% (3.2σ) |
| 64 | 582.0 ± 0.8 | **641.1 ± 0.3** | **vLLM +10.2% (69σ)** |

Both arms read `settled` at C=32 and C=64 (drift −0.3% each). Neither is
capacity-bound at C=64: a 1280-token sequence needs 2 of TokenSpeed's 163
pages, so 64 sequences need 128 — close to the wall but not against it.

**Three things this number is not.**

1. *Not TokenSpeed's fast path.* flashinfer's XQA decode refuses page_size 832,
   which is what the GDN state row forces the attention block size to, so the
   run is on `--attention-backend triton` — the PORTABLE band at priority 4,
   the fallback tier. On sm_100 this model would run fa4/flashinfer at
   priority 12+. This measures TokenSpeed-on-fallback against vLLM-on-normal.
2. *Not a workload that exercises what TokenSpeed paid for.* Its pages are 74%
   recurrent-state snapshot by bytes, which is what buys prefix-cache reuse
   across GDN state — something vLLM cannot do on hybrids at all. The harness
   prefixes every prompt with `[w{id}r{n}]` specifically to defeat prefix
   caching. So the benchmark charges TokenSpeed the cost of that machinery and
   measures none of its benefit. A shared-prefix (RAG) shape is the missing
   half of this comparison and should be run before any verdict on the design.
3. *Not measured with the gate the contract asks for.* See below.

**The period-blocked mean was blocking the wrong period.** `ladder.py` pairs
consecutive windows, which assumes period 2. TokenSpeed's C=64 series is
period 3 with lag-3 autocorrelation +0.84:

```
658.8, 633.9, 470.3, 655.2, 633.9, 474.0, 651.7, ...
```

Blocking that into pairs aliases rather than cancels: SE came out 7.8 where
blocking at the true period gives 0.8, a factor of ten. vLLM's C=64 series is
period 5 (ac +0.72), so the two arms do not even share a period and one fixed
blocking cannot serve both. Added `eval/period.py`, which measures the period
before anything blocks by it. The contract's gate 2 says to use period-blocked
means; it did not say to establish the period first, and it should.

**Client undercount, caught by the counter that was nearly absent.**
`client_captures_frac` on the TokenSpeed arm ran 0.947 at C=32 against 0.9999
for vLLM: TokenSpeed packs several tokens into one SSE chunk, so a
chunk-counting client is low by the packing factor. Every TokenSpeed number
here is therefore taken from server-side counts. Reaching those counts needed
two fixes first — `server_counters()` matched only `vllm:` metric names, and
TokenSpeed's engine metrics turn out to be unreachable over HTTP in `serve`
mode (the SMG gateway exports only its own `smg_*`). Both failures return
"no counter", which reads downstream as a passing gate. The per-request
`usage` chunk, which both engines send, is now the fallback authority.

**Not measured.** `batch_running` is unavailable on the TokenSpeed arm for the
same metrics reason, so "both arms FULL" is asserted from the page arithmetic
rather than observed. `sm%` is excluded throughout (E22).

**Verdict.** On this shape, on this hardware, on the fallback kernel path,
TokenSpeed is 10% behind stock vLLM at C=64 and the gap widens with
concurrency. That does not settle the bet — points 1 and 2 above are both
large and both unmeasured — but it does mean no current claim of a TokenSpeed
throughput advantage on sm_120 survives contact with a gated measurement.

## E24 — The deficit is the attention path, and it is a wiring gap

**Objective.** E23 left the cause of vLLM's win open. Separate it: full
attention costs more per token as the prompt grows, GDN linear attention does
not. If the gap widens with prompt length, it is the attention path.

**Result.** Qwen3.5-27B, C=32, server-side counts, blocked at each arm's
measured period:

| ILEN | TokenSpeed | vLLM 0.26 | gap |
|---|---|---|---|
| 1024 | 459.3 ± 1.4 | 467.9 ± 2.3 | +1.9% |
| 4096 | 209.7 ± 8.1 | 248.9 ± 6.8 | **+18.7%** (3.7σ) |

Tenfold growth in the gap for a fourfold prompt. It is the attention path.

**What each engine actually runs.** From the logs, not from assumption:

| | vLLM 0.26 | TokenSpeed |
|---|---|---|
| 48 GDN layers | Triton/FLA | Triton/FLA |
| 16 full-attn layers | **FlashAttention-2** | Triton |
| block size | 784 | 832 |

The linear-attention majority is the same code on both sides. The entire
difference sits in 16 of 64 layers.

**Why TokenSpeed had no FlashAttention on this GPU.** Four separate dead ends,
none of them hardware:

1. `ops/attention/flash_attn/__init__.py` registers FA4 only, every entry
   `min_arch == max_arch == (10, 0)` — sm_100 exclusively.
2. The image's `flash_attn` distribution is an empty namespace directory:
   `__file__` is None, `dir()` is empty.
3. FlashInfer's trtllm XQA decode rejects page sizes outside
   {16, 32, 64, 128}; the GDN state row forces 832.
4. FlashInfer's AOT `BatchDecodeWithPagedKVCacheWrapper` reports
   `Unsupported group_size: 6` at head_dim 256 — 24 query heads over 4 KV
   heads. This one has nothing to do with page size, and the earlier
   attribution of the decode failure to page size was wrong.

**But the kernel exists.** FlashInfer's *general* paged prefill wrapper — the
family vLLM's FA2 backend is built on — runs at exactly this geometry.
Verified on the device: page_size 832, head_dim 256, GQA group 6, finite
output, correct shape. It was never registered.

**Landed.** `unieinfra-kernel-sm120`, an out-of-tree plugin, registers that
wrapper for `attention/mha_extend_with_kvcache` at `Priority.PLUGIN` (16),
above in-tree trtllm (12) and Triton (4). Against the Triton kernel it
displaces: max abs diff 6.1e-05, mean 3.5e-06, cosine 0.99999678. No fork, no
patch to tokenspeed-kernel.

**A fourth mechanism that was built and never connected.** `discover_plugins()`
is called nowhere in the TokenSpeed runtime. The registry reserves priority
band 16-19 for plugins and the entry-point group is defined, but the engine
never walks it, so the band is permanently empty and any platform whose best
kernel lives out of tree silently runs the in-tree fallback. Added the call to
`engine.py::_set_envs_and_config`, wrapped so a third-party plugin cannot stop
the engine from starting.

**Also observed.** `flashinfer_trtllm_mha_extend_with_kvcache` is admissible on
sm_120 and dies at runtime with `TllmGenFmhaRunner: Unsupported architecture`.
That is the E21 false-admission prediction landing for the third time, and it
makes shape/arch trait declaration a correctness requirement rather than a
performance one.

**Not yet measured.** Whether the plugin recovers the 18.7%. The registration
is verified; the throughput is not.

## E25 — The FA2 extend kernel bought nothing, and a better one already existed

**Objective.** E24 showed the vLLM gap widening with prompt length (+1.9% at
ILEN 1024, +18.7% at 4096) and traced it to the full-attention path: vLLM on
FlashAttention-2, TokenSpeed on Triton. Swap in an FA2-class extend kernel and
measure how much of the 18.7% comes back.

**Path.** `unieinfra-kernel-sm120`, an out-of-tree plugin registering FlashInfer's
general paged prefill wrapper for `attention/mha_extend_with_kvcache` at
`Priority.PLUGIN` (16). Numerics verified against the Triton kernel it displaces
(cosine 0.99999678, and 0.99999583-0.99999690 across padded-query shapes).
Forced on with `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_EXTEND_WITH_KVCACHE`,
since `--attention-backend triton` hard-filters candidates by solution string and
would otherwise have excluded the plugin silently.

**Result.** Qwen3.5-27B, C=32, ILEN 4096, OLEN 256, server-side counts, blocked
at each arm's measured period:

| arm | tok/s | SE |
|---|---|---|
| TokenSpeed, Triton extend | 209.7 | 8.1 |
| TokenSpeed, FlashInfer FA2 extend | **210.1** | 5.7 |
| vLLM 0.26 | 248.9 | 6.8 |

**+0.2%, against a combined SE of 9.9. That is zero.** The kernel swap recovered
none of the 18.7%.

The kernel was definitely running: before the padded-query fix the server
crashed *inside* it (`q.shape[0] (16) does not match qo_indptr[-1] (1)` —
TokenSpeed pads q up to a CUDA-graph bucket and the in-tree Triton kernel
absorbs that through its grid bounds), and after the fix it serves. So this is a
real null, not a kernel that was never called.

**Most likely cause, stated as a hypothesis and not yet measured.** The plugin
plans on every call. FlashInfer's wrappers are built to plan once per forward
pass and run per layer; planning inside each of the 16 full-attention layers
makes them all pay for scheduling one of them needed, and `plan()` copies index
tensors to the host. That overhead plausibly cancels the kernel's advantage. It
was recorded as a known cost before the measurement, which is the only reason
this reads as a hypothesis rather than a surprise.

**The finding that matters more.** A better implementation of this exact kernel
already existed in the tree and was missed, because it lives in the vendored
older engine at `experiment/legacy/`, not in the upstream clone that was
searched:

`experiment/legacy/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/flashinfer/fa2.py`
— 532 lines, +101 of them uncommitted local work, registering
`flashinfer_fa2_mha_extend_with_kvcache` and
`flashinfer_fa2_mha_decode_with_kvcache` under `solution="flashinfer"`, with
comments explicitly about sm_120 and a per-`(batch, attention_group)` wrapper
cache — which is precisely the plan-reuse this null result points at. It also
needs no env override, since it registers under a solution string the engine's
`--attention-backend flashinfer` already selects.

The same search miss produced a second duplication: Gemma 4 is already
implemented at `experiment/legacy/python/tokenspeed/runtime/models/gemma.py`
(608 lines, Gemma4ForConditionalGeneration text decoder against HF's
modeling_gemma4), while an agent was dispatched to write it from scratch.

**Verdict.** Reject the plugin's extend kernel as written — it is correct and
worthless. Do not re-derive the fix; qualify the existing `fa2.py` instead, and
search `experiment/legacy/` before concluding anything is absent from this
engine. The 18.7% is still unexplained: the attention-path attribution in E24
survives, but "swap the extend kernel" is now known not to be the lever.

---

## E27 — The FlashInfer decode kernel now survives CUDA graphs, and the plan hoist is load-bearing

**Objective.** E25 left the sm_120 FlashInfer paged decode kernel registered but
unusable: numerically correct (cosine 0.99999523) and fatal in the live server
with `torch.AcceleratorError: CUDA error: operation not permitted when stream is
capturing`, because FlashInfer's `plan()` does host-side scheduling and
TokenSpeed captures decode. Make it capturable, and find out whether the fix
belongs in the kernel or in the runtime.

**Path.** Surveyed all four vendored engines on the single question of how they
run paged-KV decode attention under a captured graph.

| engine | host "plan" object | where it runs | bucket granularity |
|---|---|---|---|
| vLLM FA2 | none | n/a | any |
| vLLM FA3 | `scheduler_metadata` + fixed `max_num_splits` | `build()`, before capture and before replay | one buffer, sliced |
| vLLM FlashInfer | FlashInfer `plan_info` | `fast_plan_decode` in `build()` | one wrapper per batch size |
| SGLang FlashInfer | FlashInfer `plan_info` | `init_forward_metadata_out_graph()` | one wrapper per batch size |
| LMDeploy | none (own Triton kernel) | n/a | pow2 then step-256 |
| TensorRT-LLM | none (XQA schedules on GPU) | n/a | per (batch, draft-len) |

Two engines need no plan because their kernels assign tiles on the GPU. The
three that do use a host pass converge on one protocol, so that is what was
built: persistent CSR buffers per bucket, wrapper constructed with
`use_cuda_graph=True` over them, `plan()` outside the captured region at capture
and again before every replay, only `run()` captured.

Read `include/flashinfer/attention/scheduler.cuh` to check *why* that is sound
rather than assuming it. Under `enable_cuda_graph`, `PrefillSplitQOKVIndptr`
derives `cta_tile_q` from `total_num_rows - batch_size + 1` and sets
`padded_batch_size = max(2*SM/num_kv_heads, total_num_tiles_q)` — neither
depends on KV length — so every workspace offset in the `plan_info` the graph
bakes is replan-invariant. Everything length-dependent (`kv_chunk_size`,
`request_indices`, `kv_tile_indices`, `merge_indptr`, `block_valid_mask`) goes
to the device int workspace, which a re-plan overwrites at the address the graph
already holds.

Two things were done differently from every source:

1. *No device-to-host sync building the index.* SGLang and this repo's own
   legacy `flashinfer/fa2.py` both compact page ids with a mask and use the
   result's length as a Python int. Here the page ids are scattered to their CSR
   destinations with out-of-range columns aimed at a trailing scratch slot, so
   the whole index is device-side.
2. *One vendor-neutral hook, not four vendor-specific ones.* The legacy tree
   exposes `fa2_decode_is_selected` / `make_fa2_decode_cuda_graph_wrapper` /
   `fa2_decode_cuda_graph_plan` / `fa2_decode_cuda_graph_run` and makes the MHA
   backend own FlashInfer wrappers and branch between two call paths. This adds
   one op, `mha_decode_prepare`, which dispatches to a `prepare` attribute on
   whichever kernel selection picked and returns `False` when there is none. The
   runtime's only kernel boundary stays `tokenspeed-kernel`.

**Result.** All at the live geometry (24 q heads / 4 kv heads / head_dim 256 /
page_size 832, bf16), against `solution="triton"` through the ops dispatch.

| check | result |
|---|---|
| eager, `max_seqlen_q` 1 and 4, batch 1/8/33 | cosine 0.99999428-0.99999893 |
| plan hoisted out, same shapes | cosine 0.99999428-0.99999851 |
| capture + 3 replays, batch 8 and 33, fresh page tables and lengths per replay | capture OK, cosine 0.99999452-0.99999851 |
| **replay without re-planning** (1→6, 6→2, 2→5 pages) | cosine **0.246 / 0.427 / 0.465** |
| padding rows (`seq_len` 0 and 1 mixed in) | finite, live rows cosine 0.99999440 |
| capture with the hook missing | `RuntimeError` naming `prepare_paged_mha_decode` |

Per-call latency vs Triton with the plan hoisted, GPU 3: 4.8x at batch 1, 5.4x
at batch 8, 1.13x at batch 32, 1.10x at batch 128, 1.53x at batch 32 with
`max_seqlen_q=4`.

**Verdict.** Accept. The negative control is the entry that matters: a stale
plan does not fault and does not produce NaNs — it produces plausible-looking
output at cosine 0.25-0.47. A capture/replay test that only checked "it runs and
the numbers look finite" would have passed on a broken build, which is the
failure mode this project has already retracted a result over.

**The graph problem cannot be solved inside the kernel alone,** and that is not
a limitation of this implementation. Every decode call site TokenSpeed executes
is inside the captured region; the kernel has no moment of its own outside it.
The three metadata hooks (`init_forward_metadata`,
`init_forward_metadata_capture_cuda_graph`,
`init_forward_metadata_replay_cuda_graph`) are the only decode-adjacent code
that runs outside, so the hoist has to be driven from there. The kernel can
detect the situation and refuse with a directed error, and it does.

**Not measured.** Serving-level throughput. The latency numbers include Python
dispatch and were taken on a GPU shared with other agents' work; they establish
that the kernel is not slower, not what a server gains. The E23 18.7% deficit
remains unattributed — E25 showed the extend kernel was not the lever, and this
entry does not yet show that the decode kernel is.

**Known gap.** DFLASH block decode is not served: its `seq_lens` are written by
the drafter *inside* the captured graph, so no schedule computed outside can
describe the step that runs. That bucket gets the refusal error rather than a
wrong answer.

---

## E28 — Gemma 4 is already implemented, and the KV pool silently corrupts it

**Objective.** Establish what it would take to serve `gemma-4-31B-it` and
`gemma-4-26B-A4B-it` on sm_120. The premise handed to me was that TokenSpeed
had no Gemma implementation at all and one had to be written from scratch
against four hard config fields (`attention_k_eq_v`, `global_head_dim` 512 vs
`head_dim` 256, `final_logit_softcapping`, `sliding_window` 1024).

**Path.** Three passes, in this order, because each one changed what the next
one was for.

1. *Verify the premise against the artifacts, not the description.* Read both
   `config.json` files and both safetensors indices directly.
2. *Survey every engine before writing anything* (skill §2). vLLM 0.26, SGLang
   and TensorRT-LLM all ship Gemma 4; all three were read end to end and
   tabulated on design choices, not prose.
3. *Enumerate what actually wins selection on the device* (skill §0), then
   exercise each admissible candidate rather than trusting `satisfied_by`.

Mid-task the premise was corrected: a 608-line implementation already exists at
`experiment/legacy/python/tokenspeed/runtime/models/gemma.py` (+28 lines
uncommitted). The task became a gap analysis of existing code. The from-scratch
files I had started were withdrawn from the upstream tree and kept as clearly
labelled proposals under `docs/gemma4-proposals/`.

**Result.**

*Five premise corrections, all from the artifacts.*

| stated | actually |
|---|---|
| no Gemma implementation exists | 608 lines exist, and are registered |
| "appears NOT to be registered — grep finds nothing in registry.py" | it **is** registered; `registry.py` holds no model names *at all*, it auto-discovers via `pkgutil` + each module's `EntryClass` (`gemma.py:608`). That grep returns nothing for every model in the package. |
| `attention_k_eq_v` — K and V are the same tensor, model-wide | full-attention layers only. 25 of 30 layers in the 26B have a real `v_proj`; the 5 that do not are exactly the full-attention positions. |
| omni model, has `audio_config` | `audio_config: null` in **both**. Vision only. |
| `global_head_dim: 512` is the full-layer difference | full layers also drop to `num_global_key_value_heads` = 4 (31B) / 2 (26B). At 4x512 = 2048 elems they are **cheaper** per token than sliding layers at 16x256 = 4096 — the opposite of the intuition the 512 invites. |

*The existing implementation is better than all three references on k_eq_v.*
vLLM `weight.clone()`s `k_proj` into a `v_proj` slot; SGLang calls
`weight_loader(param, w, "v")` a second time; TRT-LLM does
`weights.setdefault(v_key, value)`. All three then run the projection twice.
`gemma.py:271-283` builds no `v_proj` at all and reuses the K activation
(`v = k_raw`, before k-norm and RoPE — the ordering TRT-LLM comments as a trap).
Saves ~220 MB of duplicated weight and one GEMM per token across the 31B's 10
full layers. It is also correct on three things easy to get wrong: plain
RMSNorm (Gemma 4 dropped the `1+w` convention that `GemmaRMSNorm` in this repo
still implements), the per-layer `layer_scalar`, and proportional RoPE's
frequency denominator (`head_dim` 512, not `rotary_dim` 128).

*The blocker is in the KV pool, and it does not raise.* Built the real
`MHATokenToKVPool` at 31B text parameters (60 layers, 5:1, `head_num=16`,
`head_dim=256`, `page_size=64`) and exercised the store path:

| probe | result |
|---|---|
| `layer_kv_head_counts` | `None` — the hetero-KV mechanism is not engaged |
| `k_buffer[5]` (full layer) row view | `(4160, 16, 256)` = 4096 elems/token |
| what a full layer produces | `[T, 4, 512]` = 2048 elems/token |
| store, layer 0 (sliding 16x256) | OK |
| **store, layer 5 (full 4x512)** | **OK — no error raised** |
| read-back: leading 2048 elems | match source exactly |
| read-back: trailing 2048 elems | 0 of 16384 nonzero |

The K tensor lands in the leading half of a row sized for a sliding layer. Half
the full-layer cache is dead and any read at the layer's real 4x512 geometry
straddles two tokens. `kv_cache/mha.py` carries a single scalar `head_dim`
(`:95`, `:267`) and `_layer_row_view` (`:557`) can vary the head *count* per
layer but not the head *dim*. A grep for `gemma` across the whole legacy runtime
returns only `GemmaRMSNorm` used by *other* models — there is zero Gemma 4
plumbing outside the model file.

*Selection says the head-dim question has a worse answer than expected.*
Per skill §0, on sm_120:

| op | winner | band | admissible |
|---|---|---|---|
| `mha_prefill` | triton | **4 (PORTABLE)** | triton only |
| `mha_extend_with_kvcache` | flashinfer | 12 | triton, flashinfer |
| `mha_decode_with_kvcache` | flashinfer | 12 | triton, flashinfer |

Exercising each candidate at the two head dims Gemma 4 actually uses:

| entry point | solution | hd=128 | hd=256 | hd=512 |
|---|---|---|---|---|
| `mha_decode_with_kvcache` | default (= flashinfer wins) | — | PASS | **FAIL** |
| `mha_decode_with_kvcache` | `triton` | — | PASS | **PASS** |
| `mha_decode_with_kvcache` | `flashinfer` | — | PASS | **FAIL** |
| `mha_prefill` | `triton` | PASS | **FAIL** | **FAIL** |

Two independent failures. The decode failure is a closed allowlist —
`Invalid head_dim: 512, must be divisible by 16 and in range [16, 256]` — in the
kernel that *wins selection*, so the full-attention layers hit it by default;
the Triton fallback handles 512 fine but only if `solution="triton"` is forced.
The prefill failure is shared-memory over-subscription:

```
hd=128: PASS
hd=256: OutOfResources: required 114688, limit 101376
hd=512: OutOfResources: required 102400, limit 101376
```

**The binding constraint is head_dim 256, not 512.** Gemma 4's 50 *sliding*
layers are hd=256, so prefill is broken regardless of anything to do with the
512-wide layers, and this is model-independent: any hd=256 model hits it on
sm_120. hd=512 misses by 1024 bytes. Exactly the skill §3 failure — tiles sized
for 227 KB meeting a 100 KB SM.

Meanwhile the *library* is not the problem. FlashInfer 0.6.15 called directly
runs head_dim 512 on sm_120 and is numerically correct: paged prefill, paged
decode and ragged prefill all PASS, verified against an fp32 reference at
rel err 0.0044 / 0.0000 / 0.0038, plus the windowed hd=256 sliding path at
0.0043. This contradicts TensorRT-LLM's own test suite, which patches configs
down to 128/256 citing a "FlashInfer head_dim=512 limitation on B200"
(`test_gemma4_e2e_dummy.py:17-19`). Stale, or B200-specific.

**Verdict.** Accept the gap analysis; reject the original task. Writing a new
model file would have been wasted work — the existing one is equal or better on
every axis except MoE, and none of its gaps are in the model file.

The entry that matters is the silent one. A store that writes half a row and
returns success is worse than a crash, and it is invisible to every check short
of tensor-level comparison. It also explains how this model reached CUDA graph
capture (the uncommitted `new_tensor` fix in `gemma.py` is a capture-time crash
that "took gemma-4-31B down at startup") without anyone noticing the KV path was
wrong: capture precedes any correctness check.

The fix is small and the mechanism already exists. Normalising the two
geometries to a common head dim of 256 makes `4x512` into `8x256` — the same
byte count — so the Inkling hetero-KV path (#647) expresses Gemma 4 with
`layer_kv_head_counts = (16,16,16,16,16,8)*10`, `kv_alloc_head_count = 16`,
`group_page_sizes = {"full_attention": 2*page_size}`, and the full layers get 2x
tokens per slot, which is what you want since they are the unbounded ones. Only
`_layer_row_view` needs anything new: the ability to return `(-1, 4, 512)`
instead of `(-1, 8, 256)`. Same bytes, contiguous, pure reshape, ~6 additive
lines that no-op when the new tuple is `None`.

**Rejected candidate, kept per contract.** Storing one tensor per token for the
k_eq_v layers by aliasing V to K. Wrong: the two diverge after the norms —
`v_norm` has no learned weight, `k_norm` does, and only K is roped, so
`V = n` and `K = RoPE(n * w_k)`. No engine aliases them; all three store both.
The halving is still available, because reconstructing K from V needs no
reduction, only a broadcast multiply by a `[512]` vector and a rotation at the
token's own position — both already in every kernel's K-tile load path. That is
2x on the dominant cache term (80 -> 40 KiB/token at any context length, since
the sliding layers are capped at their 1024-token window). It is a kernel
project in a file owned by another agent, and it trades bandwidth for ALU on the
read path, so it should win on bandwidth-bound decode and may not on prefill.

**Not measured.** No end-to-end serving number, and no full-model run: GPU 3 is
shared and the 31B will not fit alongside other tenants, so every result here is
per-layer or small-shape. No throughput claim is made or implied. The prefill
kernel's shared-memory fix lives in `ops/attention/triton/mha_prefill.py` and
the head-dim allowlist in the kernel dispatch — both owned by other agents;
neither was touched. Whether the legacy tree's uncommitted
`ops/attention/flashinfer/fa2.py` (532 lines, registers FA2 extend+decode for
sm_120) carries the same 256 allowlist was not checked.

**Hand-offs.** (1) `mha_prefill` Triton tiles exceed sm_120 shared memory at
hd=256 and hd=512; hd=512 by 1024 bytes. (2) The `[16, 256]` head_dim check in
the winning flashinfer MHA decode kernel is a closed allowlist, not a capability
probe — the same class of bug as the sm_120 capability tables in E24.

## E28 — Fusing GDN's `chunk_h`+`chunk_o` is worth 1.14-1.54x, but only when the batch has more than one sequence

**Objective.** GDN linear attention is 48 of Qwen3.5-27B's 64 layers, so its
chunked prefill is the dominant prefill cost. On sm_120 it runs on
`triton_gdn_chunk_prefill` at `PORTABLE` (4) — a fallback band. vLLM logs the
same Triton/FLA kernel for the same model, so both engines are on identical
math. Question: does anyone implement this better, and can we beat both?

**Path.**

*Survey first* (skill §2, all seven vendored engines). The finding that decided
the work: **six of seven engines materialize the per-chunk state checkpoint
tensor `h` in HBM; lmdeploy's TurboMind does not.** Its sm_120 kernel does
`wy_fast` + `chunk_delta_h` + `chunk_o` in one persistent warp-specialized CTA.
At this model's geometry `h` is `[1, NT, 48, 128, 128]` bf16 — **201 MB for a
single 8192-token sequence** — written by `chunk_h` and read straight back by
`chunk_o`, and by nothing else. Per-stage timing confirmed those two kernels
are 58% of the pipeline (chunk_h 383.5 us, chunk_o 332.4 us out of 1230.7 us at
T=8192).

Second survey finding: **TokenSpeed is the only engine storing the recurrent
state K-first.** vLLM, SGLang and TRT-LLM all use `[N,H,V,K]`, matching the SSM
pool, so none of them transpose. TokenSpeed's wrapper calls
`initial_state.transpose(-2, -1).contiguous()` on every call of all 48 layers
because the FLA `.jit` kernel has no stride support for `h0`.

*Implementation.* `unieinfra-kernel-sm120/unieinfra_kernel_sm120/gdn.py`, a new
Triton kernel registered at `Priority.PLUGIN` (16). One program owns
`(sequence, v-head, V-tile)` and walks all chunks holding its `[K, BV]` slice of
the state in registers, so `h` is never allocated and `v_new` (another 100 MB)
disappears with it. `h0`/`ht` are addressed K-last through a block pointer with
strides `(1, K)`, so the transpose disappears too. The five upstream stages
(l2norm, cumsum, kkt, `solve_tril`, `wy_fast`) are imported from
`tokenspeed_kernel` and reused unchanged — `solve_tril` is a 64x64 forward
substitution and would stall the chunk loop.

*Correctness.* Against `triton_gdn_chunk_prefill` through the ops dispatch:
**max abs 0.0, mean abs 0.0, cosine 1.0000000000 on all 16 shapes** — bit
identical, not merely within bf16 tolerance, because the fused loop does the
same operations in the same order and rounds `v_new` at the same point. A
negative control (perturb `scale` by 0.1%) moves max abs to 4.9e-4, so the
comparison can fail.

**Result.** `gdn_chunk_prefill` end to end, both arms through the dispatch,
interleaved, min of 9 reps:

| shape | seqlen | triton | plugin | speedup |
|---|---|---|---|---|
| N=1 T=2048 | 2048 | 381.9 us | 387.0 us | 0.99x |
| N=1 T=8192 | 8192 | 1252.2 us | 1252.5 us | 1.00x |
| N=1 T=16384 | 16384 | 2584.0 us | 2584.0 us | 1.00x |
| N=2 T=2048 | 1024 | 395.3 us | 309.4 us | **1.28x** |
| N=4 T=4096 | 1024 | 648.9 us | 536.0 us | **1.21x** |
| N=8 T=8192 | 1024 | 1327.8 us | 1050.2 us | **1.26x** |
| N=16 T=16384 | 1024 | 2745.5 us | 2136.8 us | **1.28x** |
| N=4 T=16384 | 4096 | 2684.7 us | 2354.0 us | **1.14x** |
| N=16 T=4096 | 256 | 766.1 us | 515.0 us | **1.49x** |
| N=32 T=4096 | 128 | 891.1 us | 576.9 us | **1.54x** |
| N=32 T=32768 | 1024 | 5542.0 us | 4292.6 us | **1.29x** |

**The N=1 rows read 1.00x because the fusion is gated off there, not because it
is neutral.** Ungated it measures 0.90x-0.98x at N=1. The fused kernel's only
parallelism is `(V/BV) x N x Hv` = 192 blocks at N=1 against 188 SMs, and the
chunk loop is strictly serial — each chunk consumes the previous chunk's state.
The split kernels keep an extra NT-way parallel dimension in `chunk_o` and that
is what pays for the 201 MB round trip. `FUSED_MIN_SEQS = 2` delegates
single-sequence batches to the in-tree kernel unchanged.

**Three negative results worth keeping.**

1. **Retuning the in-tree tiles is not the lever.** A sweep of `chunk_h`
   (BV x warps x stages) and `chunk_o` (BK x BV x warps x stages) at the real
   geometry found TokenSpeed's hardcoded BV=32/w4/s2 and BK=128/BV=64/w4/s2
   within **1-3%** of the best reachable config at every shape. vLLM autotunes
   these; on this device the autotuning would find almost nothing.
2. **Shared memory, not compute, is what sm_120 takes away.** Opt-in limit is
   101376 B against sm_100's ~227 KB. Every `num_stages>=3` config of the fused
   kernel needs 105-140 KB and will not launch; `chunk_h` at BV=64/stages=3
   needs 107016 B and at BV=32/stages=4 needs 123660 B, so **a third of vLLM's
   12-config `chunk_h` autotune space is silently unreachable here**. Independent
   corroboration from the survey: lmdeploy drops chunk size from 64 on sm_90 to
   **32 on sm_120** for the same kernel. Config selection therefore branches on
   the driver-reported `shared_memory_per_block_optin`, never the arch name.
3. **`num_stages=1` beats `num_stages=2` by ~2x** (N=1 T=2048: 268 us vs 884
   us). The serial state dependency gives the pipeliner nothing to overlap while
   the duplicated operand buffers cost shared memory. An earlier measurement of
   this kernel that reported it losing everywhere was taken at the Triton
   default `num_stages=2` and was wrong for that reason.

**Verdict.** Accept, gated. The kernel is correct to the bit and worth
1.14-1.54x on multi-sequence prefill, which is the common serving shape; it is
correctly neutral on single-sequence prefill, which is where the design cannot
help. Wiring is outstanding: `register_gdn()` is exported but not called from
the package `register()`.

**Not measured.** Serving-level throughput. All numbers here are kernel-level
wall clock including Python dispatch. `nvidia-smi` utilisation is not quoted —
it reads 100% on idle GPUs 2 and 3 on this host (E22).

**Still on the table, in order of size.** (a) TurboMind's `ContextParallelPlan`
— split one long sequence into segments, compute per-segment states, stitch with
a correction pass — is the principled fix for the N=1 case and the only surveyed
design that solves it. (b) SGLang fuses kkt+`solve_tril` into one launch; those
are 165 us of the 8192-token pipeline in two. (c) vLLM fuses split + l2norm +
gating into one post-conv kernel; l2norm here is 44 us and is almost entirely
launch overhead. (d) The `A`, `w`, `u` intermediates are still materialized;
TurboMind avoids those too, which is the rest of its 8-launches-to-3.

---

## E29 — The GEMM "reference wins bmm" alarm was a label, but it hid a real all-NaN kernel

**Objective.** A kernel inventory reported `gemm/bmm -> solution "reference",
priority 7 (PORTABLE)` on sm_120, and `gemm/mm -> "flashinfer", priority 15,
unverified`. Read as: every model using `bmm` runs a naive path, and the `mm`
winner may be one of the flashinfer/trtllm kernels that are admitted on sm_120
and then die with `TllmGenFmhaRunner: Unsupported architecture`. Audit both,
exercise every registered kernel on the GPU, and fix what is broken.

**Path.** Enumerated all registrations for `gemm/mm` and `gemm/bmm`, then
*exercised* each on GPU 3 at Qwen3.5-27B shapes (hidden 5120, intermediate
17408, head_dim 256) rather than trusting `capability.satisfied_by`. Surveyed
all seven vendored engines for their dense-GEMM and bmm dispatch. Numerics
against FP32-accumulated `torch.matmul` / `torch.bmm`; timings min-of-9 blocks
because GPU 3 carried up to five co-tenant processes.

**Result — the alarm itself is a false alarm.**

`gemm/bmm` has **3** registrations tree-wide and all three are the `torch_*`
references. This is not a gate bug, a trait mismatch, or a capability table
that omits 120 — no flashinfer, triton, cutlass or trtllm bmm kernel is
registered for `bmm` on *any* architecture. The winner is `torch_bmm`, which is
`torch.bmm`, which is cuBLAS strided-batched:

| shape | torch_bmm | cosine vs fp32 |
|---|---|---|
| B16 M1024 N5120 K512 | 296.7 TFLOP/s | 0.99999869 |
| B40 M4096 N128 K128 | 160.1 TFLOP/s | 0.99999869 |
| B8 M512 N512 K256 | 124.0 TFLOP/s | 0.99999863 |

against a `torch.mm` bf16 8192^3 ceiling of **415.7 TFLOP/s** on this card.
Dense `gemm/mm` is the same story: also `torch_mm`, also the only candidate,
376-413 TFLOP/s — ~99% of the ceiling. `solution` is a provenance label, not a
speed claim; the PORTABLE band it sits in is not evidence of a slow path.

Every one of the 9 admissible `gemm/mm` kernels **ran** at all four M values —
no `Unsupported architecture`, no runtime death. The `mm` half of the alarm is
clean. (`flashinfer_mm_nvfp4` is not registered on sm_120 at all: the import
guard at `ops/gemm/flashinfer.py:287` is `platform.is_blackwell`, which is
`major == 10`. Registering it by hand shows it works — 1273 TFLOP/s — but
`cublaslt_mm_nvfp4` outranks it 15 > 14 and is faster at every measured M
(1296 TFLOP/s, and 24.2 vs 7.6 at M=16). Left alone: the gate is inconsistent
with its own declared `min_arch_version=(10,0)`, but fixing it changes nothing.)

**Result — what was actually broken was invisible to the inventory.**

`deep_gemm_mm_fp8_blockscale` is admissible on sm_120 and returns **all-NaN**.
Verified raw through `deep_gemm.fp8_gemm_nt`, bypassing TokenSpeed, at six
shapes — `nan_frac` 0.25 (M1 N512 K512) rising to 1.000 by M64. DeepGEMM needs
tcgen05/TMEM; sm_12x has none; the package compiles and launches anyway.

It was registered `min_arch_version=(9,0)` with **no upper bound**, so sm_120
inherited a Hopper-era gate — the version-monotonic failure the port-kernel
skill warns about. It sits at priority 14 against Triton's 11, so for any shape
with `N % 64 == 0` but `N % 128 != 0` (which drops FlashInfer on its
`n_align_128` trait but not DeepGEMM on its `n_align_64`), it *wins*, and
`tokenspeed_kernel.mm(quant="mxfp8")` returns all-NaN end to end. Confirmed at
N=5184, K=5120.

**Fixes.**

1. *In-tree, `ops/gemm/deep_gemm.py`.* Skip the registration entirely on
   sm_12x, not merely bound it. A `max_arch_version` alone is insufficient:
   `runtime/layers/dense/fp8.py` sets
   `override="deep_gemm_mm_fp8_blockscale"` from `_use_deep_gemm_fp8` with no
   arch check, and `_resolve_override` never consults capability — so the
   override path would still have been poisoned. Skipping registration turns it
   into a loud `NoKernelFoundError`. After: N=5184 selects
   `triton_mm_fp8_blockscale`, cosine 0.999669, no NaN.
   SGLang reached this independently (`deep_gemm_wrapper/configurer.py:21-22`).
   vLLM enables DeepGEMM on sm120, but only against a `nv-dev` branch it pins
   for that purpose — a build TokenSpeed does not vendor.
2. *In-tree, `ops/gemm/__init__.py`.* `_kernel_handles_online_mxfp8` keyed on
   `solution == "reference"`, so the only way for an out-of-tree mxfp8 kernel to
   receive unquantized activations was to mislabel itself as a reference — the
   very mislabelling that started this audit. Now also honours a declared
   `self_quantizes_mxfp8` trait. Four lines, backwards compatible.
3. *Plugin, `unieinfra_kernel_sm120/gemm.py`, `register_gemm()`.* Two kernels,
   both `Priority.PLUGIN`:
   - `unieinfra_bmm_fp8_scaled_sm120` — FP8 bmm on FlashInfer's `bmm_fp8`.
     The reference it replaces dequantizes both operands to FP32 and runs
     `torch.bmm` in FP32. **2.20x-14.17x**; 449.8 TFLOP/s at B16 M1024 N5120
     K512 versus 17.5 for the reference. Cosine 0.9999986+ at every shape.
   - `unieinfra_mm_fp8_blockscale_sm120` — chooses between the two in-tree FP8
     block-scale backends by M. **1.20x-1.54x** for M <= 256, 1.00x above;
     peak 690 TFLOP/s. Numerics identical to whichever backend it routes to.

**Verdict. PROMOTED**, with the bmm alarm explicitly retracted.

**Rejected candidates, kept per contract.**

- *A Triton batched GEMM to displace `torch_bmm`.* Written and tuned over six
  tile configs. It loses at every small shape — Triton has a ~20 us launch
  floor here against cuBLAS's ~8 us, so no kernel body can win below a few
  hundred microseconds of work — and wins only 5-7% at the two largest shapes,
  inside run-to-run spread. cuBLAS strided-batched is genuinely competitive on
  sm_120 for dense bf16. The survey agrees: **no engine has a hand-written bf16
  bmm that runs on sm_120.** TRT-LLM has the only one
  (`cute_dsl_bf16_bmm_blackwell`) and it raises *"CuteDSL BF16 BMM only
  supports SM 100 family"*; SGLang's TGV CuTe-DSL bf16 GEMM likewise refuses
  anything outside SM 100/103. Everyone else calls `torch.bmm` / `F.linear`.
- *`flashinfer.gemm.tinygemm_bf16` for small-M dense `mm`.* Imported in tree
  but never registered, so it looked like a free win. Measured 0.66x-1.08x
  against `F.linear` across M=1..16 — it is not faster. Not registered.
- *Registering `flashinfer_mm_nvfp4` on sm_120.* Works once the scale swizzle
  is right, but `cublaslt_mm_nvfp4` already outranks it and is faster at every
  measured M.

**A wrong turn worth recording.** The first nvfp4 probe raised
`CUDA error: an illegal memory access`, which read as "flashinfer's fp4 kernel
is broken on sm_120" and would have been reported as such. It was **my scale
layout** — `mm_fp4` needs `swizzle_blockscale_2d(weight_scale)` and the `.T`
convention that `test_nvfp4_gemm_swiglu_nvfp4_quant.py` documents. With the
correct call it runs clean at 1273 TFLOP/s. An IMA also poisons the CUDA
context, so every later probe in that process died too and looked like further
evidence. Isolate each risky probe in its own process, and check the in-tree
test for the calling convention before blaming a kernel.

**Not measured.** No serving-level number and no end-to-end model run —
everything here is kernel-level. GPU 3 had up to five co-tenant processes
throughout (49.7 GiB resident, `nvidia-smi` showing 18%), so a first timing
pass produced numbers 5-25x off before switching to min-of-9 blocks; the
absolute TFLOP/s should be treated as lower bounds. The `(1,32)` ue8m0 mxfp8
signatures, `gemm/decode_gemv` (0 registrations — the `triton_gemv.py`
registrations are never imported by `ops/gemm/__init__.py`), and the `kimi3_*`
projection entry points were not audited.

**Hand-offs.** (1) `register_gemm()` is standalone and is not called from the
package's `register()`; `unieinfra_kernel_sm120/__init__.py` is owned
elsewhere. (2) `runtime/layers/dense/fp8.py` sets `_use_deep_gemm_fp8` purely
from scale format and shape divisibility, with **no arch check** — it will now
fail loudly on sm_120 rather than return NaN, but it should not be trying at
all. (3) The real fix for the `gemm/mm` M-split is the SM120 CUTLASS FP8
blockwise GEMM that vLLM, SGLang and TRT-LLM each ship independently, all three
with a small-M swapAB tile — none is vendored here. (4) `ops/gemm/flashinfer.py`
guards two imports on `platform.is_blackwell` (`major == 10`) while the
registrations declare `min_arch_version=(10,0)`; the two disagree on sm_120.
Same closed-allowlist class as E24.

## E29 — sm_120 admits three FlashInfer attention kernels that cannot run, and the registry already had the mechanism to stop it

**Objective.** `tokenspeed-kernel` gates kernels on architecture, vendor, dtype
and traits. The architecture gate is version-monotonic — `platform.py`'s
`_get_cuda_sm_features` grants every feature at `arch_version >= ArchVersion(10,
0)` — so sm_120 satisfies every capability requirement written for sm_100. Three
kernels were confirmed passing `capability.satisfied_by(platform) == True` on the
RTX PRO 6000 and then taking the engine down at the first call:

1. `flashinfer_trtllm_mha_extend_with_kvcache` — `TllmGenFmhaRunner: Unsupported
   architecture` (`fmhaRunner.cuh:37`)
2. FlashInfer's trtllm XQA decode — `Invalid page_size: 832, only 16, 32, 64,
   128 are supported`
3. FlashInfer AOT batch decode — `Unsupported group_size: 6` (24 q heads over 4
   kv heads)

All three reproduced on GPU 3 first, before touching anything. Selection at the
live geometry (page_size 832 / head_dim 256 / 24-4 heads, bf16) picked
`flashinfer_trtllm_mha_extend_with_kvcache` (pri 12) over
`triton_mha_extend_with_kvcache` (pri 4) and died; decode did the same.

**Path — survey first.** All four vendored engines were read on one question:
how does each separate "this kernel cannot run on this **arch**" from "this
kernel cannot run this **shape**"?

| engine | applicability form | arch vs shape | on "unsupported" | GQA group size | page-size set |
|---|---|---|---|---|---|
| vLLM | declarative class traits + `supports_*()` predicates, folded by `validate_configuration -> list[str]` | **same** mechanism; `supports_compute_capability` sits in the same list as `supports_head_size` | user-pinned -> `ValueError`; auto -> skip + `debug_once`; none left -> `ValueError` | **not in the selector contract**; backends smuggle it via global config or assert in `__init__` | per-backend `get_supported_kernel_block_sizes() -> list[int \| MultipleOf]`, validity by *divisibility* |
| SGLang | imperative `assert`/`raise` in factory closures | **separate**; arch in an if/elif ladder, page size in config passes | arch -> raise; page size -> **silently rewritten** to fit, with a warning | perf-only (`gqa_group_size >= 4` picks tensor cores); never a validity rule | literal lists inside a mutating post-process pass |
| LMDeploy | one `if/elif/else` + one `lru_cache`d bool | separate, both degenerate | never raises; terminal fallback is always Triton | `assert` inside impl `__init__`, after selection committed | scalar equalities at engine startup |
| TensorRT-LLM | class-level `frozenset` tables + a reason-returning predicate over a C++ cubin hash table | **separate by lifetime**: `is_available(cls, attn)` = arch, once; `is_supported(self, q,k,v,...)` = shape, per forward | Python: log + next FMHA lib, `RuntimeError` if chain exhausted. C++ arch: `TLLM_CHECK` abort. C++ shape: warn + `(false, reason)` | **first-class**: `MAX_HEADS_RATIO_GENERATION = 32`, divisibility in `is_available` | `SUPPORTED_TOKENS_PER_BLOCK = {16,32,64}` + power-of-2 test |

Three things were taken from this rather than invented.

- *Arch and shape are different gates and must not share a mechanism.*
  TensorRT-LLM's split by **lifetime** is the cleanest: an arch mismatch can be
  decided once at construction, a shape mismatch has to be decided per call.
  TokenSpeed already has exactly this shape — `CapabilityRequirement` is
  evaluated per platform, `traits` per call — so the fix was to *use* both, not
  to add a layer.
- *The arch predicate must be able to say "exactly this", not only "at least
  this".* vLLM carries three: `has_device_capability` (`>=`),
  `is_device_capability` (`==`) and `is_device_capability_family` (major match,
  "mirrors CUDA 13 family semantics"). It excludes sm_120 from trtllm-gen with
  `is_device_capability_family(100)` — deliberately *not* `>=`. TRT-LLM's
  `is_sm_100f(sm)` is `sm == 100 or sm == 103`, the literal mirror of the C++
  `TLLM_CHECK(mSM == kSM_100 || mSM == kSM_103)`. TokenSpeed's
  `CapabilityRequirement` already has `max_arch_version`; it simply was not set.
- *`bool` is not enough — return why.* vLLM and TRT-LLM converged independently
  on a reason string (`list[str]`, `Tuple[bool, str]`, a `SHOULD_NOT_USE(msg)`
  macro), because the moment there is more than one candidate the final choice
  has to be explainable. SGLang and LMDeploy do not need it only because their
  ladders are total.

**What changed, and why each one.**

*(a) Arch bound on the two trtllm-gen kernels.* `max_arch_version=ArchVersion(10,
3)` on `flashinfer_trtllm_mha_{extend,decode}_with_kvcache`. Read off
`fmhaRunner.cuh:37` rather than guessed: no shape makes trtllm-gen run on
sm_120, so this belongs on the arch gate. This is case 1, closed.

*(b) The decode kernel was two kernels wearing one name.*
`trtllm_batch_decode_with_kv_cache(backend="auto")` resolves to `"trtllm-gen" if
get_compute_capability(...)[0] == 10 else "xqa"` — so on sm_120 the registered
"trtllm" kernel was silently running XQA, a different kernel with a different
constraint surface, under a capability gate and trait set written for
trtllm-gen. Split into `flashinfer_trtllm_mha_decode_with_kvcache` (10.0-10.3,
`backend="trtllm-gen"`) and `flashinfer_xqa_mha_decode_with_kvcache` (>= 12.0,
`backend="xqa"`), sharing one body. Backend is now passed explicitly so the gate
and the kernel that runs cannot drift. This mirrors SGLang, which makes the same
sm_90/sm_120-vs-sm_100 XQA split — but as a runtime branch *inside* one backend
object (`self.is_xqa_impl = is_sm90_supported() or is_sm120_supported()`), which
is why its constraint asserts then have to be re-derived at every call site.

*(c) Trait vocabulary: a rule, not just a set.* XQA's real limits were measured
on the device, not read off a doc: page_size in {16,32,64,128}, head_dim any
multiple of 16 in [16,256] — 32, 96 and 192 all verified working. page_size is a
short list and the existing `frozenset` form already handles it. head_dim is a
*rule*; enumerating its 16 values hides the rule and goes stale. Added
`TraitConstraint` + `TraitRange(minimum, maximum, multiple_of)` in `registry.py`,
accepted anywhere a `frozenset` is by `spec_matches_traits`,
`ref_compatible_with_spec` and the shape-trait path. This is vLLM's `MultipleOf`
generalised — vLLM needed exactly this escape hatch for the same reason.

*(d) Two derived traits the dispatch was not building.* `group_size =
num_q_heads // num_kv_heads` (returns 0 when not divisible, which matches no
declared set, so a non-divisible config filters out rather than dispatching into
an assert) and `q_len_per_req = max_seqlen_q`. The second came out of sweeping
the decode kernel's *declared* trait surface on the device rather than only the
reported failure — `q_len_per_req > 1` on sm_120 hits `Mask is required for
speculative decoding`, and fp8 q hits `Output and query must have the same
dtype`. Both were admitted-but-broken and neither was in the original report.
The XQA registration now declares `q_len_per_req: {1}` and drops fp8 from its
signatures.

*(e) Diagnostics.* `explain_selection` reported `max_arch_version` exclusions and
every trait exclusion as `"unknown"`. Both now name the reason, per the survey's
third point. `describe_trait_value` renders numeric trait sets in numeric order.

**Result — selection.** On the real GPU, at the geometry that was crashing.

| call | before | after |
|---|---|---|
| `mha_extend_with_kvcache` page 832 | `flashinfer_trtllm_...` -> `Unsupported architecture` | `triton_mha_extend_with_kvcache` |
| `mha_decode_with_kvcache` page 832 | `flashinfer_trtllm_...` -> `Invalid page_size: 832` | `triton_mha_decode_with_kvcache_cached`, runs, finite |
| `mha_decode_with_kvcache` page 64 | `flashinfer_trtllm_...` (actually XQA) | `flashinfer_xqa_mha_decode_with_kvcache`, runs |
| `mha_decode_with_kvcache` page 64, q_len 4 | `flashinfer_trtllm_...` -> `Mask is required` | `triton_mha_decode_with_kvcache_cached`, runs |

`explain_selection` at page 832 now reads:

```
  - flashinfer_trtllm_mha_decode_with_kvcache: arch mismatch (requires <= 10.3)
  - flashinfer_xqa_mha_decode_with_kvcache: trait mismatch: page_size=832 not in {16, 32, 64, 128}
```

**Result — numerics.** The kernel selection now lands on, against
`solution="triton"` through the ops dispatch. Bar is cosine > 0.9999.

| case | max abs | mean abs | cosine |
|---|---|---|---|
| 24/4 head_dim 256 (live geometry) | 6.10e-05 | 6.28e-06 | 0.9999946 |
| 24/4 head_dim 128 | 6.10e-05 | 9.01e-06 | 0.9999912 |
| 32/8 head_dim 128 | 1.22e-04 | 9.14e-06 | 0.9999911 |
| 16/16 head_dim 64 | 6.10e-05 | 1.11e-05 | 0.9999883 |
| window_left=128 | 1.22e-04 | 1.40e-05 | 0.9999953 |
| sinks | 6.10e-05 | 9.00e-06 | 0.9999914 |

**Result — tests.** 324 passed / 165 skipped across `test_selection.py`,
`test_registry.py`, `test_kernel_api_selection.py`, `test_plugins.py`,
`test_numerics.py`, `ops/test_attention.py`, `ops/test_quantization.py`,
`ops/test_gemm.py`. New: an `rtx_pro_6000_platform` fixture, two golden
selection cases under a `blackwell-sm120` arch key, eight sm_120 regression
tests, and `TestTraitRange` / `TestTraitConstraintInSpecs` in the selection unit
tests. `ops/test_attention.py` went from **6 pre-existing failures to 0** on this
host — four disappear because the configurations are now correctly unregistered,
and two (`q4-flashinfer`) became skips once the `require` fixture was taught to
filter on traits as well as solution+dtype. A test that forces
`solution="flashinfer"` at a shape flashinfer cannot serve on this arch should
skip, not fail — that is vLLM's user-pinned-vs-auto distinction applied to the
test harness.

**Verdict. Landed for cases 1 and 2; case 3 is not reachable through selection
and was closed differently.** The AOT fa2 batch decode
(`BatchDecodeWithPagedKVCacheWrapper`, `DISPATCH_GQA_GROUP_SIZE` -> {1,2,3,4,8})
is **re-exported as a public symbol and never registered as a kernel**, so
selection cannot gate it by construction — confirmed by enumerating every
admitted `attention` spec in the installed package: `mha_decode_with_kvcache` has
exactly two registrations. Its limit is recorded as
`FLASHINFER_AOT_BATCH_DECODE_GROUP_SIZES` at the export site so the next reader
does not rediscover it by crashing, and the `group_size` trait now flows so a
one-line declaration suffices the day it is registered. Both directions are
tested: a `Priority.PLUGIN` spec declaring `{1,2,3,4,8}` loses at group 6 and
wins when the ratio is one it supports.

**Not measured.** Any throughput number. Nothing here is a performance change —
`mha_extend_with_kvcache` at head_dim 256 now runs a PORTABLE Triton kernel where
it previously ran nothing at all, and whether that is fast is a separate
question.

**Still admitted-but-broken after this, in order of size.**

(a) **`triton_mha_extend_with_kvcache` at head_dim 256 on sm_120.**
`OutOfResources: shared memory, Required: 114688, Hardware limit: 101376`.
head_dim 64/128/192 all run; only 256 fails — and 256 is the live geometry. This
is the fix landing on a fallback that is itself tuned for the 227 KB SM, exactly
the §3 trap. Pre-existing and latent (any trait combination that already excluded
FlashInfer hit it), now on the critical path. It is a tile-size problem in
`ops/attention/triton/mha_prefill.py`, owned by another agent. The registration
should declare the limit off the driver-reported
`shared_memory_per_block_optin`, never off the arch name.

(b) **The root cause is untouched.** `_get_cuda_sm_features` still grants sm_120
every sm_100 feature. Each sm_100-only kernel has to remember an explicit
`max_arch_version`; forget it and the kernel is admitted. The principled fix is a
`compute:tcgen05` feature granted only for major == 10, which every trtllm-gen /
tcgen05 kernel then requires — one line each, and a *positive* statement of what
the kernel needs rather than a negative bound on where it must not go. Not done
here: `platform.py` is shared and concurrently edited, and a mistake in
`_get_cuda_sm_features` is a silent global false-*rejection*, which is worse than
the loud false-admission it replaces.

(c) **Unverified page-size bound on the sm_100 side.** trtllm-gen's own tables
say `SUPPORTED_TOKENS_PER_BLOCK = {16,32,64}`; FlashInfer's vendored copy may
differ, and there is no B200 here to test on. Deliberately **not** declared —
adding an unverified restriction converts a loud crash into a silent fallback,
and this project has a standing note that closed allowlists are how sm_120
features go missing in the first place. Only what was measured on the device is
declared.

**Method note.** `capability.satisfied_by()` returning `True` was the entire bug,
so no candidate was believed until it had been *exercised* at the shape in
question. Sweeping the declared trait surface — not just the reported failure —
is what found (d) above; two of the four holes closed here were not in the
report.

## E30 — MXFP4 quantization was unreachable on sm_120, and the kernel behind the gate was silently wrong

**Objective.** Port the quantization kernels for sm_120. A registry inventory
said `quantization/fp8` and `fp8_with_scale` were sitting in the PORTABLE band
(priority 4), `mxfp8` was fine on FlashInfer (8), and `mxfp4` had **no
admissible kernel at all**. Also to decide whether the
`max_arch_version=ArchVersion(10, 9)` cap on `trtllm_quantize_fp8_with_scale`
(`ops/quantization/trtllm.py:72`) is a real hardware limit or an untested
guess.

**Path.**

*Inventory, then exercise.* `satisfied_by()` is not evidence. Every registered
implementation was called directly on the GPU, bypassing selection. That
immediately turned up more holes than the inventory reported: at the *public
API* level, `quantize_fp8_with_scale(granularity="tensor")` and
`granularity="token"` both raise `NoKernelFoundError` — the in-tree Triton
kernel declares only `token_group_{32,128}` and the TRT-LLM one is capped out —
and `quantization/nvfp4` has **zero** kernels registered, because
`ops/quantization/flashinfer.py:68` guards on `platform.is_blackwell`, which is
`arch_version.major == 10`. sm_120 reports 12.

*The 10.9 cap is justified, and for a worse reason than "it is slow".* All four
TRT-LLM FP8 quantize ops — `quantize_e4m3_per_tensor`,
`quantize_e4m3_activation`, `fp8_quantize_1x128` with and without UE8M0 — fail
on this card with `cudaPeekAtLastError() == 209`,
`cudaErrorNoKernelImageForDevice`: the shipped `libtrtllm_kernel.so` has no
sm_120 cubin and no forward-compatible PTX. The launch return code is **not
checked**, so the op returns normally with *uninitialized* output and leaves a
pending CUDA error that detonates on the next unrelated torch call. My own
first probe recorded `fp8_quantize_1x128` as `[WORKS]` for exactly this reason —
it printed plausible shapes and `torch.cuda.synchronize()` did not raise. Only
running each op in a fresh subprocess and peeking at the error state before
touching the results showed what was happening. The cap is the only thing
standing between sm_120 and silently corrupt activations, and it should stay.

*Why mxfp4 has nothing.* `triton_quantize_mxfp4` is registered
`vendors=frozenset({"amd"})` (`ops/quantization/triton.py:522`). It is the only
MXFP4 entry, so NVIDIA gets nothing. There is a second, quieter gate behind it:
even with the vendor gate corrected the kernel declares
`scale_layout {linear}` while `quantize_mxfp4`'s own default is `"128x4"`, so
the default call still finds nothing.

*Survey before writing anything* (§2). All five vendored engines plus `modular`
and the contest tree. The MXFP4 landscape on sm_120 is worse than expected:

| engine | MXFP4 act-quant on sm_120 | shared exponent |
|---|---|---|
| TRT-LLM / FlashInfer | **yes**, `__CUDA_ARCH__ >= 1000` covers 1200 | `ceil(log2(amax/6))`, never clips |
| vLLM | **no** — `mxfp4_experts_quant.cu:380` is literally `cap >= 100 && cap < 120`, and the source is absent from `FP4_SM120_SRCS` (`CMakeLists.txt:998-1014`) | OCP `floor(log2(round_up(amax))) - 2`, can clip |
| SGLang | **no** — `mxfp4.py:359` raises `"requires SM90 or SM100"` | reference path only |
| lmdeploy / tilert | none — FP4 is weight-only | — |

Another instance of the closed-allowlist pattern from E17, this time in vLLM
rather than TokenSpeed. NVFP4 is the opposite: vLLM builds it for `12.0a;12.1a`
under `-DENABLE_NVFP4_SM120=1` and admits `sm >= 120 && sm < 130`, so it is a
first-class path there.

**Result — the thing I was not looking for.** Widening the vendor gate makes
the in-tree Triton MXFP4 kernel selectable on NVIDIA. Checked against a float32
reference at one magnitude (`randn*4`) it matched 99.8% of scale bytes and
looked fine. Swept across magnitudes it does not:

| input scale | block amax | ref scale byte | kernel byte | dequant max err / input RMS |
|---|---|---|---|---|
| 1e-3 | 0.0023 | 116 | **127** | **6012x** |
| 1e-2 | 0.0235 | 119 | **127** | **598x** |
| 0.1 | 0.233 | 123 | **127** | **59x** |
| 1.0 | 2.37 | 126 | **127** | **4.7x** |
| 4.0 | 9.25 | 128 | 128 | 0.5 |
| 1024 | 2401 | 136 | 136 | 0.5 |

The kernel computes `scale_e8m0_unbiased.to(tl.uint8) + 127`. The shared
exponent `floor(log2(amax)) - 2` is **negative for any block with amax < 4** —
which is essentially all real activations — and Triton's float-to-uint8
conversion *saturates* negatives to 0 on the NVIDIA backend rather than
wrapping. So the stored byte collapses to 127, scale `2^0 = 1`. The E2M1
nibbles are computed from the *float* exponent and stay correct, so the
corruption is invisible in the data and lives entirely in the scale: the tensor
dequantizes `2^-e` too large, with no NaN, no error, and no shape change.

The fix is to bias before converting, `(scale_e8m0_unbiased + 127).to(tl.uint8)`.
On a backend whose conversion wraps in two's complement this is
behaviour-preserving — `(-2).to(u8) + 127 = 254 + 127 mod 256 = 125`, same
answer — which is why the kernel has presumably been correct on AMD all along
and why the pre-existing test asserting `scale == [[127, 125], ...]` passed.
That test used a fixed input and, being behind the AMD gate, had never run on
NVIDIA. A 10-point magnitude sweep now fails at 6 of 10 points without the fix.

**Result — what was built.** New `unieinfra_kernel_sm120/quant.py`,
`register_quant()`, four registrations at `Priority.PLUGIN` (16). All numerics
are against an independent float32 model of the *format* — the E4M3 and E2M1
grids, the UE8M0 exponent — built from torch primitives, never against another
kernel.

| kernel | fills | verified |
|---|---|---|
| `unieinfra_flashinfer_quantize_mxfp4_sm120` | mxfp4, all 3 layouts incl. the API default `128x4` | scale bytes **100%**, E2M1 codes **100%** vs fp32 ref; error exactly 0.5x the format step |
| `unieinfra_flashinfer_quantize_nvfp4_sm120` | nvfp4 — had **zero** kernels | SF bytes **100%**; error exactly 0.5x the step |
| `unieinfra_triton_quantize_fp8_with_scale_sm120` | `granularity` tensor + token — had **none** | codes **100%**; 0.99-1.00x of the format bound; rel err 0.058 vs the 2^-4 bound |
| `unieinfra_triton_quantize_fp8_sm120` | fp8 cast, PORTABLE -> PLUGIN | **bit-identical** to in-tree at 4 shapes; 100% vs fp32 ref |

Speed, interleaved A/B on a shared GPU, medians of 7, with A-series drift
quoted because the machine is not stationary:

| op | shape | in-tree | plugin | speedup | drift |
|---|---|---|---|---|---|
| mxfp4 | (4096, 1024) | 139.1 us | 6.2 us | **22.6x** | 4.2% |
| mxfp4 | (4096, 7168) | 1008 us | 14.8 us | **68x** | 5.4% |
| mxfp4 | (1024, 512) | 10.3 us | 4.1 us | 2.5x | 0.2% |
| fp8 cast | (8192, 1024) | 6.3 us | 4.3 us | 1.47x | 0.9% |
| fp8 cast | (4096, 7168) | 804.6 us | 12.8 us | **63x** | 8.8% |
| fp8 cast | (16384, 128) | 2.2 us | 2.1 us | 1.02x | **1076%** |

Both large multipliers are launch-geometry cliffs, not tuning. The Triton
MXFP4 kernel launches `grid = (M, N/32)` with `num_warps=1` — 917,504 programs
of 32 elements each at `[4096, 7168]`. The in-tree FP8 cast tiles as
`BLOCK_M x next_pow2(N)`, which at the same shape puts 131072 elements in one
program and spills. The last row is pure noise and is shown to say so.

**Stated precision difference.** The MXFP4 plugin is **not** bit-identical to
the Triton path and is not meant to be. FlashInfer uses `ceil(log2(amax/6))`;
the in-tree kernel uses OCP `floor(log2(amax)) - 2`. They disagree on **11.8%**
of blocks and change **3.2%** of dequantized elements by up to one full E2M1
step — and the disagreeing blocks are exactly the ones the OCP recipe has to
clip (both 11.804% on the same input). The trade goes both ways: the OCP recipe packs the block max nearer 6.0
and wins ~4% on mean absolute error (cosine 0.9937 vs 0.9933 against the fp32
input), but must clip that max in **11.8%** of blocks; the FlashInfer recipe
never clips. Both stay inside the half-step bound. Anything that pairs an
activation quantized by one recipe with a weight scale produced by the other
loses accuracy silently — vLLM and TRT-LLM have exactly this incompatibility
between them today.

**Verdict. Landed.** The tree fix is unambiguous and independent of the plugin:
`triton_quantize_mxfp4` was returning wrong answers for realistic input on
whichever backend saturates, and the sweep test now pins it. The vendor
widening and the scale-byte fix must ship together — widening alone would have
promoted a silently wrong kernel to being the PORTABLE fallback on every NVIDIA
card. The four plugin registrations fill holes where the alternative is
`NoKernelFoundError`, except the FP8 cast, which is bit-identical and therefore
risk-free.

**Not measured.** Any serving-level throughput, for any of this. Per the
contract that makes this a verified port, not a performance finding. The 63x
and 68x are kernel-level microbenchmarks on a contended GPU and reproduced at
38x and 22x under different load; treat the order of magnitude, not the digit.

**Not done, deliberately.** Nothing is fused. Every surveyed engine fuses
activation quantization into the preceding RMSNorm or SiLU — vLLM
`rms_norm_dynamic_per_token_quant`, TRT-LLM `fused_add_rms_norm_quant`, SGLang
`dsv4_norm_rope.cu`, lmdeploy `v4_swiglu_and_quant_fp8` — and that is the
single largest remaining win. It is not reachable from a plugin: TokenSpeed's
`quantize_*` are standalone ops whose signatures contain no producer, so
fusing needs a new op in the tree. `scale_encoding="ue8m0"` for
`fp8_with_scale` is also left unserved rather than guessed: the contract makes
its encoded integer layout backend-specific and there is no consumer on this
card to validate against.

**Wiring gap.** `unieinfra_kernel_sm120/__init__.py::register()` does not call
`register_quant()`. Until it does the module is inert in the live engine — the
third instance of this in a row, after `register_moe()` in E25 and
`discover_plugins()` in E24.

## E31 — A shared card was quietly halving every number, and the process list said we were alone

**Objective.** Build the kernel-level measurement contract the porting work
depends on (`unieai-kda/eval/kernel_bench.py`), and produce a baseline sweep
over the operators Qwen3.5-27B actually hits: `attention/mha_{prefill,
extend_with_kvcache,decode_with_kvcache}`, `gemm/{mm,bmm}`, `quantization/fp8`
at 24 q heads / 4 kv heads / head_dim 256 / page_size 832 / hidden 5120 /
intermediate 17408, bf16. Full results in
[`kernel-baseline.md`](kernel-baseline.md) + `kernel-baseline.json`.

**Path.** Enumerate candidates in four stages (registered -> capability ->
format signature -> traits), then *exercise* every admitted one through the
public dispatch with `override=`, so marshalling and profiling scopes are the
production ones. CUDA-event timing, median of 15 samples x 3 whole-case
repeats, `kda.gates.gate_stationary` imported (not copied) from the serving
harness. Every candidate compared against one explicitly named reference.

**Result 1 — the card was shared, and the only instrument that noticed was
arithmetic.** Three consecutive sweeps produced internally consistent,
stationary, low-spread numbers that were **about half the truth**: prefill GEMM
at 186-248 TFLOP/s. The fourth read 390-409. Nothing in the run distinguished
them except a bf16 8192^3 calibration anchor borrowed from E22:

| | contaminated runs | clean run |
|---|---|---|
| anchor peak | 203-265 TFLOP/s | **437.1** |
| `card_share` vs E22's free-card 427.1 | 0.47-0.62 | **1.02** |
| power before our work | 359-433 W | **88 W** |
| NVML compute processes (in container) | **1** | **1** |

That last row is the finding. `nvmlDeviceGetComputeRunningProcesses` only sees
PIDs in the caller's namespace, so **inside a container it reports "just us"
while another tenant holds half the GPU**. The host showed PID 873043 at 359 W
and 44 GB on GPU 3 the whole time. This is a second broken instrument on this
host alongside E22's utilisation counter, and it fails in the same direction:
it reads as *all clear*. Detect co-tenancy by measuring how much of the card you
are actually getting, never by asking who else is on it. The harness now runs
the anchor before and after every sweep and refuses to call absolute numbers
comparable when `card_share` drops.

(E29's GEMM audit independently worked around this with min-of-9 blocking under
"up to five co-tenant processes" — same hazard, different mitigation. Its
415.7 TFLOP/s ceiling and 376-413 TFLOP/s for `mm` agree with the clean numbers
here to within 5%, which is the corroboration both entries needed.)

**Result 2 — stationary and reproducible are different properties, and only one
had a gate.** `mha_decode_b32_ctx4096` measured **1433 GB/s** and **799 GB/s**
in two processes at the same shape, both passing `gate_stationary` with p10-p90
spread under 3%. A steady neighbour shifts a series by a constant and leaves it
perfectly flat. `kernel_bench.py` therefore re-runs each whole case N times with
fresh tensors and refuses a single number when the medians disagree by >10%.

**Result 3 — the benchmark's own loop length was worth 1.8x.** Sizing the inner
timing loop from a single noisy probe flipped `quantize_fp8` on a 4096x17408
tensor between `inner=1` (5.09 ms/call, every call cold) and `inner=2`
(2.79 ms/call, second call partly L2-fed) across repeats of an identical case —
clock pinned at 2422 MHz, no neighbour, both stationary. Any kernel whose
working set straddles the 128 MiB L2 has two defensible answers; the harness now
pins the choice on a coarse ladder off a median-of-5 probe and flags when it
differs. Related: five cases report GB/s **above** the card's 1451 GB/s DRAM
bandwidth because their weights live in L2 across the loop — flagged, not
silently quoted.

**Result 4 — `head_dim 256`, not page_size 832, is what kills Triton MHA.** The
obvious suspect was the page size the GDN state row forces. It is not:

| head_dim | `triton_mha_prefill` | `triton_mha_extend` (page 128 *and* 832) |
|---|---|---|
| 64 / 128 / 192 | ok | ok |
| **256** | **OutOfResources 114688 > 101376** | **OutOfResources 114688 > 101376** |

112 KiB required against a 99 KiB opt-in limit — the same sm_100 -> sm_120
shared-memory cliff that made a third of vLLM's `chunk_h` autotune space
unreachable in the GDN entry above. **`attention/mha_prefill` and
`attention/mha_extend_with_kvcache` have ZERO working kernels at this model's
geometry**, since the only other candidate is the flashinfer trtllm extend that
E29 already showed is arch-rejected everywhere. Fix follows the precedent
already set here: branch tile size on `shared_memory_per_block_optin`, never on
the arch name.

**Result 5 — flashinfer decode works at head_dim 256; page_size is the only
blocker.** Separating the two axes:

| head_dim | page_size | `flashinfer_trtllm_mha_decode` |
|---|---|---|
| 128 / 256 | **128** | **ok** |
| 128 / 256 | 832 | `Invalid page_size: 832` |

So the priority-12 kernel is reachable on this card and this head_dim; only the
832 page size (E29's item 2) stands in front of it. Decoupling the attention
page size from the GDN state row promotes a priority-4 fallback to a
priority-12 specialized path with no kernel writing at all. Worth tempering:
`triton_mha_decode_with_kvcache_cached` already runs at **1432 GB/s against a
1451 GB/s roofline — 98.7%** — so the prize is what the specialized path
*enables* (fp8 KV, CUDA-graph behaviour), not bandwidth.

**Result 6 — baseline, clean run, card to ourselves.**

| operator | best working kernel | measured | vs roofline |
|---|---|---|---|
| `mha_decode` b32/ctx4096 | `triton_..._cached` (pri 4) | 1431.6 GB/s | **98.7%** of 1451 |
| `mha_prefill`, `mha_extend` | **none** | — | — |
| `gemm/mm` bf16 M4096 | `torch_mm` (pri 7) | 390-409 TFLOP/s | 89-94% of 437 |
| `gemm/mm` mxfp8 M4096 | `flashinfer_mm_fp8_blockscale` (pri 15) | 721.9 TFLOP/s | **1.85x** the bf16 path |
| `gemm/bmm` bf16 | `torch_bmm` (pri 7) | 412.3 TFLOP/s | 94% of 437 |
| `quantization/fp8` 4096x17408 | `triton_quantize_fp8` (pri 4) | 76.1 GB/s | **5.2%** of 1451 |

`quantize_fp8` at 5.2% of memory roofline on a pure streaming cast, bit-exact
against the reference, single PORTABLE-band implementation, no competitor
registered — the largest and lowest-risk headroom in the sweep.

**Corroboration of E29/E30's all-NaN kernels.**
`deep_gemm_mm_fp8_blockscale` (pri 14) was independently caught here and
bisected: it is **exactly correct when all scales are 1.0 and returns all-NaN
for any non-unit scale**, at every shape tried, without raising, while being the
fastest of the four candidates. All-ones scales are precisely what a smoke test
writes. It does not currently win selection only because flashinfer sits one
priority point above it — an autotune-based policy would pick it.

**Verdict.** The instrument is the deliverable. Its guards are: named-mandatory
reference, `rel_l2` alongside cosine (cosine is blind to a scale error),
card-share calibration, cross-repeat reproducibility, pinned loop length,
L2-residency flags, input checksums, staged enumeration with every admitted
candidate executed, and refusal — a rejected number is emitted as `null` with a
reason, never as a caveat next to a value. Re-run cost is ~30 s of GPU time.

**Not measured.** Serving-level anything. `max_seqlen_q > 1` decode
(speculative draft blocks). Tolerance judgements for quantized paths — the
harness reports the diffs and declines to rule on them. Triton autotune state is
per-process and candidate order is fixed, so a kernel measured second may
inherit a warmer allocator. Numbers are for the image's installed
`tokenspeed_kernel`, not the `/workspace/roy/tokenspeed` working tree; use
`--plugin` to measure an out-of-tree port in the same image. `nvidia-smi`
utilisation is not quoted anywhere — E22 — and neither is the NVML process
count, for the reason in Result 1.

---

## E32 — `moe/apply` on sm_120: 4 of 7 admissible candidates raise, and FP8 has nothing

**Objective.** Two of the four target checkpoints are MoE — Qwen3.6-35B-A3B
(`qwen3_5_moe`, E=256, D=2048, ispp=512, top-k 8) and gemma-4-26B-A4B
(E=128, D=2816, ispp=704, top-k 8). E24 established that `flashinfer_trtllm`
attention is admissible-and-broken on this arch. `moe/apply` selects the same
family at priority 12. Find out what actually runs, then fill whatever hole
that leaves.

**Path.** Enumerated all 18 `moe/apply` registrations, then *exercised* each of
the 7 the capability gate admits, one per process so a sticky CUDA error could
not be mistaken for the next candidate's. Real geometry, 64 tokens, GPU 3.

**Result — what runs.**

| prio | kernel | outcome |
|---|---|---|
| 12 | `flashinfer_trtllm_fp8_moe_apply` | **raises** `unexpected hidden_states_scale shape (1024,); expected (16, 64)` |
| 8 | `flashinfer_cutlass_unquant_moe_apply` | runs, cos 0.999996 |
| 8 | `flashinfer_cutlass_fp8_moe_apply` | **raises** `NotImplementedError: FP8 block scaling not yet implemented for Blackwell` |
| 8 | `flashinfer_cutlass_nvfp4_moe_apply` | runs |
| 8 | `flashinfer_cutedsl_deepep_nvfp4_moe_apply` | **unreachable** — see below |
| 4 | `triton_mxfp4_moe_apply` | **raises** `ptxas: Feature '.tile::scatter4' not supported on .target 'sm_120a'` |
| 4 | `triton_mxfp8_precomputed_moe_apply` | runs, **wrong answer** — see below |

So a **block-scaled FP8 MoE has no working apply kernel on sm_120 at all**: the
SPECIALIZED entry that wins selection raises, and the PERFORMANT entry behind it
raises too. Unquantized bf16 is fine and always was — the trtllm unquant entries
carry `max_arch_version=(10, 3)` and were never admitted here, so bf16 has been
landing on cutlass all along. This is the *opposite* of the E24 attention
finding and worth stating plainly: the trtllm MoE entry is only reachable for
fp8, and only fp8 is broken.

**Two bugs that are not about sm_120.**

1. *DeepEP is unreachable on every architecture.* `moe_plan` sets
   `solution = "flashinfer_cutedsl_deepep"` when `a2a_backend == "deepep"`, but
   the kernel registers `solution="flashinfer_cutedsl"`. `solution=` is a hard
   filter (§4 trap 3 of the port-kernel skill), so the plan can never reach its
   own kernel. Confirmed: pinning `solution="flashinfer_cutedsl"` by hand selects
   `flashinfer_cutedsl_deepep_nvfp4_moe_apply` immediately.
2. *`triton_mxfp8_precomputed_moe_apply` returns wrong output for
   `activation="silu"`.* Its `(1, 32)` branch rewrites a `[gate | up]` checkpoint
   into row-interleaved order for triton_kernels' fused swiglu, but the
   `swiglu_arg is None` path then falls back to `_silu_gate_up`, which
   `chunk(2, dim=-1)`s the gemm1 output as if it were still concatenated.
   `MoELayer`'s default is `activation="silu"`, which leaves `swiglu_arg` None,
   so this is the live path. Measured: cos **-0.0017** against a correct
   `[gate | up]` oracle and cos **0.9989** against an oracle that reproduces the
   interleave-then-chunk permutation — i.e. the kernel is computing exactly the
   wrong thing, and the error is a layout permutation, not precision.
   E21's false-admission theme again, one layer down: the kernel is *selected*
   correctly and *wrong* silently.

**Path — what was built.** Surveyed vLLM, SGLang, lmdeploy, TensorRT-LLM and
tilert before writing anything. The survey ruled out every non-Triton option on
this arch by the vendors' own guards, not by inference:

- TRT-LLM's trtllm-gen MoE has **no sm_120 cubins at all** (`CudaArch` enumerates
  Sm90a/Sm100a/Sm100f/Sm103a; `isSMCompatible` throws for anything else), and
  its Python wrapper says so: `if sm_version >= 120: raise NotImplementedError`.
  Its CUTLASS grouped GEMM *does* build for sm_120 but only instantiates
  NVFP4xNVFP4 and FP8xFP4 — never FP8xFP8 or BF16.
- vLLM excludes its own CUTLASS grouped-GEMM MoE on 120
  (`cap < 90 or cap >= 110 -> False`) and every `trtllm_*` experts class behind
  `is_device_capability_family(100)`, which is false for 12.x because the family
  test is `cap // 10`.
- SGLang's default for unquantized MoE on sm_120 is its Triton `fused_moe_kernel`,
  and it explicitly disables the persistent/TMA-gather path there
  (`if is_sm120_supported(): update_opt_flags_constraints({"is_persistent": False})`).

That leaves the Triton design all three share, and **none of the three fuses the
two GEMMs** — the only single-kernel MoE fusions in any of these trees are vendor
kernels with no sm_120 build (FlashInfer `b12x`, trtllm-gen). So the port
reproduces the shared shape: `moe_align_block_size` -> gather A by
`offs_token // top_k` -> GEMM1 -> separate SwiGLU -> GEMM2 with the router weight
folded into the accumulator -> top-k sum.

**Two things taken from the survey rather than invented.**

- *Tiling.* Both vendored trees already ship tuned tables **for this exact GPU
  and this exact geometry**: SGLang's `E=256,N=512,device_name=NVIDIA_RTX_PRO_
  6000_Blackwell_Server_Edition.json` (bf16) and vLLM's `...,dtype=fp8_w8a8,
  block_shape=[128,128].json`. Both are Qwen3.6-35B-A3B's shape. Transcribed
  rather than guessed. The tables are visibly shaped by the 100 KB SM:
  `BLOCK_N <= 128`, `BLOCK_M <= 64` until M is in the thousands, where the H100
  tables in the same directories use 128x256x128.
- *The shared-memory clamp.* For geometries the tables do not cover, stages are
  reduced until the pipelined A/B tiles fit 96 KB. SGLang gates its defaults on
  `shared_memory_per_block_optin < 128 KB`; TRT-LLM's
  `cutlass_heuristic.cpp:558` does the same thing for its sm_120 CUTLASS tiles,
  with a comment naming this GPU and keeping only 128x128x64. vLLM has no such
  guard, which is the one place its defaults are wrong here.

**Result — numerics.** `unieinfra-kernel-sm120`, new module `moe.py`, two
registrations at `Priority.PLUGIN` (16). Bar is cosine > 0.9999.

| case | reference | max abs | mean abs | cosine |
|---|---|---|---|---|
| bf16, T=1 | in-tree cutlass | 7.8e-03 | 1.2e-03 | 0.9999871 |
| bf16, T=64 | in-tree cutlass | 7.8e-03 | 1.2e-03 | 0.9999869 |
| bf16, T=512 | in-tree cutlass | 1.2e-02 | 1.2e-03 | 0.9999869 |
| bf16, T=512 | fp32 oracle | 9.9e-03 | 1.1e-03 | 0.9999910 |
| gemma-4 shape, T=512 | in-tree cutlass | 1.6e-02 | 2.0e-03 | 0.9999871 |
| fp8 block, T=512 | W8A8-matched oracle | 3.2e-02 | 3.4e-03 | 0.9999163 |
| fp8 block, T=512 | bf16 oracle | 1.2e-01 | 1.7e-02 | 0.9979854 |
| EP=2, rank0+rank1 | full fp32 oracle | 1.0e-02 | 1.2e-03 | 0.9999894 |

The fp8 rows are the point of reporting two references. Against a plain bf16
oracle it looks like a 0.998 failure; against an oracle that applies the *same*
fp8 rounding the kernel applies, it is 0.99992. The 4.4% L2 gap is the W8A8
recipe, not the kernel — which is exactly the distinction "the output looks
reasonable" cannot make.

**Verdict. Landed for FP8, provisional for bf16.** The FP8 registration fills a
hole where both in-tree candidates raise, so there is nothing to lose to. The
bf16 registration displaces a kernel that *works* (cos 0.999996) purely on an
arch-portability argument — no FlashInfer JIT, no startup autotune, and tiles
from a table tuned for this device, whereas the cutlass path logs
`No tuned config covers trtllm::fused_moe::gemm1 ... falling back to tactic=-1`
at every token count tested. **That argument is unproven.** If a throughput A/B
says otherwise, drop the bf16 registration and keep the FP8 one.

**Not measured.** Any throughput number, for either registration. Per the
contract that makes this a verified port and not a performance finding.

**Still broken after this.** Three gaps the plugin deliberately does not cover,
because covering them dishonestly is worse than not covering them:
`routing_mode="kernel_routing"` for fp8 still selects the broken
`flashinfer_trtllm_fp8_moe_apply`; fp8 with `ispp` not a multiple of 128 (the
gemma-4 shape, ispp=704) falls through to the broken
`flashinfer_cutlass_fp8_moe_apply`, because the loader's w13 scale grid is
`2 * ceil(ispp/128)` rows which only tiles the N axis when ispp is 128-aligned;
and `triton_mxfp4_moe_apply` remains the only mxfp4 path and remains
uncompilable on sm_120.

**Two more surveys, completed after the first write-up.** Modular MAX and the
mlsys2026 contest repo were missing from the first pass; both are in the skill's
table and both changed nothing about the design, which is itself worth recording:

- **Modular MAX has no sm_120 MoE path at all.** `_is_sm12x_gpu` exists in
  `gpu/host/info.mojo` and is used in exactly two files, neither of them the MoE
  or grouped-matmul path. `grouped_matmul`'s dispatch forks four ways —
  `== H100` (WGMMA+TMA, ~192 KB smem config), `_is_sm10x_gpu` (tcgen05+TMEM,
  grid hardcoded to `B200.sm_count`, budget hardcoded to
  `B200.shared_memory_per_multiprocessor`), an AMD path, and `else`. sm_120 lands
  in `else`, which is `naive_grouped_matmul`: one scalar dot product per thread,
  no shared memory, no tensor cores. The `_is_sm12x_gpu` docstring says the quiet
  part out loud — consumer Blackwell "has no SM100 warp-specialized path".
  Two ideas there *are* portable and were not used: the persistent ragged
  `TileScheduler` (needs no tcgen05) and the single 64-bit atomic that fuses
  slot-assign with the prefix sum.
- **The contest repo contains no implementations** — only task prompts and two
  skills. Its `moe-fp8` track targets B200 and names `tcgen05`/TMEM explicitly.
  It does name two directions this port did not take: GEMM1/SwiGLU epilogue
  fusion, and expert compaction.

**Reachability, and it is worse than the skill's trap 3.** `solution=` is a hard
filter, and for MoE the requestable set is a **closed enum**: `MoeBackend` is
`{auto, triton, gluon, flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl,
deep_gemm_mega_moe, mega_moe}`. No plugin member exists, so these kernels are
reachable only under `--moe-backend auto`. Any explicit `--moe-backend` in a
benchmark silently measures something else. Same closed-allowlist shape as the
capability-keyed dicts that omit 120.
Separately: `--moe-backend triton` for a bf16 MoE on sm_120 raises
`NoKernelFoundError`. There is no in-tree NVIDIA Triton unquantized MoE apply —
only AMD ones and the mxfp4/mxfp8 entries. That flag is simply broken here.

**Awkward cases, added after auditing against §5.** fp16 was declared in
`signatures` and untested; it passes at cosine 0.9999999 (T=1/33/130). Token
counts that are not multiples of `BLOCK_M` (3/17/63/129/511): 0.9999909–0.9999911.
Degenerate routing with every token sent to one expert — 255 empty expert runs
and one with `T * top_k` rows: 0.9999920–0.9999934. Non-contiguous hidden states
(`stride(0) = 2 * hidden_size`): 0.9999910. `T = 0` returns `(0, 2048)` without
launching. Unpinned selection (no `solution=`) reaches the plugin and matches.

**Wiring gap.** `unieinfra_kernel_sm120/__init__.py::register()` does not call
`register_moe()`. Until it does, the module is inert in the live engine — the
same class of gap as `discover_plugins()` in E24, one level down.

**What was not measured.** Every throughput and latency number, for both
registrations. No `ncu`, no kernel_bench, no steady_bench. This entry is a
verified port and a set of bug findings; it contains no performance claim, and
the bf16 registration's rationale is explicitly an unproven one.

## E33 — The Gemma 4 KV truncation is fixed, and the guard that makes it impossible found a second silent bug

**Objective.** Fix the silent KV corruption E28 found on Gemma 4 (full-attention
layers write 2048 elements into a 4096-element row and the store *succeeds*),
and — the part that matters more than the fix — make a write whose element count
does not match the row it lands in **raise** rather than truncate.

**Path.**

1. *Reproduce it myself before touching anything.* Built the real
   `MHATokenToKVPool` at both checkpoints' geometry, read straight out of
   `/workspace/models/google/gemma-4-{31B-it,26B-A4B-it}/config.json`
   (`text_config.num_key_value_heads` x `head_dim` for sliding,
   `num_global_key_value_heads` x `global_head_dim` for full), and exercised
   `set_kv_buffer` on one layer of each kind.
2. *Read the heterogeneous-KV path* (`layer_kv_head_counts`, `_layer_row_view`,
   `_layer_heads_per_rank`, `_check_slab_guards`) and how Inkling turns it on
   (`registry.py::_apply_inkling_hetero_kv`, gated on
   `_INKLING_ARCHITECTURES`).
3. *Validate the proposed fix before implementing it*, then implement, then
   verify against HF transformers.

**Result 1 — reproduced, and it is worse than "the tail stays zero".**

| checkpoint | sliding row | full row | store on a full layer | tail |
|---|---|---|---|---|
| gemma-4-31B-it | 16 x 256 = 4096 | 4 x 512 = 2048 | **no exception** | 2048 elems, **not zero** |
| gemma-4-26B-A4B-it | 8 x 256 = 2048 | 2 x 512 = 1024 | **no exception** | 1024 elems, **not zero** |

`layer_kv_head_counts` is `None` for both. The tail is not zero because the
hybrid-slab layout aliases layer 0 (`sliding_attention`, occurrence 0) and
layer 5 (`full_attention`, occurrence 0) onto the *same slab tensor*: the dead
half of every full-attention row holds a **different layer's KV**. E28 saw
zeros only because it probed the full layer first.

**Result 2 — the six-line claim is wrong, but the reasoning behind it is right,
and better than it knew.**

The proposal was to normalise both geometries onto a common head dim of 256 and
teach `_layer_row_view` to return `(-1, 4, 512)`. The normalisation is exactly
right — `4 x 512` and `8 x 256` are the same bytes in the same order, so the
existing head-*count* path expresses Gemma 4 with no new pool geometry at all.
But the `(-1, 4, 512)` view is **unnecessary**: `backends/mha.py::_get_kv_cache`
already re-views the pool's tensor at the layer's own
`(tp_k_head_num, qk_head_dim)`, so what the pool must get right is the row
**width**, not the head split. `(-1, 8, 256)` and `(-1, 4, 512)` produce a
byte-identical `k_cache` after that re-view — verified, not argued.

So the six lines in `_layer_row_view` are **zero** lines. The real surface is
elsewhere:

- `kv_cache/utils.py`: `normalize_layer_kv_geometry()` — the re-count, and a
  hard `ValueError` when the geometries have no common head dim (a model that
  cannot ride this path must not be silently approximated).
- `kv_cache/mha.py`: `_validate_layer_kv_head_counts()` — refuse counts whose
  reinterpretation is not *exact*. `head_num % heads_l != 0` still reshapes
  successfully when `rows * head_num` happens to divide, and then every layer
  row straddles two slab rows. Same bug class, one level down.
- `kv_cache/mha.py`: `_assert_store_row_geometry()` — **the guard** (below).
- `registry.py`: `_apply_gemma4_hetero_kv()` + `gemma4_layer_kv_geometry()` —
  without an enablement gate the pool stays uniform and the fix is unreachable.
  `_apply_inkling_hetero_kv` was the model.

**Result 3 — the guard, which is the point.**

`store_kv_cache` copies `src.numel() // n_tokens` elements per token at
`dst.stride(0)` and **never compares the two**. Narrow source: truncation.
Wide source: overrun into the next token. Neither raises. The check now lives on
the pool (which owns the row geometry) immediately before every store, on the
same views the store targets — base `set_kv_buffer`, the MXFP8 one, and
`quantize_and_set_kv_buffer`. Message names the layer, its type, both element
counts, and whether hetero KV is on.

It is deliberately *not* in the kernel: `tokenspeed-kernel` is owned elsewhere,
and the pool is the only party that knows what a row means.

**Result 4 — verified against HF transformers, at three levels.**

*Round-trip, both checkpoints' real geometry* (page 128 sliding / 256 full,
slot 256, aliased slab):

| checkpoint | layer | width served | K/V round-trip | slab-mate intact |
|---|---|---|---|---|
| 31B | 0 sliding 16x256 | 4096 | exact | — |
| 31B | 5 full 4x512 | 2048 | exact | yes |
| 26B-A4B | 0 sliding 8x256 | 2048 | exact | — |
| 26B-A4B | 5 full 2x512 | 1024 | exact | yes |

*Whole-model, against HF `Gemma4TextModel` (eager) with identical weights.*
The 31B does not fit beside the other tenants on GPU 3, so this is a tiny
random-weight model that keeps the shape signature that causes the bug
(4x256 sliding / 1x512 full, 5:1, `attention_k_eq_v`, proportional RoPE), run
through TokenSpeed's real `Gemma4Model` with the real hetero pool in the loop:

| configuration | max abs diff | cosine |
|---|---|---|
| A. hetero KV ON (fixed) | 1.16e-04 | **1.00000000** |
| B. hetero OFF, guard bypassed (**pre-fix**) | 1.93e+00 | 0.90263206 |
| C. hetero OFF, guard active | — | **raises `ValueError`** |
| D. hetero ON, kernel's default softmax scale | 4.53e+00 | 0.43927708 |

Row B is the reproduction of the bad output at model level; row A is the fix.
Row D is a **second, independent bug** — see below. (Passing 5 sliding layers
and 1 full layer of a 6-layer stack through the same aliased-slab pool means
row A also exercises `gemma.py`'s own math against HF: RMSNorm without the
`1+w` convention, `layer_scalar`, `v = k_raw` before k-norm, and proportional
RoPE all match.)

*The new test fails before the fix.* With `kv_cache/mha.py` and
`kv_cache/utils.py` restored to `git HEAD` and nothing else changed:
**15 failed / 7 passed**, including `test_narrow_store_into_wide_row_raises` at
both geometries. After: **11 passed, 14 subtests passed**.

*No regression.* The 10 existing tests that touch the MHA pool or the hetero
path (`test_unified_kv_slab_pool`, `test_mxfp8_kv_pool`,
`test_kv_profile_layout_grid`, `test_mha_pool_spec_decode_groups`,
`test_flat_state_slabs`, `test_flat_host_{executor,mirror}`,
`test_inkling_activation_parity`, `test_group_aware_mha`,
`cache/test_mla_kv_buffer`) give **byte-identical results before and after**:
186 passed, 1 failed (`test_inkling_activation_parity`, failing identically at
`HEAD`). `test_deepseek_v4_config`: 105 passed / 2 failed, identical both ways.

**Result 5 — two of E28's three kernel blockers are gone, and the third is not
a kernel blocker.** Re-established on the current image (which has the in-tree
patches baked in) rather than inherited; every PASS is against a torch SDPA
reference, `softmax_scale=1.0`, bf16, page 128 (hd<=256) / 256 (hd=512):

| op | solution | hd=128 | hd=256 | hd=512 |
|---|---|---|---|---|
| `mha_decode_with_kvcache` | default | PASS .9999980 | PASS .9999988 | **PASS .9999991** |
| `mha_decode_with_kvcache` | `flashinfer` | PASS | PASS | **FAIL `NoKernelFoundError`** |
| `mha_decode_with_kvcache` | `triton` | PASS | PASS | PASS |
| `mha_prefill` | default | PASS | **PASS .9999992** | **PASS .9999994** |
| `mha_prefill` | `flashinfer` | FAIL | FAIL | FAIL |
| `mha_prefill` | `triton` | PASS | PASS | PASS |

E28 recorded decode hd=512 as failing by default and prefill hd=256/512 as
failing on shared memory. Neither reproduces here. Default decode at hd=512
passes *because* flashinfer's capability gate declines and selection falls
through to triton — a graceful fallback, not a hole. What is real:
`--attention-backend flashinfer` pins `solution="flashinfer"`
(`backends/mha.py:191`), and that **does** break Gemma 4's full-attention
layers.

**Result 6 — the guard's byproduct: `backends/mha.py` throws away
`layer.scaling`.** Grep it: `scaling` and `softmax_scale` appear **zero times**
in `backends/mha.py`. Every other backend (`mla`, `dsa`, `inkling`, `flashmla`,
`deepseek_v4`, `tokenspeed_mla`) passes `softmax_scale=layer.scaling`. The dense
MHA backend does not, so the kernel default `1/sqrt(head_dim)` always wins.
Invisible for every model shipped so far, because their scaling *is*
`1/sqrt(head_dim)`. **Gemma 4 sets `scaling = 1.0`** (no `query_pre_attn_scalar`;
q and k are unit-RMS after qk-norm), so it would silently get 0.0625 on sliding
layers and 0.0442 on full ones. Row D above: cosine **0.439** — a bigger error
than the KV bug this entry set out to fix. One line, in a file owned by another
agent; not touched. See hand-offs.

**Verdict.** Land the guard and the fix. The guard is the durable part: it turns
every future per-layer-geometry mistake in this pool from a confidently-wrong
model into a startup `ValueError` that names the layer and both element counts.
The Gemma 4 enablement is correct at the pool level and verified against HF, but
it does not make Gemma 4 servable on its own — see below.

**Still blocks Gemma 4 from serving.**

1. **`softmax_scale` (Result 6).** `backends/mha.py` must pass
   `softmax_scale=layer.scaling` to `mha_decode_with_kvcache` /
   `mha_extend_with_kvcache` / `mha_prefill`. Until it does, Gemma 4 is wrong at
   cosine 0.44 regardless of the KV cache.
2. **The MHA backend is single-geometry.** `self.head_dim` and
   `self.tp_kv_head_num` are set once from the model config (256 / 16 for the
   31B) and baked into `self.plan` and into every `fa2_decode_cuda_graph_plan(
   ..., self.tp_kv_head_num, self.head_dim, ...)`. Those FA2 wrappers are
   already *per-group* (`self._group_page_size(gid)`) but not per-geometry, so a
   full-attention layer's `(pages, 256, 4, 512)` cache is planned as
   `16 x 256`. Not exercised here — this entry never ran the real backend.
3. **`--attention-backend flashinfer` raises** on the full-attention layers
   (Result 5). The default (`mha`) does not.
4. **The 26B-A4B MoE variant is not implemented.** `gemma.py:380-386` documents
   it: `enable_moe_block=True` adds a routed-expert branch parallel to the dense
   MLP with two extra post-FFN norms. Its KV geometry is now handled; its FFN is
   not.
5. **fp8 KV bypasses the guard.** `_save_kv_cache` routes `is_fp8` to
   `fused_fp8_set_kv_buffer`, which takes `page_size=self.page_size` (the base
   page) and never reaches `set_kv_buffer`. `--kv-cache-dtype fp8` with a hetero
   layout is unguarded and probably wrong. mxfp8 is guarded but bails at
   `head_dim != 128` anyway.
6. **Radix-built scheduler ext.** The per-group page sizes this layout needs are
   only honoured by a flat-built `tokenspeed_scheduler`. The gate now declines
   (with a warning) on a radix build rather than emitting locs in the wrong
   units — which means the pool's guard then refuses the first full-attention
   write. Loud, but still not servable.
7. **Vision / audio towers.** Text-only, as documented in `gemma.py`.

**What was not measured.** Every throughput and latency number — none, for
anything. No full-checkpoint run of either model: with 59-83 GB in use by other
tenants on GPU 3, the 31B (~62 GB bf16) does not fit, so the model-level
reference comparison is a tiny random-weight stand-in with the same shape
signature, not the checkpoint. No real attention-backend forward: the
model-level comparison uses a stub backend that drives the real KV pool and then
does eager attention, so blocker 2 above is **read, not observed**. TP > 1 was
not exercised: `_layer_heads_per_rank`'s scaling is unit-tested arithmetic here,
not a distributed run. `test_slab_capacity_prefix_hits` was skipped — it boots
servers, which this task was told not to do. Whether the pre-existing
`test_inkling_activation_parity` and `test_deepseek_v4_config` failures matter
was not investigated; they are identical at `HEAD`.

**Hand-offs.**
1. **`backends/mha.py` (owned by another agent, currently modified):** pass
   `softmax_scale=layer.scaling` on all three MHA entry points. This is the
   largest remaining Gemma 4 error and it is one line per call site.
2. **`backends/mha.py`:** `self.head_dim` / `self.tp_kv_head_num` need to become
   per-layer (or at least per-cache-group) for any model with two head dims.
3. **`configs/flat_memory_plan.py` (owned by another agent):** nothing needed
   from it for this fix — `_apply_gemma4_hetero_kv` reuses Inkling's
   `slot_tokens` / `group_page_sizes` shape and
   `_floor_tokens_to_layout_grid` unchanged. Flagging only that Gemma 4 now adds
   a second consumer of that grid.
4. **`registry.py` is shared.** This entry added `_GEMMA4_ARCHITECTURES`,
   `gemma4_layer_kv_geometry`, `_apply_gemma4_hetero_kv` and one `elif` at the
   Inkling call site; another agent has concurrent edits to the same file's
   imports and `create_attn_components`.

## E34 — The GDN state row is priced per *block id*, not per page: `--mamba-track-interval` is worth up to 2x, and shrinking the attention page for FlashInfer costs 5x

**Objective.** Two claims arrived together: (1) decoupling the attention page
size from the GDN state row would promote paged decode from a priority-4 Triton
fallback to a priority-12 FlashInfer kernel "for free", because FlashInfer's
decode whitelist is `{16, 32, 64, 128}` and the equalizer forces 832; and (2)
`--mamba-track-interval` is dead code whose help text ("interval to track the
mamba state during decode") describes the fix for a 3.5x KV-capacity deficit
against vLLM. Establish the geometry from the code, decide whether the two
granularities can be separated, and implement whichever is real.

**Which architecture this applies to.** `FLAT_KVCACHE = True` in
`tokenspeed:sm120-dev` (`python3 -c "import tokenspeed_scheduler as s;
print(s.FLAT_KVCACHE)"`). Everything below is the **flat** `KvCacheCoordinator`
path. The radix path keeps a separate `MambaChunkAllocator` pool and none of
this applies to it.

**Provenance warning, and it is the same class of error the last two agents
hit — inverted.** The image's installed package is
`/usr/local/lib/python3.12/dist-packages/tokenspeed`, built from
`/workspace/roy/tokenspeed` (HEAD `3f88dcc2`). The tree named in the task,
`UnieInfra/experiment/legacy/python/tokenspeed`, is a **quarantined fork**
(UnieInfra commit `9290433`, "chore: quarantine research residue under
experiment/"). The two have diverged: the live tree has
`configs/hybrid_cache_plan.py` and `configs/flat_cache_runtime.py`, the legacy
tree does not, and `mamba_track_interval` sits at `server_args.py:96` in the
live tree vs `:123` in the legacy one — the line numbers in the task brief are
the **live** tree's. `runtime/configs/flat_memory_plan.py` is byte-identical
between them; `paged_cache_spec.py`, `registry.py` and
`backends/hybrid_linear_attn.py` are not. The implementation below landed in
the legacy tree (the path the task assigns, and where the sibling
heterogeneous-KV agent is working); it was **verified against a port onto the
live files**, bind-mounted over the image's package, because the legacy tree is
not what the image runs.

---

### Path — the geometry, from the code

`registry.create_attn_components` calls `equalized_block_size`, which is
`components_from_layers` + `solve_page_geometry` in
`runtime/configs/flat_memory_plan.py`:

```
needed      = ceil(max_const / max_linear)          # widest state row / widest KV row
block_size  = alignment * ceil(needed / alignment)  # alignment defaults to the configured P
block_bytes = max(max_linear * P, max_const)
```

For Qwen3.5-27B (64 layers, `full_attention_interval` 4 → 16 full + 48 GDN,
`head_dim` 256, 4 KV heads, 48 linear value heads, `mamba_ssm_dtype` fp32):

| quantity | value |
|---|---|
| KV bytes/token/full layer (`max_linear`) | `2 * 4 * 256 * 2` = **4096** |
| conv row/GDN layer | `(2*16*128 + 48*128) * 3 * 2` = 61,440 |
| ssm row/GDN layer | `48 * 128 * 128 * 4` = 3,145,728 |
| state bytes/GDN layer (`max_const`) | **3,207,168** |
| `needed` | `ceil(3207168 / 4096)` = 783 |
| **equalized P** | `64 * ceil(783/64)` = **832** ✓ |
| block row | `16*832*4096 + 48*3207168` = **208,470,016 B** |
| … of which KV | 54,525,952 (**26.2%**) |
| … of which GDN state | 153,944,064 (**73.8%**) |

That reproduces the reported 208 MB page and 74/26 split exactly, and at a
budget of `163 * 208470016` = 31.65 GiB reproduces `max_total_num_pages = 163`
and the advertised 135,616 KV tokens.

**Where the state row enters, and it is not where the brief assumes.**
`FlatStateSlabs.allocate` (`layers/attention/kv_cache/flat_state_slabs.py:186`)
allocates `size // page_size + 1` rows of `(conv, ssm)` **per state layer**,
indexed by **page id** — and `KvCacheCoordinator` is documented as
"Multi-group fan-out over the per-attention managers, **one shared BlockPool**"
(`csrc/cache/kv_cache_coordinator.h:44`), with `block_pool_{config_.device_
allocator.total_pages}` in `scheduler.cpp:66` and
`AvailableKvPages()` reporting "one flat block is one pool row, interpreted at
each group's own block_size".

So the coupling is **not** "one state snapshot per attention page". It is:
**every block id in the single shared pool carries a KV page row in every
full-attention slab AND a recurrent-state row in every GDN slab, whichever
group actually holds it.** A T-token request draws `ceil(T/P)` block ids for
`full_attention` and another `~T/N` for `linear_attention`; each of those ids
costs a full 208 MB row. Resident cost is therefore

```
bytes/token = (1/P + 1/N) * (L_full * k * P + L_state * s)
```

which is what `flat_memory_plan.capacity_model` now writes down, bytes-exact
against `plan_component_tensors`.

---

### Design table — how the other engines price recurrent state

| | **TokenSpeed (flat)** | vLLM | SGLang | TensorRT-LLM (Cpp mgr) | lmdeploy (PyTorch) |
|---|---|---|---|---|---|
| where state lives | dense slab row per **block id**, shared BlockPool with KV | mode-dependent (`mamba_cache_mode ∈ none/align/all`) | **per-request slot**, own free-list (`MambaSlotAllocator`) | **per block** in the shared KV pool, with *placeholder* blocks for non-snapshot positions | per-request runtime slot + separate checkpoint slots (`StateManager`) |
| states resident per request | 2 live pages + retained snapshots | `1+spec` (none) / `2+spec` (align) / `cdiv(len, P)+spec` (all) | 1 (+ ping-pong buffer) | 1 live + snapshot blocks | 1 runtime (+ ≤1 decode checkpoint) |
| state-resumable prefix caching | **yes** | yes (`align` for GDN; `all` only for Mamba2 SSD — `qwen3_next.py:796` raises for GDN) | yes (`MambaRadixCache`) | yes | yes |
| page-size ↔ state coupling | **byte-equalize P so the KV row covers the state row** (`solve_page_geometry`) | byte-equalize the same way (`Platform._align_hybrid_block_size`, `interface.py:884-912`, then `mamba_page_size_padded`) | none — separate pools, split by `--mamba-full-memory-ratio` | `statesSnapshotInterval % tokensPerBlock == 0` (`kvCacheManager.h:262`) | checkpoint steps `% block_size == 0` |
| does a non-snapshot page still cost a state row? | **YES** — the dense slab is the whole finding | no (states are their own tensor) | no | **no** — "only snapshot-position blocks are real; intervening positions are placeholders" (`kvCacheManager.h:478`) | no |
| snapshot-interval knob | `--mamba-track-interval` (**was dead**) | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (sparse *retention*, `single_type_kv_cache_manager.py:1326`) | `--mamba-track-interval` (256, `% page_size == 0`) | `mamba_state_cache_interval` (256, `% tokensPerBlock == 0`) | `prefix_cache_decode_state_interval` (0 = off, `% block_size == 0`) |
| hit below a snapshot | **truncate** — bound aligns DOWN to the groups' lcm (`SweepThenConverge`) | truncate | truncate (`value = value[:best_value_len]`) | truncate (partial reuse explicitly disabled for hybrids) | truncate (`_match_state_checkpoint` refuses KV-only hits) |
| per-group tokens/block | supported (`PagedCacheGroupSpec.block_size`, gcd/lcm in `MakeCoordinator`) | supported (`resolve_kv_cache_block_sizes`, lcm/gcd) | v1 forces `page_size == 1` | supported | n/a |
| per-group **bytes**/block | not uniform, but the id space is | **forced uniform**, mamba padded up | n/a (separate pools) | uniform | n/a |

Two conclusions the table forces:

1. **Every engine that decouples the snapshot interval from the page does it by
   making the interval a multiple of the block size and truncating the prefix
   hit to it.** TRT-LLM's `statesSnapshotInterval % tokensPerBlock == 0` is
   literally the design TokenSpeed's dead flag was named for (SGLang, whose
   flag name TokenSpeed inherited verbatim, uses it for a decode-time donate
   interval instead). So: implement `--mamba-track-interval` as the **state
   group's own `block_size`**. No new C++ is needed — `MambaStateManager` is
   `SwaManager(spec.block_size, window=2)` with `RegistersAlignedFinalPageOnly`,
   so raising its block size *already* means "publish a snapshot only every N
   tokens".
2. **TokenSpeed is the only engine where a non-snapshot block still pays for a
   state row.** TRT-LLM's *placeholder blocks* are exactly the mechanism that
   makes the attention page size free to shrink, and TokenSpeed does not have
   them. That is what kills FINDING 1.

---

### Result 1 (NEGATIVE) — the attention page cannot be decoupled downward

At a fixed 31.65 GiB budget on Qwen3.5-27B, varying the attention page `P` and
the snapshot stride `N` (all numbers from `flat_memory_plan.capacity_model`,
bytes-exact against `plan_component_tensors`):

| `P_attn` | `N` | block bytes | pages | nominal tokens | B/token | **effective tokens** | vs today |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 154,992,640 | 219 | 3,504 | 19,374,080 | 1,753 | 0.03x |
| 64 | 64 | 158,138,368 | 214 | 13,696 | 4,941,824 | 6,876 | 0.10x |
| **128** | **128** | 162,332,672 | 209 | 26,752 | 2,536,448 | **13,396** | **0.20x** |
| 128 | 6,656 | 162,332,672 | 209 | 26,752 | 1,292,613 | 26,288 | 0.39x |
| **832** | **832** | 208,470,016 | **163** | **135,616** | 501,130 | **67,808** | **1.00x** |
| 832 | 6,656 | 208,470,016 | 163 | 135,616 | 281,886 | 120,547 | **1.78x** |
| 1,664 | 6,656 | 262,995,968 | 129 | 214,656 | 197,563 | 171,998 | 2.54x |

`P = 128` — the largest page in FlashInfer's paged-decode whitelist — is a
**5x capacity regression**, and the coarsest snapshot stride does not rescue it
(0.39x). The reason is one term: `L_state * s / P` is the state bytes charged
per token of attention history, 185 KB/token at P=832 and **1.2 MB/token** at
P=128. The equalizer's 832 is not arbitrary; it is close to the point where
that term stops dominating.

**What couples them, precisely.** Not the kernels, not the prefix cache, not
the CUDA-graph bucket shapes — all three already handle heterogeneous per-group
page sizes (`PagedCacheGroupSpec.block_size`, `flat_group_page_sizes`,
`compute_max_logical_pages_for_capture`; Inkling already ships full=256/swa=128
against a base of 128). The coupling is **two lines of layout**:

- `csrc/scheduler/scheduler.h:206` — `BlockPool block_pool_;`, one free list
  shared by every group, so a block id given to the attention group is drawn
  from the same space as one given to the state group; and
- `flat_state_slabs.py:190` — the `(conv, ssm)` slabs are **dense** over
  `size // page_size + 1` rows, so the state row exists for a block id whether
  or not the state group ever owns it.

Decoupling `P` downward requires either a second `BlockPool` (state ids from
their own space) or TRT-LLM-style placeholder blocks. Both are C++ changes in
`tokenspeed-scheduler`, outside this task's file ownership. **Getting
FlashInfer's paged decode on GDN hybrids is a scheduler change, not a page-size
change.** Recorded as a rejected approach, not attempted.

**Rejected approach 2: raising `P` instead.** The table says `P = 1664` is
2.54x. It is also up to 1,664 tokens of tail waste per request across a
262 MB block row — ~262 MB of dead memory per concurrent request. Not pursued;
no measurement of the fragmentation term was taken, so the 2.54x is an upper
bound that ignores it.

---

### Result 2 — `--mamba-track-interval`, wired

The flag now resolves to the **state group's page size** N:

```
flat_memory_plan.state_snapshot_interval(block_size=P, requested_interval=arg)
    -> smallest multiple of P that is >= max(P, arg)
paged_cache_spec.state_group_page_sizes(...) -> {"linear_attention": N}
    -> group_specs_from_layer_types(page_sizes=...) -> rows_per_page = block_size = N
```

Everything downstream already keyed off `rows_per_page`:
`compute_paged_cache_group_page_counts` sizes the state group at `T/N` instead
of `T/P`; `compute_max_logical_pages_for_capture` narrows the captured state
block table; `MakeCoordinator` computes `base = gcd(P, N) = P` and
`lcm = N`; `MambaStateManager(N)` publishes snapshots only at N-aligned chunk
ends. One consumer had to be corrected —
`MambaAttnBackend.set_kv_pool` indexed the state block table at the pool's
**global** page size, which with N ≠ P reads the wrong snapshot row; it now
reads `paged_cache_spec.state_page_size(specs, default)`.

**The default is a deliberate no-op.** The shipped default is 256, which is
*below* the equalized 832, so `state_snapshot_interval` returns 832 and the
geometry is byte-identical to before. The flag can only be opted into.

---

### The cost, stated plainly

The amortisation does **not** introduce a state-replay cost, because TokenSpeed
does not replay: the recurrent state evolves in place inside its working row
(`compute_state_page_indices` gives `state_in == state_out` within a window and
only crosses on a boundary), and a prefix hit that lands between snapshots is
**refused**, not reconstructed. `SweepThenConverge`
(`kv_cache_coordinator.cpp:73`) says so in its own comment: *"Bounds align DOWN
to align_tokens (the groups' lcm): a servable boundary must land on a whole
block of EVERY group."* Raising N raises the lcm.

So the price of `--mamba-track-interval N` is: **prefix-cache hit granularity
coarsens from P to N tokens, for every group, including attention.** At
N = 6656 on Qwen3.5-27B, a request sharing a 6,000-token prefix with a cached
one gets **zero** reuse and recomputes all 6,000 tokens through all 64 layers —
where today it would have reused 5,824 of them. That is strictly worse than the
"replay the recurrence from the nearest snapshot" the brief hypothesised, and
it is what the code does. It is the same trade every other engine makes
(SGLang truncates to `best_value_len`, TRT-LLM disables partial reuse for
hybrids, lmdeploy refuses KV-only hits), but the granularity is far coarser
here because P is already 832.

Concretely: `--mamba-track-interval` is a **long-context, low-prefix-sharing**
knob. On a RAG shape with a shared 4k system prompt it is a large regression
(E14/prefix-caching measured +31.5% from *enabling* prefix reuse on these
models); on independent long documents it is up to +78% resident capacity.
**No throughput or hit-rate measurement was taken for either shape.**

---

### Verification

**Environment.** Image `tokenspeed:sm120-dev`, GPU 3 only (shared: 46-73 GiB
resident to other tenants throughout, which is why the byte plan is pinned
below rather than compared raw — see E31). `FLAT_KVCACHE = True`. The change
was bind-mounted over the image's installed package as a port onto the LIVE
files (`docs/e34-mamba-track-interval-live-tree.patch`); the legacy tree's copy
is not what the image runs.

**Model.** Qwen3.5-2B (24 layers = 18 GDN + 6 full, `head_dim` 256, 2 KV heads,
16 linear value heads). Same GDN family as the 27B, equalized page **P = 576**,
block row **26,615,808 B** (26.8% KV / 73.2% state). Qwen3.5-27B does not fit
alongside a 46-73 GiB co-tenant, so the 27B numbers are the planner evaluated
on the real HF config; the 27B `max_total_num_pages = 163` / 135,616 tokens the
brief quotes is reproduced exactly by `plan_component_tensors` at a
31.65 GiB budget.

**Capacity — deterministic, from the published group specs.** The byte plan
(`max_total_num_pages`, `block_bytes`) is **unchanged** by the flag: the same
number of block ids are allocated, each the same size. What changes is how many
of them a request pins. `compute_paged_cache_group_page_counts` at
`max_live_requests=8`, `max_context_len=8192`:

Qwen3.5-27B, P = 832, `max_total_tokens` = 135,616:

| N | `full_attention` pages | `linear_attention` pages | total | B/token | vs N=P |
|---:|---:|---:|---:|---:|---:|
| 832 (today) | 172 | **172** | 344 | 501,130 | 1.00x |
| 1,664 | 172 | 91 | 263 | 375,847 | 1.33x |
| 3,328 | 172 | 50 | 222 | 313,206 | 1.60x |
| 6,656 | 172 | **30** | 202 | 281,886 | **1.78x** |
| 13,312 | 172 | 20 | 192 | 266,225 | 1.88x |

Qwen3.5-2B, P = 576, `max_total_tokens` = 57,600: 109/109 -> 109/22 at
N = 4608, 92,416 -> 51,984 B/token, **1.78x**. The engine logs exactly this at
startup:

```
GDN state snapshot stride: 4608 tokens (attention page 576, requested
--mamba-track-interval 4608); resident cost 92416 -> 51984 bytes/token
(1.78x capacity), prefix-cache hit granularity coarsens 576 -> 4608 tokens
```

and at N = 2304: `57760 bytes/token (1.60x capacity) ... 576 -> 2304 tokens`.
Both match the offline model to the byte. `Flat GDN KV profile: block_bytes=
26615808 ... max_total_num_pages=396` is identical across N, as designed.

**Correctness — and the first answer was wrong.** Greedy (`temperature 0`,
32 new tokens), 4 prompts, the last 7,318 tokens long so it crosses 12 snapshot
boundaries at N = 576, 3 at N = 2304 and 1 at N = 4608.

Round 1 (default `chunked_prefill_size`) said N = 2304 and N = 4608 both
**diverge** from N = 576 on the long prompt — the exact "fluent wrong text"
signature. Round 2 found why, and it is not the state index:

| run | N | prefill chunk split | output |
|---|---:|---|---|
| `L-noop` | 576 | one chunk, 7,325 | reference |
| `L-n2304` | 2,304 | 6,912 + 413 | differs from `L-noop` |
| `L-n4608` | 4,608 | 4,608 + 2,717 | differs from `L-noop`, **identical to `L-n2304`** |
| `C-576` | **576** | **4,608 + 2,717** (forced) | **byte-identical to `L-n4608`** |
| `C-2304` | 2,304 | 4,608 + 2,710 (forced) | byte-identical to `C-576` |
| `C-4608` | 4,608 | 4,608 + 2,717 (forced) | byte-identical to `C-576` |

The scheduler aligns the prefill chunk end to the state group's block size — it
must, or `RegistersAlignedFinalPageOnly` publishes no snapshot. So changing N
changes the chunk decomposition of the GDN recurrence. Holding the split fixed
at 4,608 + 2,717 and varying N from 576 to 4,608 gives **byte-identical
output**; holding N fixed at 576 and changing the split from one chunk to
4,608 + 2,717 gives **different output**. The divergence is fp reassociation in
chunked prefill, the same class as changing `--chunked-prefill-size`, and it is
**independent of the snapshot stride**. The three short prompts are identical
across every configuration and every pool size.

**Unit tests.** `experiment/legacy/test/runtime/test_mamba_track_interval.py`,
16 cases, all passing, plus the 18 pre-existing `test_flat_memory_plan.py`
cases still green. They pin: the multiple-of-P rounding; the default (256) is a
no-op against P = 832; only state labels move; `capacity_model` is bytes-exact
against `plan_component_tensors` (block 208,470,016, 73.8% state); the state
group's page count strictly decreases and the capture table narrows; and the
negative result — every FlashInfer-legal page (16/32/64/128) is below a quarter
of today's capacity.

**What this does NOT verify.** No throughput, latency or prefix-hit-rate
number. No 27B runtime. `max_total_num_pages` at a raw (unpinned) memory
profile is not comparable across runs on this card — it moved 649 -> 396 -> 234
between otherwise identical runs as co-tenants grew, which is E31 again.

---

### Interface exposed to the heterogeneous-KV agent

Neither `layers/attention/kv_cache/mha.py` nor `runtime/models/gemma.py` was
touched. Per-layer geometry is available from `runtime/configs/paged_cache_spec.py`:

```python
layer_group_ids(layer_types=..., sliding_window_tokens=...) -> list[str]
layer_page_sizes(layer_types=..., sliding_window_tokens=...,
                 page_size=P, page_sizes={...})            -> list[int]   # NEW
state_page_size(specs, default) -> int                                    # NEW
state_group_page_sizes(layer_types=..., page_size=P,
                       state_page_size=N) -> dict[str, int]               # NEW
```

`layer_page_sizes` returns one page size per layer, aligned with `layer_types`,
resolving each layer's group through the same `_layer_specs` path that
`group_specs_from_layer_types` uses — so a per-layer KV geometry and the
published group specs cannot disagree. `state_page_size` is the accessor any
consumer of a state block table must use instead of the pool's global
`page_size`.

### What was not measured

- Any throughput, latency or prefix-hit-rate number, at any N, on any shape.
  The capacity numbers are the planner's arithmetic and a live `max_total_num_
  pages`; the *cost* (coarsened hit granularity) is argued from the
  coordinator's own alignment rule and is **not** measured.
- Qwen3.5-27B at runtime. GPU 3 was shared (58 GiB resident to another tenant),
  so the 27B numbers are the planner evaluated on the real HF config, and the
  runtime check ran on Qwen3.5-2B (same GDN family, P = 576).
- The `P = 1664` fragmentation term (see rejected approach 2).
- FlashInfer paged decode was never exercised at any page size; FINDING 1 was
  rejected on capacity arithmetic before reaching a kernel.

---

## E33 — FA2 does run on sm_120, and it is *not* a faster kernel than FlashInfer; it wins the extend path only because the incumbent plans on every call

**Objective.** Settle by measurement a question that had been guiding decisions
on assumption: is a FlashAttention-2 kernel faster than FlashInfer's paged
prefill for MHA on sm_120, at Qwen3.5-27B geometry (24 q heads, 4 kv heads,
head_dim 256, GQA group 6, page_size 832, bf16)? An earlier swap of FlashInfer
in for Triton produced zero end-to-end change, but with a ±9.4% noise floor it
settled nothing. Prior established facts, not re-derived here: FA3 raises
`no kernel image is available` (sm_90 cubins), FA4 needs `tcgen05` which sm_120
does not have, and `flash_attn` in the image is an empty namespace directory.

**Path.**

*Getting an FA2 that runs.* Four routes were checked; three are dead. PyPI has
nothing usable — `vllm-flash-attn` is abandoned at 2.6.2 (torch 2.4/cu121) and
`flash-attn` 2.8.3 is sdist-only against an image with no `nvcc`. Building the
vendored source at the pinned commit would work but yields the identical binary,
for the reason below. sgl-kernel's `infllm_ops` genuinely compiles `sm_120a` FA2
SASS, but only bf16 head_dim 64/128 — this model is 256. What worked, in about
ten minutes: **transplant the extension from `vllm/vllm-openai:cu130-nightly`**,
which is torch 2.11.0+cu130 / py3.12 / cxx11-ABI, an exact match for
`tokenspeed:sm120-dev`. It loads unmodified. The `v0.26.0-cu129` image's copy
does not — it links `libcudart.so.12` against a CUDA-13-only image.

*The finding that reframes everything else.* `cuobjdump --list-elf` on the
shipped `_vllm_fa2_C.abi3.so` returns **52 `sm_80` cubins and nothing else**;
`--list-ptx` returns 52 matching `compute_80` images. The fork's CMake clamps
FA2 unconditionally — `cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" ...)`
— so building for `12.0` changes nothing. **"FA2 on sm_120" is the driver
JIT-compiling Ampere FlashAttention-2**: `mma.m16n8k16`, no TMA, no warp
specialisation. Nobody ships a native sm_120 FA2. First launch of each shape
costs 31 s (extend) and 53 s (decode) on a cold JIT cache; the driver caches it
under `CUDA_CACHE_PATH` thereafter.

*Correctness before speed* (skill §5). 22 comparisons against the in-tree Triton
kernels through the ops dispatch, plus a direct comparison against FlashInfer on
identical inputs. All pass at cosine > 0.99999 with `rel_l2` 2-3e-03 — which is
bf16 output rounding (2^-9 = 2.0e-03), confirmed by the fp16 rows dropping to
3.5e-04. Awkward cases specifically: 7 and 64 padding rows, ragged q and ctx
with partial last pages (831/832/833/1663), non-causal, fp16, `return_lse`,
speculative `q_len 4`, ragged contexts. Under CUDA-graph capture FA2 captures
and replays with **no plan and no prepare hook**, cosine 1.00000000 against
eager, and still tracks eager after `cache_seqlens` is mutated in place.

*Measurement.* `eval/kernel_bench.py`, cases added through a driver that swaps
`build_cases` rather than editing the shared file. Four candidates per case:
FA2, FlashInfer *as registered*, FlashInfer *`run()` only* (a bench-only
registration that caches the plan — the floor that path can reach, since the
runtime hoists one plan per step across all attention layers), and in-tree
Triton. **Card share was 0.32-0.47 throughout and drifted 0.47 → 0.36 within the
first sweep**; three other agents' processes were on GPU 3 at 425 W. So the
sweep was repeated with `--repeats 3` and ratios computed *within* each repeat,
which cancels a steady neighbour. Those ratios reproduce to a few percent while
the absolute numbers do not — 14 of 32 absolute figures were rejected by the
reproducibility gate, and none of them are quoted.

**Result.** Time relative to FA2 within the same repeat; >1 means slower.

| shape | FlashInfer as registered | FlashInfer `run()` only | in-tree Triton |
|---|---|---|---|
| extend b4 q96 ctx4096 | 13.68x | **0.61x** | 2.06x |
| extend b1 q2048 ctx4096 | 4.66x | 0.98x | 2.25x |
| extend b1 q4096 ctx4096 | 3.03x | 1.01x | 2.24x |
| decode b8 ctx4096 q1 | 40.3x | **0.60x** | 2.97x |
| decode b32 ctx4096 q1 | 6.89x | 0.99x | 0.97x |
| decode b64 ctx2048 q1 | 5.98x | **0.85x** | **0.84x** |
| decode b8 ctx4096 q4 | 9.36x | **0.17x** | **0.61x** |
| decode b32 ctx4096 q4 | 1.94x | **0.28x** | **0.42x** |

Prefill, where FlashInfer has no registered candidate: FA2 is **3.30x** faster
than `triton_mha_prefill` at b4/s1024 and **3.23x** at b1/s4096.

Where FlashInfer's time goes, measured separately: `plan()` 5.9-8.1 ms and
`_paged_kv_layout` 3.3-5.0 ms per call, against a `run()` of 0.67-7.1 ms. The
CSR rebuild is a boolean-mask gather (`page_table[keep]`), which is a
device-to-host sync — the decode path's provenance describes a sync-free scatter
for exactly this, and the extend path does not use it.

**Verdict — the question, answered plainly.**

*Is FA2 a faster kernel than FlashInfer's paged prefill here? **No.*** FlashInfer
`run()` is equal or faster at every shape swept: a tie within 2% at long extends,
and 1.6x-5.9x faster at short-query and decode shapes. FA2 wins only one column,
against Triton, and even there it loses at decode.

*Does the answer differ for extend versus decode? **Yes, decisively.***

- **Extend: FA2 wins the path, not the kernel — 3.0-13.7x.** That entire margin
  is host-side overhead the incumbent pays per call per layer, because nothing
  hoists the extend plan the way E27 hoisted the decode one. Until someone does,
  FA2 is the faster extend path by a wide margin, and it is also the only one of
  the three that accepts TokenSpeed's padded `q` — FlashInfer's extend kernel
  **raises** on it (`q.shape[0] (192) does not match qo_indptr[-1] (128)`),
  which is the exact failure its own provenance entry documents an
  absorber-sequence fix for. That fix is not in the code as it stands. Registered
  at `Priority.PLUGIN + 1` (17).
- **Decode: FA2 loses, and structurally.** `flash_api.cpp::mha_varlen_fwd` gates
  `set_params_splitkv` on `seqlenq_ngroups_swapped`, which is true only when
  `max_seqlen_q == 1`, under the comment "Only apply split-k for decoding"; the
  else-branch hard-checks `num_splits <= 1, "num_splits > 1 is not supported for
  varlen paged KV"`. So a speculative draft block gets **no KV parallelism at
  all** and wastes all but 4 rows of a 64-row query tile — 3.6-5.9x slower than
  FlashInfer and 1.6-2.4x slower than Triton at `q_len 4`. It cannot be forced
  on. Written and verified, **not registered by default**
  (`UNIEINFRA_FA2_REGISTER_DECODE=1`).
- **Prefill: FA2 wins outright, 3.2-3.3x**, with no FlashInfer competitor.
  Registered at 17.

*So why is vLLM 10-19% faster end to end?* Not because FA2 is a better attention
kernel — it is not. This closes that hypothesis. What FA2 does have is a calling
convention with no host-side scheduling in it, and on the extend path that is
worth 3-13.7x against how FlashInfer is currently wired here.

**Not wired.** `unieinfra_kernel_sm120/__init__.py::register()` does not call
`register_flash_attn2()`; these kernels are inert until that one line is added
to the module loop at the end of it. Same class of gap as E24's missing
`discover_plugins()` and E32's missing `register_moe()`. That file has a
separate owner and was not touched, per the task's constraints — nor were
`flashinfer_mha.py`, `gdn.py`, `gemm.py`, `moe.py` or `quant.py`.

**What was NOT measured.**
- **Anything at serving level.** No `steady_bench`, no `ladder`, no `period`, no
  end-to-end tokens/s. Every number here is kernel-level. The claim that this
  helps a real workload is *unproven*; a prefill-heavy shape should benefit most,
  and a decode-bound one should see nothing at all, since decode is unchanged.
- **`ncu`.** No occupancy, register or memory-pipe counters. The split-KV
  explanation for the `q_len 4` result is read off FA2's source and is consistent
  with the timings, but was not confirmed against a profile.
- **A free card.** Card share never exceeded 0.47. Absolute TFLOP/s and GB/s in
  the JSON are depressed by roughly that factor and are not comparable to any
  other run; only the within-repeat ratios above are.
- **Sliding window, logit softcap, attention sinks, the mxfp8 block-scaled
  path.** FA2 implements the first two natively and they are declared `False`
  anyway, because unverified.
- **`fp8` KV.** Not attempted.
- **The FA4 CuteDSL path already in this image.** `flash_attn.cute` ships
  `flash_fwd_sm120.py` with an SM80-MMA branch and `nvidia-cutlass-dsl 4.6.0` is
  installed; `import flash_attn.cute.interface` succeeds. That is the only
  *natively* sm_120-targeted FlashAttention on this box and it was not
  benchmarked. Given that FA2 here is JIT'd Ampere PTX and still ties FlashInfer
  at long extends, this is the most promising unexplored lead in the family.

## E35 — Retuning TokenSpeed's Triton MHA tiles for sm_120: the (64,64) that was borrowed to fit turns out to be optimal, and the win was hiding in `num_stages`

**目的 / Objective.** The Triton MHA prefill kernel had been crashing on sm_120
with `OutOfResources: shared memory, Required: 114688, Hardware limit: 101376`:
tile selection branched on architecture *family*, sm_120 satisfies
`is_hopper_plus`, and so it took the Hopper `(BLOCK_M, BLOCK_N) = (128, 64)`
branch that needs 112 KB. The fix in place branched on the device's reported
shared memory instead, which routes sm_120 into the sm_89 table — `(64, 64)` at
head_dim 256. That number was **chosen to fit, not to be fast**, and nobody had
swept the space. This kernel is the fallback path's floor: even once
FlashAttention-class kernels land, Triton is what runs whenever a request needs
a feature the fast kernel does not implement. So the floor is worth knowing.

Question: is `(64, 64)` actually the right tile, and what else is on the table?

### 方法 / Path

**1. Derive the shared-memory formula, so "what fits" is predictable rather than
discovered by crashing.** Reading `_fwd_kernel`, the query tile stays live across
every loop iteration; the key tile is dead after the first `tl.dot`, so its
buffer is reused for the value tile; the peak is at the second `tl.dot`, where Q,
V and the softmax probabilities P are all live simultaneously:

```
smem(ns=1) = 2·BM·(BD+BDPE) + max( 2·BN·(BD+BDPE), 2·BN·BDV + 2·BM·BN )
```

Checked against the original crash: `2·128·256 + (2·64·256 + 2·128·64)` =
`65536 + 32768 + 16384` = **114688**, exactly the number Triton refused to
launch. Verified exact for every dense shape swept, and conservative by ≤2 KB for
the MLA shapes that carry a non-zero `BLOCK_DPE`. Erring high is what you want
from a fit test.

The stage cost is separable and also exact — Triton's pipeliner spends the extra
stage on the **K loads only**:

```
smem(ns=2) = smem(ns=1) + 2·BN·(BD+BDPE)
```

confirmed to the byte on all six head-dim shapes (128/192/256/288/512/576).

**2. Survey the other engines** (`third_party/{vllm,sglang,lmdeploy,TensorRT-LLM,
tilert,mlsys2026-flashinfer-contest}`) before touching anything:

| engine | sm_120 rule for this kernel | `num_stages` |
|---|---|---|
| **SGLang** `kernels/ops/attention/extend_attention.py` (direct ancestor) | `cc==12`: Lq≤128 → (64,128); ≤256 → (64,64); else (32,32) | **1**, hard-pinned at the call site |
| **TRT-LLM** `_torch/attention_backend/triton_prefill.py` (fork of the same) | identical `cc==12` table | 1 |
| **lmdeploy** `pytorch/kernels/cuda/flashattention.py` | the *only* tree with an explicit sm_12x **and** head_dim-512 row: DK≤128 → 128/128; ≤256 → 64/…; ≤512 → 64-or-32/64 | **3 / 3 / 2** |
| **vLLM** | no head_dim-512 tile at all; its one tuned large-head branch is gated on `head_size==256 ∧ capability_family(100)` — sm_120 never reaches it | mostly Triton default |
| tilert | no Triton attention kernels | — |
| flashinfer contest repo | prompts/docs only, zero kernel source; all Blackwell content targets **sm_100**, nothing for sm_120 | — |

Two things fell out of that table. First, TokenSpeed, SGLang and TRT-LLM all
carry `num_stages = 1` and none of them ever revisited it — it is pinned at the
call site, outside the per-arch helper, so an arch-specific tuning pass never
touches it. Second, **lmdeploy is the only engine that pipelines on sm_12x**
(stages 2-3), and it is also the only one with a deliberate 512 row. That made
`num_stages` the first thing to sweep rather than the last.

**3. Sweep on the real GPU.** Target geometry Qwen3.5-27B: 24 q heads, 4 kv
heads, head_dim 256, page_size 832, bf16, paged extend. Every
`(BLOCK_M, BLOCK_N, num_stages, num_warps)` over
`{16,32,64,128,256}² × {1,2,3} × {2,4,8}` = 225 configs, at extend lengths
96/512/2048/4096, plus head_dim 128 and 512, plus `q_len=1`, plus the grouped
decode kernel. Timing under CUDA graphs to remove Triton's ~30 µs host launch
overhead.

### 效果 / Result

**What fits.** At head_dim 256, 93 of 225 configs launch and 105 die of
`OutOfResources` (the rest were predicted-out and skipped). At head_dim 512 only
**48** launch — and only six distinct tile shapes: (16,16), (16,32), (16,64),
(32,16), (32,32), (64,16). `(32,64)` and `(64,32)` miss by **exactly 1024 bytes**
(102400 needed vs 101376 available) — 1 KB of driver reservation is the entire
reason gemma-4's global-attention layers cannot use a wider KV tile on this card.

**head_dim 256 — the borrowed number was already right.** `(64, 64)` was the
fastest of all 93 launchable configurations at 512/2048/4096; the nearest rivals
`(64,32)` and `(32,64)`, both pipelined, came in 4-13% behind. It also cannot be
pipelined: at 73728 B, double-buffering K would ask 106496 B, over budget. **This
is a real negative result** — the tile that was picked because it fit is the tile
you would have picked if you had measured. Left unchanged.

One trap worth recording: the first, non-interleaved sweep reported
`(64,32,3,4)` beating `(64,64,1,8)` by 7% at L=4096. Re-measuring round-robin
reversed it (0.957x). The card was shared and the sweep took ~20 minutes, so
config order aliased onto drift. Same failure mode as the period-blocked-means
finding — a single-shot ranking across a long sweep is not a ranking.

**Everything else was leaving 1.2-2.6x on the floor, almost all of it in
`num_stages`.** Before/after, median of 9 interleaved rounds, batch 4:

| Lq | old tile | new tile | L96 | L512 | L2048 | L4096 |
|---|---|---|---|---|---|---|
| 128 | (64,128,**1**,8) | **(128,64,2,8)** | 1.42x | **2.60x** | 1.68x | 1.73x |
| 192 | (64,64,**1**,8) | (64,64,**2**,8) | 1.10x | 1.10x | 1.20x | 1.20x |
| **256** | (64,64,1,8) | **unchanged** | 1.00 | 1.00 | 0.98 | 1.00 |
| 288 | (32,32,**1**,8) | (32,32,**2**,8) | 1.07x | 1.35x | 1.24x | 1.26x |
| **512** (gemma-4) | (32,32,**1**,8) | (32,32,**2**,8) | 1.11x | 1.25x | 1.26x | **1.34x** |
| **576** (MLA) | (32,32,1,8) | **unchanged** | 1.00 | 1.06 | 1.01 | 1.00 |

At head_dim 128 the tile was also wrong, not just the stage count: turning
`(64,128)` on its side to `(128,64)` keeps the same footprint but streams the
prefix once per 128 query rows instead of once per 64, *and* frees enough room to
double-buffer K. sm_80 already uses `BLOCK_M=128` here; the small-shared-memory
parts had been given a narrower M tile than the budget actually required.

**gemma-4's head_dim 512 does run on this kernel** — which is not true of every
engine surveyed; vLLM has no 512 tile at all. It runs at `(32,32)`, one of only
six legal shapes, and is now 1.34x faster at 4K.

**`num_stages` cannot be raised blindly, and the sweep nearly shipped a
regression.** The `Lq > 256` branch also catches the MLA shapes, and `Lq=576`
(`BLOCK_DMODEL=512 + BLOCK_DPE=64`) needs **108544 B at ns=2** — it would have
crashed exactly the way the original bug did, on a different shape. So the stage
count is now *derived from the budget* rather than hardcoded, using the two
formulas above; `Lq=576` correctly stays at 1 while `Lq=288/512` go to 2. The
branching stays shared-memory-derived, per the fix it builds on.

**The decode kernel needs nothing.** Sweeping `_fwd_grouped_kernel_stage1`
(BLOCK_N × BLOCK_H × stages × warps, head_dim 128/256/512, contexts
1K/4K/16K) moves long-context latency by **≤1.5% in either direction** — it is
bandwidth-bound on the KV stream and the tile is not what limits it. The shipping
`(BLOCK_N=32, BLOCK_H=16, ns=2, warps=4)` is within noise of the best config at
every geometry. `mha_decode.py` was **not** touched.

**Numerics.** Against an fp64 PyTorch reference at every shape: cosine
**> 0.999998**, max abs 8e-3 to 1e-2 (bf16 round-off). Against the *old* tile:
**bitwise identical (max abs 0.0)** at Lq 192/256/288/512/576 — changing only
`num_stages` is a scheduling change, not a math change, which is exactly the
signal you want. Only Lq=128 differs (max abs 1.95e-3, cos 0.9999997), because
`BLOCK_N` changed 128→64 and the running-softmax accumulates in a different
order.

### Verdict

Landed in `tokenspeed-kernel/.../ops/attention/triton/mha_prefill.py`. Net: no
shape regresses, head_dim 128 gains 1.4-2.6x, head_dim 512 gains 1.1-1.3x,
head_dim 256 is confirmed-optimal and untouched, and the MLA path is protected by
a budget check instead of a hardcoded constant.

The transferable finding is that `num_stages` is where the win was, and it was
invisible to every ancestor of this code because it is pinned at the call site
*outside* the per-arch tile helper — SGLang, TRT-LLM and TokenSpeed all inherited
`num_stages = 1` and no per-arch tuning pass could ever have reached it. lmdeploy,
the one engine that puts stages inside its per-arch table, is the one engine that
pipelines on sm_12x. Tuning tables should carry every knob the launch takes, or
the ones left outside never get tuned.

### What was NOT measured

- **Anything at serving level.** All numbers are kernel-level. No
  `steady_bench`, no `ladder`, no end-to-end tokens/s. Whether a 1.7x on the
  head_dim-128 Triton prefill moves a real workload depends entirely on how often
  the fallback path is taken, which was not measured.
- **A free card.** GPU 3 carried another tenant holding 45-56 GB throughout.
  Absolute latencies are inflated roughly 2x versus an idle card and are **not**
  comparable to any other run; per-round spread reached 1.4x at head_dim 128 and
  3-4x at head_dim 512. Only the within-round interleaved medians above are
  trustworthy, which is why every headline number is one.
- **sm_86 / sm_89.** The retuned table is keyed on the shared-memory budget, so
  those parts (identical 101376 B) now inherit these tiles. The measurements are
  from sm_120 only. The mechanism is arch-generic and sm_80 already prefers
  `BLOCK_M=128` at head_dim 128, but this is **inherited, not verified**.
- **`ncu`.** No occupancy or memory-pipe counters. The "prefix streamed once per
  128 rows instead of 64" explanation is consistent with the timings and with the
  smem accounting, but was not confirmed against a profile.
- **A length-dependent `BLOCK_M`.** At head_dim 256 and extend length 96,
  `(32,64,2,4)` is **1.36x** faster than the shipped `(64,64)` — `max_len_extend`
  is already available at the call site, so a clamp is possible. Deliberately not
  landed: it doubles the JIT variant count and the same tile is 0.87x at 4K.
  Recorded here so the next person does not have to re-find it.
- **fp16, non-causal, sliding window, sinks, logit softcap, custom masks, fp8
  KV.** Swept causal bf16 only.

---

## E36 — The GPU is busy 98% of the time on both engines: the deficit is not kernels and not host time, it is tokens per GPU-second

**Objective.** Before any more kernel work: find out where TokenSpeed's serving
time actually goes on sm_120. The stated worry was two-sided — kernels
optimised but never selected, or an engine blocked on the host. A kernel-level
sweep had put decode at 98.7% of the memory roofline while E23 had TokenSpeed
10-19% behind vLLM 0.26 end-to-end, so the deficit had to be somewhere nobody
had looked. Establish, in order: (1) which kernels are *selected*, not
registered; (2) the GPU-busy / host split of a decode step, for TokenSpeed and
for vLLM; (3) the synchronisation points; (4) the same decomposition for vLLM.

**Setup.** Qwen3.5-27B bf16, `/workspace/models/Qwen/Qwen3.5-27B`, **GPU 1
alone** for every arm, sequentially, one server at a time. Image
`tokenspeed:sm120-dev` and `vllm/vllm-openai:v0.26.0-cu129`, both at the E23
defaults (`gpu_memory_utilization 0.88`, `max_num_seqs 160`, prefix caching on),
TokenSpeed launched with **no** `--attention-backend` so selection is free.
Isolation anchor taken on the card with the server idle: bf16 8192³ GEMM
**405.1 TFLOP/s** (free-card band 405-427, E31) and D2D copy **1462 GB/s**.
`sm%` excluded throughout (E22). All token counts are the per-request `usage`
chunk, not SSE chunk counts (E23).

**Instrument.** `TOKENSPEED_KERNEL_VERBOSE=1` for the census; `/start_profile`
with `activities=["CPU","GPU"]` on TokenSpeed and vLLM 0.26's
`--profiler-config '{"profiler":"torch", ...}'` — the same torch profiler, the
same chrome trace, the same analysis script on both sides. GPU-busy is the
**union** of device-event intervals (kernels ∪ memcpy ∪ memset) inside a step,
so overlapping streams are not double counted; both engines turned out to run
one device stream. Decode steps are delimited by a kernel that fires exactly
once per step (`fused_decode_input_prep_kernel` on TokenSpeed,
`fused_recurrent_gated_delta_rule_packed_decode_kernel` ÷48 on vLLM), and the
profiling window is **pure decode** — 32 requests fired at once with
`min_tokens=max_tokens=8000`, so no prefill enters the window and the per-step
wall needs no attribution. nsys was not used; the chrome trace was the only
instrument.

---

### Result 1 — the selected-kernel census: 5 of 35 operators, and the dominant one is `torch_mm`

35 `(family, mode)` operators and ~90 kernels are registered on this platform.
Exactly **five** are ever dispatched for this model, and the verbose log fires
once per cache miss so this is the complete set:

| operator | winner | pri | band |
|---|---|---|---|
| `attention.mha_decode_with_kvcache` | `triton_mha_decode_with_kvcache_cached` | 4 | PORTABLE |
| `attention.mha_extend_with_kvcache` | `triton_mha_extend_with_kvcache` | 4 | PORTABLE |
| `attention.gdn_decode_step` | `flashinfer_gdn_decode_step` | 12 | |
| `attention.gdn_chunk_prefill` | `triton_gdn_chunk_prefill` | 4 | PORTABLE |
| `gemm.mm` | `torch_mm` | 7 | REFERENCE |

The other **30 operators never run**: `dsa_*` (6), `kda_*` (4), `mla_*` (2),
`msa_*` (2), `rel_mha_*` (3), `mha_prefill`, `gdn_decode_mtp`,
`attn_merge_state`, `attn_res.fwd`, `embedding.rope`, `embedding.rope_mla`,
`gemm.bmm`, `moe.apply`, `moe.sigmoid_bias_topk`, `quantization.{fp8,
fp8_with_scale, mxfp4, mxfp8}`, `transform.hadamard_transform`. That is not a
bug — they belong to other architectures or other dtypes — but it means roughly
**85 of the 90 registered kernels are dead weight for the model this project
benchmarks**, and any kernel effort not aimed at those five operators cannot
move this number.

The one that matters most is the least TokenSpeed-owned: **`gemm.mm` resolves to
`torch_mm`**, i.e. `torch.mm` → cuBLASLt, in the REFERENCE band at priority 7.
Result 2 shows that path is **80% of decode GPU time**. Not one line of
TokenSpeed kernel code executes in it. cuBLAS answers with
`cutlass_80_tensorop_bf16_s16816gemm_*` — Ampere-generation CUTLASS — plus 144
`splitKreduce_kernel` calls per step, on a Blackwell card. vLLM gets exactly the
same treatment (Result 4), so this is a cuBLAS property, not a TokenSpeed one.

**The out-of-tree plugin: 2 of 11 selected, 1 fatal, 8 unreachable.**
`unieinfra-kernel-sm120` registers exactly 11 kernels. With the plugin
installed and no backend pinned:

| plugin kernel | outcome |
|---|---|
| `unieinfra_flashinfer_paged_mha_decode_sm120` | **selected** (pri 16) |
| `unieinfra_gdn_chunk_prefill_sm120` | **selected** (pri 16) |
| `unieinfra_flashinfer_paged_mha_extend_sm120` | **selected, and it kills the server** |
| `unieinfra_bmm_fp8_scaled_sm120` | never — `gemm.bmm` is not dispatched at all |
| `unieinfra_mm_fp8_blockscale_sm120` | never — bf16 signature selects `torch_mm` |
| `unieinfra_triton_fp8_block_moe_apply_sm120` | never — model has no MoE |
| `unieinfra_triton_unquant_moe_apply_sm120` | never — model has no MoE |
| `unieinfra_triton_quantize_fp8_sm120` | never — no fp8 |
| `unieinfra_triton_quantize_fp8_with_scale_sm120` | never — no fp8 |
| `unieinfra_flashinfer_quantize_mxfp4_sm120` | never — no mxfp4 |
| `unieinfra_flashinfer_quantize_nvfp4_sm120` | never — no nvfp4 |

The fatal one is not subtle. With the plugin installed and default selection,
the server **dies during startup warmup**:

```
File ".../unieinfra_kernel_sm120/flashinfer_mha.py", line 745, in flashinfer_paged_mha_extend
ValueError: q.shape[0] (16) does not match qo_indptr[-1] (1).
```

That is the exact padded-query failure E25 recorded as *fixed*; the fix is not
in `unieinfra-kernel-sm120` as it stands (`flashinfer_paged_mha_extend` slices
nothing and constructs a fresh `BatchPrefillWithPagedKVCacheWrapper` and
`plan()`s on every call). Every plugin-arm number below therefore forces
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_EXTEND_WITH_KVCACHE=triton_mha_extend_with_kvcache`
so the engine can boot. **As shipped today, installing the plugin makes
TokenSpeed unable to serve this model at all.**

---

### Result 2 — a decode step is 98.2% GPU-busy. TokenSpeed is not host-bound.

Batch 32, pure decode, 40 profiled steps, 39 usable inter-step intervals:

| | TokenSpeed in-tree | TokenSpeed + plugin decode | vLLM 0.26 |
|---|---|---|---|
| wall / step | **48.978 ms** (sd 0.022) | 50.250 (sd 0.343) | **50.844** (sd 0.018) |
| GPU busy, compute kernels | 48.114 (**98.2%**) | 48.205 (95.9%) | 50.152 (**98.6%**) |
| GPU busy, incl. copy/memset | 48.180 (98.4%) | 48.275 (96.1%) | 50.179 (98.7%) |
| GPU idle (host-attributable) | **0.798 ms (1.6%)** | 1.975 (3.9%) | **0.665 ms (1.3%)** |
| observed throughput | 665.6 tok/s | 653 | 640-646 |

The step wall's standard deviation is 22 µs on 49 ms — 0.04%. This is not a
noisy measurement.

Same picture on the mixed 1024-in / 256-out workload at C=32, where prefill
takes a large share of the wall: **TokenSpeed 97.5% GPU-busy, vLLM 97.9%**.

**The "engine may be CPU-blocked" hypothesis is dead at every operating point
measured.** Host code costs TokenSpeed 1.6% of a decode step and 2.5% of a
mixed step. Even eliminating it entirely would be worth under 2%.

---

### Result 3 — the ranked stalls, and what the 1.6% is actually made of

Per decode step, in-tree arm (48.978 ms):

| rank | stall | cost / step | % of step |
|---|---|---|---|
| 1 | GEMM below achievable bandwidth (see below) | **4.4 ms** | **9.0%** |
| 2 | inter-kernel gaps: 849 gaps × ~0.77 µs, i.e. ~830 kernel launches/step | 0.655 ms | 1.34% |
| 3 | pure Python / allocator gaps (no CUDA call in the gap) | 0.102 ms | 0.21% |
| 4 | `cudaDeviceSynchronize` | 0.018 ms | 0.04% |
| 5 | `cudaGraphLaunch` | 0.014 ms | 0.03% |
| 6 | `cudaLaunchKernel` | 0.005 ms | 0.01% |
| 7 | D2H copies (`cudaMemcpyAsync`, 2 pinned DtoH/step) | 0.002 ms | 0.004% |

Rank 2 is the largest thing that is genuinely "not compute", and it is **not**
Python: it is the ~0.8 µs bubble between consecutive kernels of a graph replay,
multiplied by the ~830 kernels a 64-layer step launches. vLLM pays the same
tax with a different shape — 1078 gaps at 0.57 µs = 0.615 ms/step. The only
lever on it is *fewer kernels per step*, not faster host code.

**The `.item()` / `.cpu()` search came back nearly empty on the decode path.**
`aten::item` runs 33×/step but costs 23 µs of CPU and blocks the GPU for
nothing measurable; there are exactly 2 pinned `Memcpy DtoH` per step, 1.6 µs
of device time. `model_executor.py:1090`'s
`int(results.output_lengths.sum().item())` is real but is off the critical path
(its own comment says the results are already synced). The FlashInfer
`plan()`-copies-three-int32-tensors sync that motivated the search **does not
appear in the in-tree arm at all**, because the in-tree decode kernel is Triton
and has no plan.

**It appears the moment the plugin decode kernel is selected, and it is the
whole cost of that kernel.** Plugin arm, same step:

| | in-tree | + plugin decode |
|---|---|---|
| GPU idle during `cudaMemcpyAsync` | 0.002 ms/step | **1.449 ms/step** |
| GPU idle during `cudaLaunchKernel` | 0.005 | 0.355 |
| pageable `Memcpy DtoH` (a hard sync) | 0/step | 3/step |
| full-attention kernel time, 16 layers | 2.944 ms (Triton) | 3.012 ms (FlashInfer) |
| **step wall** | **48.978 ms** | **50.250 ms** |

So the out-of-tree FlashInfer paged decode kernel is **2.3% slower on the GPU
than the priority-4 Triton kernel it displaces** at this geometry, and adds
1.45 ms/step of host-induced GPU idle on top. Net **+2.6% slower end to end**,
653 vs 666 tok/s. This is the first serving number any of the eleven
out-of-tree kernels has produced, and it is negative.

---

### Result 4 — vLLM, decomposed the same way: it is *behind* TokenSpeed on the decode step

Same batch, same window construction, same script:

| per decode step, batch 32 | TokenSpeed | vLLM 0.26 |
|---|---|---|
| linear-layer GEMM (cuBLAS + splitK) | **39.45 ms** | 41.61 ms |
| GDN linear attention, 48 layers | 4.275 | **4.142** |
| full attention, 16 layers | **2.944** (Triton `_fwd_grouped_kernel_stage1`) | 3.198 (FA2 `flash_fwd_splitkv` + combine) |
| everything else on device | 1.44 | 1.20 |
| host-attributable idle | 0.798 | **0.665** |
| **total** | **48.978** | 50.844 |

Two things fall out of this table that overturn earlier framing.

**The GEMMs are the same code and TokenSpeed drives them better.** Both engines
land on `cutlass_80_*` Ampere-class cuBLAS kernels. Linear-layer weight bytes
read per decode step, computed exactly from the checkpoint shapes (48 GDN layers
× 766.6 MB + 16 full-attention layers × 744.6 MB + lm_head 2.543 GB) =
**51.25 GB**. Against the measured 1462 GB/s:

| | GEMM time | achieved | % of achievable |
|---|---|---|---|
| TokenSpeed | 39.45 ms | 1299 GB/s | **88.9%** |
| vLLM | 41.61 ms | 1232 GB/s | 84.2% |

Whole-step: TokenSpeed 1047 GB/s (71.6% of achievable), vLLM 1009 GB/s (69.0%).
**The "decode runs at 98.7% of roofline" claim does not survive at the serving
level** — the step-level figure is 72% — but the shortfall is shared with vLLM
and slightly smaller on TokenSpeed's side.

**The priority-4 Triton fallback beats FlashAttention-2 here.** TokenSpeed's
full-attention decode is 2.944 ms for 16 layers against vLLM's 3.198 ms, and
TokenSpeed's window was at the *longer* context of the two (~2 500 vs ~1 850
tokens), so the margin is understated. E23's caveat that this measures
"TokenSpeed-on-fallback against vLLM-on-normal" is, for the decode path, not a
handicap at all.

---

### Result 5 — where the deficit actually is: batch 53 vs 64, and one prefill sequence at a time

If the decode step is faster and the GPU is 98% busy, the 8-10% end-to-end
deficit has to be in how many useful tokens each GPU-second carries. It is.
Mixed 1024-in / 256-out, steady-state windows only (t ≥ 100 s of a 300 s run):

| C | TokenSpeed | vLLM 0.26 | |
|---|---|---|---|
| 32 | 457-461 tok/s | 465-468 tok/s | vLLM +1.6% |
| 64 | **591.9 tok/s** | **639.4 tok/s** | **vLLM +8.0%** |

which reproduces E23 (582.0 / 641.1, +10.2%) on a different day and a different
card. And the scheduler logs say why:

```
TokenSpeed  Decode batch. #running-req: 53, #pages(active/cached/total): 159/160/163,
            page ratio: 0.98, #queue-req: 11              [78 of 82 sampled decode batches]
vLLM        vllm:num_requests_running 64.0   vllm:num_requests_waiting 0.0
            vllm:num_requests_waiting_by_reason{reason="capacity"} 0.0
```

**TokenSpeed runs 53 of the 64 offered sequences and queues 11, permanently,
with its page pool 98% full. vLLM runs all 64 with nothing waiting.** KV
capacity: TokenSpeed 163 pages × 832 = **135,616 tokens**; vLLM **478,162
tokens** — 3.5×. Per byte: TokenSpeed's pool is 163 × 208.5 MB = 34.0 GB for
135,616 tokens = **250 KB/token**, against roughly 63 KB/token on the vLLM side.
The cause is already established elsewhere in this log — 74% of each 208 MB /
832-token page is a GDN recurrent-state snapshot (E34, which prices that row per
*block id* and lands `--mamba-track-interval`) — but this entry is the first
measurement of **what it costs at the serving level**: an 11-sequence batch
deficit at C=64, on a step whose wall is set by the 51.25 GB of weights and
barely moves with batch size.

**And prefill runs one sequence per forward pass.** Across 300+ observed
prefill batches, without exception:

```
Prefill batch. #new-seq: 1, #new-token: 1025, #cached-token: 0, #running-req: 1, #queue-req: 10
```

Ten requests queued, a 7488-token budget (`chunked_prefill_size` 8192 floored to
9 × 832 by `aligned_max_scheduled_tokens` — the `#new-token: 7488` rows in the
same log confirm the budget is real), and one 1025-token prompt admitted. With
`enable_mixed_batch=False` the scheduler additionally `break`s out of the decode
branch as soon as a prefill is pushed
(`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:1232,1246`), so
**decode is fully suspended for the whole prefill**. Under load those prefills
arrive every ~339 ms; idle, the same 1024-token prefill takes 217 ms.

What that costs is visible in TokenSpeed's own prefill scaling (C=1, idle
server, unique leading prefixes, median of 9):

| ILEN | TokenSpeed TTFT | vLLM TTFT | TokenSpeed MFU |
|---|---|---|---|
| 1024 | **216.7 ms** | 236.6 ms | 255 TFLOP/s = 63% |
| 2048 | 363.5 | **350.5** | 304 TFLOP/s = 75% |
| 4096 | 683.8 | **677.3** | 323 TFLOP/s = 80% |
| 8192 | 1452.3 | **1323.3** | 305 TFLOP/s = 75% |

A 1024-token prefill runs at 63% of the card's measured 405 TFLOP/s; the same
work packed to 4096 runs at 80%. Packing the queue into the budget the scheduler
already has is a ~20% cut in prefill GPU time, and mixed batching would stop
decode being suspended while it happens. Neither is a kernel.

---

**Verdict.** The GPU is busy 98.2% of a decode step and 97.5% of a mixed step.
Host time is 1.6% and there is no significant synchronisation stall on the
in-tree decode path — the one large sync in the whole study is *introduced* by
an out-of-tree kernel. TokenSpeed's decode step is 3.7% **faster** than vLLM's
at the same batch, its GEMM path reaches 88.9% of achievable bandwidth against
vLLM's 84.2%, and its priority-4 Triton attention kernel beats
FlashAttention-2 at this geometry. None of the remaining deficit is inside a
kernel.

It is in tokens per GPU-second. Ranked, with the evidence:

1. **KV bytes per token — worth ~+21% at C=64, at zero GPU cost.** Batch is
   capped at 53/64 by a 98%-full page pool while vLLM runs 64/64 with 3.5× the
   token capacity. The decode step is weight-bound (51.25 GB of weights against
   a batch-32 KV working set), so restoring the missing 11 sequences lengthens
   the step by almost nothing and adds 21% of tokens. E34 has already priced and
   landed the mechanism (`--mamba-track-interval`); **this is the measurement
   that says to spend the next hour finishing it and re-running C=64**, not on
   kernels.
2. **Prefill packing and mixed batching — worth ~20% of prefill GPU time.**
   `#new-seq: 1` against a 7488-token budget, with decode suspended for the
   duration. TokenSpeed's own TTFT curve prices the packing at 63% → 80% MFU.
3. **Retract or fix the plugin's extend kernel before anything else ships.** As
   registered it is selected by default and prevents the server from starting;
   the decode kernel next to it is a measured 2.6% regression. Eight of the
   eleven cannot be reached by a dense bf16 model at all.
4. **Kernels, last.** The entire headroom inside the GPU is the 4.4 ms/step
   (9.0%) between cuBLAS's 1299 GB/s and the card's 1462 GB/s — a cuBLAS tile
   selection problem in the REFERENCE-band `torch_mm` path, not a TokenSpeed
   kernel, and one on which TokenSpeed is already ahead of vLLM. The second
   item is the 0.655 ms/step of inter-kernel bubbles, addressable only by
   launching fewer than ~830 kernels per step.

**What was not measured.**
- *No nsys, no CUDA events.* The torch-profiler chrome trace was the sole
  instrument; kernel durations inside CUDA graphs are CUPTI's word and were not
  cross-checked against an independent timer.
- *No prefill kernel-level split.* Prefill is characterised by TTFT and by the
  mixed-window GPU-busy fraction only; the per-kernel breakdown of a prefill
  step was not produced for either engine.
- *The two throughput estimates in the verdict are arithmetic, not A/Bs.* +21%
  from batch 53→64 follows from the measured decode step being weight-bound;
  ~20% from prefill packing follows from TokenSpeed's own TTFT curve. Neither
  was run.
- *The mechanism behind `#new-seq: 1` was not isolated.* The loop in
  `forward.cpp:1188-1214` permits several prefills per batch and the budget is
  demonstrably 7488, so the refusal happens inside
  `schedulePrefillFirstChunk`; which predicate declines was not traced.
- *vLLM's KV bytes/token* is inferred from its reported 478,162-token cache and
  the memory budget, not measured directly.
- *Concurrency 1, 32 and 64 only; one model; bf16 only; TP=1.* C=128 does not
  fit TokenSpeed's page pool at this prompt length and was not attempted.
- *The plugin arm used a snapshot* of `unieinfra-kernel-sm120` taken at
  18:13 while another agent was editing that tree, with its extend kernel
  overridden to Triton so the server could boot. It is not a clean read of
  whatever that tree contains now.
- *Sequence lengths differed between the two pure-decode windows* (~2 500
  tokens for TokenSpeed, ~1 850 for vLLM). This penalises TokenSpeed's
  attention time, so the attention comparison in Result 4 is a lower bound on
  its margin; the GEMM comparison is unaffected.

---

## E37 — Placeholder blocks: the GDN state row was priced per *shared* block id; giving it its own id space costs zero bytes and takes C=64 from 53 running to 64

**Objective.** E36 priced the deficit: at C=64 TokenSpeed runs 53 of 64
sequences with 11 permanently queued and its page pool 98% full, while vLLM
runs 64/64. E34 established why — every block id in the single shared
`BlockPool` carries a row in *every* slab, so an attention page funds 48 GDN
state rows it never writes and a state snapshot funds 16 KV pages it never
reads — and landed `--mamba-track-interval`, which buys 1.78x by *coarsening
prefix-cache hit granularity from 832 to N tokens*. This entry implements the
third approach, the one that does not pay that price: TensorRT-LLM's
**placeholder blocks**.

---

### Design comparison — how five engines represent "no snapshot here"

Read from the vendored checkouts: TRT-LLM `bfb0fca859`, vLLM `d02df748b`,
SGLang `a690e5e0b3`, lmdeploy `9140d372`.

| | **TokenSpeed (flat, before)** | **TensorRT-LLM** | vLLM `"all"` | vLLM `"align"` | SGLang | lmdeploy (pytorch) |
|---|---|---|---|---|---|---|
| "no snapshot" representation | *none* — the slab is dense over the id space | `KVCacheBlock` with a **negative id** (−2,−3,…), `mIsPlaceholder=true`, pool index `nullIndex` (`kvCacheManager.h:478-483`) | n/a (dense) | shared **`null_block`, id 0**, `is_null=True` (`block_pool.py:187-191`) | `TreeNode.mamba_value is None` (`mamba_radix_cache.py:82`) | **absence** from `_state_checkpoint_index[(adapter, step)]` (`block_trie.py:219-222`) |
| where the empties live | — | `mAllPlaceholderBlocksById`, indexed by `abs(id)` (`kvCacheManager.h:1407-1410`), plus eviction **level 2** (`evictionPolicy.cpp:36-37`) | — | inline in `req_to_blocks[...]` | — | — |
| **is the state pool smaller than the KV id space?** | **NO** — the finding | **YES**: own `WindowBlockManager` at window `kRecurrentStates`, own tensor `{L, mNumPrimaryBlocks, 1, blockSize}` (`kvCacheManager.cpp:1137-1141`), sized `max_bs*pp+1(+draft) + max_tokens/interval` (`resource_manager.py:1955-1977`) | no | 2 (+spec) pages resident | own `MambaPool`, `--max-mamba-cache-size` | `max_batches + 2 + prefix_cache_state_budget` (budget default **0**) |
| read on a placeholder | — | `setOffsets` writes `KVCacheIndex::nullIndex` (`kvCacheManager.cpp:1352-1387`); a read there is a hard error, never silent | — | null block | node has no value | key absent |
| **hit needing a missing snapshot** | truncate (align DOWN to the groups' lcm, `SweepThenConverge`) | **truncate** to `latestMatchingNonPlaceholderBlockIdx` (`kvCacheManager.cpp:1979-1985`); partial reuse hard-refused for hybrids (`:611`) | truncate | truncate | truncate to `best_value_len` | hard miss, no fallback |
| interval knob | `--mamba-track-interval` | `mamba_state_cache_interval` = **256**, `% tokensPerBlock == 0` (`kvCacheManager.h:268-269`) | — | — | `mamba_cache_chunk_size` | `cache_checkpoint_interval` = 4096 (TurboMind only) |

Two conclusions the table forces, and they are the whole design:

1. **TokenSpeed already has the placeholder *representation*.** Its state
   group has its own block table with its own page size, and a position with
   no snapshot is already a null hole (page id 0) — `compute_state_page_indices`
   raises on reading one rather than resuming from zero. What TokenSpeed did
   **not** have is TRT-LLM's other half: a state pool *sized independently of
   the KV block-id space*. `blocksPerWindow[RECURRENT_STATES] = (max_snapshots, …)`
   is the piece that was missing.
2. **Not one of the five engines replays the recurrence across a gap.** All
   five truncate. See "what a request does when the snapshot is missing" below.

**Rejected: an indirection table (block id → state row) inside one pool.**
Modelled and dropped. Keeping the KV slabs dense while only the state slabs
are sparse leaves every *state* id paying for 16 unused KV pages: at the
27B geometry that is **1.58x**, not 2.0x. The symmetric fix — a second id
space — costs the same bytes and gets the whole factor, and needs no index
translation at all because each group's block table already indexes only its
own family's tensors.

---

### Path — what changed

`KvCacheManager` was already written pool-agnostically ("every operation acts
on the pool it is handed, identically for any tier"), so the change is at the
`CacheGroup`/coordinator layer, not below it.

- `KvCacheSpec.separate_state_pool` (`cache_types.h`) marks mamba-style state
  groups. `MakeSpecsFromConfig` sets it from `SchedulerConfig::state_allocator.total_pages`.
- `Scheduler` gains `flat_state_pool_` (`scheduler.h`), sized 1 and left
  unbound when off. `KvCacheCoordinator` takes it as its **last** parameter —
  placing it before `host_pool` would have silently re-bound twelve existing
  three-argument call sites' host pool as a state pool.
- Every `pool_` use in the coordinator becomes `devicePoolFor(i)`;
  `probeTierWithKeys` / `acquireTierWithKeys` now take a tier flag instead of a
  pool reference.
- `Acquire` and `AcquireDecodeDestination` check **each** pool
  (`BlocksNeededByPool` / `FreeBlocksByPool`); a total-vs-total check would
  admit a request the state pool cannot serve and trip their own post-check
  assert.
- Python: `plan_component_tensors(..., state_blocks_per_history_block=r)`
  sizes the state components against their own id space;
  `FlatStateSlabs(num_state_pages=...)` sizes the slabs from it;
  `capacity_model(..., placeholder_blocks=True)` writes down the arithmetic.
  `--enable-placeholder-blocks` / `--placeholder-state-block-ratio` (default
  1.0). **Default OFF; with the flag off the build is byte-identical.**

**The property that makes this cheap: at r = 1 nothing is reallocated.**
`plan_component_tensors` returns the same 163 history ids, the same 163 state
rows, the same 208,470,016-byte block row and the same 31.647 GiB total as
the shared id space. The tensors do not change size. What changes is that a
state snapshot stops consuming a KV page id and vice versa:

```
shared id space      bytes/token = (1/P + 1/N) * (kv + state) = 501,130
placeholder blocks   bytes/token =  kv/P + state/N            = 250,565
```

exactly 2.00x at N == P, at **zero** additional memory.

---

### Result 1 — capacity, offline and bytes-exact

Qwen3.5-27B (`/workspace/models/Qwen/Qwen3.5-27B`, 64 layers = 16 full + 48
GDN, head_dim 256, 4 KV heads, 48 linear value heads, fp32 SSM). Reproduced
E34's geometry to the byte before changing anything: P = **832**,
block_bytes = **208,470,016** (26.2% KV / **73.8% state**),
`max_total_num_pages` = **163** at the 31.647 GiB profile, 135,616 nominal
tokens. `capacity_model` is bytes-exact against `plan_component_tensors` at
every ratio.

| geometry | history ids | state ids | B/token | effective tokens | vs today |
|---|---:|---:|---:|---:|---:|
| shared id space (today) | 163 | 163 | 501,130 | 67,808 | 1.00x |
| **placeholder r = 1.0** | **163** | **163** | **250,565** | **135,616** | **2.00x** |
| placeholder r = 0.5 | 258 | 129 | 250,565 | 135,616 | 2.00x |
| placeholder r = 0.25 | 365 | 91 | 250,565 | 135,616 | 2.00x |
| `--mamba-track-interval 6656` | 163 | 163 | 281,886 | 120,547 | 1.78x *(E34, reproduced)* |
| both, N = 6656 | 163 | 163 | 88,665 | 383,248 | **5.65x** |

The two knobs compose: `r` stops history ids funding state rows, `N` lowers
the state pin rate. r only *moves* the split between the two pools; it does
not change bytes/token.

The engine logs the same numbers at startup:

```
Flat GDN KV profile: block_bytes=208470016 (112 component tensors, block_size=832), max_total_num_pages=163
Placeholder blocks ON: recurrent state moves to its own block-id space (163 history ids,
  163 state ids, ratio 1.000); resident cost 501130 -> 250565 bytes/token (2.00x capacity);
  prefix-cache hit granularity UNCHANGED at 832 tokens
State slabs: 48 (conv, ssm) pairs x 164 page rows (row 0 = null page; id space: own state pool)
Scheduler config: block_size=832 num_device_pages=163 num_state_pages=163 ...
```

### Result 2 — serving: 53 → 64 at C=64, and the queue empties

Qwen3.5-27B bf16, **GPU 3 alone** (0 MiB resident to anyone else at both
launches; GPU 2's `--mamba-track-interval 6656` benchmark was left untouched).
`gpu_memory_utilization 0.88`, `max_num_seqs 160`, prefix caching on,
`--disable-kvstore` on **both** arms (see the guard below), no
`--attention-backend`. C = 64 closed loop, **1019 prompt tokens** (word count
calibrated against the server's own `usage.prompt_tokens`; guessing gives
~3.7x the intended length and a different batch entirely) and exactly 256
output tokens, unique prefixes, 260 s, medians over the saturated window.

| | OFF (shared id space) | **ON (placeholder blocks)** |
|---|---:|---:|
| `#running-req` | **53** | **64** |
| `#queue-req` | **11** | **0** |
| `#pages(active/total)` | 159/163 | 128/163 |
| page ratio | **0.98** | **0.79** |
| scheduler `gen throughput` | 987.6 tok/s | **1120.6 tok/s** (+13.5%) |
| end-to-end completion tok/s | 584.5 | **671.1** (+14.8%) |

`53 of 64 with 11 queued and the pool 98% full` is E36's measurement
reproduced exactly, on a different day, from a different load driver. With
placeholder blocks the same box runs **all 64 offered sequences with nothing
waiting and 21% of the pool still free** — the pool has stopped being the
binding constraint at this shape.

### Result 3 — correctness: byte-identical, once the probe is well conditioned

Greedy (`temperature 0`, 48 new tokens), four prompts, two independent passes
per arm, `POST /flush_cache` plus a discarded warm-up before each pass.

**Two probes had to be thrown away first, and both are worth recording.**
(1) Long prompts made of random 5-digit numbers leave the model with
near-tied logits: greedy output flips run-to-run *on one arm*, and it looks
exactly like a correctness bug. Replaced with long repeated English carrying
a determinate answer. (2) A 6-token prompt (`"The capital city of France
is"`) is likewise unstable — the **control arm itself** produced
`a1ea01de…` on pass 0 and `9786e6e2…` on pass 1. It is reported below as
unusable, not as a difference.

| prompt | prompt tokens | OFF | ON | |
|---|---:|---|---|---|
| `primes` | 10 | `69068367398ada21` | `69068367398ada21` | identical |
| `med_en` | 3,207 | `a931ccf42e887ac4` | `a931ccf42e887ac4` | identical |
| `long_en` | **11,294** | `57aaae8510a0e94a` | `57aaae8510a0e94a` | identical |
| `short` | 6 | *flips within the OFF arm* | — | **probe rejected** |

`long_en` is 11,294 tokens: 13 attention pages, and its prefill splits
7,488 + 3,806 — one full aligned chunk plus a remainder, i.e. it exercises a
mid-sequence state snapshot and a cross-page state read. Answers are also
right on the merits ("1847", "42 meters").

**Why byte-identical is the correct bar here, and why E34's trap does not
apply.** E34 found that moving the state block size changes the GDN
recurrence's chunk decomposition, which changes output through fp
reassociation alone. Placeholder blocks move **neither** P (832) **nor** N
(832) **nor** `chunked_prefill_size` (7,488) — the logged geometry is
identical on both arms — so the decomposition is fixed by construction and
anything other than an exact match would be a real defect.

### Result 4 — prefix-cache granularity: unchanged, which is the entire point

State groups publish a snapshot only at an aligned chunk end
(`RegistersAlignedFinalPageOnly`) and `chunked_prefill_size` floors to
9 × 832 = 7,488, so the probe is a shared prefix longer than one chunk plus a
distinct tail. The server's own `#cached-token`:

| | shared id space | **placeholder blocks** | `--mamba-track-interval 6656` |
|---|---:|---:|---:|
| `#cached-token` on the repeat | **7,488** | **7,488** | ≤ 6,656 by construction (lcm = N) |
| hit granularity | 832 (1 attention page) | **832 (unchanged)** | 6,656 |

Identical, and it must be: the coordinator aligns every hit DOWN to the
groups' lcm, `MakeCoordinator` computes that lcm from the group **block
sizes**, and placeholder blocks change which *pool* a group draws ids from,
never its block size. That is the difference from `--mamba-track-interval`,
which buys its 1.78x precisely by raising N and therefore the lcm.

### Verification of the code itself

- **C++**: 6 new tests in `tokenspeed-scheduler/tests/cpp/test_placeholder_blocks.cpp`
  — the feature is off by default; the state group draws exactly one id from
  each pool; **32 usable ids per pool admit 32 one-page requests against the
  shared pool's 16** (the 2x, at the allocator); `Acquire` refuses when only
  the state pool is exhausted and allocates nothing; the two id spaces
  overlap numerically on purpose; prefix-hit length is identical split vs
  shared. Suite: **419 passed** with the change vs **413** without, and
  **103 pre-existing failures in both** (all in the radix
  `test_paged_cache_replay.cpp` path, unrelated and present on `3f88dcc2`).
- **Python**: 15 new tests in `test/runtime/test_placeholder_blocks.py`
  pinning the published geometry (208,470,016 / 73.8% / 163 pages), the exact
  501,130 → 250,565 B/token, r = 1 allocating byte-identically to the shared
  space, the plan matching `capacity_model` at r ∈ {None, 1, 0.5, 0.25}, state
  tensors sized by the state id space, N untouched, and the slab row count.
  73 passed across the placeholder, flat-memory-plan, flat-state-slab,
  GDN-state-paging, scheduler-config-guard and page-zeroing suites.

---

### What this costs — stated plainly

1. **It does not compose with the flat host tier (KVStore/L2), and the build
   refuses rather than guessing.** The host tier is one id space shared by
   every group; a state group loading a host extension would land a host page
   keyed in the shared host index onto a device page drawn from the state
   pool. Unverified, so `--enable-placeholder-blocks` raises
   `NotImplementedError` unless `--disable-kvstore` is passed, and the
   coordinator asserts as a backstop. **Both arms above therefore ran with
   `--disable-kvstore`**; the numbers are not comparable to an L2-enabled run.
   Making them compose means giving the host tier a state segment too.
2. **The admission gates are conservative, not family-exact.** Eight call
   sites in `forward.cpp` compare one scalar demand against `flatFreeBudget`.
   That budget is now `min(free_history, free_state)`, which implies both
   per-pool conditions and so never over-admits — but it is tight only while
   the two pools drain together (the r = 1, N = P case measured above) and
   pessimistic when they do not. A family-exact gate means threading
   `PoolDemand` through all eight. Not done, not measured.
3. **`r` is a static split chosen at startup.** At r = 1 a running request
   pins `ceil(T/832)` history ids but only ~2 live state ids plus retained
   snapshots, so on long-context/low-sharing shapes the history pool binds
   first and r < 1 would be strictly better. The right default is workload
   dependent and **no shape other than 1019-in/256-out was measured**.
4. **The two id spaces overlap numerically.** Safe only because each group's
   block table indexes only its own family's tensors. `ActiveKvPages` had to
   stop deduping on page id alone (it now reports the binding pool) and
   `flat_page_ids_to_zero` had to split into two lists. The state list is
   plumbed but **unconsumed**: `MHATokenToKVPool` does not implement
   `zero_pages` and does not declare `flat_kv_requires_page_zeroing`, because
   a fresh request forces `state_in = 0` and never reads a stale row. Any
   future pool that *does* need sanitisation must consume
   `flat_state_page_ids_to_zero` as well.
5. **Kimi-K3's `FlatHybridCachePool` is not wired.** It plans through
   `_create_kimi_k3_flat_plan`, publishes no `flat_state_pool_pages`, and so
   silently keeps the shared id space. Correct, but the win is GDN-only today.

### What a request does when the snapshot does not exist — recommendation

**Keep refusing (truncate to the last real snapshot). Do not replay.**

- Every engine surveyed truncates: TRT-LLM to
  `latestMatchingNonPlaceholderBlockIdx` with partial reuse hard-disabled for
  hybrids, SGLang to `best_value_len`, lmdeploy to a hard miss, vLLM to the
  last cached non-null block.
- With placeholder blocks at N = P the question **barely arises**: every state
  page position still holds a snapshot, so the truncation boundary is
  unchanged at 832 tokens. Replay would buy nothing here. It only becomes
  interesting stacked on `--mamba-track-interval`, where the gap is N − P.
- The cost of replaying is not bounded by anything the scheduler knows. Re-running
  the recurrence from the nearest snapshot means up to N − P tokens through all
  48 GDN layers *before the first admitted token*, on the critical path of a
  request that was admitted on the promise of a cache hit — and TokenSpeed's
  state evolves in place inside its working row (`state_in == state_out`
  within a window), so a replay also needs a scratch row per replaying
  request, i.e. more of the exact resource this work is trying to free.
  Recommend implementing it only behind a measured bound on N, and only after
  someone has measured prefix-hit rate on a real RAG shape — which nobody in
  this log has done yet.

### What was NOT measured

- **Any shape other than C=64 / 1019-in / 256-out.** No C=32, no C=128, no
  long-context, no RAG/shared-prefix throughput. E36's own note that
  TokenSpeed wins only at high concurrency applies: this is one operating
  point.
- **`r ≠ 1` at runtime.** The 258/129 and 365/91 splits are the planner's
  arithmetic only; no server was started with `--placeholder-state-block-ratio`.
- **Prefix-hit *rate*** under load. Result 4 shows the granularity is
  identical (7,488 = 9 × 832 on both arms); it does not measure how often a
  real workload hits.
- **The two knobs together.** The 5.65x row is the model; placeholder blocks
  and `--mamba-track-interval` were never run in the same server.
- **vLLM.** No vLLM arm was run in this entry; the 478,162-token / 63 KB-token
  comparison is quoted from E36, and at 250,565 B/token TokenSpeed is still
  ~4x behind vLLM per token on this model.
- **Kernel-level anything.** No profiler, no ncu, no isolation anchor beyond
  `nvidia-smi` showing GPU 3 at 0 MiB before each launch (E22: `sm%` is
  useless on this box and was not used).
- **PD / disaggregation, speculative decoding, TP > 1.** The
  decode-destination path is per-pool-correct by construction and unit-tested,
  but was never exercised on hardware.

---

## E38 — Placeholder blocks compose with the flat host tier: the mirror was safe only by id-exclusivity, and two ownership asserts caught the composition bugs

**Objective.** E37 shipped `--enable-placeholder-blocks` with a hard refusal
against the flat host tier (KVStore/L2), so buying the 2x capacity cost the
host tier. Establish the real mechanism, decide whether composition is a fix
or a redesign, then compose them.

*(Sections 1-2 below were written before any code and are left as they were;
the implementation and the four-arm matrix follow them.)*

### Correction first: where the guard actually lives

It is not in `registry.py` and it is not pre-existing. **Both copies were
added by E37**, deliberately:

- `tokenspeed-scheduler/csrc/cache/kv_cache_coordinator.cpp:56` —
  `_assert(host_pool_ == nullptr, "placeholder blocks and the flat host tier are not composable yet")`
- `python/tokenspeed/runtime/engine/event_loop.py:476` — the
  `NotImplementedError` that raises unless `--disable-kvstore` is passed.

`registry.py:~1235` only computes `placeholder_ratio` and forwards
`state_blocks_per_history_block`; it has no opinion about the host tier. So
the question "is there a real incompatibility, or was the guard merely
precautionary?" is the right one to ask, and the answer is: **the guard is
real, and the mirror's own docstring states the invariant it protects.**

### Result 1 — the mechanism, and it is one root cause wearing three hats

`FlatHostMirror._copy_pages` (`python/tokenspeed/runtime/cache/flat_host_mirror.py`):

```python
for (dev, mirror), p in zip(self.tensor_pairs, self.row_spans):
    for device_page, host_page in pairs:
        dev_rows  = dev[device_page * p : (device_page + 1) * p]
        host_rows = mirror[host_page * p : (host_page + 1) * p]
```

One `(device_page, host_page)` pair copies that page's row range on **every**
mirror pair — every K slab, every V slab, and all 96 GDN state slabs
(48 conv + 48 ssm). The class docstring names exactly why that is safe today:

> "Slab tensors are enumerated once each — a page's rows are exactly its
> owner group's layers, so byte copies are **group-safe by id-exclusivity**."

**Id-exclusivity is precisely what placeholder blocks abolish.** With two
device pools, device id 5 exists in both and names different live memory.
Storing history page 5 would also copy state slab row 5 — a *different,
running* request's GDN snapshot — into the host page, and a restore would
write it back over that request's state. That is silent recurrent-state
corruption: fluent wrong text, not a crash. The guard is correct.

The coordinator's three hypotheses are all true and all the same cause — the
host tier is an **isomorphic mirror of the shared device id space**:

| hypothesis | verdict |
|---|---|
| host pages indexed by the shared block-id space, and now there are two | **yes** — `_copy_pages` indexes every device tensor by one id |
| eviction/restore assumes one pool | **yes** — `flat_host_pool_` is one pool; `FlatStoreLedger` / `FlatLoadTicket` carry no family |
| the host page layout bakes the dense state row | **yes** — `flat_bytes_per_host_page` = `kv_bytes + state_bytes`, i.e. the union row, the same 2x waste E37 removed on device |

### Result 2 (POSITIVE, and it changes the cost) — the split channel already exists on the wire

The expensive part of this would have been a wire-format change. It is not
needed:

- `scheduler/operations/cache.h:52` — `enum class CacheKind { kKV = 0, kMamba = 1 }`.
- `:64` — `TransferPair { CacheKind kind; int32 src; int32 dst; }`; `TransferPairHash` already hashes `kind`.
- `bindings/python_module.cpp:424-425, 431-432` — **`src_pages_by_kind` and
  `dst_pages_by_kind` are already bound** on both `LoadBackOp` and `WriteBackOp`.
- `flat_memory_executor.py:184` — `self.pools = {CacheKind.KV: self.mirror}`,
  a dict keyed by `CacheKind` with exactly one entry.

The kind-partitioned payload is already end to end; the radix path uses it,
and the flat path collapses it because `scheduler.cpp:487` hard-codes
`TransferPair{CacheKind::kKV, ...}` for every store. **Composition is teaching
the flat path to use a channel that is already there.**

### Design — mirror the split on the host, r for r

- **C++.** Add `flat_state_host_pool_`; `tierPoolFor(i, host_tier)` returns
  the family's host pool. `StoreCandidate` and the loadback path carry the
  family (available from `drawsFromStatePool(i)` at the point of construction
  in `CacheFullBlocks` / `LoadHostExtension`), so `scheduler.cpp:487` emits
  `CacheKind::kMamba` for state groups and draws their host page from the
  state host pool. Delete the assert.
- **Python.** `FlatHostMirror` takes a family selector and is built twice: a
  KV-only mirror (row span `page_size`) and a state-only mirror (row span 1).
  `pools = {CacheKind.KV: kv_mirror, CacheKind.MAMBA: state_mirror}`; the
  executor routes each pair through `*_pages_by_kind` instead of the
  flattened lists. Delete the `NotImplementedError`.
- **Sizing preserves E37's key property.** A history host page becomes
  KV-bytes only, a state host page state-bytes only; at r = 1 and equal page
  counts the host total is unchanged, so *nothing is reallocated on the host
  either* and the feature stays byte-identical when off.

**Interaction with r:** the two host pools should be sized in the same ratio
as the device pools, or the host tier becomes the binding constraint at
r ≠ 1. Since only r = 1 has ever been run, this is unverified in both
directions.

### Rejected designs

1. **Disjoint device id ranges** (state pool numbered 163..325) to restore
   id-exclusivity without touching the mirror. Fails: `_copy_pages` would
   index `k_buffer[200*832 : 201*832]` on a 163-page buffer — out of bounds.
   It converts a silent corruption into a loud crash and still forces the
   mirror to learn families. Buys nothing.
2. **One host pool, union-row host pages, route only the device side.**
   Works, but wastes 74% of every history host page and 26% of every state
   host page — it reintroduces on the host exactly the 2x that E37 removed on
   the device, halving effective L2 capacity.
3. **Biased device ids on the wire** (`src = id + num_device_pages` for
   state) to avoid a family field. Unnecessary: `CacheKind` is already there
   and already bound. Rejected as an encoding hack with no upside.
4. **Drop state from L2** (mirror history pages only). Cheapest, but a
   semantic regression: a prefix hit above the device boundary would need a
   state snapshot that was never stored, so the coordinator would truncate.
   That is giving up state-resumable prefix caching — the one thing E34's
   design table records TokenSpeed doing that vLLM cannot on hybrids.

### Verdict

**A contained fix, not a redesign.** The device-side abstraction (per-group
pools, `drawsFromStatePool`) already generalises; the host tier simply has
not been taught the same lesson, and the channel it needs is already bound.
Estimated: ~5 C++ sites and ~3 Python sites, plus a second mirror.

### What was NOT done, and what it will cost to finish

- **No code was written for this entry.** E37's guard is still in force;
  `--enable-placeholder-blocks` still requires `--disable-kvstore`.
- **Nothing here was measured.** No arm was run. The four-arm matrix the
  request asks for (neither / placeholder only / kvstore only / both) is
  ~4 server starts and ~4 loads on GPU 3, plus a correctness pass per arm and
  the eviction/restore evidence, which needs `--enable-cache-report` plus the
  executor's own store/load counters — allocation alone is not evidence, and
  E37 did not exercise the host tier at all.
- **Two E37 caveats will need revisiting during that work**, not before:
  `flatFreeBudget = min(free_history, free_state)` becomes wrong-shaped if
  the host pools are sized asymmetrically, and the executor's
  layerwise-loadback fencing assert
  (`flat_memory_executor.py:~155`, "last layer's V mirror is not the last KV
  tensor pair") is written against **one** mirror's tensor ordering and will
  have to be restated per mirror.

---

---

### IMPLEMENTED — the composition, and the two bugs the ownership asserts caught

Sections above were the design. It was built as scoped: `CacheKind` carried
the family, no wire-format change, no new binding.

**C++ (6 sites).** `BlockTransfer.state_family` and
`KvCacheCoordinator::StoreCandidate.state_family` carry the family from where
the group is known; `hostPoolFor(i)` selects the host pool; `MakeSpecsFromConfig`
/ `SchedulerConfig::state_host_allocator` size it; `scheduler.cpp` emits
`CacheKind::kMamba` and draws from `flat_state_host_pool_`; `forward.cpp` tags
loadbacks the same way. The ctor assert was **narrowed, not deleted** —
`host_pool_ == nullptr || state_host_pool_ != nullptr` — because a single host
pool under two device pools is still the aliasing hazard.

**Python (3 sites).** `FlatHostMirror` gained a `families` vector and
per-family pair lists; `FlatMemoryExecutor` reads the already-bound
`src_pages_by_kind` / `dst_pages_by_kind` (the flat path had been using
`src_pages`, which the header documents as the **backward-compatible KV-only
view** — tagging state as `kMamba` without this change would have silently
dropped every state transfer); the `NotImplementedError` became a positive
consistency check.

**Two bugs, both caught by asserts that already existed. Neither was silent.**

1. `BlockPool::CacheFullBlock` — *"block reference belongs to another pool"*.
   Publish-at-ack in `outside_event_handler.cpp` hashed every completed store
   into `flat_host_pool_`, including state pages drawn from the state host
   pool. Fixed by carrying `state_family` on `FlatStoreTicket`.
2. `KvCacheManager::AcquireMatchedBlocks` — *"cached block disappeared between
   match probe and acquisition"*. `acquireTierWithKeys` used `*host_pool_`
   while `probeTierWithKeys` used `tierPoolFor(i, host_tier)`, so a state
   group probed the state host pool and acquired from the history one.

Both are the same mistake — routing one half of a pool pair — and both would
have been silent data corruption had the asserts not been there. Recorded
because the next person composing a family-split pool will make one of them.

### Result 3 — the four-arm matrix, C=64, 1019-in / 256-out

Qwen3.5-27B bf16, **GPU 3 alone** (0 MiB to anyone else at each launch),
`gpu_memory_utilization 0.88`, `max_num_seqs 160`, prefix caching on,
`--enable-cache-report`, one 260 s closed-loop window per arm, sequential,
medians over the saturated window.

| arm | `#running` | `#queue` | pages | ratio | scheduler gen tok/s | e2e tok/s |
|---|---:|---:|---:|---:|---:|---:|
| **A** neither | 53 | 11 | 159/163 | 0.98 | 987.3 | 584.5 |
| **B** placeholder only | **64** | **0** | 128/163 | 0.79 | 1120.1 | 670.8 |
| **C** KVStore only | 53 | 11 | 159/163 | 0.98 | 987.2 | 581.1 |
| **D** **both** | **64** | **0** | 128/163 | 0.79 | **1119.5** | **669.4** |

- (a) **D reaches 64 running with an empty queue** — the composition keeps
  placeholder blocks' entire win. D vs B is 669.4 vs 670.8, indistinguishable.
- KVStore alone (C vs A) is worth nothing at this shape, as expected: unique
  prefixes, no reuse to restore. It is not a regression either.
- The features are orthogonal at this operating point: neither pays for the
  other.

### Result 4 — correctness, including the cross-family id collision

(b) Same probe as E37 (greedy, `/flush_cache` + discarded warm-up, two passes
per arm). All four arms **byte-identical** on the three well-conditioned
prompts:

| prompt | tokens | A | B | C | D |
|---|---:|---|---|---|---|
| `primes` | 10 | `69068367398ada21` | = | = | = |
| `med_en` | 3,207 | `a931ccf42e887ac4` | = | = | = |
| `long_en` | **11,294** | `57aaae8510a0e94a` | = | = | = |

(`short`, the 6-token prompt, again splits `a1ea01de` / `9786e6e2`; E37 showed
it flipping **within** its own control arm, so it stays a rejected probe.)

**The collision case, constructed deliberately rather than hoped for.**
`test/runtime/test_placeholder_host_tier.py` drives `FlatHostMirror` directly:
request A holds *history* page 5, request B holds *state* page 5 in the other
pool with a live snapshot; A's page is stored to host, B's snapshot then
evolves, A's page is restored.

| | state row 5 after A's restore | verdict |
|---|---|---|
| split families (the fix) | **9.0** — B's live value, untouched | pass |
| shared id space (**negative control**) | **7.0** — rolled back to A's store time | the bug, reproduced |

The negative control is the point: it proves the test can see the failure.

**Regression:** 419 C++ tests pass (unchanged from E37; the same 103
pre-existing radix-replay failures in both arms), and **99 Python tests pass**
across the placeholder, flat-memory-plan, flat-state-slab, GDN-state-paging,
host-mirror, host-executor, page-zeroing and scheduler-config-guard suites.
One existing test needed updating (`test_flat_host_executor.py`'s
`load_pages_with_events` spy) because the mirror API grew a defaulted
`state_pairs` parameter. `test_flat_host_tier_e2e.py` never completes in this
environment — verified identical on the **stock** image, so pre-existing and
not caused by this change.

### Result 5 — the host tier demonstrably works (c)

Allocation is not evidence, so: warm an 11,294-token prompt, churn the
163-page device pool with 90 sequential ~1,700-token requests (sized to sit
**between** the 163-page device pool and the 327-page host pool — both are LRU
over the same freed order, and a first attempt with 90 s of C=64 traffic
evicted the host copy too and showed nothing), then re-request it:

```
Flat host tier census: kv stored=193 loaded=9 | state stored=1 loaded=1 (split_families=True)
Prefill batch. #new-token: 3806, #cached-token: 7488
```

`loaded=9` KV pages (= 9 x 832 = the 7,488-token first chunk) **and
`state loaded=1`** — the recurrent-state channel restores, it is not silently
empty. `#cached-token: 7488` after the device pool was churned is the same
number reaching the scheduler. Counters are `FlatMemoryExecutor.stats`, logged
every `TOKENSPEED_FLAT_HOST_CENSUS_PAGES` pages (default 512, set to 1 here).

Note `state stored=1` against `kv stored=193`: state groups publish only at an
**aligned** chunk end, and the churn prompts (~1,700 tokens) never end on an
832 boundary. The state channel is rare by design, not broken — which is
exactly why it needed the deliberate test above rather than a benchmark.

### E37's safety property: kept

At r = 1 the host tier logs `bytes_per_host_page=208470016` — the union row —
and splits it 327/327. **Byte-identical to the shared layout**, on the host as
well as the device. Off is unchanged: `families` collapses to all-`"kv"`, every
tensor consumes one pair list, and the C++ emits `kKV` throughout.

### The two E37 caveats, revisited

1. **`flatFreeBudget = min(free_history, free_state)`** — *not* made
   wrong-shaped by the host tier. It gates **device** pages only; a full host
   pool degrades gracefully (`if (!host_block) { cand.block.reset(); continue; }`
   drops the store and unpins). It remains exact when the two device pools
   drain together and conservative otherwise — unchanged by this work, and
   still not family-exact. See E40 Item 2 for why r matters more than I
   assumed.
2. **The layerwise-loadback fencing assert** — restated, and a second assert
   added that state mirrors trail every KV mirror. The reasoning survives
   because the split changes *which pair list* each tensor consumes, not
   `tensor_pairs` order, and both families still copy on one serial stream, so
   `events[-1]` still covers every copy. **A related trap was found and
   avoided**: adding `CacheKind.MAMBA` to `FlatMemoryExecutor.pools` looks
   right but is wrong — `_setup_layerwise_loadback` iterates that dict and
   `get_producer_index` **pops**, so a second entry would read `None` and its
   `set_consumer(-1)` would overwrite the first's real consumer indices. The
   dict keeps exactly one entry; the split is internal to the mirror.

### Interaction with r (and a correction from E40)

The executor derives its host split from the **device** page counts
(`state_ratio = device_state_pages / device_history_pages`), so the host pools
track `r` automatically: at r = 0.5 the host would be 327 KV / 163 state. That
was the open question in the design and it is handled — but **E40 (another
agent, same day) found the larger gap**: `r` and `--mamba-track-interval` (N)
are independent knobs and `plan_component_tensors` never sees N, so raising N
alone is byte-identical and inert. E40's `r = P/N` coupling with N = 2P is the
real operating point. Nothing in this entry changes that; the host split will
follow whatever r ends up being.

### What was NOT measured

- **One 260 s window per arm, not repeated runs** — no standard error, no
  significance test. The coordinator's gated harness put placeholder blocks at
  +10.4% / 9.7 sigma; the D-vs-B equivalence here (669.4 vs 670.8) is a single
  window each and should be read as "no detectable cost", not as a measured
  bound.
- **`period.py` was not run**; the limit cycle was not sampled at any arm.
- **Only r = 1.** Every host-split number above is the symmetric case. r != 1
  has still never been run on hardware, on device or host.
- **No host-tier throughput benefit was demonstrated at all.** Arms C and D
  used unique prefixes, so there was nothing for L2 to restore under load; the
  restore evidence is the deliberate probe, not the benchmark. A shared-prefix
  / RAG shape would be the shape that shows what the host tier is *for*, and
  it was not run.
- **No PD/disaggregation, speculative decoding, or TP > 1**, and no Kimi-K3
  `FlatHybridCachePool` (still refuses the host tier for unrelated reasons).
- **Host memory pressure was not explored**: 68.17 GB pinned at
  `kvstore_ratio 2.0`; the split does not change the total at r = 1 but no
  larger ratio was tried.

## E39 — fp8 KV on Qwen3.5-27B/sm_120: the arithmetic says +15%, no kernel accepts it at page 832, and turning it on would silently double the attention page

**Objective.** Decide whether `--kv-cache-dtype fp8` is worth landing for
Qwen3.5-27B on one RTX PRO 6000 (sm_120), starting from the page arithmetic
rather than from the folklore that fp8 KV buys "2x capacity". E37 took C=64
from 53 to 64 running and a sweep put the new ceiling at 80, which matches
`max_total_num_pages = 163` with `ceil(1280/832) = 2` pages per 1024-in/256-out
request. Halving the attention KV is the obvious next lever.

**Verdict up front: do not land it.** The payoff is +15% pages at fixed page
size, the path does not exist (zero registered kernels accept an fp8 KV cache
at this geometry), the in-tree Triton kernel that could be forced to run it
carries two defects that double the error over an ideal fp8-KV kernel, and the
flag has an undocumented side effect that is larger than the flag itself.

---

### Path 1 — the decomposition, from the code, before any GPU work

`registry.py:1240` plans the flat GDN cache with `plan_component_tensors`
(`runtime/configs/flat_memory_plan.py:423`):

```
per_block = sum(kv_row_bytes) + r * sum(state_row_bytes)      # r = 1.0 (E37 default)
kv_row_bytes[i]    = cache_cell_size() * P                    # full-attention layers
state_row_bytes[i] = conv_bytes + ssm_bytes                   # GDN layers, CONSTANT in P
```

From the HF config (`num_key_value_heads=4`, `head_dim=256`,
`linear_num_value_heads=48`, `linear_key_head_dim=128`,
`linear_conv_kernel_dim=4`, `mamba_ssm_dtype=float32`):

| term | bytes | share |
|---|---:|---:|
| 16 full-attn layers, K+V, bf16, P=832 | 54,525,952 | **26.15 %** |
| 48 GDN state rows (conv 61,440 + ssm 3,145,728 each) | 153,944,064 | **73.85 %** |
| `block_bytes` | **208,470,016** | |

This reproduces the engine's startup line **to the byte** — confirmed against a
live baseline boot on GPU 1 (container `ts-e39-base`, the exact run config in
the brief):

```
Setting attention block size to 832 tokens to cover the GDN state row (configured block size 64)
Flat GDN KV profile: block_bytes=208470016 (112 component tensors, block_size=832), max_total_num_pages=163
Placeholder blocks ON: ... resident cost 501130 -> 250565 bytes/token (2.00x capacity)
KV Cache is allocated. K size: 4.16 GB, V size: 4.16 GB.
```

**The brief's decomposition is correct.** ~54.5 MB attention KV / ~151 MB GDN
state was right; the exact split is 54.5 MB / 153.9 MB, 26.15 % / 73.85 %. It
also agrees with E34's unit tests (`block 208,470,016, 73.8% state`).

Implied cache budget: `163 x 208,470,016 = 33.98 GiB` (bracketed
[31.65, 31.84] GiB by the floor division).

### Result 1 — the payoff, at fixed page size

fp8_e4m3 halves only `cache_cell_size` (4096 -> 2048 B/token/layer):

| arm | block_bytes | pages | B/token | pages/req @1280 | ceiling |
|---|---:|---:|---:|---:|---:|
| bf16, P=832 (baseline) | 208,470,016 | **163** | 250,565 | 2 | **81** (obs. 80) |
| fp8, P **pinned** at 832 | 181,207,040 | **187–188** | 217,797 | 2 | **93–94** |

**+15.0 % pages, +15.5 % sequence ceiling.** The brief's estimate (~190 pages,
~95 seqs) was right within the floor-division rounding. This is not "2x": fp8
halves 26 % of the block, so the ceiling is `1/(1-0.5*0.2615) = 1.147x` by
construction. **On a hybrid, attention KV is not the memory — the GDN state
rows are 74 % of it, and fp8 does not touch them.**

### Result 2 — the hazard: `--kv-cache-dtype fp8` silently doubles the page

The page size is not a free parameter. `registry.py:962` derives it from
`equalized_block_size`, which consumes **`config.cache_cell_size()`** — the
same quantity fp8 halves:

```
needed = ceil(max_const / max_linear) = ceil(3,207,168 / max_linear)
P      = 64 * ceil(needed / 64)
```

| `kv_cache_dtype` | max_linear | needed | **P** |
|---|---:|---:|---:|
| bf16 | 4096 | 783 | **832** |
| fp8  | 2048 | 1566 | **1600** |

So `--kv-cache-dtype fp8` does not just halve the KV — it **doubles the
attention page from 832 to 1600 tokens**, and with it the state snapshot
stride N (still `N = P`, since `--mamba-track-interval 256` stays inert).
Consequences, none of them announced by the flag:

- prefix-cache hit granularity coarsens **832 -> 1600 tokens** (E34's
  documented price for `--mamba-track-interval`, here charged silently by a
  dtype flag);
- a 1280-token request drops from 2 pages to 1, so the *apparent* ceiling
  jumps 80 -> ~164 — but **that is the page-size lever, not fp8**;
- this lands squarely in the `P = 1664` fragmentation regime that E34
  explicitly listed as **not measured**.

Separating the two levers at the same P is the honest comparison:

| arm | block_bytes | pages | B/token | ceiling @1280 |
|---|---:|---:|---:|---:|
| bf16, P=832 | 208,470,016 | 163 | 250,565 | 81 |
| bf16, P=1600 | 258,801,664 | 131 | 161,751 | 131 |
| fp8, P=1600 (what the flag actually does) | 206,372,864 | 164–165 | 128,983 | 164 |

At equal P, **fp8 is worth 1.25x, and the page size is worth 1.55x for free.**
Of the 80 -> 164 the flag appears to buy, 80 -> 131 is page size (zero numerics
cost, reachable today with `--block-size`) and 131 -> 164 is fp8. Anyone who
reads "fp8 took us from 80 to 164 running" has attributed a bf16 win to a
quantization.

### Path 2 — which kernels accept an fp8 KV cache at 24q/4kv/d256/page-832

Exercised on the real device through the real dispatch (`probe_fp8kv_sm120/`),
not read off a table. `explain_selection` on sm_120, page_size 832:

| op | bf16 | fp8_e4m3 |
|---|---|---|
| `mha_extend_with_kvcache` | `triton_mha_extend_with_kvcache` (P4) — **only candidate** | **none** — signature mismatch |
| `mha_decode_with_kvcache` | `triton_mha_decode_with_kvcache_cached` (P4) | **none** — signature + trait |

Why each is filtered:

- `flashinfer_trtllm_mha_{extend,decode}`: `arch mismatch (requires <= 10.3)`.
  Not available on sm_120 at any dtype.
- `flashinfer_xqa_mha_decode_with_kvcache`: `page_size=832 not in
  {16,32,64,128}` **and** signature `{float16, bfloat16}`. Its registration
  comment states the reason for the latter outright: *"XQA reads and writes the
  query dtype; the fp8 path would need out_dtype == q.dtype, which this op's
  contract does not provide."* Both filters are independent, so no page-size
  change would rescue it — and E34 already showed every XQA-legal page is below
  a quarter of today's capacity.
- `triton_mha_{extend,decode}_with_kvcache`: `signatures=format_signatures(
  ("q","k_cache","v_cache"), "dense", {torch.float16, torch.bfloat16})`.

**Live confirmation, not a table read:** calling the ops with genuine fp8
tensors at this geometry raises `NoKernelFoundError` for both extend and decode,
at `solution=None` and at `solution="triton"`. The engine's backend
(`backends/mha.py:812,860`) passes exactly these dtypes, so
`--kv-cache-dtype fp8` would fail at the first full-attention layer of the
first forward, not at startup.

**Re-verifying the prior lead (it does not transfer).** The note
"fp8 KV works on sm_120 via Triton" is a **vLLM** result — vLLM's `TRITON_ATTN`
backend, Qwen3-32B, dense. TokenSpeed's Triton MHA is a different
implementation and does *not* inherit it. The closed-allowlist diagnosis is
still the right shape (the block is a hardcoded dtype set, and it is the *only*
registered kernel at this geometry), but here the allowlist is **honest**: the
kernel body genuinely does not implement fp8.

### Path 3 — forcing the Triton kernel anyway, to price the "just widen the signature" fix

Bypassing the signature filter with
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_{DECODE,EXTEND}_WITH_KVCACHE`, the
kernel **runs and returns finite output**. It is nonetheless not a correct
fp8-KV kernel. Three defects, all in the body/contract, not the gate:

1. **No descale.** `mha_decode.py` accepts `q_scale`/`k_scale`/`v_scale` and
   **silently ignores all three** — they appear only in the signature, nowhere
   in the body. Measured: storing `k/4, v/4` and reading back gives output-norm
   ratio **0.243** and `cos = 0.9829` vs the bf16 reference (a descaling kernel
   would give 1.0). This is inert *today* only because Qwen3.5-27B's checkpoint
   has **0 scale keys**, so `layer.k_scale` stays `None`
   (`paged_attention.py:62`) and `fused_fp8_set_kv_buffer` skips its division.
   Load any compressed-tensors checkpoint that maps
   `.k_proj.output_scale -> .attn.k_scale` and the output is silently wrong.
   Declaring fp8 in the signature while ignoring the scales is precisely the
   "declare a trait you do not implement" failure the port-kernel skill forbids.
2. **The attention output is written in e4m3.** `out = torch.empty_like(q)`
   (`mha_decode.py:348`, `mha_prefill.py:674`) — and the backend casts `q` to
   fp8 first, so the *output* tensor is `float8_e4m3fn`. Verified directly:
   `q+kv fp8 -> out.dtype = torch.float8_e4m3fn`. With a measured output RMS of
   0.0143, most elements land in e4m3's low-exponent/subnormal range. This is a
   precision loss on the residual stream, entirely separate from KV
   quantization, and it is the same gap XQA's comment names.
3. **Softmax probabilities are quantized before the PV matmul.**
   `acc += tl.dot(p.to(v.dtype), v)` (`mha_decode.py:411`) casts fp32 `p` to
   the *cache* dtype, i.e. e4m3 (3 mantissa bits) once the cache is fp8.

### Result 3 — the numerics cost, decomposed and bounded

BS=8, SEQ=1280, QLEN=64, real geometry, reference = the in-tree bf16 Triton
kernel it would displace. Errors are relative L2 over the whole output tensor:

| arm | decode | extend | what it isolates |
|---|---:|---:|---|
| A — ideal fp8-KV (quantize+dequantize, bf16 math) | cos 0.999612, **2.79 %** | cos 0.999612, **2.79 %** | irreducible cost of e4m3 KV |
| B — real fp8 KV tensors, q bf16 | cos 0.999259, **3.85 %** | cos 0.999236, **3.91 %** | + defect 3 (`p -> e4m3`) |
| C — fp8 KV **and** fp8 q (**the engine path**) | cos 0.998414, **5.63 %** | cos 0.998414, **5.63 %** | + defect 2 (e4m3 output) + fp8 q |

**The engine path costs 5.63 % relative L2 per full-attention layer, 2.0x the
2.79 % that fp8 KV intrinsically costs.** Both A and C are below the
port-kernel skill's `cos > 0.9999` bar; A is what a correct implementation
would have to hit, and even that is not free. This is a per-layer figure across
16 full-attention layers whose errors enter a shared residual stream; the
end-to-end effect was **not** measured (see below).

Kernel speed (same shapes, x16 layers), for completeness — fp8 is genuinely
faster, which is worth recording because it means the kernel is not silently
dequantizing:

| | bf16 | fp8 | x16 layers |
|---|---:|---:|---|
| decode | 0.149 ms | 0.139 ms (1.08x) | 2.387 -> 2.217 ms |
| extend | 0.441 ms | 0.327 ms (1.35x) | 7.051 -> 5.231 ms |

---

### Verdict

**Do not land fp8 KV for this model on this hardware.** Priced honestly:

- **Buys** +15 % pages / +15 % sequence ceiling at fixed page size (163 -> 187,
  81 -> 93). Not 2x — attention KV is 26 % of a hybrid block.
- **Costs** 5.63 % relative L2 per attention layer as the code stands, of which
  only 2.79 % is intrinsic; the other half is two fixable defects.
- **Costs, in work:** it is not a flag. It needs (a) fp8 added to two in-tree
  kernel signatures, (b) a real descale in the Triton body, (c) an `out_dtype`
  in the op contract so the output stops being e4m3 — the same contract gap
  that already makes XQA refuse — and (d) the backend to stop casting `q` to
  fp8. That is a kernel-and-contract change affecting every model that uses
  these two kernels, to buy 15 % on one.
- **Risks** a silent 2x page inflation (832 -> 1600) that coarsens prefix-cache
  granularity and moves the engine into a fragmentation regime E34 left
  unmeasured — charged by a flag whose name says "dtype".

**Take the other lever first.** At equal page size fp8 is worth 1.25x while the
page/state-stride geometry is worth 1.55x with **zero** numerics cost, using
machinery E34 already built and tested. If the goal is to get past 80 running
sequences, `--mamba-track-interval` / the placeholder-block ratio `r` (E37) and
the page size are the cheap, lossless moves, and they should be measured before
anyone pays 5.63 % per layer for 15 %.

### Rejected approaches

1. **Widen the two Triton signatures to include `float8_e4m3fn` and ship.**
   The kernel runs and the numbers look plausible (cos 0.9984) — which is
   exactly the trap. It ignores `k_scale`/`v_scale` entirely and returns an
   e4m3 output tensor. Shipping it would declare a capability the body does not
   implement and would be a live wrong-answer bug the moment a checkpoint
   carries KV scales.
2. **Route fp8 decode through FlashInfer XQA.** Refused on two independent
   grounds (page_size whitelist `{16,32,64,128}` vs 832; dtype signature), and
   E34 already established that every XQA-legal page costs more than a quarter
   of capacity. Rejected on arithmetic before reaching a kernel — the same way
   E34 rejected its FINDING 1.
3. **`--kv-cache-dtype mxfp8`.** No block-scaled MHA kernel is registered on
   sm_120 at all: `mha_extend_with_kvcache` has exactly one candidate (Triton,
   bf16/fp16) and `mha_decode_with_kvcache` two. The tsmha blockscaled path
   wants page 128 and FA4/tcgen05, neither of which exists here.
4. **Reach the 2x by letting fp8 inflate P to 1600.** Real, but it is the page
   size doing the work, and it arrives bundled with a numerics cost that has
   nothing to do with it. If P=1600 is the right operating point, set it in
   bf16 and measure it there.

### What was NOT measured

- **No 27B engine run with `--kv-cache-dtype fp8`.** All four cards were
  occupied for the whole session (GPU 0 another tenant at 83 GiB; GPU 1 the
  prefill-admission agent's `ts-e39-base` at 91 GiB; GPU 2 `ts-sm120` at 93
  GiB; GPU 3 the third agent's, briefly at 0 then 90 GiB), leaving < 7 GiB on
  my assigned card — not enough for 54 GiB of weights. **`max_total_num_pages`,
  bytes/token and `#running-req` under fp8 are therefore the planner's
  arithmetic, not a live log.** The arithmetic is bit-exact against the live
  bf16 boot (208,470,016 / 163 / 250,565 reproduced exactly), and the
  fp8 fp8-boot would in any case abort at the first full-attention layer with
  `NoKernelFoundError`, which is a code-path proof (the backend's dtypes are
  read from `backends/mha.py:812,860`) rather than an observed crash.
- **No end-to-end generation comparison against bf16 or HF transformers.** The
  5.63 % figure is per-layer kernel output, not token agreement or a scored
  task. Same reason. Given the verdict is "do not land", the end-to-end number
  would only refine a trade already rejected on the capacity side.
- **No serving throughput, latency or prefix-hit-rate number**, at any arm — by
  instruction. The 1.08x/1.35x above are isolated kernel timings, not serving
  deltas, and the engine's limit cycle was not sampled (`period.py` not run).
- **The bf16 P=1600/1664 arm was not booted.** 131 pages / 131 seqs is the
  planner's number. Its fragmentation cost on non-1280-token shapes is
  unmeasured — still open from E34.
- **`--kv-cache-quant-method`** was not exercised; only `--kv-cache-dtype`.
- No file owned by a concurrent agent was touched: `kv_cache_coordinator.{h,cpp}`,
  `flat_state_slabs.py`, the host-mirror path and the C++ admission logic were
  read only. **No production code was changed by this entry** — the probes live
  in `tokenspeed/probe_fp8kv_sm120/` and nothing was landed.

---

## E40 — The GDN state is 74% of the page and `--mamba-ssm-dtype bfloat16` halves it for ~1.8% bounded error; `--mamba-track-interval` is *still* inert, now because `r` was never coupled to `N`

**Objective.** E39 priced fp8 KV at +15% and found no kernel for it. fp8 only
touches 26% of the page. Price the other 74% instead: (1) `--mamba-ssm-dtype
bfloat16`, (2) whether `--mamba-track-interval` composes with E37's placeholder
blocks now that the density premise E34 blamed has been removed.

**Verdict up front.** bf16 SSM state is the lever: **+57% ceiling for a bounded,
non-compounding ~1.8% per-layer error** — roughly 3x fp8 KV's capacity at a
third of its numerics cost, from a flag that is already correctly wired. But
**both** state levers carry a landmine that silently returns most of the win,
and they are different landmines.

---

### Item 1 — `--mamba-ssm-dtype`: the flag is real, the arithmetic needs one correction, and the equalizer eats the win

**Plumbing (traced end to end, not assumed).** `server_args.py:742`
(`__post_init__`) does `envs.TOKENSPEED_MAMBA_SSM_DTYPE.set(self.mamba_ssm_dtype)`;
`qwen3_5_text_base_config.py:344` reads exactly that env var
(`ssm_dtype = dtype_map[envs.TOKENSPEED_MAMBA_SSM_DTYPE.get()]`);
`flat_memory_plan.py:135` sizes with `ssm_dtype.itemsize`; `registry.py:956`
feeds it to **both** `equalized_block_size` and `plan_component_tensors`; and
`flat_state_slabs.py:234` allocates with `dtype=self._ssm_dtype`.
**It is consumed, not dropped** — this is *not* the "declared but still sized in
fp32" failure. Verified by driving the real objects
(`ServerArgs -> env -> ModelConfig -> mamba2_cache_params -> planner`):

| `--mamba-ssm-dtype` | `state_const_bytes` | per GDN layer |
|---|---|---:|
| `float32` (default) | `{conv: 61,440, ssm: 3,145,728}` | 3,207,168 |
| `bfloat16` | `{conv: 61,440, ssm: 1,572,864}` | 1,634,304 |

The fp32 arm reproduces the live baseline boot **bit-exactly** (`block_bytes=
208470016`, `max_total_num_pages=163`, 250,565 B/token), which validates the
whole chain.

**Correction to the estimate.** The state row is *not* halved: `conv` is
already bf16 and does not move. 3,207,168 -> 1,634,304 is a **49.0%** cut, so
state goes 153.9 -> 78.4 MB and the page 208.47 -> **132.97 MB** (not 130).
Pages 163 -> **255**, ceiling 81 -> **127**. The estimate (261 pages / ~130)
was right to ~2%.

**The landmine.** `equalized_block_size` derives P from the *state row*:
`P = 64 * ceil(ceil(state_row / kv_cell) / 64)`. Shrinking the state row
shrinks P with it:

| arm | P | block_bytes | pages | B/token | ceiling@1280 |
|---|---:|---:|---:|---:|---:|
| fp32 state (baseline) | 832 | 208,470,016 | 163 | 250,565 | 81 |
| bf16 state, **default `--block-size 64`** | **448** | 107,806,720 | 315 | **240,640** | **105** |
| bf16 state, **`--block-size 832`** | 832 | 132,972,544 | **255** | **159,823** | **127** |

At the equalizer's own choice the page drops to 448, a 1280-token request needs
**3** pages instead of 2, and **bytes/token moves only 4%** (250,565 ->
240,640). You get 105, not 127, and almost none of the resident-cost win.
**`--mamba-ssm-dtype bfloat16` must be paired with `--block-size 832`.**
This is the same structural trap E39 found on `--kv-cache-dtype fp8` (which
inflates P 832 -> 1600), inverted: **any change to a row size silently retunes
P, and the capacity outcome depends entirely on whether P is pinned.**

### Item 2 — `--mamba-track-interval` is still inert, and the reason is new

E34 measured N alone as worthless (bytes/token 1.78x, concurrency 53 -> 54) and
blamed `FlatStateSlabs.allocate` making state slabs dense over the whole id
space. E37 removed exactly that. **It is still worthless, for a different
reason.** `plan_component_tensors` charges

```
per_block = sum(kv_row_bytes) + r * sum(state_row_bytes)
```

where `r = placeholder_state_block_ratio`, **default 1.0**, an *independent*
CLI knob. **`N` does not appear in the sizing at all.** Raising N without
lowering r changes nothing:

| arm | block_bytes | pages (hist/state) | ceiling |
|---|---:|---:|---:|
| baseline (N=P=832, r=1) | 208,470,016 | 163/163 | 81 |
| `--mamba-track-interval 1664`, r=1.0 (default) | **208,470,016** | **163/163** | **81** |
| `--mamba-track-interval 3328`, r=1.0 (default) | **208,470,016** | **163/163** | **81** |
| `--mamba-track-interval 6656`, r=1.0 (default) | **208,470,016** | **163/163** | **81** |

Byte-identical. The premise E34's conclusion rested on *did* change, but the
knob that would express it was never wired to N, so the null result reproduces.

**With `r = P/N` it works, and it has an optimum:**

| N | r | block_bytes | pages (hist/state) | ceiling | bound by |
|---:|---:|---:|---:|---:|---|
| 832 (=P) | 1.00 | 208,470,016 | 163/163 | 81 | history |
| **1664 (2P)** | **0.50** | 131,497,984 | 258/129 | **129** | history |
| 3328 (4P) | 0.25 | 93,011,968 | 365/91 | 91 | **state** |
| 6656 (8P) | 0.125 | 73,768,960 | 460/57 | 57 | **state** |

Past N = 2P the state pool becomes the binding constraint and the ceiling
*falls*. **N = 2P with r = 0.5 is the operating point**, worth +59% on its own.

### Composition (P pinned at 832, ceiling at the 1024-in/256-out shape)

| arm | block_bytes | B/token | ceiling | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 208,470,016 | 250,565 | 81 | 1.00x |
| fp8 KV only (**blocked**, E39) | 181,207,040 | 217,797 | 93 | 1.15x |
| `--mamba-ssm-dtype bfloat16` | 132,972,544 | 159,823 | 127 | **1.57x** |
| bf16 state + N=1664 + r=0.5 | 93,749,248 | 112,679 | **181** | **2.24x** |
| + fp8 KV (blocked) | 66,486,272 | 79,911 | 255 | 3.15x |

**bf16 state alone is 3.7x the capacity gain of fp8 KV, and it composes.**

### Numerics — the real risk, and it does not behave as feared

The GDN decode kernel (`triton/gated_delta_rule.py`) accumulates the recurrent
state in **fp32 registers** (`b_h = tl.zeros([BK,BV], dtype=tl.float32)`,
promoting on load at :214) but **stores back through the cache dtype every
decode step** (:256). So bf16 state requantizes the recurrence **once per
token** — the compounding hazard is real in principle.

Measured on the real geometry (48 v-heads x 128 x 128) with the **real
Qwen3.5-27B gate parameters** read from the checkpoint (`A_log`, `dt_bias`;
layer 0 median per-step retention **0.9977**, max 0.9999 — genuinely long
memory), bf16 state vs fp32 state, 1024 decode steps:

| layer | state relL2 | out relL2 @32 | @256 | @512 | @1024 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.77 % | 0.59 % | 1.50 % | 1.72 % | **1.78 %** |
| 30 | 1.44 % | 0.56 % | 1.27 % | 1.41 % | **1.45 %** |
| 62 | 1.62 % | 0.57 % | 1.38 % | 1.57 % | **1.62 %** |

**The error saturates.** 256 -> 512 adds 15%; 512 -> 1024 adds **3.7%**. The
signature to look for — divergence growing with output length — is **absent**.
This is the `eps/(1-a_bar)` equilibrium: a requantization error injected at step
t is multiplied by the forget gate at every later step, so it decays rather than
accumulates. A synthetic gate sweep confirms the mechanism is the gate and
nothing else — the plateau tracks memory length monotonically (0.09% at fast
decay, 0.20%, 0.41%, 0.88%, **1.40%** at `dt_bias = -6`) and the growth ratio
over 256 steps rises with it (1.03x -> 2.94x) while never leaving a plateau.

For scale: E39 measured fp8 KV at **5.63%** relative L2 per full-attention
layer. **bf16 state costs ~1.8% per GDN layer and buys 3.7x more capacity.**

Supporting evidence that this is an *anticipated* configuration, unlike fp8 KV:
`hybrid_linear_attn.py:243-251` already carries a bf16-state routing rule
("the portable Triton kernels guard negative reads and writes for both FP32 and
BF16 state ... route BF16 state through Triton"), so a bf16 state pool selects a
kernel written for it rather than one that never declared it.

---

### Verdict

1. **Land `--mamba-ssm-dtype bfloat16` + `--block-size 832`.** +57% ceiling
   (81 -> 127), 1.57x resident cost, ~1.8% bounded per-layer error that does
   not grow with generation length. The flag exists and is correctly consumed.
2. **Couple `r` to `N`, then `--mamba-track-interval 1664 --placeholder-state-block-ratio 0.5`.**
   Another +42% on top (127 -> 181). As shipped, raising N alone is a no-op and
   will read as "track-interval does nothing" for the second time.
3. **fp8 KV stays third and stays blocked** (E39): no kernel accepts it at
   page 832, and it is worth 1.15x for 5.63% per-layer error.

**The strongest single recommendation is that `r` should be derived from
`P/N` rather than left as an independent knob defaulting to 1.0.** Every user
who reaches for `--mamba-track-interval` today gets a byte-identical plan and
concludes the feature is dead.

### Rejected approaches

1. **Let the equalizer pick P under bf16 state (i.e. ship the flag alone).**
   P collapses to 448, bytes/token moves 4%, ceiling 105 instead of 127. Half
   the win, and it looks like the flag underperformed rather than like the
   page moved.
2. **Push N to 4P or 8P for more capacity.** The state pool becomes binding and
   the ceiling *falls* (91 at N=4P, 57 at N=8P). N=2P is the optimum.
3. **`--block-size 1280` for bf16 state** (ceiling 209, best in the P sweep).
   Rejected as benchmark-fitting: 1280 exactly equals the 1024-in/256-out shape,
   so it shows zero fragmentation at precisely the measured shape and would
   degrade on any other. Bytes/token, not ceiling@1280, is the shape-independent
   metric.
4. **Judging bf16 state by whether the plateau is "small".** The test that
   matters is whether it *grows*, because a wrong recurrent state produces
   fluent wrong text. Measured to 1024 steps with the checkpoint's own gates
   for that reason.

### What was NOT measured

- **No engine boot at any arm.** All four cards were occupied for the entire
  session (GPU 0 another tenant; GPU 1 the prefill agent's `ts-e39-base`
  at 91-93 GiB; GPU 2 the owner's measurement; GPU 3 the third agent's),
  leaving < 5 GiB on my assigned card against ~54 GiB of weights. **So
  `max_total_num_pages`, bytes/token and `#running-req` under bf16 state are
  the planner's arithmetic, not a live log** — driven through the real
  `ServerArgs`/`ModelConfig`/`mamba2_cache_params` objects, with the fp32 arm
  reproducing the live boot bit-exactly, but not observed.
- **No end-to-end generation comparison against the fp32 arm**, and therefore
  no test with the chunked-prefill split held fixed. E34 showed that changing
  state block sizing changes the chunk decomposition and alters output through
  fp reassociation alone; **bf16 state at a pinned `--block-size 832` should
  not move the split, but that was not verified.** This is the single most
  important missing measurement and it needs one free card.
- **Input distribution is synthetic.** Only the *gates* (`A_log`, `dt_bias`)
  are the checkpoint's; q/k/v/b are random. Real activations could sit at a
  different point on the plateau curve.
- **Prefill was not measured**, only decode. Chunked prefill writes state once
  per chunk rather than per token, so it should be strictly less exposed —
  argued, not shown.
- No throughput, latency or prefix-hit-rate number at any arm (by instruction);
  the limit cycle was not sampled (`period.py` not run).
- **The coarsened prefix-cache granularity that N > P buys is not measured**,
  same open item as E34.
- No production code was changed. Probes live in `tokenspeed/probe_fp8kv_sm120/`.
  `kv_cache_coordinator.{h,cpp}`, `flat_state_slabs.py`, the host-mirror path
  and the C++ admission logic were read only.

---

## E41 — `#new-seq: 1` is not an admission refusal at C=64, it is an empty queue: the scheduler admits 100% of what is waiting, and the budget it leaves unused can only be filled by a mixed batch the MHA backend refuses to execute

**Objective.** E36 named prefill packing as the second serving deficit —
`#new-seq: 1` against a 7,488-token budget with ten requests queued — and
explicitly recorded that *"the mechanism behind `#new-seq: 1` was not isolated…
which predicate declines was not traced."* Trace it. Then compare against the
other three engines' admission loops, fix what the trace says is broken, and
price the fix at C=64 and at the long-prompt shape where the remaining gap
lives.

**Setup.** Qwen3.5-27B bf16, `/workspace/models/Qwen/Qwen3.5-27B`, **GPU 1**,
image `tokenspeed:sm120-dev` with the scheduler extension rebuilt from this
checkout to carry the trace below. Launch exactly as specified:
`--disable-kvstore --enable-placeholder-blocks --enable-metrics`, no
`--attention-backend`, no out-of-tree kernel plugin. The engine's own startup
line, quoted because every number below is relative to it:

```
Scheduler config: block_size=832 num_device_pages=163 num_state_pages=163
  max_scheduled_tokens=7488 decode_input_tokens=1 overlap_schedule_depth=1
  disable_l2_cache=True max_batch_size=160 (global max_num_seqs=160, dp_size=1)
  enable_mamba=True disable_prefix_cache=False
  paged_cache_groups=['linear_attention', 'full_attention']
```

`max_scheduled_tokens=7488` is `chunked_prefill_size` 8192 floored to 9 × 832 by
`aligned_max_scheduled_tokens` — E36's number, reproduced.

**Artifacts.** Traces, bench JSON and the probe live in
`/workspace/roy/tmp/e39/` (`dbg_c64.log`, `dbg_i4096_only.log`, `A_c64.json`,
`A_c32i4096.json`, `probeA.json`, `probe.py`).

**Instrument.** Reading was ambiguous, so the scheduler was made to say what it
did. `TOKENSPEED_SCHED_TRACE` (new, off by default,
`csrc/scheduler/operations/forward.cpp`) emits two lines:

```
round cand=64 sub=1 prefilling=0 decoding=63 other=0 | ops=1 prefill_ops=1
      tokens=1030 budget_left=6458 break=decode_suspended free_h=34 free_s=97
firstchunk-reject rid=… reason=flat_admit chunk=4101 prefill=4101 hit=0 need=10
      dev_free_hit=0 ext=0 budget=1 free_h=1 free_s=133 resv_other=0
```

`round` is one planning round: how many candidates sat in each schedulable
state, what the round built, how much budget went unused, and which predicate
ended the loop. `firstchunk-reject` is one line per waiting request the
first-chunk path declined, tagged with the predicate and its inputs. **The
discriminator is `sub` vs `prefill_ops`** — the engine's own `Prefill batch`
line reports the batch that was built and never the candidates considered, so
it cannot tell "eight were refused" from "one was waiting". Nothing else in the
engine could answer this question, which is why an earlier session guessed.

---

### Result 1 — at C=64 / ILEN=1024 nothing is refused. `#new-seq` equals the queue depth exactly

3,651 planning rounds traced over a 300 s closed-loop run at C=64,
1,030-token prompts, 256 output tokens.

| | |
|---|---:|
| planning rounds traced | 3,651 |
| rounds that built a prefill | 655 |
| **`firstchunk-reject` lines, all reasons** | **0** |
| rounds where `prefill_ops == sub` | **655 / 655 (100%)** |
| rounds where a waiting request was not admitted | **0** |
| max `sub` (waiting requests) ever observed | **3** |

| `sub` (waiting) | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| rounds | 2,996 | 543 | 109 | 3 |
| **resulting `#new-seq`** | — | **1** | **2** | **3** |

`#queue-req` read **0 on all 655 `Prefill batch` log lines**. The scheduler
admitted every request it had, every round, and `#new-seq: 1` is the queue
depth, not a clamp. The task premise — nine queued while one is admitted — does
not reproduce at this operating point.

**It reproduced in E36 because E36 ran without placeholder blocks.** E36's
`#queue-req: 10` is the pool-exhausted regime (53 of 64 running, page ratio
0.98) that E37 fixed; the *symptom* `#new-seq: 1` survived E37, its *cause* did
not. This is the "mechanism present vs. mechanism reached" trap, and it caught
this entry's own first hypothesis.

Two loop-level facts from the same trace matter more than the packing:

| on the 655 prefill rounds | |
|---|---:|
| `break=decode_suspended` | **647 (98.8%)** |
| mean tokens placed, of a 7,488 budget | **1,194** |
| **mean budget left unused** | **6,294** |
| mean requests in `Decoding` skipped by that break | **63** |
| decode-round batch size | 64 (2,237 rounds), 63 (456), 62 (112) |

Every prefill round throws away ~6,294 budget tokens and freezes ~63 decodes,
because `enable_mixed_prefill_decode` is false and the loop breaks at the first
decode candidate once a prefill is pushed
(`forward.cpp`, `if (!config_.enable_mixed_prefill_decode && pushed_prefill) break;`).
Sixty-three decode rows cost 63 of those 6,294 tokens.

### Result 2 — at C=32 / ILEN=4096 the predicate *is* reached, and this is it

The same instrument on the long-prompt shape, 3,148 rounds:

| | |
|---|---:|
| `firstchunk-reject reason=flat_admit` | **6,893** |
| every other reject reason | **0** |
| rounds with ≥1 waiting request where ≥1 was refused | **3,011 / 3,075** |
| median `need` (pages) | 10 |
| median `budget` | **1** |
| median `free_h` / `free_s` | **1 / 133** |
| rejections with `free_h < free_s` | **6,893 / 6,893 (100%)** |

The predicate, verbatim (`csrc/scheduler/operations/forward.cpp`,
`Scheduler::flatAdmitFirstChunk`):

```cpp
if (blocks_needed + device_free_hit_blocks + ext_real_pages > flatFreeBudget(request->Id())) {
    return std::nullopt;
}
```

and `flatFreeBudget` (`csrc/scheduler/scheduler.h`):

```cpp
std::int32_t free_blocks = block_pool_.NumFreeBlocks();
if (config_.FlatStatePoolEnabled()) {
    free_blocks = std::min(free_blocks, flat_state_pool_.NumFreeBlocks());
}
return free_blocks - reserved;   // reserved = OTHER requests' decode reservations
```

So it is a **page** budget, not a token budget, not a per-batch sequence cap and
not a policy about mixing with decode. A waiting request is declined when the
pages its first chunk plus one decode token would pin exceed the free pool.
Three things follow from the numbers:

1. **The history pool binds, alone, in 100% of refusals.** `free_h ≈ 1` while
   `free_s ≈ 133` of 163. At ILEN=4096 a running request pins
   `ceil(4352/832) = 6` history ids for its whole life but only ~1 live state
   id, so at `--placeholder-state-block-ratio 1.0` the state pool sits **~82%
   idle while the history pool is full**. This is E37's caveat 3 —
   *"on long-context/low-sharing shapes the history pool binds first and r < 1
   would be strictly better… no shape other than 1019-in/256-out was
   measured"* — reaching hardware for the first time. It is independent runtime
   evidence for **E40's recommendation to couple `r` to `N`**: E40 argues it
   from the capacity model, this is the pool occupancy a live server actually
   shows at ILEN=4096, and the two agree.
2. **The conservative `min()` gate is second-order here, not the cause.** E37's
   caveat 2 (the gate compares a summed demand against the *binding* pool
   rather than checking each pool) would flip only **1,376 of 6,893 = 20.0%** of
   these refusals if made family-exact, assuming the 50/50 split the two
   832-token groups imply. `free_h == 1` in 73% of them: the history pool is
   genuinely out of pages. Making the gate family-exact is worth doing and is
   *not* worth doing first.
3. **Packing at this shape is token-budget bound, and already near the cap.**
   Mean 5,643 tokens placed of 7,488; `break=token_budget` on 85 of 150 prefill
   rounds; `#new-seq` ∈ {1: 62, 2: 29, 3: 56, 4: 3}. Two 4,101-token prompts do
   not fit in 7,488, so the scheduler chunks. Decode batch sizes are 6/11/23/26
   — a third of the 32 offered sequences are running at all.

### Result 3 — the design comparison: TokenSpeed is the *most* permissive of the four on the axis the deficit was assumed to be on

Read from the vendored checkouts. The question is what happens to candidate
*k+1* when candidate *k*'s KV gate refuses.

| | **TokenSpeed** | **vLLM v1** | **SGLang** | **lmdeploy** | **TensorRT-LLM** |
|---|---|---|---|---|---|
| new seqs per step | unbounded; `token_budget` + `max_batch_size` only | unbounded; `max_num_seqs` only (`scheduler.py:674-676`). **No `max_num_partial_prefills` in this tree** | `prefill_max_requests` (default unlimited) + allocatable-req cap (`schedule_policy.py:1021`, `scheduler.py:3008`) | `max_batches − ready − running` (`paging/scheduler.py:826,848`) | `mMaxNumRequests` (`capacityScheduler.cpp:271`), `maxBatchSizeRuntime` (`microBatchScheduler.cpp:445`) |
| **KV gate refuses → ?** | **`continue`** — the loop tries the next candidate | **`break`** (`scheduler.py:933-940`, `:611-613`) | **`break`** on `NO_TOKEN` (`scheduler.py:3043-3069`) | **`break`** always for KV/eviction (`paging/scheduler.py:352-357`); *skip* only for token-budget overflow (`:629`) | **`break`** in GuaranteedNoEvict (`capacityScheduler.cpp:420-427`, *"If one requests fails to be scheduled, break"*); MaxUtilization pauses the tail instead (`:568-585`); V2 breaks on full ctx, **skips on chunked ctx** (`scheduler_v2.py:530,417-420`) |
| prefill/decode budget | **separate; decode suspended** while a prefill is in the batch unless `enable_mixed_batch` | **one budget**, decode charged 1 token (`:504-511`, `:621`, `:1019`) | **separate; prefill suspends decode** (`scheduler.py:2844-2852`) unless `--enable-mixed-chunk` (default off) | **separate**; a step is prefill *or* decode (`paging/scheduler.py:936-939`); decode has no token budget at all | **one budget**; gen charged `beam_width + draft` (`microBatchScheduler.cpp:428`). V2 budgets **all** gen before any ctx (`scheduler_v2.py:292-298`) |
| reserve for running decodes | `flat_reserved_pages_` = the decode pages promised at admission | `watermark` (default **0.0**, off) but `scheduler_reserve_full_isl=True` demands the whole ISL fit (`kv_cache_manager.py:468-484`) | per-request `remaining_tokens × new_token_ratio`, a closed-loop controller; over-admits and retracts (`schedule_policy.py:489-494`, `schedule_batch.py:2617-2645`) | `prealloc_size` + eviction; starvation handled by *turn* policy (`inputs_maker.py:1164-1180`) | blocks-**to-completion** for every running request before any new one (`capacityScheduler.cpp:279-296`) |
| multiple partial prefills at once | yes | yes | **no** — one `new_chunked_req` slot (`schedule_policy.py:1189`) | **no** — `if seq.kv_token_limit is not None: break` (`:871-872`) | yes, a vector, three chunking policies (`microBatchScheduler.cpp:465-486`) |

Two conclusions, and they invert the working hypothesis:

- **On break-vs-continue, TokenSpeed is already the most permissive of the
  five.** vLLM, SGLang, lmdeploy and TRT-LLM's default C++ scheduler all *stop*
  admitting on the first KV refusal; TokenSpeed keeps trying. Result 2's 6,893
  refusals are 6,893 candidates that *were* considered. There is no
  head-of-line stall to remove — SGLang has to carry an explicit escape hatch
  for exactly that livelock (`schedule_policy.py:651-661`), TokenSpeed does not
  need one.
- **The axis TokenSpeed is actually behind on is the shared budget.** vLLM and
  TRT-LLM put prefill and decode tokens in one pot and charge a decode 1 token.
  That is precisely what would consume Result 1's 6,294 wasted tokens. SGLang
  suspends decode like TokenSpeed and gates the fix behind the same kind of
  opt-in flag; lmdeploy suspends it too but forces a prefill turn every
  `prefill_interval` decodes so neither starves.

### Result 4 — the fix the trace points at exists, is one flag, and is dead on arrival

`--enable-mixed-batch` sets `enable_mixed_prefill_decode`, which reprioritises
decode to 0 (ahead of prefill), removes the `decode_suspended` break, and makes
the 7,488 tokens one shared pot with decode charged
`decode_input_tokens = 1` each — vLLM's design, already implemented in
`newForwardOperation`. It also starts cleanly and logs
`mode=1 (mixed prefill+decode batches supported)`.

It then **kills the engine on the first mixed batch**:

```
Mix batch. #new-seq: 1, #new-token: 1029, #cached-token: 0, #running-req: 2, #queue-req: 0
ERROR - Scheduler hit an exception: …
  File ".../layers/attention/backends/mha.py", line 197, in init_forward_metadata
    assert not forward_mode.is_mixed(), "mha backend does not support mixed batch"
AssertionError: mha backend does not support mixed batch
```

8 s after the first request; **54,371 client failures** in the 180 s smoke run;
SM 11.7%. Grepping every backend: **only `mha.py:197` and `msa.py:200` assert.**
`mla`, `dsa`, `deepseek_v4`, `flashmla`, `trtllm_mla` all handle MIXED
(`flashmla.py:161`: *"Decode and prefill metadata are split so MIXED batches can
plan both"*). Qwen3.5's full-attention half is MHA, so **the second deficit E36
named is blocked in the attention backend, not in the scheduler.** The
scheduler-side work is done; what is missing is one metadata split in MHA —
extend metadata for rows `[0, num_extends)` and decode metadata for the rest,
which `init_forward_metadata` builds one of, never both.

**Update, same session, written after the fact.** While this entry was being
written another agent **implemented the split** (E42). It landed in
`/workspace/roy/tokenspeed` on top of the guard below: `MHAAttnBackend` now sets
`supports_mixed_batch = True`, computed **per instance** by
`mixed_batch_supported(spec_num_tokens, is_draft, draft_block_decode)` — because
capability is not uniform, a speculative-decoding rank's decode tail is a verify
window rather than one token per row — `init_forward_metadata` builds both
metadata halves with a dedicated `_compute_flat_mixed_out_cache_locs`, and the
`assert` became a `NotImplementedError` so `python -O` cannot skip it into
silent corruption. A fuller version of the same work has been sitting in
`/workspace/roy/UnieInfra/experiment/legacy/` since before this session.

So Result 4's conclusion — *"blocked in the attention backend, one metadata
split away"* — was correct as a diagnosis and is **already obsolete as a
prediction**. What it never was is measured: **nobody has yet run
`--enable-mixed-batch` end-to-end and reported a throughput number.** Result 1
says what it is worth going after (6,294 budget tokens and ~63 frozen decodes
per prefill round) and Result 6 says the packing is reachable; the A/B is the
open item, and TTFT must be taken with it because that is the side of the trade
this entry left unpriced.

---

### What changed in the tree

1. **`TOKENSPEED_SCHED_TRACE`** (`csrc/scheduler/operations/forward.cpp`) — the
   admission trace above. Off by default, `getenv` read once into a
   function-local static, so a non-tracing build pays a predictable branch per
   round. Measured cost at C=64, both arms gated **settled** and blocked at the
   period 5 both arms show: **634.7** tok/s traced against **634.8** untraced —
   0.02%, i.e. nothing.
2. **`--enable-mixed-batch` is now refused at startup** instead of asserting
   under load. `AttentionBackend.supports_mixed_batch` (default **True**, so
   backends opt out and a new backend is not silently declared incapable),
   `False` on `MHAAttnBackend` and `MSAAttnBackend`;
   `BaseAttnConfig.enable_mixed_batch` stamped generically in
   `registry._create_attn_config` so no arch's `generate` has to thread it; and
   `registry._check_mixed_batch_support`, called from **both**
   `_create_attn_backend` and `_create_attn_backend_with_name` (the hybrid
   full-attention half goes through the second), raising

   > `--enable-mixed-batch is not supported by attention backend MHAAttnBackend:
   > it cannot serve a ForwardMode.MIXED batch (extend and decode rows in one
   > forward). Re-run without --enable-mixed-batch, or select a backend that
   > supports it.`

   This is a diagnosis fix, not a performance fix, and it is worth its own line:
   the previous behaviour was a server that passes its health check, serves
   correctly for as long as the queue happens to stay empty, and then fails
   every in-flight request at an unpredictable moment.

   *Where it fires:* at backend construction, which is after weight load and
   after `plan_component_tensors`. Still startup, not serving; moving it before
   weight load would mean resolving the backend class before the config exists.
   Not done.

**Tests.** `test/runtime/test_mixed_batch_backend_guard.py`, 9 tests: the
default is capable; MHA and MSA are not; the guard raises with an actionable
message; it is silent when the flag is off and when the backend is capable; it
runs **before** the backend's `__init__` on both creation paths (pinned with a
backend whose `__init__` raises a distinguishable error); the name swap in
`_create_attn_backend_with_name` is undone even when the guard raises; the
config is stamped from server args; an unknown arch still raises first. All 9
pass, plus 38 pre-existing tests in the adjacent
`test_server_args_attention_backends` / `test_placeholder_blocks` /
`test_aligned_max_scheduled_tokens` suites, unchanged. The C++ scheduler
extension rebuilds clean from this checkout with `TOKENSPEED_FLAT_KVCACHE=ON`,
and its gtest suite reads **419 passed / 103 failed of 522** — E37's numbers
exactly, with every failure in the same pre-existing radix
`PagedCacheOverlapRetractTest` set (`test_paged_cache_replay.cpp`) and none in
the flat admission path the trace touches. `clang-format` clean at the repo's
`.clang-format`.

### Result 5 — throughput and latency of the arm that shipped

Control = the run configuration as given. Gated with `stationarity.py` and
blocked at the period `period.py` measured, not at the period `ladder.py`
assumes.

| shape | client steady tok/s | server tok/s, period-blocked | period | stationarity |
|---|---:|---:|---:|---|
| C=64, ILEN=1024, OLEN=256 | 602.7 | **634.8 ± 1.4** | 5 | **settled** (skip 120 s, n=24, drift +0.1%, swing +0.9%) |
| C=32, ILEN=4096, OLEN=256 | 225.8 | **225.2 ± 3.4** | 2 | settled at skip 180 s only — see below |

C=64 reproduces E36 (591.9) and E37 (671.1) on a third day. **The C=32/ILEN=4096
shape does not settle cleanly in 480 s**: the verdict is ENVELOPE-MOVING at skip
120, settled at 180, DRIFTING at 240, settled at 300, while the mean stays
within 222–228 across all four. Reported as a level with that caveat attached,
not as a measurement fit to resolve a few percent.

Idle TTFT, median of 9, unique leading prefix per request, and greedy
correctness, two independent passes with `/flush_cache` and a discarded warm-up
before each:

| | |
|---|---:|
| TTFT ILEN=1024 | **291.0 ms** |
| TTFT ILEN=4096 | **722.6 ms** |

| prompt | prompt tokens | pass 0 | pass 1 | |
|---|---:|---|---|---|
| `primes` | 13 | `0f8f359521b5aa1b` | `0f8f359521b5aa1b` | identical |
| `med_en` | 1,787 | `f99d9e90c1f83266` | `f99d9e90c1f83266` | identical |
| `long_en` | 9,932 | `8d57ca30e95c6639` | `8d57ca30e95c6639` | identical |

E37's probe-conditioning lesson was applied: prompts carry determinate answers
in ordinary English, no random digits, no 6-token prompts. **These are the
control-arm references.** The two shipped changes cannot move them — the trace
is `getenv`-gated and writes only to stderr, and the guard raises or does
nothing — but the references exist so the next change to this path has a bar to
clear.

### Result 6 — give the scheduler a queue and it packs 8 per batch, filling 90% of the budget. `#new-seq: 1` and `#queue-req: 9` never co-occur

`steady_bench.py` staggers its workers over 20 s specifically to break the
cohort, which is why Result 1's queue is empty. Re-run the *identical* control
arm with `STAGGER_S=0` — a lockstep closed loop, the most likely way a real
driver produces a standing queue — and the queue does get deep: `#queue-req`
reads 5, 13, 20, 27, 34, 42, 49, 56, and `sub` reaches **64**.

What the scheduler does with it, over 3,202 rounds / 123 prefill rounds:

| `#new-seq` | 1 | 6 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|
| prefill rounds | **10** | 12 | **75** | 25 | 1 |

| | |
|---|---:|
| rejections, all reasons | **0** |
| rounds ending on `token_budget` | **101 / 123** |
| mean unused budget, of 7,488 | **743 (90% packed)** |
| the 10 rounds with `#new-seq: 1` | all 10 have `sub == 1` |

**Eight**, because 8 × 1,030 = 8,240 exceeds the 7,488 budget and the eighth
prompt is chunked; seven whole prompts is 7,210. That is the budget being
packed to its limit, by the code as it stands, with no change.

So the two halves of the reported observation are mutually exclusive on this
build: **`#new-seq: 1` happens when and only when one request is waiting, and
whenever nine are waiting `#new-seq` is 8.** The one regime that produces
`#new-seq: 1` *with* a standing queue is the page-gate regime of Result 2 — long
prompts, or `--enable-placeholder-blocks` off as in E36.

**A throughput number, and why it is not an A/B.** The lockstep arm completed
**727.8 tok/s** against the staggered arm's 646.6. Do **not** read that as
"packing is worth +12.5%": it is a different arrival process, `stationarity.py`
refuses the run (**DRIFTING**, drift +12.1%, n=12), and its server counter
yielded 8 samples. It is reported only as directionally consistent with E36's
MFU argument — 8-sequence batches at 90% budget occupancy alongside a higher
completion rate than 1-sequence batches at 16% — and as the reason the
mixed-batch work in Result 4 is worth finishing.

### Result 7 — `--placeholder-state-block-ratio 0.5` removes the gate completely and buys ~1.5%. This is the entry's most useful number

The fix Result 2 points at, measured. Same box, same image, same probes, GPU 1
alone. `r = 0.5` publishes `259 history ids, 129 state ids`.

**The mechanism does everything it was supposed to.** At C=32 / ILEN=4096, from
the trace:

| | r = 1.0 (control) | **r = 0.5** |
|---|---:|---:|
| `flat_admit` rejections | **6,893** | **1** |
| decode-round batch size (mode) | 6 / 11 / 23 / 26 | **32 of 32, in 3,496 of 3,538 rounds** |
| `#new-seq` | {1: 62, 2: 29, 3: 56, 4: 3} | {1: 43, 2: 77, **3: 153**, 4: 4} |
| mean unused budget, of 7,488 | 5,470 | **829** |
| rounds ending on `token_budget` | 85 / 150 | 209 / 277 |
| binding pool in refusals | history, 6,893/6,893 | state (`free_h=123, free_s=9`), n = 1 |

The page gate is gone, every offered sequence runs, the budget is 89% full and
the bottleneck has moved across to the state pool — i.e. `r = 0.5` is close to
balanced for this shape. **This is prefill packing, achieved.**

**And it is worth almost nothing.**

| shape | metric | r = 1.0 | r = 0.5 | |
|---|---|---:|---:|---|
| C=64, ILEN=1024 | server tok/s, period-blocked (period 5, both **settled**) | **634.8 ± 1.4** | **634.8 ± 2.8** | **0.0%** |
| C=64, ILEN=1024 | completion rate, t ≥ 180 s | 646.6 | 645.8 | −0.1% |
| C=32, ILEN=4096 | completion rate, t ≥ 120 / 180 / 240 s | 222.5 / 225.2 / 225.0 | 227.5 / 218.4 / 238.8 | **≈ +1.5%, inside the spread** |
| TTFT idle, ILEN=1024 / 4096 | median of 9 | 291.0 / 722.6 ms | 291.5 / 726.0 ms | unchanged |
| greedy `primes` / `med_en` / `long_en` | sha256[:16] | `0f8f3595…` / `f99d9e90…` / `8d57ca30…` | **identical** | ✔ |

**Do not read the window medians on the ILEN=4096 arm.** `steady_bench` reported
**311.1 tok/s** for `r = 0.5` against 225.8 for the control — **+38%, and it is
an artifact.** Freeing the pool lets all 32 requests run, which puts them in a
hard **period-2 lockstep**: the client's window series reads
`542, 1, 516, 28, 331, 214, 144, 401, 1, 542, 3, 388`, and the server counter
(`stream_usage`, which only advances on completion) produced 11 usable samples
out of 24 with SE 45–81. `stationarity.py` refuses the arm at every skip
(DRIFTING at 120/180/240; at 300 the envelope reads −1019%). The completion
rate over the tail is the only metric the cycle cannot alias, and it says +1.5%.
**The capacity fix manufactured the limit cycle that then inflated its own
result** — which is exactly the failure `period.py` and `stationarity.py` exist
to catch, catching it.

**Why restoring the sequences buys nothing here, and what that corrects.** E36
priced the 53→64 batch deficit at ~+21% by arguing the decode step is
weight-bound (51.25 GB of weights against a batch-32 KV working set), so extra
sequences ride nearly free. **That argument holds at ILEN=1024 and fails at
ILEN=4096.** Going from ~11–26 to 32 concurrent sequences — roughly 2x the
resident KV — moved throughput by ~1.5%, so the step must have lengthened
almost proportionally: at 4,096-token contexts attention, not weights, sets the
decode step. The standing extrapolation *"restoring the missing sequences adds
21% of tokens"* should be read as shape-specific to ~1k prompts, not as a
property of this engine. No profiler was run to confirm the attention/weight
crossover directly; this is inference from the two batch sizes and the two
throughputs.

**Recommendation.** `r = 0.5` is free (0.0% at C=64, byte-identical output,
TTFT unchanged) and it removes a gate that refuses 6,893 admissions per 5-minute
run. Ship it as the default *for its own sake* — a scheduler that can admit what
is offered is easier to reason about, and E40 wants `r` coupled to `N` anyway —
but **do not ship it expecting throughput.** At ILEN=4096 TokenSpeed's ~14.7%
gap to vLLM is not the page pool.

---

### Rejected approaches

- **"Pack more prefills per batch."** The stated fix for the stated
  observation. Rejected at C=64/ILEN=1024 by Result 1: there is nothing to pack,
  100% of waiting requests are already admitted and the deepest queue in 3,651
  rounds was 3. Rejected at C=32/ILEN=4096 by Result 2 point 3: the budget is
  already 75% full and two prompts do not fit in 7,488 tokens.
- **Making the loop `break` rather than `continue` on refusal, to match vLLM.**
  Backwards. Result 3 shows TokenSpeed is already the permissive one; every
  other engine breaks. Changing this could only lose admissions.
- **A family-exact (per-pool) admission gate**, i.e. E37's caveat 2. Correct,
  but priced at **20.0%** of the ILEN=4096 refusals and 0% at C=64, and it means
  threading `PoolDemand` through eight call sites in `forward.cpp` plus
  `kv_cache_coordinator.{h,cpp}` — a file another agent is editing concurrently.
  Deferred, with the number attached so it can be ranked honestly.
- **Flipping `enable_mixed_batch` to default True.** Cannot: it does not run
  (Result 4). Even once MHA supports MIXED, the default touches every model and
  every path — spec decoding, PD, radix — none measured here.
- **Raising `chunked_prefill_size` to pack more per round.** E36's own TTFT
  curve peaks at ILEN=4096 (80% MFU) and falls back to 75% at 8192, so a bigger
  budget buys packing at a worse point on the curve. Not measured, not pursued.
- **Deferring a prefill by one round to let the queue accumulate.** Would
  manufacture the packing opportunity Result 1 says does not exist, by spending
  TTFT — the one thing the task said not to trade blindly. Not implemented.

### What was NOT measured

- **`r` other than 1.0 and 0.5.** Result 7 measured 0.5 only. 0.25 (365/91)
  was not run, and Result 7's finding that the bottleneck has already crossed
  to the state pool at 0.5 (`free_h=123, free_s=9`) suggests 0.25 would be
  worse, not better, at this shape. Untested.
- **`r` composed with `--mamba-track-interval`.** E40's actual recommendation is
  the pair (`N = 1664, r = 0.5`); Result 7 ran `r` alone. The two were never in
  the same server, here or in E40.
- **Any fix for the actual deficit.** Result 1's 6,294 wasted budget tokens and
  63 frozen decodes per prefill round are still there. Nothing in this entry
  makes the engine faster; the shipped changes make the problem *visible* and
  make one failure mode *loud*.
- **MHA mixed-batch support.** Scoped (one metadata split, plus whatever the
  GDN half of `hybrid_linear_attn.py` needs for a mixed batch) but not
  attempted. The GDN side was not even surveyed.
- **A valid throughput A/B for packing itself.** Result 6's lockstep arm is a
  different arrival process, not a control, and it does not pass stationarity.
  Packing's worth is still unpriced; only its *reachability* is established.
- **No vLLM arm.** The 641.1 comparison is quoted from E23/E36.
- **No profiler.** No ncu, no nsys, no torch trace. The isolation anchor was not
  re-taken; GPU 1 was verified idle by `nvidia-smi` memory (not `sm%`, per E22)
  before each control launch, and the one run where that was violated is
  discarded below.
- **TTFT under load, TPOT, and p99 anything.** TTFT is idle-server only, so the
  latency half of the packing trade-off is *unpriced*. If MHA ever supports
  MIXED, that is the number to take first.
- **One model, bf16, TP=1, two shapes.** No C=128, no C=1, no shared-prefix or
  RAG shape, no `--mamba-track-interval` interaction.

### Three shared-box hazards, recorded because each cost someone real time

1. **A co-tenant benchmarked my server and would have called it vLLM.** Another
   agent's `docker run … --name vllm-bench -p 8101:8000` failed to bind
   because my TokenSpeed container already held host port 8101 — its container
   sat in `Created` while *its load driver ran anyway*, at CONC=80, against
   127.0.0.1:8101, i.e. against **TokenSpeed**, writing the result to
   `vllm_c80.json`. It is visible in my engine log as `#running-req: 80` on a
   C=64 run. My first C=64 arm is discarded from t≈300 s (windows 0–19,
   602–717 tok/s, are clean and agree with the final number); their arm is
   presumably worthless. The port was freed and their container started. **On a
   shared box, bind a distinctive port and check `docker ps -a` for `Created`,
   and treat `#running-req` > your own concurrency as proof of a second
   client.**
2. **I destroyed two other agents' containers with a careless cleanup glob.**
   `docker rm -f $(docker ps -aq --filter name=ts-e4)`, intended to remove my
   own `ts-e41-*`, matched **`ts-e42-mix`** and **`ts-e43-B`** — two other
   agents' servers, both a few seconds into weight loading. `docker rm -f`
   destroys the config, so they could not be restarted and their launch
   commands are lost; those agents have to relaunch from their own notes.
   Nothing was mid-measurement, but that was luck. **Name containers with a
   unique suffix and delete them by exact name, one at a time, never by prefix
   glob on a shared box** — `ts-e4` is a prefix of every E4x entry anyone will
   ever run.
3. **GPU memory is not released when a container is removed.** Starting the next
   arm immediately after `docker rm -f` gave
   `Failed to CUDA calloc 536870912 bytes` — the previous engine's 90 GB was
   still held by an orphaned `tokenspeed::scheduler_0` that outlived its
   container and had to be found through `nvidia-smi
   --query-compute-apps` and killed. Poll for free memory between arms.

---

**Verdict.** The predicate that stops prefill admission is
`Scheduler::flatAdmitFirstChunk`'s page gate,
`blocks_needed + device_free_hit_blocks + ext_real_pages > min(free_history, free_state) − others' reserves`.
**At C=64 / ILEN=1024 it never fires** — 0 refusals in 3,651 rounds, `#new-seq`
equals the queue depth exactly, and the queue is empty because the scheduler
admits instantly. `#new-seq: 1` there is arrival-limited and is not a bug. At
C=32 / ILEN=4096 it fires 6,893 times, and in **100%** of them the history pool
is the binding one while the state pool sits ~82% idle — a
`--placeholder-state-block-ratio` problem, i.e. E37's untested caveat, not a
packing problem.

The reachable lever at C=64 is not packing at all: it is the **6,294 budget
tokens and 63 frozen decodes** thrown away by every prefill round because
prefill suspends decode. TokenSpeed already implements the fix
(`enable_mixed_batch`, vLLM's shared-budget design) and **the MHA attention
backend cannot execute the resulting batch**, so the flag is a latent crash.
That is now a startup refusal with a message, and the scheduler can now be asked
what it did instead of guessed at. The split itself was implemented by another
agent (E42) while this entry was being written; **the throughput A/B for it is
still unrun**, and it is the single most valuable measurement left on this path.

And the packing that *was* achievable was achieved, and measured, and it is
worth nothing: `r = 0.5` takes the ILEN=4096 refusal count from **6,893 to 1**
and the decode batch from ~11 to **32 of 32**, for **+1.5%** — while
`steady_bench`'s window medians claimed **+38%** because the fix put the
workload into a period-2 limit cycle. Two conclusions worth more than the flag:
**the long-prompt gap to vLLM is not the page pool**, and E36's "restoring the
missing sequences is worth ~21%" is a property of ~1k prompts, not of this
engine.

---

## E43 — MHA can serve mixed batches, and doing so ends the decode freeze without reclaiming the wasted budget: −7.5% at C=64/ILEN=1024, not resolvable at C=32/ILEN=4096

*(E42 is the concurrent `--placeholder-state-block-ratio` entry appended above; numbered E43 to avoid a collision rather than a fourth duplicate number in this file.)*

**Objective.** E41 ended by scoping the fix it did not attempt: "*one metadata
split in MHA — extend metadata for rows `[0, num_extends)` and decode metadata
for the rest*", plus "*whatever the GDN half of `hybrid_linear_attn.py` needs for
a mixed batch. The GDN side was not even surveyed.*" Implement it, and price it
at both shapes — including the latency half of the trade, which E41 recorded as
idle-only and named the first number to take once MHA supported MIXED.

**Setup.** Qwen3.5-27B bf16, `/workspace/models/Qwen/Qwen3.5-27B`, **GPU 2**,
image `tokenspeed:sm120-dev`, `--disable-kvstore --enable-placeholder-blocks
--enable-metrics`, no `--attention-backend`, no kernel plugin, `--enable-mixed-batch`
on the treatment arm only. Both arms reported the identical scheduler geometry,
so the comparison is not confounded by sizing:

```
Scheduler config: block_size=832 num_device_pages=163 num_state_pages=163
  max_scheduled_tokens=7488 decode_input_tokens=1 overlap_schedule_depth=1
  max_batch_size=160 enable_mamba=True disable_prefix_cache=False
  paged_cache_groups=['linear_attention', 'full_attention']
```

Treatment additionally logged `prefill breakable graph ENGAGED (text): bucket=16
mode=1 (mixed prefill+decode batches supported)` — i.e. the flag engaged rather
than being silently ignored. Edited Python was bind-mounted over the installed
package (`mha.py`, `hybrid_linear_attn.py`); no rebuild, so both arms share one
compiled image and differ only in those two files plus the flag.

**Artifacts.** `/workspace/roy/tmp/e42/` — `{ctl,mix}_c64i1024.json`,
`{ctl,mix}_c32i4096.json`, `probe_{ctl,mix}.json`, `ttft_loaded_{ctl,mix}.json`,
`{ctl,mix}_log_*.txt`, `budget.py`, `ttft_loaded.py`.

**Instrument.** Two new probes, because nothing existing answered the question.
`budget.py` reads the engine's own round lines and accounts `#new-token` against
`max_scheduled_tokens`. It must match **`Mix batch.`** as well as `Prefill
batch.` — `model_executor.py:1284` logs `mode = "Prefill" if num_extends == bs
else "Mix"`, so a regex written against the control arm sees ~3% of the
treatment's rounds and reports a phantom collapse in prefill activity. That
mistake was made and caught here. `ttft_loaded.py` holds C background streams in
flight and injects probes, because idle TTFT is structurally blind to a packing
trade: with nothing to mix against, nothing mixes.

---

### Result 1 — the scoping was half right: MHA was one split, the GDN half was an unsurveyed second blocker

MHA needed exactly what E41 said. `hybrid_linear_attn.py` needed an independent
fix that E41 explicitly never looked at, and it fails in the *worse* direction.
`model_executor.py:2008-2018` passes `extend_prefix_lens_buf[:num_extends]`
while `seq_lens` stays `[bs]`; the mamba metadata then computes

```python
extend_lens = (seq_lens[:bs] - extend_prefix_lens[:bs]).to(torch.int32)
```

| case | behaviour |
|---|---|
| `num_extends == bs` (pure EXTEND) | correct — today's path |
| `1 < num_extends < bs` | `RuntimeError`, size `bs` vs `num_extends` |
| **`num_extends == 1`, `bs > 1`** | **silently broadcasts** — every decode row gets row 0's prefix, `query_start_loc` runs past the real token count, conv/scan read out of bounds |

The silent case is the common one: prefill rounds here admit 1–3 requests, and
`#new-seq: 1` was 684 of 975 control rounds. Had only MHA been fixed, the engine
would have run and produced quietly wrong linear-attention state.

Also found: `HybridLinearAttnBackend` inherited `supports_mixed_batch = True`
from the base class and never consulted its children, so E41's startup guard was
protecting Qwen3.5 only by the accident that the **MHA sub-backend** is built
through the guarded factory (`registry.py:506`). The composite now states the
AND of its halves, pinned by a test.

### Result 2 — design survey: split, don't fuse. Three of four engines agree, and this tree already measured it

| engine | mixed metadata | one call or two | budget sharing | anti-starvation |
|---|---|---|---|---|
| **vLLM v1** | one `CommonAttentionMetadata`; `split_decodes_and_prefills` derives counters (`utils.py:564`) | **backend-dependent** — FlashAttention ONE, FlashInfer/MLA **TWO** into slices of one output | one budget, RUNNING drained first (`scheduler.py:471`) | running-first + `long_prefill_token_threshold` (default 0 = off) |
| **SGLang** | one unified batch; decode appended as extend rows of length 1 | **ONE** varlen call, `max_seqlen_q` from the longest prefill | decode reserved up front: `rem_input_tokens -= num_mixed_decode_tokens` | reservation + `chunked_prefill_size`; `enable_mixed_chunk` off by default |
| **lmdeploy** | **no mixed batch** — `is_decoding: bool` per batch | one call, homogeneous | temporal alternation (`prefill_interval=16`) | interval policy both directions |
| **TRT-LLM** | one metadata + split index `num_contexts` | **TWO** — `PhasedFmha.run_context` then `run_generation`, own `input_seq_length` each | phase 1 generation, phase 2 context fills remainder | gen-before-ctx, implicit skip not break, deadlock detector |
| **TokenSpeed (this change)** | **two objects**, pre-sliced | **TWO** | scheduler-side, decode-first priority | scheduler's, unchanged |

The discriminator is not "is this row a decode" but **whether the decode kernel
is a different algorithm**. Here it is: `mha_decode_with_kvcache` /
`mha_decode_prepare` versus `mha_extend_with_kvcache` / `mha_extend_prepare`.
TRT-LLM's `PhasedFmha` deliberately keeps the longest prefill's q-length out of
the generation launch shape; vLLM's FlashInfer decode wrapper is a separate
split-KV kernel that *requires* uniform q-length.

Decisive, and found only because it was searched for: the quarantined fork at
`/workspace/roy/UnieInfra/experiment/legacy/` contains an older, fuller
implementation that reached the **same** two-call design and left the
measurement behind it:

> Measured on the B64 shape (1 × 3072 prefill + 63 decode rows over 3k of cache,
> Qwen3-8B/TP1/sm_120, triton): **5.37 ms fused into one extend call vs 2.91 ms
> split** — the decode tail costs 3.3× more inside the prefill kernel (3.74 vs
> 1.12 ms) because that kernel tiles queries at BLOCK_M=128 with no split-K, so a
> `seqlen_q == 1` row occupies a whole tile and scans its cache with one block.
> Over 36 layers that is **+89 ms per mixed step**.

So the fused variant was not re-measured here; it is already refuted in-tree.
The legacy fork also gates MIXED per *instance* on spec/draft config, which this
change adopts as an explicit refusal.

### Result 3 — one ordering correction, which inverts any slicing copied from vLLM

The brief (quoting `split_decodes_and_prefills`) states the batch is ordered
`decode → short_extend → long_extend → prefill`. **That is vLLM's ordering. The
function does not exist in TokenSpeed** (zero hits repo-wide). This engine is
**prefill-first**, from `FlatForwardOperation`'s constructor:

```cpp
std::stable_partition(ops.begin(), ops.end(),
    [](const ForwardOperation& a) { return std::holds_alternative<PrefillOperation>(a); });
```

pinned by `test_fsm_and_scheduling.py`: *"Layout is prefill-first/decode-second
(FlatForwardOperation::stable_partition)"*, `op.request_ids == ["r1", "r0"]` with
r1 the prefill. Rows `[0, num_extends)` are extend, `[num_extends, bs)` decode —
which is what `mha.py`'s own comment said and what MLA/DSA/flashmla already
slice by. Verified before following; the slices are the mirror image of vLLM's.

### Result 4 — the change

1. **`mha.py`** — `_init_mixed_forward_metadata` builds extend metadata over
   `[0, num_extends)` and decode metadata over `[num_extends, bs)`, each
   pre-sliced (rather than carrying a `num_extends` discriminator as MLA does)
   so `_prepare_decode_kernel` plans the rows that actually run. Extend-metadata
   construction was factored into `_build_extend_metadata`, shared with the pure
   path. `_forward_mixed` calls the two kernels over disjoint token slices and
   assembles one output. `supports_mixed_batch = True`; spec/draft configs raise.
2. **Padding.** `_scrub_extend_padding` reads `cu_extend_seq_lens_cpu[-1]`, which
   on a mixed batch ends **where the decode rows begin** — applying it to the
   full `q` would zero every decode row. `_forward_mixed` scrubs once by the
   combined count (`_scrub_mixed_padding`) and hands each half an exact-length
   view, on which the original is a no-op. Both counts are host ints from the
   pinned mirrors; no device read was added to the step path.
3. **No double-planning.** `_prepare_extend_kernel` and `_prepare_decode_kernel`
   each run exactly once per step, as on the homogeneous paths.
4. **`hybrid_linear_attn.py`** — `_widen_mixed_extend_kwargs` widens
   `extend_prefix_lens` to `[bs]` before any reader (decode row prefix =
   `seq_len - 1`, the identity `_state_page_bounds` already uses). One edit fixes
   `query_start_loc`, `extend_seq_lens_cpu`, the `has_initial_state` gate, the
   padding scrub, the state-page bounds and the radix track block. MIXED+spec
   raises rather than bypassing the verify machinery.

**Tests.** `test/runtime/test_mixed_batch_backend_guard.py` — 11 pass; MHA moved
to the capable side, MSA kept as the incapable case so the guard's refusal branch
is still exercised, and a new test pins the composite as the AND of its halves.

### Result 5 — correctness

| prompt | control | treatment | |
|---|---|---|---|
| `med_en` (1,787 tok) | `f99d9e90c1f83266` | `f99d9e90c1f83266` | **identical** |
| `long_en` (9,932 tok) | `8d57ca30e95c6639` | `8d57ca30e95c6639` | **identical** |
| `primes` (13 tok) | `953c…` / `0f8f…` | `0f8f…` / `953c…` | unstable **on both arms** |

Both long references match E41's, so the control reproduces and the treatment is
bit-identical to it. `primes` returns two different hashes across two passes of
the *same* binary on *both* arms — the same two values, in swapped order. That is
E37's near-tied-logit instability, not a regression, and it disqualifies `primes`
as a discriminator. `--chunked-prefill-size` was held at its default 8192
(`max_scheduled_tokens=7488`) throughout. 0 failed requests in every run.

### Result 6 — throughput, gated

`client_captures_frac` left 1.0 ± 2% on three of four runs (0.844–1.036), so per
the contract every number below is **`server_tok_s`, blocked at the period
`period.py` measured**. The two arms do not share a period — E23's exact hazard.

| shape | arm | period | period-blocked server tok/s | stationarity |
|---|---|---|---|---|
| C=64 / ILEN=1024 | control | 5 | **636.5 ± 1.6** | settled @120 |
| C=64 / ILEN=1024 | treatment | 2 | **588.7 ± 2.2** | ENVELOPE-MOVING @120 (settled @60: 586.6 ± 2.1) |
| C=32 / ILEN=4096 | control | 1 | **221.8 ± 7.8** | DRIFTING at every skip |
| C=32 / ILEN=4096 | treatment | 2 | **230.4 ± 7.5** | settled @300 only |

**C=64 / ILEN=1024: −7.5%.** The treatment mean is stable across skips
(586.6–588.7 period-blocked; 569–571 window mean), so the sign is not a skip
artifact even though the arm fails the gate.

**C=32 / ILEN=4096: +3.9%, not resolvable.** 230.4 − 221.8 = 8.6 against a
combined SE of ~10.8. The raw `steady_server_tok_s` fields say 256.0 vs 227.5
(**+12.5%**) — that is the aliasing trap, and it is the number this entry would
have reported without gating. Control reproduces E41's C=64 figure (636.5 vs
634.8) and E41's C=32 band (221.8, inside its 222–228).

### Result 7 — the mechanism worked; the metric it was aimed at did not move

| C=64 / ILEN=1024 | control | treatment |
|---|---|---|
| round kinds | 975 `Prefill`, 0 `Mix` | **774 `Mix`**, 5 `Prefill` |
| mean batch size on prefill rounds | **1.42** | **56.6** |
| median `#new-token` of 7,488 | 1,030 | 1,092 |
| **implied wasted budget, median round** | **~6,458** | **~6,396** |

| C=32 / ILEN=4096 | control | treatment |
|---|---|---|
| round kinds | 15 `Prefill` | **186 `Mix`** |
| mean batch size on prefill rounds | 2.27 | **25.5** |
| mean wasted budget | 1,698 (22.7%) | **1,143 (15.3%)** |

**The decode freeze is gone.** Prefill rounds went from carrying 1.42 requests to
56.6 at C=64: ~55 decodes per round that were previously suspended now ride along,
and separate decode rounds fell from 190 to 95. `break=decode_suspended` cannot
occur — the branch is compiled out by the flag.

**The wasted budget did not shrink at the shape it was measured at.** 6,458 →
6,396 at C=64/ILEN=1024, which is nothing. This is not a failure of the
implementation; it is E41's own Result 1 read correctly. That result found
**zero** admission refusals in 3,651 rounds and `#new-seq` equal to the queue
depth, with `#queue-req: 0` on all 655 prefill lines — the budget is unused
because **there is no queued prefill work**, not because prefill is blocked.
Mixing moves decode *into* the prefill round; it cannot invent prefill work to
fill a budget that is idle for want of arrivals. The 6,294 tokens and the 63
frozen decodes were one sentence in E41 but they are two different deficits, and
only the second was ever addressable this way.

**A correction to the brief.** It attributes "a mean 6,294 of its 7,488 budget
tokens … at C=32 with 4096-token prompts". The 6,294/98.8% figures are E41's
**Result 1**, which is **C=64 / ILEN=1024**. E41's C=32/ILEN=4096 result wasted
1,845 of 7,488. The control here reproduces both (~6,458 and 1,698), so the
deficit named in the brief lives at the shape where this change *loses*.

### Result 8 — the latency trade, which is flat

| | control | treatment |
|---|---|---|
| idle TTFT, ILEN=1024 | 289.1 ms | 291.4 ms |
| idle TTFT, ILEN=4096 | 719.6 ms | 720.5 ms |
| **loaded TTFT median**, C=32/ILEN=4096 | **884.3 ms** | **865.5 ms** |
| loaded TTFT p90 | 1,395.8 ms | 1,269.4 ms |
| loaded TPOT median | 51.15 ms | 51.57 ms |
| loaded TPOT p90 | 303.6 ms | 316.0 ms |

Idle TTFT reproduces E41 (291.0 / 722.6) and is unchanged — as it must be, since
an idle server never builds a mixed batch. Under load **TTFT does not regress**;
median and p90 both move slightly in the treatment's favour and TPOT slightly
against, all within the spread of n=12 probes. The feared trade — packing prefill
into decode rounds buying TTFT at TPOT's expense — is **not visible at this
shape**. The regression at C=64 is therefore a throughput cost, not a latency
one, and is not explained by the packing trade.

### Why it costs throughput at C=64 (mechanism, partly inferred)

Two costs, both paid per decode token and both scaling with the 64-layer depth:

1. **Mixed batches are eager.** Decode CUDA graphs are refused for MIXED
   (`init_forward_metadata_capture_cuda_graph` asserts `not is_extend_or_mixed`).
   Control ran 190 decode rounds as captured replays; treatment folds that decode
   work into 774 eager `Mix` rounds. At ILEN=1024 a round carries ~1,092 tokens,
   so there is little prefill work to amortise the launch overhead against.
2. **The GDN half loses its decode specialisation.** 48 of 64 layers are linear
   attention, and a decode row inside a mixed batch goes through
   `causal_conv1d_fn` + `gdn_chunk_prefill` (padded to a 16- or 64-token chunk)
   instead of `causal_conv1d_update` + `gdn_decode_step`.

At C=32/ILEN=4096 each round carries ~6,345 tokens, the overhead amortises, and
the result lands at "not resolvable" instead of −7.5%. **This ordering is
consistent with the two shapes but was not confirmed with a profiler** — no
ncu/nsys/torch trace was taken, and the split between cause 1 and cause 2 is
unmeasured.

---

### Rejected approaches

- **Fusing both halves into one varlen extend call** (SGLang's design, and
  vLLM's plain-FlashAttention design). Refused on the legacy fork's in-tree
  measurement: 5.37 ms fused vs 2.91 ms split on a 1×3072 + 63-decode batch,
  +89 ms per step over 36 layers, because a `seqlen_q == 1` row occupies a whole
  BLOCK_M=128 tile. Not re-measured; the existing number is specific enough.
- **Carrying `num_extends` into the decode kernel call** (MLA's pattern,
  `page_table[num_extends:]` at call time) instead of pre-slicing. Rejected
  because `_prepare_decode_kernel` runs from the metadata hook: an unsliced
  metadata plans the decode kernel over the extend rows' sequence lengths, i.e.
  schedules for rows that will not run. MLA can afford it because drafters
  re-point the discriminator; MHA has no such caller on this path.
- **Fixing MHA only, per the brief's scoping.** Would have run and been silently
  wrong — see Result 1's `num_extends == 1` broadcast.
- **Making `_check_mixed_batch_support` per-instance.** Attempted by the
  concurrent agent, broke these tests, reverted. The composite property added
  here achieves the same coverage without moving the guard.
- **Defaulting `--enable-mixed-batch` on.** Refused, and now on evidence rather
  than on the absence of it: it loses 7.5% at the shape TokenSpeed had just drawn
  level with vLLM on.

### What was NOT measured

- **A profiler.** No ncu/nsys/torch trace. The eager-vs-graph and GDN-decode-
  specialisation costs are inferred from two shapes, not attributed. Splitting
  those two is the next measurement and would say whether the C=64 loss is
  recoverable (e.g. by capturing a graph for the decode half of a mixed batch).
- **A rebuilt image.** Both arms ran the shipped `tokenspeed:sm120-dev` with two
  bind-mounted Python files. Equivalent for pure-Python edits and it keeps the
  compiled half identical across arms, but `docker build` was not exercised.
- **The spec/draft refusals.** Both new `NotImplementedError` paths are reasoned,
  not run — this configuration has `speculative_algorithm=None`. They were added
  after the benchmark arms were launched; unreachable in this config, so the
  measured binary and the final source differ only there.
- **C=128 and C=1.** The two shapes disagree in sign, so the crossover between
  them is unlocated and nothing here extrapolates to higher concurrency, where a
  prior result has TokenSpeed ahead.
- **vLLM.** No vLLM arm was run; the 641.1 / 639.9 comparisons remain E23/E36's.
- **Multi-request prefill rounds under mixing.** `#new-seq` fell 1.42 → 1.15 at
  C=64 (the decode-first priority inversion costs prefill admission slots). Its
  effect on tail latency for queued requests is unpriced.
- **Prefix caching, PD, DP, and the fused-RoPE-prewrite models.** MHA gates
  prewrite to decode, so `select_out_cache_loc` on a MIXED batch is untested;
  no model in this run uses it.
- **`--chunked-prefill-size` sweeps.** Held fixed by design, so nothing here says
  whether a smaller chunk changes the C=64 sign.

---

**Verdict.** MHA can serve mixed batches, and the assert was not guarding an
invariant worth keeping — it was guarding an unwritten metadata split, which is
now written and correct (bit-identical greedy output on both long references).
E41's scoping was accurate for MHA and incomplete for the engine: the GDN half
needed an independent fix that fails *silently* when exactly one prefill is
admitted, which is the common case, so shipping the MHA half alone would have
corrupted linear-attention state rather than crashing.

The feature does what it was designed to do and does not do what it was wanted
for. **The decode freeze is eliminated** — prefill rounds go from 1.42 to 56.6
requests, ~55 previously-suspended decodes per round. **The 6,294 wasted budget
tokens do not come back**, because E41's own Result 1 already established that
the budget is idle for want of arrivals, not for want of permission; mixing
relocates decode work, it does not manufacture prefill work. Those were two
deficits in one sentence and only one was reachable from here.

Priced: **−7.5%** at C=64/ILEN=1024 (636.5 → 588.7), **+3.9% and not resolvable**
at C=32/ILEN=4096 (221.8 → 230.4, SE ±7.8/±7.5). TTFT does not regress at either
end — loaded median 884.3 → 865.5 ms, TPOT 51.15 → 51.57 ms — so this is a
throughput cost, not the latency trade that was anticipated. The likely cause is
that mixed batches cannot use the decode CUDA graph and the GDN half loses
`gdn_decode_step` across 48 of 64 layers; both amortise at 4k prompts and do not
at 1k, which matches the sign flip. **`--enable-mixed-batch` should stay off by
default**, and the next move is a profiler splitting those two costs, not another
A/B.

---

## E44 — Coupling `r = P/N` makes `--mamba-track-interval` reach the sizing at last: the ceiling moves 80 → 126 running exactly as E40's arithmetic said, and buys ~10% at 1σ, because capacity has stopped being the binding constraint

**Objective.** E34 landed `--mamba-track-interval` (stride N for retained GDN
state snapshots) and it has been inert twice, for two different reasons. E34
blamed dense slabs; E37 removed the density with placeholder blocks; E40 found
the actual cause and E41 corroborated it from the opposite direction. Neither
owned the fix. The cause: `plan_component_tensors` charges a history block id
`sum(kv_row) + r * sum(state_row)` where `r = placeholder_state_block_ratio`
**defaulted to 1.0 and was an independent knob**, and **N does not appear in
the sizing expression at all**. Raising N lowers the state group's *pin rate*
(1/N ids per token instead of 1/P) while the pool it draws from keeps its size,
so the pages the coarser grid stops needing are never handed to the history
pool. This entry couples them and boots the result.

---

### Result 0 — the gap, confirmed from the code before changing anything

Driving the real `flat_memory_plan` objects against Qwen3.5-27B's config
(64 layers = 16 full + 48 GDN, head_dim 256, 4 KV heads, 48 linear value heads,
fp32 SSM; conv `(10240, 3)` bf16, ssm `(48,128,128)` fp32) reproduces E34/E37's
published geometry to the byte before any edit: equalized **P = 832**,
`kv_bytes_per_block` = 54,525,952 (26.16%), `state_bytes_per_block` =
153,944,064 (**73.84%**), `block_bytes` = **208,470,016**, 163 pages.

With `r` pinned at 1.0, N = 832 / 1664 / 3328 / 6656 produce **byte-identical
plans** — same `block_bytes`, same 163/163 page split, same ceiling 81. That is
the inertness, demonstrated rather than inferred, and it is now pinned by
`test_ratio_pinned_at_one_makes_the_interval_inert`.

**Why `r = P/N` is the right coupling, and not a tuning constant.** A request of
T tokens pins `ceil(T/P)` history ids and `ceil(T/N)` state ids — a demand ratio
of **P/N** for T >> N. Size the two pools in that proportion and they exhaust
together; leave r = 1 at N = 2P and the state pool sits half idle while the
history pool binds, which is precisely the asymmetry E41 measured
(`flatAdmitFirstChunk` refusing 6,893 times with `free_history ≈ 1` and
`free_state ≈ 133/163` in **100%** of refusals).

**Why the ceiling turns over past N = 2P, i.e. why this is not monotone.** Once
`T < N` the state demand floors at one snapshot per request while supply keeps
shrinking with r. The crossover is at T ≈ N, so at ILEN 1024–1280 the optimum
is N = 2P and every larger N *lowers* the ceiling. This is a discretisation
effect, not a modelling artifact, and it is the reason the change warns instead
of just deriving.

---

### Path — what changed

Three files, pure Python, no kernel or scheduler change.

- **`configs/flat_memory_plan.py`**: new `resolve_state_block_ratio(block_size,
  state_interval, override=None) -> StateBlockSplit`. It is the single place
  the coupling is written down, returns `ratio` **and `source`** (`"derived"` /
  `"override"`), and exposes `matched_ratio` (P/N) and
  `interval_past_turnover` (N > 2P). Module docstring gains a section saying r
  and N are *not* orthogonal, since believing they were is what made this inert.
- **`utils/server_args.py`**: `placeholder_state_block_ratio` default
  `1.0 -> None`. `None` means *derive*; any explicit value still wins, so
  `--placeholder-state-block-ratio` remains a direct override (E41's arm still
  reproduces exactly).
- **`layers/attention/registry.py`**: the flat-GDN branch resolves the split
  through the new function instead of reading the raw arg, and the startup log
  now carries the resolved value **next to the page counts** — the whole reason
  this was inert twice is that nothing said so.

```
Flat GDN KV profile: block_bytes=131497984 (112 component tensors, block_size=832),
  max_total_num_pages=259, state_pool_pages=129, state_block_ratio=0.5000 (derived)
Placeholder blocks ON: ... (259 history ids, 129 state ids, ratio 0.5000 derived,
  demand-matched P/N = 0.5000); attention page P=832, snapshot stride N=1664;
  resident cost 375847 -> 158050 bytes/token (2.38x capacity);
  prefix-cache hit granularity 1664 tokens (COARSENED from 832)
```

Two log defects fixed on the way: the old line printed
`prefix-cache hit granularity UNCHANGED at %d tokens` while passing it N, so it
claimed "unchanged" *for the exact configuration that changes it*; and
`max_total_num_pages` was printed without the state count or the ratio.

Past N = 2P the engine now emits a **warning** naming the turnover:

```
--mamba-track-interval 3328 is past 2x the attention page (832): a request shorter
than N=3328 pins a floor of one state snapshot no matter how coarse the grid, while
the state pool keeps shrinking with r=P/N, so the concurrency ceiling TURNS OVER.
```

**Rejected: clamping N at 2P.** The turnover is request-length dependent
(crossover at T ≈ N), and at ILEN 4096 the offline model still prefers N = 4P
or more. A clamp would silently overrule a user who knows their shape; a warning
that names the crossover length does not. **Rejected: deriving r from a measured
request-length distribution.** It would make startup geometry depend on traffic,
which is not knowable at profile time and would make two runs of the same
command incomparable.

**Default behaviour is unchanged.** `mamba_track_interval` defaults to 256 < P,
so N = P, so the derived r = 1.0 and the plan is byte-identical to E37
(`test_n_equals_p_is_byte_identical_to_today`). Only someone who explicitly
raises the stride sees a different geometry — which was the entire point.

---

### Result 1 — sizing, booted. E40's table is confirmed

Image `tokenspeed:e43-ratio` (tag predates the renumber). **Deviation, recorded:** this is a *derived*
image (`FROM tokenspeed:sm120-dev` + `pip install ./python`), not a full
`docker build` of the tree. The change is pure Python; a full CUDA rebuild at
`MAX_JOBS=32` would have saturated the box's CPU while two other agents were
mid-benchmark, which is a measurement hazard for them and for me. Everything
else in the tree is the `sm120-dev` build.

| N | r (derived) | history pages | state pages | B/token | E40 predicted |
|---:|---:|---:|---:|---:|---|
| 832 (= P) | **1.0000** | **163** | **163** | 250,565 | 163 / 163 |
| 1664 (2P) | **0.5000** | **259** | **129** | 158,050 | 258 / 129 |
| 3328 (4P) | **0.2500** | **367** | **91** | 111,793 | 365 / 91 |

E40's arithmetic is confirmed. The ≤ 2-page excess is because E40 (and my
offline driver) assumed the budget sat exactly on the lower edge of the
interval that yields 163; the real profiled budget is slightly above it.
`--mamba-track-interval` now moves `max_total_num_pages`. It did not before.

---

### Result 2 — the ceiling really moves, and it really turns over

Qwen3.5-27B bf16, TP=1, **GPU 3 alone**, `gpu_memory_utilization 0.88`,
`max_num_seqs 160`, prefix caching on, `--disable-kvstore`, no
`--attention-backend`. `--chunked-prefill-size` **pinned at 6656** on all three
arms — 6656 = 8×832 = 4×1664 = 2×3328, the smallest common multiple of every N
tested, so the GDN chunk decomposition is identical across arms by construction
(E34's fp-reassociation trap, avoided rather than tripped). C = 128, ILEN 1024,
OLEN 256, unique prefixes, ~1000 s per arm. `#running-req` is the **engine's
own** count from its decode-batch log, not inferred from bytes/token.

| arm | N | pages (hist/state) | predicted ceiling | **measured `#running-req`** | `#queue-req` | page ratio |
|---|---:|---:|---:|---:|---:|---:|
| A | 832 | 163/163 | 81 | **80.0** (100% of 226 samples) | 55.2 | 0.982 |
| B | 1664 | 259/129 | 129 | **126.0** (100% of 159) | 2.0 | 0.973 |
| C | 3328 | 367/91 | 91 | **88.0** (97% of 211) | 44.9 | **0.480** |

**E40's 81 → 129 is confirmed at 80 → 126**, and the predicted turnover is
confirmed too: arm C holds **367** history pages, more than twice arm A's, and
still runs fewer sequences than arm B because its 91-page state pool binds —
its history pool is less than half used (0.480) while 45 requests wait. A
ceiling model that only counted bytes/token would have predicted arm C to be
the best of the three; it is the middle one.

At **C=80 / ILEN=1024** the two arms are indistinguishable — 78.6 mean / 80
median running on *both*, 0 queued on both. Arm A's ceiling is 81, so C=80 fits
underneath it and there is nothing for the extra pages to do. **The shape the
brief named as "the current ceiling shape" does not exercise the ceiling**;
C=128 is the smallest offered load that does. That is why C=128 was added.

At **C=32 / ILEN=4096** (E41's refusal shape) the batch fills:

| | arm A (r=1, N=P) | arm B (r=0.5, N=2P) |
|---|---:|---:|
| `#running-req` mean | **19.8** (median 21, max 31) | **31.6** (median 32, max 33) |
| `#queue-req` mean | 3.04 | **0.00** |
| page ratio | 0.981 | 0.704 |

which reproduces E41's independent `--placeholder-state-block-ratio 0.5`
result (~11 → 32/32) through the derived path instead of the hand-set flag.

---

### Result 3 — throughput: real, ordered, and much smaller than the capacity

**Instrument correction, and it matters.** Neither of `steady_bench`'s
throughput numbers is usable on this engine at these shapes:

* `output_tok_s` (client) counts SSE **chunks** when the payload carries no
  `token_ids`, and this engine coalesces several tokens per chunk under load.
  `token_mismatch` was **100% of requests** in every run and
  `client_captures_frac` ran **0.35–0.79 and moved with the arm** (0.59 for A
  at C=128 vs 0.43 for B). An arm-dependent undercount is worse than no number.
* `server_tok_s` is exact but only advances when a request *finishes*, and
  requests finish in cohorts here, so the per-window series is a comb —
  `[0, 0, 2150.1, 0, 0, 2150.1, ...]` — and `steady_server_tok_s` averages only
  the **non-zero** windows (`[x for x in tail if x.get("server_tok_s")]` drops
  `0.0`). It reports the **burst height**, not the rate. Comparing two arms on
  it compares how big their cohorts are; it is what made arm B look like
  2150 vs 1365 tok/s, a "+57%" that is not a throughput ratio at all.

Used instead: `reqs_done` is cumulative and every request is exactly 256 server
tokens (`min_tokens == max_tokens`, `ignore_eos`, confirmed against
`usage.completion_tokens`), so `Δreqs × 256 / Δt` is an exact rate. Its window
differences are then gated for stationarity and blocked at the measured period,
per the contract (criterion 3 is satisfied by the server-side route, since
`client_captures_frac` is nowhere near 1.0 ± 2%).

**C = 128, ILEN 1024** — skip 360 s, the only skip at which all three arms read
`settled` simultaneously:

| arm | `#running` | tok/s | blocked SE | vs A |
|---|---:|---:|---:|---:|
| A (N=P, r=1) | 80.0 | **730.2** | ±44.8 | — |
| B (N=2P, r=0.5) | 126.0 | **800.0** | ±53.8 | **+9.6%** |
| C (N=4P, r=0.25) | 88.0 | **768.7** | ±0.4 | +5.3% |

**C = 80, ILEN 1024** — both settled at skip 60 and 180:

| arm | tok/s (skip 60) | tok/s (skip 180) | vs A |
|---|---:|---:|---:|
| A | 718.8 ± 8.1 | 717.6 ± 9.6 | — |
| B | 720.3 ± 8.1 | 721.0 ± 9.4 | **+0.2% / +0.5% — neutral** |

**How much of this is resolved, stated honestly.** At C=128 the point estimates
are stable across skips (A: 722.7 / 729.6 / 730.2; B: 800.0 / 801.0 / 815.5;
C: 751.1 / 765.9 / 768.7 — ~1–2% skip sensitivity), and the ordering
A < C < B tracks the ceiling ordering 80 < 88 < 126 monotonically. But the
blocked SE at the one skip where all three settle is ±45–54 on A and B, so
B − A = 69.8 ± 70.0 is **1.0σ**. The effect is directionally positive at every
skip and consistent with the mechanism, and it is *not resolved* to the
contract's standard. It is reported as "+10%, 1σ", not as +10%.

**The finding that matters more than the sign.** Arm B runs **58% more
sequences** than arm A (126 vs 80) for **~10%** more tokens per second — and
arm C runs 10% more sequences for ~5%. Restoring the missing sequences no
longer pays what E36 measured at ~21%, because **capacity has stopped being the
binding constraint at these shapes**. That is consistent with E41's +1.5% for
the hand-set ratio, and with E36's note that at larger batch the decode step is
set by attention and memory rather than by weight traffic. If the next entry
wants throughput, the page pool is no longer where it is.

---

### Result 4 — correctness

Greedy (`temperature 0`, 48 new tokens), two passes per arm with
`POST /flush_cache` and a discarded warm-up before each.

| prompt | prompt tokens | A (N=P) | B (N=2P) | C (N=4P) | |
|---|---:|---|---|---|---|
| `med_en` | 2,589 | `9d20ddbc1ec0c6c5` | `9d20ddbc1ec0c6c5` | `9d20ddbc1ec0c6c5` | identical |
| `long_en` | 9,625 | `78a1a8cb8497e42d` | `78a1a8cb8497e42d` | `78a1a8cb8497e42d` | identical |
| `primes` | 9 | *flips within every arm* | — | — | **probe rejected** |

`long_en` is 9,625 tokens: it spans two prefill chunks at 6656, so it exercises
a mid-sequence state snapshot and a cross-page state read, and it answers on
the merits ("42.0"; `med_en` gives "1847"). Byte-identical is the correct bar
here **only because `--chunked-prefill-size` was pinned at a common multiple of
every N** — E34's finding is that changing the state block sizing changes the
GDN recurrence's chunk decomposition and alters output through floating-point
reassociation alone, which looks exactly like a correctness bug and is not one.
Hold the chunk size and the decomposition is fixed by construction.

`primes` (9 tokens) flips between two hashes *within a single arm*, reproducing
E37's short-prompt hazard exactly. It is reported as unusable, not as a
difference.

---

### Result 5 — the prefix-cache cost, stated rather than skipped

**This experiment does not pay the cost, and that is an artifact of how it was
configured, not a property of the change.** Measured `#cached-token` on a
9,467-token shared prefix was **6,656 on all three arms** — no measured loss.
The reason is that `--chunked-prefill-size` was pinned at 6656 to hold the GDN
chunk decomposition fixed, and 6656 is a multiple of 832, 1664 *and* 3328, so
registration grain rather than N set the hit length in every arm. Choosing the
chunk size that makes the correctness comparison clean also masked the
granularity cost.

The cost is real and structural anyway. The coordinator aligns every prefix hit
**down** to the groups' lcm, which is N, and state groups publish a snapshot
only at an aligned chunk end. So:

* the **minimum reusable shared prefix doubles** from 832 to 1,664 tokens at
  arm B, and **quadruples** to 3,328 at arm C;
* any shared prefix shorter than N yields **zero** state-group reuse and the
  whole hit truncates — which is E34's measurement that a 6,000-token shared
  prefix yields zero reuse at N = 6656;
* our load harness uses **unique prefixes by construction**
  (`f"[w{wid}r{i}] " + text`), so it cannot observe any of this. The
  throughput numbers above are therefore an upper bound for any workload with
  shared prefixes shorter than N, and say nothing about a RAG shape.

This is the asymmetry against E37: placeholder blocks at r = 1 bought 2x at
**zero** granularity cost because they left N = P. Raising N to buy pages is
**not** free in the same way, and `r = P/N` does not make it free — it only
makes the pages actually arrive.

---

### Verification of the code itself

`test/runtime/test_placeholder_blocks.py` gains a `StateBlockRatioTest` class
(8 tests): the inertness pinned as a regression guard; the derived ratio at
N = P/2P/4P/8P; E40's exact page split (163/163, 258/129, 365/91) reproduced
from the resolver; N = P byte-identical to today; explicit override winning
over the derivation; `interval_past_turnover` flagging N > 2P; the ceiling
sequence {1P: 81, 2P: 129, 4P: 91, 8P: 57} pinning that the optimum is real;
and input rejection. **55 passed** in-image across the placeholder,
flat-memory-plan and server-args suites. `black` clean; the one `isort`
complaint in `registry.py` is pre-existing and in another entry's import block,
left alone.

---

### What I did **not** measure

- **Any RAG or shared-prefix shape.** The granularity cost above is argued from
  the coordinator's align-down rule and E34's measurement, **not** measured
  here — and the one prefix probe I did run was masked by the pinned chunk size.
  This is the biggest hole in the entry.
- **C = 32 / ILEN 4096 throughput, gated.** The `#running-req` result there
  (19.8 → 31.6) is solid, but neither arm read `settled` at any skip on 420 s
  runs, so the throughput point estimates (A ~209–214, B ~237–259, i.e.
  +13–21%) are **refused** and are not quoted as a result. They would need
  ~1000 s runs like the C=128 arms got.
- **Arm C at ILEN 4096**, where the offline model says N = 4P should *win*
  (ceiling 45 vs 43 vs 32). The turnover was only tested at ILEN 1024.
- **N = 6656 (8P)**, and any N that is *not* a divisor of the chunk size —
  which is exactly where the prefix-granularity loss would become visible.
- **TTFT, TPOT, p99.** Arm A queues 55 requests deep at C=128 and arm B queues
  2; the latency consequence of that is unpriced, and it is likely the largest
  unmeasured *benefit* of the change.
- **vLLM.** No cross-engine arm; every number here is TokenSpeed vs TokenSpeed.
- **The 33 failed requests in arm A's long run** (22 in arm C, 0 in arm B).
  They are excluded from `reqs_done`, so they depress A and C slightly rather
  than flattering them, but I did not chase whether they are shutdown artifacts
  or real timeouts behind a 55-deep queue.
- **A full `docker build`** of the tree; see the derived-image deviation above.
- **bf16 SSM state** (`--mamba-ssm-dtype bfloat16`, E40's other lever), which
  composes with this and was not combined with it.

---

### Two operational notes

1. **`docker rm -f $(docker ps -aq --filter name=ts-e4)` matches other agents'
   containers.** One arm of mine was destroyed 8 seconds into weight loading by
   another agent's cleanup, in the same second as two more containers belonging
   to two other entries. Nothing of mine was mid-measurement. Container names
   were moved to the prefix `e43ratio-` afterwards, which no `ts-e4*` filter
   catches. **Name containers so that someone else's substring sweep cannot
   match them.**
2. **GPU assignment, deviation recorded.** The brief said GPU 1. At launch time
   GPU 1 held E41's container (89 GB) and GPU 2 held E42's, so this entry ran
   on **GPU 3**, which read 0 MiB before every launch and was calibrated at
   **427.6 TFLOP/s** bf16 8192³ (free-card range 405–427). All three arms ran
   on the same card so the A/B is not confounded across GPUs; moving mid-series
   to GPU 1 when it later freed would have been worse than the deviation. Host
   ports 8303/9303, used by nobody else.

---

**Verdict.** `--mamba-track-interval` reaches the sizing now. Deriving
`r = P/N` turns a flag that produced byte-identical plans into one that moves
`max_total_num_pages` 163 → 259 → 367, and the concurrency ceiling **80 → 126**
at N = 2P, confirming E40's 81 → 129 arithmetic against a booted engine and
confirming its predicted **turnover** at N = 4P (367 history pages, still only
88 running, history pool 48% idle, state pool binding). Generation is
byte-identical to the r=1/N=P arm on every stable probe.

The throughput it buys is **+9.6% at C=128 / ILEN 1024, and that is 1.0σ** —
directionally positive at every skip, consistent with the mechanism, not
resolved. At C=80 it is flatly neutral, because arm A's ceiling of 81 already
covers 80 offered sequences. **58% more running sequences for ~10% more tokens
per second is the headline result**, and it says capacity has stopped being the
binding constraint at these shapes. That is a smaller number than E40's ceiling
arithmetic implies and a larger one than E41's +1.5% for the hand-set ratio,
and it points the next entry away from the page pool.

---

## E45 — Every ILEN=4096 number before today was measured on an image without `r = P/N`: coupling it is worth **+10.3%** at that shape, and TokenSpeed lands **2.0% behind** vLLM there

> **Corrected after the 900 s arms landed.** This entry first claimed +26.2% for
> `r = P/N` and −19.8% for mixed batch, both from 210 s arms. Neither survived.
> The identical tree config that read 273.1 at 210 s reads **242.6** at 900 s — a
> 12.6% inflation from arm duration alone. The corrected numbers are in the
> tables below; the superseded ones are struck where they appear. See
> **Correction** at the end for how this got past me, since Result 3 of this very
> entry diagnoses the same failure mode in vLLM's arm.

**Objective.** Take the one shape TokenSpeed was still behind vLLM on
(C=32 / ILEN=4096, last quoted at −14.7%) and either close it or say why not.

**The finding that reframes the rest.** `tokenspeed:sm120-dev`, the image every
benchmark in this experiment has used, predates E44. Its startup says

```
Placeholder blocks ON: ... (163 history ids, 163 state ids, ratio 1.000)
Scheduler config: num_device_pages=163 num_state_pages=163
```

while the current tree says

```
Placeholder blocks ON: ... (259 history ids, 129 state ids, ratio 0.5000 derived,
  demand-matched P/N = 0.5000); attention page P=832, snapshot stride N=1664
Scheduler config: num_device_pages=259 num_state_pages=129
```

**259 KV pages against 163 — +59%.** E44's coupling was in the tree and never in
an image, so it has never been in any measurement in this document. This was
found by diffing the image's installed `tokenspeed` against the tree (10 files,
all pure python) rather than by reading the code, which had been read twice
without noticing.

**Setup.** Qwen3.5-27B bf16, GPU 1 alone, arms sequential, one server at a time.
`tokenspeed:sm120-tree` = `sm120-dev` + those 10 files, `COPY` only — no rebuild,
so every arm is bit-identical outside the change under test. The C++ scheduler
already supported mixed batches (it logs `mode=1`); what was missing was the
python attention backend. C=32, ILEN=4096, OLEN=256, WINDOW=30 s, MAX 720 s.

### Result 1 — the capacity ceiling at this shape, and its removal

The old image is capacity-bound and it is visible in one log line:

| | `sm120-dev` | tree |
|---|---|---|
| `#running-req` | **24** | **32** |
| `page ratio` | 0.99 | 0.74 |
| `#queue-req` | 4 | 0 |

24 of 32 offered requests ran; 4 sat in the queue against a 99%-full pool. The
arithmetic: 4096+256 = 4352 tokens rounds up to 6 pages of 832 = 4992, so
163/6 = 27.2 requests is the ceiling. At 259 pages it is 43, and the offered
concurrency fits with 26% of the pool spare.

The 210 s arms (~~`sm120-dev` 211.7, tree 273.1, +26.2%~~) are superseded. The
gated numbers are the fixed-duration runs with early convergence disabled
(`STABLE_NEEDED=999 TOL=0.0001`), scored on the last 300 s:

| arm | duration | server tok/s | completion tok/s |
|---|---|---|---|
| `sm120-dev` (no `r = P/N`) | 720 s | 220.0 | 219.9 |
| tree (`r = P/N`) | 900 s | 242.7 | **242.6** |

**+10.3%**, not +26.2%. Two things came with it that survive the correction and
matter more than the number. The two counters, 7.8% apart on the old image, now
agree to 0.04%. And the limit cycle is *gone* — window spread 9.81% → 0.07%, and
this shape converged for the first time. So the oscillation was never the
prefill/decode alternation: it was capacity thrash, requests queued and released
in waves. The alternation is unchanged and no longer produces a limit cycle.

E44 measured this same coupling at ~10% at 1σ at C=80/ILEN=1024. At the
corrected +10.3% the two agree, and the "worth 2.6x more at long context" story
in the first draft of this entry was an artifact, not a mechanism. The
per-request page arithmetic (6 pages at ILEN=4096 vs 2 at 1024) is still right;
it just does not show up as extra throughput once the arm is long enough for the
completion pipeline to drain.

### Result 2 — mixed batch runs for the first time; its price is *not* established

E43 measured `--enable-mixed-batch` at −7.5% at C=64/ILEN=1024 and recorded
C=32/ILEN=4096 as *not resolvable*. It is resolvable now, because the engine at
this shape is no longer capacity-bound:

| arm | duration | server tok/s | completion tok/s | spread | converged | failed |
|---|---|---|---|---|---|---|
| tree, prefill-first | 210 s | ~~273.0~~ | ~~273.1~~ | 0.07% | yes | 0 |
| tree, `--enable-mixed-batch` | 210 s | ~~219.0~~ | ~~219.7~~ | 0.41% | yes | 0 |

**What stands: it runs.** 140 `Mix batch` rounds, e.g. `Mix batch. #new-seq: 3,
#new-token: 6656, #running-req: 6` — no assertion, no `NotImplementedError`,
zero failed requests. The E42 implementation is correct, and this is the first
time the flag has executed at all on this model.

**What does not stand: the −19.8%.** Both arms were 210 s, and the prefill-first
arm of that pair is exactly the one the 900 s run cut from 273.1 to 242.6. A
ratio between two arms that are each individually inflated is not a measurement
of the difference. The mechanism argument is unchanged and still plausible —
every step builds both metadata halves and issues two attention calls per layer,
and the GDN side widens its extend-only arrays to full batch width
(`_widen_mixed_extend_kwargs`) — but it is now an untested hypothesis about
this shape. **Open: 900 s mixed-batch arm.**

**Still not measured: TTFT.** That is the side of the trade mixed batch should be
judged on and this entry does not price it, exactly as E43 said.

### Result 3 — the vLLM comparison is not yet takeable, and why

vLLM 0.26 at the same shape converges at **270.5** tok/s twice, independently,
to the digit (spread 0.41%, SM 100%, `client_vs_server` 1.0). Against
TokenSpeed's 273.0 that reads as a win. It is not takeable:

| | server counter | completions, same tail |
|---|---|---|
| TokenSpeed tree | 273.0 (`stream_usage`) | 273.1 |
| vLLM | 270.5 (prometheus) | **226.1** |

vLLM's two counters disagree by 16.5% in its *converged* tail. The cause is not
the engine: `steady_bench` stops after 3 windows of stable **token** rate, and
the completion counter lags the token counter by about one request duration
(~30 s at this shape). vLLM ramped 210 → 270 at t=180 s and exited at t=300 s
with the completion pipeline still draining that ramp.

TokenSpeed's two counters agreeing to 0.04% proves nothing about this, because
both of its sources are completion-based — TokenSpeed's `serve` puts the
engine's metrics behind a gateway that exports only `smg_*`, so `steady_bench`
falls back to per-request `usage`. Agreement between a measure and itself is not
counter agreement. Its arm was also short (210 s).

**So the only claim this entry supports is TokenSpeed against TokenSpeed**, same
harness, same counter, same shape: **+26.2%**. A fixed-duration 900 s pair with
early convergence disabled is running to settle the cross-engine number.

### What changed

Nothing in either repo. `tokenspeed:sm120-tree` is a build artifact; the code it
carries was already committed by E42 and E44.

### Hazard for whoever reads this next

**A benchmark image is a measurement instrument and it drifts.** Ten files of
committed, tested, documented work were invisible to every number here because
nobody diffed the image against the tree. E44's own conclusion ("~10% at 1σ")
undersells its change by 2.6x for that reason. Diff the image before trusting an
arm, and record the image's *content*, not its tag.

### Result 3 — resolved: TokenSpeed is **2.0% behind** vLLM at this shape

The 900 s fixed-duration arms (`STABLE_NEEDED=999 TOL=0.0001 MAX_SECONDS=900`,
early convergence disabled, last 300 s scored) settle it:

| arm | duration | completions tok/s | server counter |
|---|---|---|---|
| `sm120-dev` (no `r = P/N`) | 720 s | 219.9 | 220.0 |
| TokenSpeed tree | 900 s | **242.6** | 242.7 |
| vLLM 0.26 | 900 s | **247.5** | 245.8 |

vLLM's "converged 270.5" was itself short-run inflation — on a long arm its own
server counter reads 245.8, and its two counters, 16.5% apart before, now agree
to 0.7%. So the drain-lag diagnosis in the original Result 3 was right about
vLLM and I simply failed to apply it to my own arm.

`r = P/N` narrows the gap from −11.1% to **−2.0%**. It does not close it.
TokenSpeed is still behind at C=32/ILEN=4096.

### Correction — how a 12.6% artifact got published

I reported +26.2% and −19.8% from 210 s arms in the same entry where I refused
vLLM's number because its arm was too short to drain. The rule was stated
correctly and then not applied to my own measurement. The tree config that read
273.1 at 210 s reads 242.6 at 900 s.

**Gate, from now on:** an arm that stops on a convergence criterion has only
shown that its *rate* counter is stable, which a draining pipeline can be while
the completion counter is still climbing. Any cross-arm ratio must come from
fixed-duration arms with convergence disabled, scored on a tail at least one
request-duration long. This applies to arms I like as much as to arms I don't.

## E46 — TurboMind registers **zero** w4a16 kernels on sm_120; the cause is one half-open range, and fixing it registers 220

**Objective.** E36 put GEMM at 80.5% of a decode step, moving 1299 GB/s against
a 1462 GB/s card — the bottleneck is weight bandwidth, so w4a16 is the lever.
TurboMind has the best w4a16 GEMM of the engines surveyed (AWQ `lop3` u4→f16
unpack fused into the mainloop, offline MMA-fragment weight repack). A previous
entry recorded it as unusable here behind three blockers. Two of the three do
not exist.

**The gate.** `gemm/arch.h` declares `struct Sm80: Arch<800, 900>`. On this
device `arch = major*100 + minor*10 = 1200`, so

```
Sm80::is_compatible(1200) == (800 <= 1200 && 1200 < 900) == false
```

which does two things. Host-side, `is_arch_compatible(800, 1200)` is false, so
`Registry::Add` rejects every mma.m16n8k16 kernel. Device-side,
`gemm_universal.h`'s `if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__))`
compiles every one of their bodies away. The kernels were *already being
compiled for sm_120a* — the top-level CMakeLists appends `120a-real` on
CUDA >= 12.8 — they were just being compiled into nothing.

**Compile-level proof.** Both files, identical flags, `-arch=sm_120a`:

| file | PTX lines | entries | arch tag | registers | mma | ldmatrix | cp.async | lop3 |
|---|---|---|---|---|---|---|---|---|
| `sm80_16816_4.cu` | 1,129 | 93 | `Sm80` | **4 / 4** | 0 | 0 | 0 | 0 |
| `sm120_16816_4.cu` | 466,129 | 115 | `Sm120` | 48 / 255 | 5,008 | 1,444 | 3,581 | 4,304 |

4 registers for all 93 sm80 kernels is an empty `__global__` — parameters and
`ret`. The new file emits `mma.sync.aligned.m16n8k16` in both `.f16.f16.` and
`.bf16.bf16.` forms, with zero spill on all 115.

**Runtime proof.** A harness that constructs `Registry` on the real card, built
twice from the same objects with only the `sm120_16816_4()` call commented out:

| build | kernels | arch tags | max smem |
|---|---|---|---|
| upstream | **2** | `sm0` x2 | 0 B |
| ported | **222** | `sm0` x2, `sm1200` x220 | 99,872 B |

The 2 are `cublas_float()`, which registers with `arch = 0` (always
compatible). Upstream TurboMind on sm_120 has no quantized GEMM whatsoever —
only the cuBLAS fp16/bf16 fallback. 99,872 B sits under the 101,376 B optin
limit, and the 5-kernel gap between 115 compiled and 110 registered is
`Registry::Add`'s smem check dropping the widest tiles, as designed.

By operand type (`encode_data_type(s,e,m) = (s<<16)|(e<<8)|m`):

| A | B | C | count | |
|---|---|---|---|---|
| `kFloat16` | `kUint4` | `kFloat16` | 154 | w4a16, fp16 activations |
| `kBfloat16` | `kUint4` | `kBfloat16` | **40** | w4a16, bf16 activations |
| `kFloat4_e2m1` | `kBfloat16` | `kBfloat16` | 26 | NVFP4 weights, bf16 activations |

**Why the bf16 40 needed new code, not re-tagging.** `Config_U4_d` pins
`Operand_A<half, ...>` and `Tc = half` regardless of its `T` parameter, so
upstream's 13 bf16 sm80 entries were both dead *and* structurally incapable of
serving bf16 activations. The 40 come from a `register_u4_g_bf16` block added in
the new file, routed through `Config_U4_g`, which uses `Operand_A<T, ...>` and
`Tc = T`. Qwen3.5-27B is bf16; without that block the port would have registered
180 kernels the model can never call.

### The two blockers that turned out not to be blockers

**Weight repack.** Recorded as requiring work in `convert_v3.cu`. It does not:
`GetConverters` is keyed on the *MMA fragment layout*, not the arch tag, and its
tests are open-ended —

```cpp
if (weight_type == kUint4) { ...
    if (sm >= 80) return {W(sm8_, kCol, s16816h | B | _2), S(sm8_, kCol, s16816h | V | _1)};
```

`getSMVersion()` returns `major*10 + minor = 120`, which takes the `sm >= 80`
branch and produces exactly the layout the sm80-shaped kernels expect. The one
early return above it (`sm >= 100 && sm < 120`) is grouped-fp16 only and 120 is
deliberately outside it. **The registry keyed on the arch tag; the converter
keyed on the instruction. That asymmetry is the entire bug** — one half of the
pipeline was arch-gated and the other was not.

**Split-K workspace.** Also not arch-gated: `arch_` in `gemm.cu:46` is declared
and never read, and `barriers_size` / `partials_size` are supplied by the caller.

The one real blocker, fp16-only activations, is the one the new bf16 block
addresses.

### Also found: a CUDA 13 build break, unrelated to sm_120

`core/mma.h` uses `nv_bfloat16` without including `<cuda_bf16.h>`. Up to CUDA
12.x it arrived transitively through `cuda_runtime.h`; CUDA 13 no longer pulls
it in, so *every* kernel TU that reaches this header first — sm70, sm75, sm80,
and the new sm120 — fails with `identifier "nv_bfloat16" is undefined`. Fixed by
adding the include. This is a break in the vendored tree, not something the port
introduced; it was invisible because nothing had compiled these files under
CUDA 13 before.

### What changed

`third_party/lmdeploy`, five files:

| file | change |
|---|---|
| `gemm/kernel/sm120_16816_4.cu` | new — 55 sm80 tiles re-tagged `Sm120`, plus `register_u4_g_bf16` (11 tiles x 2 group sizes) |
| `gemm/registry.h` | declare `sm120_16816_4()` |
| `gemm/registry.cu` | call it unconditionally — `Add` already filters on arch and smem, so guarding on `arch_` would duplicate the decision |
| `gemm/CMakeLists.txt` | `GEMM2_KERNELS_SM120`, wired into `gemm2`; explicitly *not* into `GEMM2_KERNELS_SM90`, which CMake quarantines to arch 90 because it genuinely needs wgmma/TMA |
| `core/mma.h` | `#include <cuda_bf16.h>` |

### Not yet done

Numerics are unverified — this entry proves the kernels exist, compile to real
SASS, and register on the device. It does not prove they compute the right
answer or that they beat the current path. Next: a correctness check against a
dequant-then-cuBLAS reference, then a GEMM-shape benchmark at Qwen3.5-27B's
projection shapes, then extraction as a TokenSpeed plugin kernel under
`tokenspeed-kernel/ops/gemm/`.

### Hazard for whoever reads this next

**"Arch X is unsupported" is usually a range literal, not a hardware fact.**
Three engines in this experiment have now turned out to gate features on
capability-keyed tables that simply omit 120. The falsifiable test costs one
compile: build the file for the target arch and count registers per entry. A
kernel that uses 4 registers is not a kernel.

### Numerics — 28/28, after finding the second gate

The registered kernels are correct, but only after a runtime fault that the
compile-level checks above could not have caught. Against `Quantize()`'s
*pseudo-quantized* weight (so quantization error cancels and what is measured is
the kernel plus the layout conversion), at Qwen3.5-27B's projection shapes:

| M | N x K | fp16 rel_max | bf16 rel_max | rel RMS |
|---|---|---|---|---|
| 1 (decode) | 6144 x 5120 | 0.0009 | 0.0039 | ~2e-5 |
| 1 | 1024 x 5120 | 0.0010 | 0.0049 | ~2e-6 |
| 1 | 5120 x 6144 | 0.0017 | 0.0062 | ~1e-5 |
| 8 | 6144 x 5120 | 0.0075 | 0.0066 | ~3e-5 |
| 64 | 6144 x 5120 | 0.0389 | 0.0334 | ~1e-4 |
| 4096 (prefill) | 6144 x 5120 | 0.0000 | 0.0000 | 0 |
| 4096 | 5120 x 6144 | 0.0000 | 0.0000 | 0 |

x {group 128, group 32} = **28/28 pass**. The M=4096 rows are exact because the
dispatcher picks a split-K-free tile there and the reference is the same cuBLAS
accumulation order.

**The second gate: `cache_policy::Stream` on the packed-B operand faults.**
Before this, every case with M <= 64 died with `cudaErrorIllegalInstruction`.
Holding the tile fixed at `16x64x128` and varying only the two cache policies:

| PolicyA | PolicyB | result |
|---|---|---|
| D | D | pass |
| **S** | D | pass |
| D | **S** | **illegal instruction** |
| S | S | illegal instruction |

Same for every other tile tried, at any CTA_M — so the initial correlation with
small M was an artifact of which tiles the dispatcher picks for small M, not a
property of the shape.

What the evidence does *not* say: this is not "sm_120 lacks L2 cache policies".
`createpolicy.fractional.L2::evict_first.b64`, `cp.async...L2::cache_hint`, and
every `.L2::64B/128B/256B` prefetch combination all run correctly in isolation on
this card (9/9 in a micro-benchmark). And the emitted SASS is *identical* between
the passing and faulting builds -- 4 `LDGSTS.E` and 16 `LDGSTS.E.BYPASS.128`
either way, with no opcode unique to `S`. The difference is in the memory
descriptor the hint builds, not in the instruction. A zero policy operand is
separately fatal on this card (verified: `0` faults, `0xdeadbeefcafebabe` does
not) but forcing the policy non-zero did **not** fix it, so that is a true fact
about the hardware and not this bug's cause.

The fix is `TURBOMIND_ARCH_HAS_L2_EVICT_POLICY` in `core/common.h`, 0 on
`1200 <= __CUDA_ARCH__ < 1300`, which drops the hint but not the copy. An
evict-first policy is a pure performance hint, so this cannot change results;
what it may cost in L2 behaviour is unmeasured and is the natural thing to
revisit if the benchmark disappoints.

**Why the compile-level proof was not enough.** `ptxas` accepted every one of
these kernels for `sm_120a`, they contained real `HMMA`/`LDSM`/`LDGSTS`, and they
registered on the device. All three of those were true while 100% of small-M
GEMMs faulted. "It compiles for the arch and registers" is a necessary condition
and nothing more.

### What changed (revised)

| file | change |
|---|---|
| `gemm/kernel/sm120_16816_4.cu` | new -- 55 sm80 tiles re-tagged `Sm120`, plus dense **and** grouped bf16 blocks |
| `gemm/arch/config_sm80_s16816.h` | `Config_U4_d` takes `Operand_A<T>` / `Tc = T` instead of hardcoded `half` |
| `gemm/registry.h`, `registry.cu` | declare and call `sm120_16816_4()` |
| `gemm/CMakeLists.txt` | `GEMM2_KERNELS_SM120`, wired into `gemm2` |
| `core/common.h` | `TURBOMIND_ARCH_HAS_L2_EVICT_POLICY`; `#include <cuda_bf16.h>` fix is in `core/mma.h` |
| `gemm/iterator_sm80.h` | gate `createpolicy` and the hinted `cp.async` on it |

**`_d` is dense, `_g` is grouped (MoE)** -- not two spellings of group-wise
quantization. The first cut of this port registered only `Config_U4_g` for bf16,
reasoning correctly that `Config_U4_d` could not serve bf16 and incorrectly that
`_g` was the substitute. That registered 40 MoE kernels and left an ordinary bf16
linear layer with nothing: it dispatches `group_axis = -1`, `ttt_fff`, which only
the `_d` set answers. Caught by dumping registered descriptors in the same
vocabulary as the "no feasible kernel" message and diffing against the request.

## E47 — w4a16 is worth **2.3-3.3x on decode GEMMs** and **0.77x at M=128**; the first cut of this measurement was an L2 artifact

**Setup.** Same harness as E46's numerics check, retargeted: baseline is cuBLAS
on the *unquantized* bf16 weight (what the engine runs today), not the
pseudo-quantized one. 201 timed iterations, median, `cudaEvent` per call.
Qwen3.5-27B shapes: hidden 5120, 24 q heads x 256 = 6144, 4 kv x 256 = 1024,
MLP 17408.

### The instrument error, caught before it was published

The first run showed **1.43x** on `q_proj` and **6.36x** on `gate/up` and I
almost wrote that up as "speedup grows with N". It is an L2 artifact. This card
has **128 MB of L2**; `q_proj`'s bf16 weight is 6144 x 5120 x 2 = **62.9 MB** and
stays resident across iterations, so the baseline read it at **2530 GB/s** --
1.7x the ~1460 GB/s this HBM can actually do. `gate/up`'s weight is 178 MB and
does not fit, so that arm was honest. The comparison was L2-vs-L2 for small
weights and HBM-vs-HBM for large ones.

**The tell was free and I nearly missed it: a bandwidth number above the
hardware's peak.** Any measured GB/s exceeding the card's HBM bandwidth means
the working set is cached, not that the kernel is fast.

Fixed by rotating each path over enough weight copies to exceed L2 four times
over (>=512 MB, 4-193 buffers depending on shape), which is what real decode
does anyway -- 60+ layers are swept before any one weight comes round again, so
it is never resident.

### Results (cold weight, group_size 128, bf16)

| shape | | bf16 | w4a16 | speedup |
|---|---|---|---|---|
| gate/up decode | 1 x 17408 x 5120 | 0.1322 ms | 0.0431 ms | **3.06x** |
| down decode | 1 x 5120 x 17408 | 0.1424 ms | 0.0433 ms | **3.29x** |
| q_proj decode | 1 x 6144 x 5120 | 0.0523 ms | 0.0217 ms | **2.41x** |
| o_proj decode | 1 x 5120 x 6144 | 0.0516 ms | 0.0228 ms | **2.26x** |
| kv_proj decode | 1 x 1024 x 5120 | 0.0188 ms | 0.0174 ms | 1.08x |
| q_proj | M=8 | 0.0627 ms | 0.0269 ms | 2.33x |
| q_proj | M=32 | 0.0505 ms | 0.0351 ms | 1.44x |
| q_proj | M=128 | 0.0524 ms | 0.0678 ms | **0.77x** |
| q_proj | M=512 | 0.1590 ms | 0.1293 ms | 1.23x |
| q_proj | prefill M=4096 | 0.7052 ms | 0.6785 ms | 1.04x |
| gate/up | prefill M=4096 | 1.7927 ms | 1.7802 ms | 1.01x |

The ceiling is **3.76x**, the weight-byte ratio (2 B/elem vs 0.5 B/elem plus one
u32 {scale,zero} per group per column). The two MLP GEMMs reach 3.06-3.29x, so
they are close to it. `q_proj`/`o_proj` stop at ~2.4x because w4a16 only reaches
770 GB/s there against the baseline's 1203 GB/s -- at N=6144 the quantized path
is not saturating memory, so there is headroom the tile choice is not taking.

### Three things that are not wins

**M=128 is a regression (0.77x).** Past the compute-bound crossover the dequant
costs more than the bandwidth saves: 118.7 TF vs the baseline's 153.7 TF. This
sits squarely in mid-batch decode, which is not a corner case. A dispatch rule
that falls back to bf16 above some M is needed; where the crossover sits is
between M=32 (1.44x) and M=128, and is unmeasured.

**kv_proj barely moves (1.08x).** At N=1024 the call is fixed-overhead-bound:
17.4 us for a sixth of `q_proj`'s work, which takes 21.7 us.

**Prefill is flat (1.01-1.04x).** Both paths run at ~410 TFLOPS; there is nothing
for a weight-traffic optimization to win when the GEMM is compute-bound.

### Extrapolation, flagged as such

E36 put GEMM at 80.5% of a decode step. At ~2.8x on the linear layers that would
put the step at ~2.0x. **This is arithmetic on a microbenchmark, not a
measurement.** It assumes the non-GEMM 19.5% is unchanged, ignores that only 16
of 64 layers are full-attention, and does not cover the GDN layers' weight
shapes, which are not benchmarked here. The end-to-end number requires wiring
this into TokenSpeed.

### Hazard for whoever reads this next

**A GB/s above the card's peak is a caching artifact, not a result.** Print
achieved bandwidth next to every kernel timing; it costs nothing and it is the
cheapest available check that the working set is not sitting in cache. This is
the same class of error as E45's short arms -- an instrument that flatters the
arm you were hoping for.

## E48 — sm_120 runs on the re-forked tree after three gating bugs, and is **16.2% behind vLLM** without the work still in `legacy/`

**Context.** UnieInfra was re-forked from upstream tokenspeed with a shared
ancestor; the sm_120 work from the old fork was deliberately left in `legacy/`
to be re-expressed. This is the first measurement of what upstream alone does on
this card, and therefore the honest baseline for everything that gets ported
back.

### Getting it to start: three bugs, each found by running it

Upstream tokenspeed does not start on sm_120 with Qwen3.5. Each failure was
found by launching and reading the traceback, one at a time.

**1. XQA decode wins selection for a page size it cannot compile.**
`flashinfer_trtllm_mha_decode_with_kvcache` declares traits for sliding window,
sinks, logit cap and LSE, but not for page size. XQA compiles `TOKENS_PER_PAGE`
as a template parameter and accepts only 16/32/64/128. Callers already pass
`page_size` to `select_kernel`, so the constraint simply was not declared:

```
ValueError: Invalid page_size: 576, only 16, 32, 64, 128 are supported
```

Hybrid attention/linear-attention models are what hit this — their attention
page size is derived from the linear-state row rather than chosen.

**2. TRT-LLM-gen FMHA wins selection on an arch it was not built for.**
`flashinfer_trtllm_mha_extend_with_kvcache` gates on
`min_arch_version=ArchVersion(10, 0)` — an open-ended comparison that every newer
part satisfies. sm_120 passes it, then dies at the first extend with
`Unsupported architecture` from `fmhaRunner.cuh`. Bounded above at sm_103.

**3. The Triton prefill tile does not fit, and the check that should have caught
it was dead code.** The tile ladder branches on arch family, with a carve-out for
sm_86/sm_89 keyed on `arch_version.minor == 9 or == 6`. sm_120 is
`is_hopper_plus` by version and takes the datacenter tile, but has sm_89's
budget:

```
OutOfResources: shared memory, Required: 114688, Hardware limit: 101376
```

The obvious fix — ask the device for its budget — did not work, because
`PlatformInfo.max_shared_memory_per_sm` read
`getattr(props, "max_shared_memory_per_block", 0)` and **torch has no attribute
by that name**, so the field was 0 on every CUDA device ever. Nothing else read
it, which is why it was never noticed. The real name is
`shared_memory_per_block_optin`; it reports 101,376 here, exactly the limit in
Triton's own message.

That last one is the interesting one. A capability *test* existed, a capability
*field* existed, and the value flowing through them was a silent zero.

### The baseline

Qwen3.5-27B bf16, GPU 1 alone, C=32 / ILEN=4096 / OLEN=256, max_model_len 8192,
GMU 0.85. Both arms 600 s fixed-duration with early convergence disabled
(`STABLE_NEEDED=999 TOL=0.0001`), scored on the last 200 s, run back to back in
one session on the same card.

| arm | completions tok/s | server tok/s | window spread |
|---|---|---|---|
| TokenSpeed (re-forked tree) | **208.0** | 206.0 | 17.7% |
| vLLM 0.26 | **248.1** | 248.0 | 24.7% |

**TokenSpeed is 16.2% behind.** vLLM's 248.1 lands within 0.2% of the 247.5 that
E47 measured at this shape in a different session, which is the best available
check that the harness and the setup are the same as before.

vLLM needed `--max-num-seqs 256`; its default of 1024 exceeds the Mamba cache
blocks (516) this model leaves and it refuses to capture CUDA graphs.

### What the gap is made of

`num_device_pages=149`, `block_size=832`, and the log never mentions placeholder
blocks — the tree has none of the capacity work. The window sequence is the
giveaway: `238 188 219 184 220 183 227`, alternating high and low at 17.7%
spread. E45 identified exactly that pattern as capacity thrash and showed
`r = P/N` collapsing it to 0.07%.

So the ordered gap, from what has already been measured elsewhere:

| | measured | where |
|---|---|---|
| `r = P/N` placeholder-block coupling | +10.3% | E45, `legacy/divergence` (A6) |
| FA2 extend kernel for sm_120 | not isolated | `legacy/`, E-earlier |
| w4a16 decode GEMMs | 2.3-3.3x on the GEMM | E47, not yet wired into the engine |

+10.3% on 208 is ~229, still short of 248. Beating vLLM needs more than
re-porting; it needs at least one of the kernel-level wins to reach the engine.

### Hazard for whoever reads this next

**A zero from a detection layer is indistinguishable from a real answer.**
`max_shared_memory_per_sm` was 0 on every device for as long as the field has
existed, and the one consumer added later read it as "budget unknown, assume
large" — which is the safe reading of a missing value and the wrong reading of a
broken one. When a capability check does not change behaviour, check that the
value it reads is not a default.

### E48b — the code behind E45's +10.3% is not in the preserved tree

Porting the two-granularity KV work to the re-forked tree landed the
state-snapshot machinery (`state_snapshot_interval`, `state_page_size`,
`state_group_page_sizes`, `layer_page_sizes`) and it is live -- the new grain
check fires, warning that `chunked_prefill_size=8192` is not a multiple of the
state-snapshot page grain and using 7488 instead.

It did **not** reproduce E45's capacity result. `num_device_pages` is still 149.

E45 quotes this startup line from `tokenspeed:sm120-tree`:

```
Placeholder blocks ON: ... (259 history ids, 129 state ids, ratio 0.5000 derived,
  demand-matched P/N = 0.5000)
```

Searching for it:

| where | result |
|---|---|
| the re-forked tree | absent |
| `legacy/divergence/unieai-python.diff` | 0 occurrences of "placeholder" |
| the preserved pre-restructure working tree (`UnieInfra-old`) | only in `log.md` itself |
| all 76 commits of the old repository (`git log -S`) | none |

Placeholder blocks themselves are an **upstream** feature -- `scheduler.h`,
`operations/forward.cpp`, `fsm/forward_states.h` -- which is consistent with
`tokenspeed-scheduler` having zero UnieAI divergence. What is missing is the
python side that derived the ratio and turned it on.

So the string exists in exactly one place: the log entry claiming the
measurement. The code that produced it was never committed and is not in the
working tree that was preserved. `tokenspeed:sm120-tree` was built by copying 10
files from the tree at that moment; at least one of them no longer exists.

**This puts a question mark on E45 Result 1 that this entry cannot remove.** The
measurement may well have been real -- the capacity arithmetic in E45 is
self-consistent and the `#running-req: 24 -> 32` observation is hard to fake --
but it is no longer reproducible from anything in the repository, and +10.3%
should not be quoted as a banked win until it is.

Next step is not to re-port but to re-derive: placeholder blocks exist upstream,
so find how the scheduler expects the ratio to be supplied and drive it from the
already-ported `state_snapshot_interval`.

### Hazard for whoever reads this next

**A result whose code was never committed is a result you do not have.** E45 was
measured from an image built by copying files out of a working tree. The image
is gone, the tree moved on, and the only surviving trace of the mechanism is the
sentence claiming it worked. Build the artefact from a commit, or record the
diff in the entry.

## E49 — w4a16 costs **+7.2% perplexity** on Qwen3.5-27B without calibration, and that is the number to beat, not to quote

**What was measured.** The quantization *scheme*, not the kernel: weights are
quantized and dequantized in place and the model runs bf16 throughout, so this
is separable from E46's kernel correctness (28/28 against a dequant reference).
Scheme matches what the TurboMind kernel consumes -- asymmetric int4, grouped
along the input dimension, group_size 128, `scale = (max-min)/15`.
496 of 497 `nn.Linear` quantized; embeddings, lm_head and norms left in bf16.

| corpus | ppl bf16 | ppl w4a16 | delta |
|---|---|---|---|
| 17,396 tokens of natural technical prose | 17.4763 | 18.7346 | **+7.20%** |
| ~~one paragraph x12~~ | ~~1.6401~~ | ~~1.8345~~ | ~~+11.85%~~ |

Greedy continuations agreed on 1 of 3 prompts.

**The first corpus was wrong and is struck.** It was a single paragraph repeated
twelve times, which is trivially predictable -- ppl 1.64 -- and a relative delta
measured on a corpus the model has effectively memorised says nothing about real
text. Caught before quoting it. The rule is the same one that keeps biting in
this log: check that the instrument can express the thing being measured.

### What the 7.2% is and is not

It is **round-to-nearest with no calibration**: no AWQ activation-aware channel
scaling, no GPTQ error compensation, no calibration set at all. Published int4
g128 results with either technique land nearer 1-3% on this kind of corpus.

So +7.2% is the floor of the quality range, not the number a deployment would
see -- and the TurboMind kernel consumes **AWQ-format** weights specifically, so
the calibration step is part of the intended pipeline rather than an optional
extra. Quoting 7.2% as "the cost of w4a16" would understate the method by
roughly the width of the technique it omits.

It is also not free to find out: calibration needs a representative dataset and
a pass over the model, and the result depends on that dataset.

### Standing on the throughput side

E47 measured the kernel at 2.3-3.3x on decode GEMMs and 0.77x at M=128, with a
3.76x ceiling from weight bytes. Neither number is in the engine yet: no
quantized Qwen3.5-27B checkpoint exists on this box, and the kernel is still in
`legacy/patches/` rather than registered as a tokenspeed-kernel op.

### Order that follows from this

1. AWQ-calibrate Qwen3.5-27B and re-measure ppl. If calibration does not bring
   it under ~3%, the throughput win is not worth taking on a quality-sensitive
   deployment and the rest of this is moot.
2. Only then wire the kernel in as `unieinfra/kernel-sm120` `gemm/turbomind`,
   and measure end-to-end against the E48 baseline (208.0 vs vLLM 248.1).

Doing those in the other order would produce a fast engine whose output quality
nobody has agreed to.

## E52 — the capacity plan was wrong, bf16 SSM state buys nothing, and mixed batch needs code that does not exist

Three things were tried on top of the fp8-KV result (E-prev: 245.8 tok/s, +19.0%).
Two are negative results and one corrects an earlier analysis. Standing against
vLLM at C=32/ILEN=4096 is now **245.8 vs 248.1, −0.9%**.

### The #12 plan would have made things 13x worse

The task said: stop `equalized_block_size` inflating the attention page P from 64
to 832, since P is only that large to cover a 3.2 MB GDN state row, and the
`state_group_page_sizes` machinery is now ported. That analysis was wrong, and
reading the allocator is what showed it:

```python
self.num_pages_with_null = self.size // self.page_size + 1
self.state_slabs = [(zeros((num_pages_with_null, *conv_shape)),
                     zeros((num_pages_with_null, *ssm_shape)))]
```

**One full state row is allocated per page id.** At P=832 that is 149 rows. At
P=64 it would be ~1937 rows x 3.2 MB = **6.2 GB instead of 478 MB**. The
inflation is not waste bookkeeping — it is what bounds state memory. The 14.7%
KV rounding waste is the price paid for that, deliberately.

`flat_state_slabs.py` says so in a comment that should have been read first:
"the flat plan packs [conv|ssm] state rows and KV rows into **one page-id
space**". Decoupling P from the state row means giving state its own id space,
which is the placeholder-block work E48b established is lost.

### bf16 SSM state: no gain, so not adopted

`--mamba-ssm-dtype bfloat16` halves the state row, which pulls P back from 1600
(where fp8 KV pushes it — halving `kv_bytes_per_slot` doubles
`ceil(state_bytes / kv_bytes_per_slot)`) down to 832, and doubles the page count
from 149 to 294.

| config | P | pages | requests that fit at ILEN=4096 |
|---|---|---|---|
| bf16 KV, fp32 SSM | 832 | 149 | 24 |
| fp8 KV, fp32 SSM | 1600 | ~150 | 49 |
| fp8 KV, bf16 SSM | 832 | 294 | 49 |

Both fp8 configurations fit 49 requests — one via coarser pages, one via more of
them. Measured: **246.4 vs 245.8 tok/s**, i.e. the same number. It changes the
precision of the recurrent state for nothing, so it is not adopted.

### Mixed batch: the MHA half ports, the GDN half does not exist

Upstream's MHA backend asserts `not forward_mode.is_mixed()`. The fork's 14
hunks apply clean and lift that. But the run then dies in the *linear* half:

```
RuntimeError: Size does not match at dimension 0 expected index [2, 1] to be no
larger than self [1, 3] apart from dimension 1     # _flat_state_pages
```

`MambaAttnBackend` gathers state rows assuming every sequence contributes the
same number. Searching for the fix: `hybrid_linear_attn.py`'s whole divergence
is 31 lines and already applied; the diff's 67 "mixed" hits are
`validate_mixed_batch_backend`, a **guard** that rejects incapable backends, not
an implementation. Upstream has no `is_mixed` handling there either.

**So this is the second measured result whose code is gone**, after E48b. E45
reported 140 successful `Mix batch` rounds; nothing in the repository can produce
one on this model.

What landed instead: the backend declares `supports_mixed_batch = False` and
`EventLoop.__init__` refuses the flag. Before, the server started, accepted
requests, completed none, and scored 0 tok/s in all 20 windows of a 600 s arm
before the cause was visible.

### Where the remaining ~10% actually is

The fp8-KV arm oscillates between two levels rather than sitting at its mean:

```
199 212 204 212 265 273 273 273 273 273 272 247 211 211 212 212 273 273 273 273
```

Bimodal, 211 vs 273, 25.1% spread. **The high mode already beats vLLM by 10%.**
The tail mean is whichever phase the window lands in, which is also why 245.8 and
246.4 should not be read as different.

Capacity is no longer the cause — 49 requests fit against C=32. The remaining
candidate is the prefill/decode alternation, which is what mixed batch would
address, and which now needs the GDN state gather written rather than ported.

### Hazard for whoever reads this next

**Read the allocator before trusting a geometry argument.** The case for #12 was
arithmetically clean, cited the right constant, and was backwards, because it
reasoned about page *size* without checking what is allocated *per page*. The
comment naming the constraint was eleven lines from the code being changed.

## E53 — GDN mixed batch needs a forward split, not index fixes; two of three blockers cleared

Continuing #23. Three failures, each found by running the flag rather than by
reading, each revealing the next.

### 1. `_state_page_bounds` — batch-uniform mode branch (fixed, 8ecb4b24)

Branched on `forward_mode` for the whole batch: decode subtracted one from every
row, extend read `extend_prefix_lens[:bs]`. A MIXED plan advances its segments by
different amounts and `extend_prefix_lens` covers only the extend segment, so the
extend branch also produced a `before` shorter than the batch.

Segmented on the layout MHA already builds: rows `[0, num_extends)` carry a
prefix, `[num_extends, bs)` advance by one, with `num_extends` read off
`extend_prefix_lens.shape[0]` rather than new plumbing.

### 2. `init_forward_metadata` — same bug, second site (fixed, 69a5f7c5)

`query_start_loc` is built for the whole batch from `extend_start_loc` /
`extend_seq_lens` / `extend_prefix_lens`, all extend-segment-sized:

```
RuntimeError: The size of tensor a (6) must match the size of tensor b (3)
```

Both branches now default every row to one query token and overwrite the first
`num_extends`.

### 3. `_mask_fresh_initial_state` — not an index bug (open)

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (6)
```

`recurrent_state` has 2 rows — the extend segment — against a 6-row
`has_initial_states`. **The linear half runs the whole mixed batch down its
extend path, decode rows included.**

This is the structural difference from MHA, which handles MIXED by *splitting*:
`_forward_mixed` gives each segment the kernel built for it, which is why
`MHAExtendMetadata` carries `mixed_decode_rows` and a sibling
`MHADecodeMetadata`. The linear half has no equivalent. Writing one is the
remaining work; there are 23 references to the extend arrays in this file and
patching them one at a time would only move the failure.

### A correction to E52's guess

E52 said blocker 2 was the state block table arriving extend-sized. Wrong: both
construction paths build it with `num_requests=bs`, and instrumenting the gather
confirmed full-batch tables (`rows=(3, 1)` at `bs=3`). The illegal memory access
came from the *bounds*, not the table. The guess was plausible because MHA does
slice its own copy (`rows = num_extends if is_mixed else bs`) — but that is MHA's
private metadata, not the kwarg the linear half reads.

### State

`supports_mixed_batch` stays False; the startup guard keeps refusing the flag.
Standing throughput is unchanged at **245.8 vs vLLM 248.1**, and the bimodal
211/273 oscillation — whose high mode beats vLLM by 10% — is still the target
that mixed batch was meant to address.

### Hazard for whoever reads this next

**Two index fixes in a row can look like a trend and be a coincidence.** After
the second one it was tempting to keep slicing; the third failure was a
different kind of thing, and only reading what the shapes *meant* (2 = extend
rows, 6 = batch) showed the path itself was wrong. Count the remaining call
sites before assuming the next fix is the last one.

## E54 — how vLLM and TRT-LLM serve a mixed batch on a recurrent layer, and why our split is in the wrong place

Written after five failed attempts at `--enable-mixed-batch` on Qwen3.5-27B,
each fixing the previous failure and finding the next. Surveying first would
have cost less than the fifth attempt.

### Both engines split. Neither splits where we did.

**vLLM** (`model_executor/layers/mamba/mamba_mixer2.py`):

```python
hidden_states_B_C_d, hidden_states_B_C_p = torch.split(
    hidden_states_B_C[:num_actual_tokens], [num_decode_tokens, num_prefill_tokens], dim=0)
```
then `mamba_chunk_scan_combined_varlen` for prefill against
`selective_state_update` for decode. Metadata is built prefill-only, marked with
a `_p` suffix throughout (`has_initial_states_p`, `seq_idx_p`,
`cu_chunk_seqlen_p`).

**TRT-LLM** (`_torch/modules/mamba/mamba2_mixer.py`):

```python
seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
batch_split_size  = [num_prefills, num_decodes]
state_indices_p, state_indices_d = torch.split(state_indices, batch_split_size)
```

Three things fall out of the comparison:

1. **The split lives in the mixer**, i.e. the model layer, in both. Ours is in
   the attention backend. The layer owns the tensors; the backend does not, and
   every failure in E53 and after was the backend reaching for something it does
   not own -- query-length arrays, the kernel's output buffer, the state gather.

2. **Two axes, not one.** Per-token tensors split by `seqlen_split_size`,
   per-request tensors by `batch_split_size`. Our `MambaForwardMetadata.segment`
   (per-request) and `_slice_token_kwargs` (per-token) are the same decomposition
   and can be kept.

3. **TRT-LLM preallocates the output and splits *that*:**

   ```python
   # Preallocate output tensor to avoid memcpy cost for merging prefill
   # and decode outputs
   preallocated_ssm_out = torch.empty([zxbcdt.shape[0], nheads * head_dim], ...)
   preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
       preallocated_ssm_out, seqlen_split_size)
   ```

   Each half writes into its own view of one buffer. That removes the
   `torch.cat` we ended with, and it is the pattern to copy.

**Order differs between them and matters here.** vLLM puts decode first,
TRT-LLM prefill first. TokenSpeed's own scheduler and `MHAAttnBackend` put
extend first (`rows [0, num_extends)`, `mixed_decode_rows` after), so TRT-LLM's
order is the one to follow -- adopting vLLM's would mean reordering the whole
batch convention for one layer type.

### What was kept from the failed attempt

`146c4175` lands the machinery, disabled. Correct and expensive to find:

- **`.contiguous()` is a no-op on a leading-dimension slice.** PyTorch already
  considers such a slice contiguous, so the call returns the same view and the
  data pointer keeps its byte offset; the recurrent kernels reject it with
  `Misaligned Tensor data on argument #9` (the output buffer). `.clone()` is
  required, and only for a segment that does not start at zero. TRT-LLM's
  preallocate-and-split sidesteps this entirely, which is a second reason to
  prefer it.
- Per-token kwargs must be selected **by shape, not by name**: the recurrent
  forward takes a growing set of them and a name list goes stale silently, where
  a missed entry is a wrong answer rather than an error.

### Hazard for whoever reads this next

**Surveying is cheapest before the first attempt, not after the fifth.** The
instruction to look at how the other engines do it was given and half-followed:
vLLM was read, the split-into-two-halves shape was taken from it, and the part
that mattered -- *where* the split lives -- was not. Five serve-crash-read cycles
followed. The structural question ("who owns these tensors") was answerable in
one file read.

### E54b — the other three engines, and the best answer of the four

Extending the survey on request. All four engines that implement a recurrent
layer split a mixed batch. They differ in *where* and in whether they
materialise the split.

| engine | where the split lives | order | how the halves are addressed |
|---|---|---|---|
| vLLM | mixer (`mamba_mixer2.py`) | decode first | `torch.split` on the token dim |
| TRT-LLM | mixer (`mamba2_mixer.py`) | **prefill first** | `torch.split`, output preallocated and split too |
| SGLang | mixer (`mamba/mamba.py:518`) | **prefill first** | `torch.split` |
| TurboMind | layer plan (`GatedDeltaNetLayer.cc`) | decode first | **offsets, no slicing at all** |

**SGLang's hybrid backend has the same gap we do.**
`hybrid_linear_attn_backend.py` — the file our `hybrid_linear_attn.py` shares a
lineage with — dispatches `is_decode()` vs else with no mixed branch, exactly
like ours. Its mixed handling is one level down, in the mixer. That is strong
confirmation that the backend is the wrong level: the engine we are closest to
made the same structural choice and put the split elsewhere.

**TurboMind is the most interesting of the four:**

```cpp
data.decode_count = 0;
while (data.decode_count < data.batch_size && data.input_lens[data.decode_count] == 1)
    ++data.decode_count;
data.prefill_count = data.batch_size - data.decode_count;

if (data.decode_count)  { planning.physical_batch = data.decode_count; ... }
if (data.prefill_count) { planning.q_offsets.assign(host_offsets.begin() + data.decode_count, ...); }
```

Two things nobody else does:

- **It never materialises a slice.** Two launch *plans* carry offsets into the
  same buffers. That removes the concat (TRT-LLM's preallocation also does) *and*
  the alignment trap that cost two debug cycles here — there is no offset view
  to misalign, because the offset is an argument rather than a pointer.
- **It derives the boundary itself**, scanning `input_lens` for the leading run
  of length-1 rows, instead of trusting a `num_prefills` the scheduler supplies.
  Self-describing, and it cannot disagree with the batch it was handed.

### What to build, given all four

Split in the layer, prefill-first (TokenSpeed's own convention, matching TRT-LLM
and SGLang), and pass **offsets rather than slices** wherever the kernel takes a
base pointer plus a count — TurboMind's shape. Preallocate the output and hand
each half a view, per TRT-LLM, for anything that must be a tensor.

Not surveyed: tileRT, which an earlier entry established ships no CUDA or C++
source at all — pure Python shims over `libtilert_*.so` — so it has no
implementation to read.
