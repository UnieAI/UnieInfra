# Five-model adaptation on 8xH200 (roy-h200g)

Serving `Qwen3.6-35B-A3B`, `Qwen3.5-122B-A10B`, `Qwen3-Embedding-8B`,
`Gemma4-31b-it` and `Whisper-large-v3` on two of the eight H200s (SM90, cc
9.0) of the `roy` host, through the token-speed / UnieInfra engine. The
engine is pinned to logical GPUs 0,1 (host 6,7); host 0–3 is never touched.

| Family | Model | Path | TP | Backend result |
| --- | --- | --- | --- | --- |
| Dense MoE | Qwen3.6-35B-A3B | `/models/Qwen/...` | 1 | default |
| Dense MoE | Qwen3.5-122B-A10B | `/models/Qwen/...` | 1 | default |
| Embedding | Qwen3-Embedding-8B | `/models/...` | 1 | default |
| MoE + per-layer SWA | Gemma4-31b-it | `/models/google/gemma-4-31b-it` | 2 | **triton (forced)** |
| Encoder-decoder ASR | Whisper-large-v3 | `/models/openai/whisper-large-v3` | 1 | default (fp16) |

## Status

| Item | State |
| --- | --- |
| Qwen3.6 / Qwen3.5 / Qwen3-Embedding serve correctly | done |
| Gemma4-31b-it serves correctly on the *default* backend | done (see gate) |
| Whisper-large-v3 transcribes correctly (ASR endpoint) | done (token-for-token vs HF) |
| Whisper baseline (RTF / throughput) | captured (see below) |

## Correctness gates that had to be fixed

Three engine-side defects surfaced while bringing these models up. Each is a
real capability/selection bug, not a launch-flag workaround.

### 1. `kv_cache_dtype="auto"` ignored the model's dtype (Whisper)

`--kv-cache-dtype` help promises `auto` resolves to the model data type, but
`resolve_dtype("auto")` hardcoded `bfloat16`. A fp16 checkpoint (Whisper as
shipped: `torch_dtype: float16`) therefore built a bf16 pool against fp16
queries, and the paged MHA kernels are registered per-dtype with an all-same
mix required — so the first decode step died with
`No kernel found for attention.mha_decode_with_kvcache (q=float16,
k_cache=bfloat16, v_cache=bfloat16)`.

Fix: `runtime/layers/attention/configs/mha.py` resolves `auto` to
`model_config.dtype` (bf16 models unchanged; fp16 models get a fp16 cache).
The whole Whisper model then runs fp16 end-to-end, matching the fp16 kernel
registrations.

### 2. Encoder-decoder cross-attention got mixed dtypes (Whisper)

Whisper's decoder loads fp16 while its encoder keeps fp32 checkpoint conv
weights (bf16 after load), so the decoder's query projection and the encoder's
key/value projection land on different precisions. `cross_attn_varlen` fed
them to `flash_attn_varlen_func`, which aborts:

`RuntimeError: mha_fwd ... query and key must have the same dtype`

Fix: `runtime/layers/attention/mm_encoder_attention.py` normalizes q/k/v to a
single (model-side) dtype before the flash-attn call.

### 3. Non-uniform per-layer head dims selected fa3 (Gemma4)

`Gemma4ForConditionalGeneration` has 60 layers: 50 `sliding_attention` at
`head_dim=256` and 10 `full_attention` at `global_head_dim=512`. The fa3 dense
kernels are validated up to `head_dim=256`; selecting them for the 256 layers
while Triton handled the 512 layers silently corrupted output (E2E: repeated
garbage, `finish=length`).

Fixes:
- `tokenspeed_kernel/ops/attention/flash_attn/__init__.py`: the three fa3 MHA
  kernels now declare `head_dim={64,128,256}` (omitting the trait was a
  wildcard, which is how 512 was able to select a kernel that cannot run it).
- `runtime/layers/attention/backends/mha.py`: a model with non-uniform
  per-layer head dims is steered wholly to the Triton path, the only backend
  validated for that geometry. Default backend then serves `Paris` /
  `finish=stop`, deterministic across repeated temperature-0 runs.

The forced-Triton path is not a masking flag: it is the only arm that keeps
Gemma correct. A/B against per-layer backend selection (fa3 for the 256-dicts
sliding layers, Triton for the 512-dict global layers) failed the correctness
gate the way the engine warns about — the "corrupt arm" did not even match on
output *length*. Same prompt, `temperature=0`, `max_tokens=64`:

| Arm (bottom decoder) | Output | Length | Gate |
| --- | --- | --- | --- |
| Triton everywhere (default) | `The silver moon glows softly in the velvet night sky, ...` | 28 tokens, `stop` | reference |
| fa3 for 256 + Triton for 512 (per-layer) | `"// I'm sorry, but your request is not clear. ..."` | 29 tokens, `stop` | **fails** (length + text differ) |

The per-layer schedule shows the two backends interleaving on the decode step
(`fa3_mha_decode_with_kvcache_cached` then `triton_mha_decode_with_kvcache_cached`),
which is where the corruption enters. Mixing head-dim groups across two
backends in one decode step is the open defect to fix; until it is, steering
the whole model to Triton is the only correct route.

## Verification

- **Gemma4**: two identical temperature-0 responses (`Paris`, `finish=stop`,
  2 tokens, deterministic).
- **Whisper**: the engine's transcript is byte-for-byte identical to an
  independent HuggingFace `WhisperForConditionalGeneration` reference run
  (CPU, same 16 kHz features, `<|notimestamps|>` prompt, temperature 0):
  *"The New York Times from July 21st, 1969. ..."* — token-for-token
  agreement, then repeated parallel requests all agree.
- **Whisper baseline** (17.4 s clip, temp 0, steady state): single request
  latency 3.40 s -> RTF 0.195 (~5× real time); at concurrency 4,
  19.4 audio-seconds/sec (aggregate RTF 0.052), all outputs byte-identical.

## Performance baselines (diagnostic)

`tokenspeed` suites (`temperature=0`, `max_tokens=128` unless noted), run from
inside the container (the engine HTTP port is container-bound). "knee" =
concurrency where throughput flattens.

> Prompt-length note: the `conc 1` figures below are on the short `ts64`
> prompts (~250 tokens); the `conc 4..16` figures are on the `4096` / `8192`
> -token prompts and are **not** comparable to a short-prompt run. The short
> -prompt engine numbers for every concurrency (and the matched vLLM compare)
> are in the next section.

| Model | conc | req/s | out_tok/s | e2e p50 |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B (TP1) | 1 / 4 / 8 / 16 | 1.64 / 2.89 / 5.64 / 7.50 | 210 / 369 / 722 / 960 | 607 / 1374 / 1474 / 2205 ms |
| Qwen3.5-122B-A10B (TP2) | 1 / 4 / 8 / 16 | 1.18 / 2.30 / 3.99 / 4.93 | 151 / 294 / 511 / 631 | 786 / 1724 / 1844 / 3326 ms |
| Gemma4-31b-it (TP2, Triton) | 1 / 4 / 8 / 16 | 0.60 / 2.13 / 3.26 / 6.80 | 76 / 272 / 418 / 871 | 1674 / 1854 / 2089 / 2250 ms |
| Qwen3-Embedding-8B (TP1) | 1 / 4 / 16 / 32 | 46 / 78 / 99 / 116 | n/a | 21 / 48 / 161 / 241 ms |

Observations:
- Qwen3.6 and Qwen3.5 keep scaling through conc 16 (knee not yet hit); Qwen3.5
  flattens around 8→16.
- Gemma's *throughput* scales well at high concurrency (Triton batches well),
  but its **single-stream decode is the weak point** (76 out_tok/s at conc 1).
  That is the Triton penalty from the correctness fallback — the fast fa3
  decode cannot be used until the mixed-backend corruption above is fixed.
- Embedding (2048-char prompts) is I/O-light: 46→116 req/s at sub-250 ms.

### Matched A/B vs vLLM (short `ts64`, temp 0, max_tokens 128)

Both sides on the **same** short `ts64` prompts (~250 tokens), bf16 checkpoints,
`temperature=0`, `max_tokens=128`. Engine = `ts-env:gpu` on GPU6,7; vLLM =
`vllm/vllm-openai:nightly` on GPU6,7 (TP2 needs `NCCL_NVLS_ENABLE=0`,
`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `--max-num-seqs` for Qwen3.5). Every text
request emitted exactly 128 completion tokens, so `req/s` × 128 ≈ `out_tok/s`.

| Model | concurrency | engine req/s | vLLM req/s | engine out_tok/s | vLLM out_tok/s | engine e2e p50 | vLLM e2e p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B (TP1) | 1 / 4 / 8 / 16 | 1.64 / 4.70 / 7.95 / 11.72 | 1.71 / 4.44 / 7.65 / **16.07** | 210 / 601 / 1017 / 1500 | 218 / 568 / 979 / **2057** | 604 / 832 / 1006 / 1362 | 585 / 860 / 1045 / **980** |
| Qwen3.5-122B-A10B (TP2) | 1 / 4 / 8 / 16 | 1.08 / 2.54 / 4.18 / 5.35 | **1.37 / 3.19 / 4.94 / 8.84** | 138 / 326 / 536 / 685 | **175 / 408 / 633 / 1131** | 915 / 1557 / 1899 / 2976 | **721 / 1216 / 1613 / 1820** |
| Gemma4-31b-it (TP2, Triton) | 1 / 4 / 8 / 16 | 0.54 / 2.22 / 3.05 / 7.85 | **0.67 / 2.64 / 5.23 / 10.20** | 69 / 284 / 390 / 1005 | **86 / 338 / 669 / 1306** | 1778 / 1804 / 1951 / 2038 | **1487 / 1513 / 1526 / 1565** |
| Qwen3-Embedding-8B (TP1) | 1 / 4 / 16 / 32 | 22.8 / 54.2 / 78.2 / 98.1 | **57.8 / 127.0 / 207.4 / 184.8** | n/a | n/a | 42 / 70 / 184 / 240 | **17 / 31 / 58 / 104** |

`req/s` for embedding is requests/s (dim 4096). Whisker not shown; engine
Gemma4 `conc 8` e2e p90 spiked to 6208 ms vs p50 1951.

**Correctness gate (AGENTS.md: output length AND token-for-token agreement at
temp 0) — result: partial pass.**

- Output length gate passes everywhere: every request on both sides emitted
  exactly 128 completion tokens (no truncation / no degenerate fast arm).
- Token-for-token agreement (exact strings): Qwen3.6 53–64%, Qwen3.5 16–31%,
  Gemma4 36–53% of the 64 prompts. This is **not** 100%, so the A/B is not a
  strict apples-to-apples comparison.
- Divergence position: Qwen3.6 agrees for almost the whole string (median
  common prefix ≈ suffix ≈ the full ~530 chars), diverging only in the last few
  tokens → near-token-identical. Qwen3.5 and Gemma4 diverge early/mid (~char
  110/146): both stacks still produce full, substantive 128-token reasoning,
  but they format/continue the thinking differently (e.g. Gemma4 one side emits
  `$\epsilon \tau ...$`, the other `ε τ γ ...`).

**Verdict under these matched short-prompt conditions: vLLM is at or above the
engine on every model and every concurrency measured.** The engine's apparent
advantage in the baseline table above came from comparing long-prompt engine
numbers against short-prompt vLLM numbers; the matched short-prompt run shows
vLLM ≥ engine (Qwen3.6 is the closest, and flips at conc 16; embedding is the
largest gap, ~2–4×). Because Qwen3.5/Gemma4 are not token-identical between
stacks, their exact deltas are confounded (different prompt/decode formatting),
but the direction is consistent and there is no degenerate-output arm hiding a
false "win".

### Perf verdict (algorithm level)

Checked every kernel the Qwen MoE models actually dispatch, rather than
sweeping flags:

- `moe.apply -> flashinfer_cutlass_unquant_moe_apply` — the fast fused CUTLASS
  MoE path.
- `moe.softmax_topk -> triton_softmax_topk` — the only registered softmax-topk
  implementation (Triton + PyTorch fallback); no faster kernel is shadowed.
- Decode attention `fa3_mha_decode` + `flashinfer_gdn_decode_step`; prefill
  `triton_gdn_chunk_prefill`.

The prefill CUDA graph is deliberately off when the flashinfer GDN kernel is
live. `resolve_fi_gdn_prefill_graph` (`runtime/engine/event_loop.py`) records
the measured arms on Qwen3.5-27B RAG C=128: Triton+graph 511.9, Triton-no-graph
472.0, flashinfer+graph 579.97 (garbage), flashinfer-no-graph **574.9 tok/s
(correct)**. The default is already the best *correct* arm (574.9); the only
faster combination (flashinfer+graph, 580.0) answers every request with
`"!!!!!!!!"`. So the prefill path is at its correct optimum — switching to
`TOKENSPEED_ENABLE_FI_GDN_SM120=0` would be a ~2% regression, not a win.

For Gemma, `fa3` is hard-capped at `head_dim<=256` (see the fa3 registration
comment, "Hopper FlashAttention caps the head dimension at 256"), so the
512-head global layers can never run fa3; pairing fa3 (256) with Triton (512)
in one decode step is the corrupt arm shown above. There is no safe kernel-level
fix short of solving the engine's mixed-backend decode assembly. The all-Triton
gate is therefore the correct and only deliverable for Gemma; its remaining gap
is the open mixed-backend defect, not a tunable flag.

## Qwen3.6: real no-replay A/B + flashinfer SM90 GDN prefill

The short-prompt matrix above (conc 1/4/8/16) is a self-authored corpus and no
longer the benchmark. The authoritative comparison is
`UnieInfra/bench/agent-slo/` (SWE-agent multi-turn traces, no-replay: fresh
server per cell, warmup on a disjoint corpus, target corpus served exactly
once). Qwen3.6-35B-A3B, TP1, conc 8, temp 0, max_tokens 128, ignore_eos:

| run | out_tok/s | TTFT p50 | decode ms/tok | output ntok |
| --- | --- | --- | --- | --- |
| engine, Triton GDN (before) | 339.9 | 1012 ms | 14.42 | 127.7 |
| engine, flashinfer SM90 GDN (public API) | 481.2 | 588 ms | 9.37 | 127.7 |
| engine, flashinfer SM90 GDN (**raw kernel**) | **493.6** | **555 ms** | **9.24** | 128 |
| vLLM, H200 | 490.5 | 428 ms | 11.88 | 127.1 |

The raw arch-specific kernel (`_use_raw_kernel` on sm90, bypassing the public
`@flashinfer_api` wrapper's ~0.33 ms/call Python trace over 30 GDN layers)
closes the last gap: **493.6 vs 490.5 out_tok/s (engine >= vLLM)**, decode
9.24 vs 11.88 ms/tok, e2e p50 1693 vs 1904 ms. Correctness gate: GDNTRACE
`nan=0`, first-chunk absmean matches the public-API path
(0.0019638 vs 0.0019625), output lengths preserved (ntok p50 128), texts
non-garbage (unique-6gram 0.94 == vLLM). TTFT p50 remains 555 vs 428 ms (the
remaining lever).

Bottleneck found: the engine silently ran the 30 GDN layers' prefill on the
Triton fallback on sm90 even though flashinfer ships an SM90 C++ kernel
(`chunk_gated_delta_rule_sm90`, the same entry point vLLM dispatches to).
`gated_delta_rule.py` probed and registered it only under `is_blackwell_plus`,
so the sm90 kernel was never loaded. The fix probes per arch major
(`arch_version.major`) and registers the flashinfer GDN prefill at
`min_arch_version=ArchVersion(9, 0)`; `is_available()` now enables sm90. The
contract matches the engine's state already: `mamba_ssm_dtype=float32`
(flashinfer requires fp32 state on sm90) and the runtime passes a dense compact
`[N, H, V, K]` state so no pool-index support is needed for prefill.

Verified: `[KERNEL] attention.gdn_chunk_prefill ->
flashinfer_gdn_chunk_prefill`, GDNTRACE `nan=0`, output lengths preserved
(p50=128, min 124, max 128), texts non-garbage (unique-6gram 0.95 vs vLLM
0.94), and tok/s 481.2 vs vLLM 490.5 (0.98x, up from 0.69x). The prefill CUDA
graph is deliberately off for the flashinfer GDN path (not graph-safe); the
remaining TTFT gap (588 vs 428 ms) is the next lever.

## Qwen3.5-122B-A10B: SM90 GDN fix active but still 0.88x (in progress)

Same GDN architecture as Qwen3.6 (36 GDN + 12 full-attn layers), so the SM90
flashinfer fix applies directly. It is **confirmed active and correct at TP2**:
`[KERNEL] attention.gdn_chunk_prefill -> flashinfer_gdn_chunk_prefill` on both
TP ranks, GDNTRACE `nan=0` on both. The tailoring still loses.

**Both engines reject the benchmark's real prompts at their defaults — this is a
feasibility defect, not a perf flap.** The `swe_smith_agent_sess32x8` prompts are
~7.4k tokens. The engine's default `max_single_request_tokens` (paged-cache
memory budget for a 122B) caps admission at 4350 and returns HTTP 400 beyond it
(`input_processor.py:214`). vLLM instead defaults `max_seq_len` to the
`max_position_embeddings` (262144), over-allocates KV cache and dies on rank 1
at `initialize_from_config` (`max_num_seqs (1024) exceeds available Mamba cache
blocks (367)`). A fair reference needs `--max-model-len 16384` on both (and
`--max-num-seqs 128` on vLLM): engine then accepts the 7.4k prompt and serves
ok=64/64, vLLM accepts it too.

Real no-replay conc-8 A/B at `--max-model-len 16384` (temp 0, max_tokens 128):

| run | out_tok/s | TTFT p50 | decode ms/tok | e2e p50 | output ntok |
| --- | --- | --- | --- | --- | --- |
| engine, RAW flashinfer GDN | 305.9 | 1214 ms | 16.59 | 3314 ms | 128 |
| vLLM, H200 | **348.0** | **498 ms** | 17.45 | 2527 ms | 128 |

Engine decode is already **faster** than vLLM (16.59 vs 17.45 ms/tok); the loss
is entirely **TTFT, 2.44x** (1214 vs 498 ms). The GDN fix that closed Qwen3.6 is
not enough for the 122B, so the remaining prefill cost is **not the GDN layers**.

**The "prefill graph was disabled by the GDN path" hypothesis is REFUTED by
measurement.** The serve log shows the flashinfer GDN path turns the whole
prefill CUDA graph off ("Disabling the prefill CUDA graph ... silently produces
garbage under graph capture"). I re-ran with the capture forced on
(`TOKENSPEED_FI_GDN_ALLOW_PREFILL_GRAPH=1`, timing-only: GDN is not
graph-safe so output is garbage, but TTFT is a scheduler/kernel timing and is
valid). With the default bucket ladder the capture **OOMs**: `weak_ref_tensor`
is unavailable (`PermissionError: /home/runner/.cache/torch_extensions`), so
the breakable-graph falls back to identity and its memory scales with the
bucket sum; 40 buckets blew the ~16 GB a 122B leaves free (this is a real
memory defect on top of the graph-safety one). Capturing a **single** 8192
bucket fits and serves ok=64/64:

| run | out_tok/s | TTFT p50 | decode ms/tok |
| --- | --- | --- | --- |
| engine, GDN graph OFF | 305.9 | 1214 ms | 16.59 |
| engine, GDN graph ON (1x8192 bucket) | 291.8 | **1226 ms** | 17.54 |
| vLLM | 348.0 | 498 ms | 17.45 |

Graph ON does not move TTFT at all (1226 vs 1214 ms) -- slightly worse overall
(291.8 vs 305.9 tok/s). So the 2.44x TTFT gap is **not the disabled graph**. The
next lever I tested was the eager prefill forward itself, using the engine's own
per-forward timer (`TOKENSPEED_LOG_MM_TIMING=1`, `mm_timing forward_execute_ms
... forward_step=...`):

```
mode=EXTEND bs=8 total_tokens=8162 forward_step=257.4 ms
mode=EXTEND bs=3 total_tokens=8192 forward_step=248.4 ms
mode=EXTEND bs=6 total_tokens=8112 forward_step=244.4 ms
mode=EXTEND bs=1 total_tokens=8192 forward_step=255.3 ms
mode=EXTEND bs=1 total_tokens=1536 forward_step=131.9 ms   # tail chunk
```

**This refutes the "eager 122B MoE/FA3 prefill compute is the bottleneck"
hypothesis too.** A full ~8k-token prefill forward (bs 1..8) costs only
**~250 ms**; forward_step is token-count-bound and nearly batch-independent
(255 ms at bs=1 vs 257 ms at bs=8 for the same ~8.2k tokens). So the ~8k forward
is only 250 ms, yet TTFT p50 is 1214 ms -- **~75% of TTFT is scheduler/queue
latency, not compute**. Under conc-8 with `chunked_prefill_size=8192` the engine
slices 8 requests' ~59k prefill tokens into ~8 serial 8192-token forwards and a
request's first token waits for its whole prompt to be spread across several of
them (each ~250 ms), which is why TTFT is ~1214 ms while vLLM (different chunk
priority, completes one request's prompt per round) reaches ~498 ms.

So the actual lever is **prefill/chunk scheduling** -- ensuring a request's full
prompt is prefilled in a single chunk (per-request chunk priority) rather than a
strict global 8192-token round-robin across all concurrent requests -- not the
GDN kernel and not the graph. The next experiment is to hand the scheduler a
per-request chunk budget that fits the full prompt (single chunk per request) and
re-measure; that is the quantity that correctly explains the 2.44x TTFT. To also
confirm the forward is healthy, the engine's own `forward_step` (~250 ms / 8k)
is well under the TTFT it produces. Note: toggling
`TOKENSPEED_ENABLE_FI_GDN_SM120=0` (Triton+graph arm) and
`TOKENSPEED_FI_GDN_ALLOW_PREFILL_GRAPH=1` with the default bucket ladder both
crash at startup (the latter from the capture OOM above; the former
rc=-10 during `lamportInitialize`).

**The simple "bigger chunk budget" fix is also REFUTED.** Raising
`chunked_prefill_size`/`max_prefill_tokens` 8192 -> 16384 made TTFT *worse*, not
better: 1214 -> **1614 ms** (3.24x vLLM; tok/s 312.3, still ok=64/64); the
forwards themselves stayed ~220 ms/16k. So the delay is not one request's prompt
being spread across token-budgeted rounds -- enlarging the budget spreads it
further and the scheduler simply waits to fill the bigger bucket. The one hard
fact that survives every experiment: `forward_step` is ~250 ms for a full ~8k
forward, yet TTFT p50 is 1214 ms, so **~75% of TTFT is scheduler/queue latency
between the request arriving and its first forward-token**, and neither the GDN
kernel, the CUDA graph, nor the prefill-token budget moves it. The remaining
work is to trace the scheduler's request -> forward assignment (why a request
does not reach its first output token until several ~250 ms forwards later, and
how vLLM reaches it in ~500 ms) from the engine's own `Scheduler` / admission
path, then fix the assignment at the algorithm level.

**The decisive concurrency experiment (conc-1 vs conc-8) isolates schedule vs
compute.** Same engine, one fresh server, same corpus/temp0, `--max-model-len
16384`:

| run | out_tok/s | TTFT p50 | note |
| --- | --- | --- | --- |
| engine conc-1 | 104.3 | **532 ms** | single request (8 seqs) |
| engine conc-8 | 305.9 | **1214 ms** | 8 concurrent |
| vLLM    conc-8 | 348.0 | **498 ms** | 8 concurrent |

The engine's *single-request* prefill is **532 ms** for a single ~10k prompt --
that is the true per-token prefill cost, and it is only ~7% behind vLLM. Raising
the request count 1 -> 8 costs the engine **+682 ms** (532 -> 1214 ms) while vLLM
stays essentially *flat* at ~498 ms. So the fixable gap is **not per-token
compute**.

### Round-7 trace: what `newForwardOperation` actually computes (with the engine's own plan/admit trace)

The previous "fair-split / round-robin is the defect" hypothesis is **REFUTED**.
Three corrected facts, each measured:

1. **The active scheduler is the root tree, not the legacy copy.** The imported
   `tokenspeed_scheduler` changes live in
   `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` (root), and it is
   **byte-different** from `legacy/experiment/legacy/tokenspeed-scheduler/...`
   (a stale copy). All earlier code references pointed at the stale file.

2. **The real prompts are ~10.2k tokens (mean 10165, median 10641, max 13356)**
   on the Qwen tokenizer, not ~7.4k and not a ~59k pool. This is the quantity
   that sizes every forward.

3. **FCFS already works and is NOT the fix.** The default and the
   `--context-chunk-policy fcfs` path both give a single Submitted request its
   whole budget in one forward. With the engine's own plan trace (touch
   `/workspace/roy/.plan_trace.on`; it writes `.plan_trace.txt` and
   `.admit_trace.txt` -- no env var, the scheduler runs in a spawned process):
   `first_chunk=8192 state(sub=0 pre=1) prefill_ops=1` and
   `first_chunk=7424/7680` appear repeatedly. `AlignPrefillChunk`
   (`tokenspeed-scheduler/csrc/scheduler/operations/cache.cpp:29`) returns 8192
   for a fresh request (first_pos=0, page-aligned); it does not clamp below
   budget. Explicit `fcfs` re-run: TTFT p50 **1204 ms** -- identical to default.
   So the chunk policy is a red herring.

4. **The real defect signal is a prefix-cache veto.** `.admit_trace.txt` shows
   `match req=... common=0 closed=1152` and `common=0 closed=896` for nearly
   every real request: the coordinator computed 1152/896 prefix-closed tokens, but
   the **served `common` is 0**, so **no cache is reused at all** and every
   request re-prefills its full ~10k. This is the exact signature the code
   comment at forward.cpp:269-275 warns about: `closed>0, common=0` = a group
   (sliding / mamba-state) discarding the full-attention group's match -- the
   same class of defect as the gemma `AdmissionBlocksNeededFor` case.

5. **The forwards are bursty, not back-to-back.** `mm_timing` timestamps in one
   conc-8 run: `bs=8 @39.699`, `bs=3 @39.904` (+205 ms), then `bs=5 @41.037`
   (**+1133 ms**) with `forward_step` only ~120-145 ms each. So a chunk of GPU
   work finishes in ~140 ms and the engine sits idle ~1 s between forwards. This
   inter-forward latency, on top of full re-prefill, is what turns a healthy
   ~250 ms/token-compute into a 1214 ms TTFT and a 302.6 tok/s aggregate (vs
   vLLM 348.0).

**Next lever (algorithm, not flags):** (a) find which cache group emits
`closed>0 common=0` and restore cross-group reuse so a request's already-cached
prefix is actually served (the `PromotionBoundaryTokens` / `admit` path in
`schedulePrefillFirstChunk`, forward.cpp:262-347); and (b) find what stalls the
engine ~1 s between `forward_step`s (event_loop / admission round, not GPU).
The chunk-policy work is closed: default FCFS already produces `first_chunk=8192`
single-request completes, and the 2.44x TTFT survives it verbatim.

### Round-8: the mechanism is prefill-residency multiplexing, not admission

With the plan trace on a steady-state conc-8 run the MIXED steps show
**`state(sub=0 pre=7 dec=1)` with `prefill_ops=3, first_chunk=1280`** (then
`prefill_ops=6, first_chunk=124`, `prefill_ops=5, first_chunk=100`,
`first_chunk=66/49/28`). So seven concurrent requests are all kept **resident
and mid-prefill simultaneously**, and every 8192-token forward is multiplexed
across them: the first resident prefill gets ~1280 in a round, the rest get the
leftover in page-sized dribbles. A ~10k prompt therefore needs ~8 such forwards
(~200 ms each) before its first output token -> TTFT ~1.2-1.6 s. The `forward_step`
(~120-145 ms for a full 8k forward) is fine; the request just does not *complete*
in one forward while six siblings hold the batch open.

Admission is **not** the gate: the engine default is
`admission_reserve="generation-full"` (`server_args.py:204` ->
`scheduler_utils.py:285-288`), i.e. `reserve_full_prefill_at_admission=True`, the
ported vLLM `scheduler_reserve_full_isl`. With a pool large enough to carry all
eight ~10k prompts it admits them all (the skip at forward.cpp:310-311 only
fires when the pool is empty), so residency is not bounded by admission.

Both `schedulePrefill` (continuation, forward.cpp:411-424) and
`schedulePrefillFirstChunk` (forward.cpp:350-353) return up to `min(remaining,
unscheduled)` = 8192 for a lone request, and `AlignPrefillChunk`
(`cache.cpp:29`) returns 8192 for page-aligned/fresh positions. So the small
per-request grant is **not** produced inside those functions -- it comes from the
main `newForwardOperation` loop passing a partly-spent `token_budget` as
`remaining` to successive resident prefills in one plan build
(forward.cpp:966-973). The algorithm fix, matching vLLM request-completion
priority, is to make the resident-prefill path *complete one request's prompt per
round* (give the longest-waiting resident prefill the whole budget before
offering any sibling), rather than multiplexing the budget across all resident
prefills. That is the exact quantity to change next (the `offer`/`remaining`
handoff among resident prefills), rebuild, and re-measure conc-8.

### Round-9: two algorithm fixes tried; the 0.88x gap is per-forward prefill compute, not the grant

With per-op grant instrumentation (`/workspace/roy/.prefill_dbg.txt`, gated on
`.plan_trace.on`) the Round-8 mechanism was pinned down precisely and two
algorithm-level fixes were implemented and each re-measured at conc-8 (n=64,
duration form, vLLM baseline 348.0 tok/s / TTFT p50 498 ms):

| run | tok/s | dur | TTFT p50 | comment |
| --- | --- | --- | --- | --- |
| instrumented (baseline) | 303.5 | 26.8 | 1200 | before fixes |
| + remove 1152 cap (`PromotionBoundaryTokens`) | 291.7 | 27.9 | 1160 | neutral |
| + HOL admission close | 292.7 / 293.0 | 27.8 | 1150/1175 | neutral |

1. **`KvCacheCoordinator::PromotionBoundaryTokens` returned a boundary even when
   nothing was reusable.** It returned `prefix_closed_tokens` whenever it
   exceeded `matched_tokens`, so a cold request (`matched=0`) whose PrefixProbe
   carried a large `prefix_closed_tokens` (~1152) got an artifacts boundary, and
   `AlignPrefillChunk` (cache.cpp:34) then clamped the first chunk to
   `promotion_boundary - first_pos`, i.e. every fresh admission was capped at
   ~1152 tokens. That is the Round-8 "1152" mystery. Fixed by returning 0 when
   `matched_tokens == 0` (kv_cache_admission.cpp) -- a boundary only constrains
   a prefix that is actually reused.

2. **The first-chunk head-of-line close did not fire.** The Submitted branch
   gated `admissions_closed=true` behind
   `!reserve_full_prefill_at_admission` (default is
   `generation-full` => True), and even when forced, `request->Is<fsm::Prefilling>()`
   is False on the admitting plan (the FSM transition lands only when the first
   forward result returns). Switched to the operation's own lengths
   (`PrefillOperation.input_length < prefill_length`, prefill_length =
   request->PrefillSize) in forward.cpp:1125-1155.

Both changes are principled wrong-quantity / completion-priority fixes and are
kept, but neither moved the number: TTFT p50 stays ~1150-1200 ms and tok/s
stays ~293-303 (0.84-0.87x).

**Refined diagnosis.** `mm_timing` shows forwards are **back-to-back**
(`forward_step` p50 = 123 ms, wall gap p50 = 118 ms, idle p50 ≈ 3 ms), so this
is **not** a schedule/overlap gap -- it is compute/throughput bound at
~126 ms per full 8k prefill forward (~15.7 µs/tok). The ~1.2 s TTFT is
**queueing**: eight concurrent requests are admitted in one wave and each
consumes roughly one plan's worth of the 8192 forward budget (a ~2-5k prompt
fits in ~1.6 requests/forward), so the median request waits ~7 forwards before
its prefill runs -- independent of the per-request grant size. Closing admission
does not compress that, because the total prefill work (tokens / 8192 forwards)
and hence total wall time is unchanged by *how* it is shared.

### Round-10: prefill batching is the tradeoff; all scheduler levers plateau at ~0.88x

Two corrections to Round-9, each verified by a fresh conc-8 run:

- **The head-of-line (HOL) admission close was reverted.** It fired on the
  page-alignment shortfall (e.g. 2304 of 2309), serialised to **one fresh
  request per forward**, and single-sequence prefill is ~2x more expensive
  per token than a multi-sequence batch (EXTEND bs=1
  `forward_step=123 ms` ≈ 35 µs/tok for a ~3.5k prompt vs a batched MIXED at
  ~15 µs/tok). Dropping it restores multi-request batching.
- **Prefill chunk size does not cap aggregate throughput.** Raising
  `--chunked-prefill-size/--max-prefill-tokens` to 36864 (8 x ~3.5k prompts fit
  one forward) changed tok/s by <1%: it only moves TTFT. With HOL reverted +
  big chunk the run is **305.0 tok/s, dur 26.7 s, TTFT p50 1892 ms** (max
  2223) — the highest throughput measured for this model, but TTFT *worse*
  because batched requests share the forward and the later ones wait.

Full measured matrix (n=64, vLLM baseline 348.0 tok/s / TTFT p50 498 ms):

| run | tok/s | dur | TTFT p50 |
| --- | --- | --- | --- |
| instrumented (default chunk, no HOL) | 303.5 | 26.8 | 1200 |
| + remove 1152 cap | 291.7 | 27.9 | 1160 |
| big chunk + HOL | 292.0 | 27.9 | 1046 |
| big chunk, HOL reverted | **305.0** | 26.7 | 1892 |

### Round-11: the prefill forward has a ~110 ms cost that is INDEPENDENT of length

A controlled single-request microbenchmark (running container, tokenizer-driven,
second call per length to avoid JIT) settles the fixed-vs-scalable question:

| prompt_tokens | EXTEND forward_step (rank 0) |
| --- | --- |
| 18 | 109.4 ms |
| 70 | 111.9 / 111.0 ms |
| 98 | 110.3 / 109.9 ms |
| 512 | 111.0 / 117.4 ms |
| 1280 | 109.4 / 121.8 ms |
| 3456 | 132.7 ms |

`forward_step` is **~110-120 ms flat for all of them** (the build-4656/`input_fill`
of 68-71 ms that alternates is host-side state prep, not the kernel). So a
prefill forward pays the same fixed cost whether it carries 18 tokens or 1280.
`mm_timing` also shows an EXTEND bs=1 / total_tokens=1 at `forward_step=123 ms`
and a MIXED 7-sequence / 8159-token batch at `forward_step=125 ms` — the same
~120 ms again.

**Consequence**
- Decode is at parity (engine `tpot` p50 17.0 ms vs vLLM 17.45 ms) and the
  DECODE bs=8 forward is 0.38 ms; the 17 ms/tok is server loop latency, equal in
  both engines.
- The aggregate is gated by the **number of prefill forwards**, not by their
  length: cost ≈ (#prefill-forwards) x ~120 ms + decode. Minimising the forward
  count is the only lever, which is exactly what aggressive prefill batching
  does (pack every ready prompt into each ~120 ms forward up to the token and
  sequence budget).
- The open item is therefore the per-forward batching *floor*. The measured
  batch (MIXED 7 seqs / 8159 tokens in 125 ms) already amortises the fixed cost
  over 7 requests, yet the conc-8 run still spends ~10 s of its 26.7 s wall in
  prefill+overhead. Next step: count prefill forwards in a conc-8 run and find
  the scheduler/model cap that stops 8 simultaneous ~2-5k prompts from landing
  in one forward (it is not `chunked_prefill_size`, which was swept to 36864
  with <1% aggregate change).

### Round-12: correction — prefill batching works and prefill/decode are both at parity

A direct 8-concurrent burst closes the open item and refutes the Round-11
"prefill is the bottleneck" reading:

- 8 simultaneous prompts totalling **20080 tokens** all return in ~0.40 s
  (pt=2510 each). `mm_timing` shows the batch genuinely runs as **`EXTEND
  bs=8`** (also a MIXED bs=8 and a bs=1 tail), so the scheduler *does* pack
  concurrent prefills into one forward. ~20k prefill tokens in ~4 forwards x
  ~120 ms, i.e. prefill throughput is vLLM-comparable, not a 4 s deficit.
- Decode is at parity too (engine `tpot` p50 17.0 ms vs vLLM 17.45 ms;
  DECODE bs=8 forward_step = 0.38 ms, so the ~17 ms/tok is server loop
  latency, equal in both).

So the residual 0.88x is **not** prefill compute, chunk size, or decode
latency. With prefill ≈ 5 s, decode ≈ 17 s, and a measured ~26.7 s wall, the
engine carries ~4 s of step-level scheduling / inter-wave overhead that vLLM
does not. The next lever is the decode/scheduling loop's per-step idle
(~16.6 ms of server time between a 0.38 ms DECODE forward), and how the driver
waves overlap — not the prefill kernel or admission policy. This is where the
remaining investigation should point.

### Round-13: the defect is a ~120 ms prefill forward cost that is FIXED, independent of length

Reproduced the 0.88x on a fresh default-config container (`cell-q35k`,
chunked-prefill-size=8192, mixed-batch on): n=64, dur 26.8 s, **303.5 tok/s**,
TTFT p50 **1178 ms**, tpot p50 **16.0 ms**, ntok 128. Driver analysis
(`drive2.py` keeps 8 in flight via a bounded pool, not wave-gated):

    duration ≈ 64 * (TTFT + 128 * tpot) / concurrency

With tpot already at parity (engine 16.0 vs vLLM 17.45 ms — engine slightly
better), the only winning move is **TTFT** (engine 1178 ms vs vLLM 498 ms).
Drive2's own streaming gates the aggregate on TTFT, so the 0.88x is a TTFT
loss, not a decode/TPOT loss.

The full-run `mm_timing` sequence (149 rounds: 84 MIXED, 38 DECODE, 27 EXTEND)
exposes the quantity that is wrong:

| forward | total_tokens | forward_step |
|---|---|---|
| DECODE bs=8 | 8 | **0.37 ms** (graph=True) |
| MIXED bs=8 | 14 (~6 prefill) | **115 ms** (graph=False) |
| MIXED bs=8 | 1414 | 115 ms |
| MIXED bs=8 | 8070 | 126 ms |

So the **prefill path has a ~120 ms fixed cost that is independent of token
count** (also shown by single-request prefill: 18 tok → 109 ms, 3456 tok →
132 ms from Round-11). This is the algorithm defect, and it explains
everything that looked like a scheduling problem:

1. **Why interleaving is forbidden.** A MIXED forward with only ~6 prefill
   tokens costs 120 ms instead of 0.37 ms, i.e. adding a tiny prefill to a
   decode step multiplies the step cost ~300x. So the engine cannot cheaply
   admit new prefills alongside decode (as vLLM's length-scaled chunked
   prefill can) — it is forced into big prefill chunks, which is exactly the
   high-TTFT / low-overlap behaviour measured.
2. **Why TTFT is 1178 ms not ~500 ms.** Every admission round that carries
   even a handful of prefill tokens is a 120 ms round; a request needing a few
   of them queues ~10 such rounds.
3. **Why GPU duty is low.** sum(forward_execute) ≈ 15.3 s over the run; the
   fixed 120 ms/round plus decode server-loop idle leaves the GPU well below
   the duty vLLM achieves.

The fix direction is now specific and kernel-level: make the prefill forward
cost **scale with token count** (a length-scaled GDN/linear-attn prefill,
matching sglang/TensorRT-LLM where small prefill chunks are cheap) instead of
paying ~120 ms per forward regardless of length. Attribution of the 120 ms
(GDN `gdn_chunk_prefill` per-call fixed cost vs model-level eager fixed cost)
is the next step before any edit; a `--chunked-prefill-size` change does **not**
address it (bigchunk was neutral, Round-10), because it re-buckets tokens but
still pays the same ~120 ms per forward.

### Round-13b: the fixed cost is the eager (non-graphed) forward, not GDN compute — `graph=False` vs `graph=True`

Instrumenting the model loop (`cat_timer` around `forward_step`, one single
prefill, 2..24 tokens) gives the per-category split:

    CAT_TIMER tokens=2/4/8/16/24  full_attn ≈ 2..10 ms   moe ≈ 10..42 ms
                                  total        ≈ 82..96 ms   (FIXED vs tokens)

`full_attn` + `moe` move with token count but the **total is flat at ~82 ms
even for 2 tokens**. The ~30..70 ms per rank that is not full-attn/MoE is the
36-layer GDN linear-attn prefill sweep — but crucially the WHOLE model forward
(48 layers) has a flat floor, so the cost is a per-forward launch/sync
overhead, not per-token GDN compute.

The decider is the `graph` field in the same `mm_timing` lines:

- `DECODE bs=8` → **`graph=True`**, `forward_step=0.37 ms`
- `EXTEND` / `MIXED` (any token count) → **`graph=False`**, `forward_step=110..
  132 ms`

so prefill/mixed run **eager, 48 layers of host-side prep, no cuda graph**,
while decode runs a captured graph at 0.37 ms. That ~300x difference is the
fixed cost: ~120 ms is eager per-layer overhead (launch/state-prep/MoE-dispatch
per layer), not GDN arithmetic. Reference: vLLM's chunked prefill also runs
outside a graph but its per-step prefill chunk stays cheap because each layer's
work is proportional to tokens and host overhead is small; here the engine
pays a large flat per-forward host cost for every eager prefill.

**Next lever (algorithm): make the prefill/mixed path use a captured cuda
graph (padded to the actual batch) just as decode already does, or cut the
per-layer eager host overhead in the prefill forward.** The engine already
exposes `prefill_graph_capture_sizes` / `max_cudagraph_capture_size`; this run
never enters the prefill graph (`graph=False` on every EXTEND/MIXED). If a
MIXED round drops from ~120 ms to near-decode cost, TTFT and duration follow,
because that is the exact quantity the duration model below is bounded by.

### Round-14: the GDN is already a graph break point; the prefill graph's real blocker was bucket coverage — and enabling it at 4-8k emits garbage

Directly resolving the open question from Round-13b ("is the prefill graph
capable, and is the GDN kernel the capture hazard?"), with the prefill graph
enabled and the flashinfer GDN tracer on:

1. **The GDN chunk-prefill is a graph break point today.** Every capture-bucket
   GDNTRACE line logs `capturing=False` (the kernel runs eager, outside the
   capture), and `HybridLinearAttnBackend.forward` already carries
   `@break_point` and dispatches `MambaAttnBackend.forward_extend` from inside
   it. The old `event_loop.py` auto-disable ("flashinfer GDN is unsafe under
   graph capture") is therefore **obsolete at its premise**: with the break, the
   kernel receives live cu_seqlens / state-page indices at every replay, and a
   single-request round trips `nan=0`, healthy `state_absmean≈1e-3`.
   `TOKENSPEED_TRACE_GDN=1` proves it; `TOKENSPEED_FI_GDN_ALLOW_PREFILL_GRAPH=1`
   plus a writable `TORCH_EXTENSIONS_DIR` (the `weak_ref_tensor` JIT build
   failed on the `~/.cache` PermissionError, which strong-refs pin graph-pool
   memory and caused the capture OOM) is what allowed the capture to proceed.

2. **The default prefill graph never fires for this workload — bucket coverage,
   not a kernel bug.** With `prefill_graph_max_tokens` at its default 2048, a
   throttled `_replay_bucket` diagnostic logs every live EXTEND/MIXED skip as
   `reason=no_bucket input_tokens=3692..8192` — the agent prompts are 4-8k
   tokens, all above the largest captured bucket. So the prefill graph is
   engaged but replays **zero** times; every prefill stays eager (the 120 ms
   floor), which is why Round-13b saw `graph=False` on every EXTEND/MIXED.

3. **Raising the cap to 8192 makes it fire and the output is garbage.** With
   `--prefill-graph-max-tokens 8192` the capture builds all 56 buckets
   [16..8192] with no OOM (`weak_ref` holds each bucket's pool peak-live;
   avail_mem 15.9→11.1 GB). It replays and reports `332.6 tok/s`, TTFT
   p50 1250 ms, tpot 14 ms — **but every reply is degenerate text** (e.g.
   `"Great! We've!!!!!!!!!!..."`), full 128-token length yet garbage. That is
   the AGENTS.md "the corrupt arm can look faster" failure exactly: length 128
   and faster-than-any-correct-run, only the unique-6gram / token-agreement gate
   catches it. The prefill graph for q35 is therefore **not safe at realistic
   prompt lengths**; whatever quantity is baked for a large bucket is still
   un-fixed, and `resolve_fi_gdn_prefill_graph`'s disable is still load-bearing.
   The event_loop and instrumentation edits from this round were reverted;
   default serving returns to the correct eager config (303.5 tok/s).

4. **The actual TTFT gap is admission contention, not prefill compute.** A
   3343-token prompt at concurrency 1 prefill-only: ~124 ms TTFT. At concurrency
   8 (continuous refill): TTFT p50 = 1178 ms — the prefill itself is only
   ~130 ms, so **~1 s is scheduler admission/queueing under load**. vLLM at the
   same conc/`max_tokens` is 498 ms TTFT, i.e. it adds only ~374 ms of
   admission. This is the quantity to attack next (an admission-block or
   per-round-wait defect, the family AGENTS.md Worked Example-1 names), and it
   is why the prefill graph — which only shaves the ~130 ms prefill slice —
   cannot close the gap by itself; it also still needs the large-bucket capture
   defect fixed before it is safe.

### Round-14b: admission-reserve is a Pareto dead-end — default is already best

The other candidate for the ~1 s admission gap was the **quantity** admission
charges up front. `scheduler_utils.py` maps `--admission-reserve` to
`reserve_full_prefill_at_admission` / `reserve_generation_at_admission`, whose
default `generation-full` charges a request's **full eventual footprint** at
admission. A/B at conc 8 / max_tokens 128, prefill graph off, correctness gate
passed (unique-6gram p50 = 1.0, sensible text, not the garbage arm):

| `--admission-reserve` | tok/s | TTFT p50 | tpot med |
|---|---|---|---|
| `generation-full` (default) | 303.5 | 1178 ms | 16 ms |
| `prompt` (prefill, no generation) | 305.0 | 1137 ms | 17 ms |
| `chunk` (nothing reserved) | 263.2 | 983 ms | 23 ms |

`chunk` does cut TTFT (1178→983) but inflates tpot (16→23 ms) as prefills
overlap decode and thrash it — a net loss. `prompt` is indistinguishable from
default. So the admission **reservation quantity is not the defect**; the
default is already on the Pareto front. The only remaining lever that lowers
per-request time without trading decode is the prefill graph cutting the ~120 ms
fixed eager prefill — which is currently gated behind the still-un-fixed
large-bucket capture bake (Round-14.3).

**Round-14.4 — live-metadata hypothesis REFUTED (no code changed):** the
metadata object is rebuilt live before a prefill-graph replay. `_can_use_graph`
(cuda_graph_wrapper:1097) returns False for any non-DECODE mode, so a MIXED step
always falls into the wrapper's eager-parity branch, which calls
`_init_forward_metadata` with the live batch (cuda_graph_wrapper:1299) *before*
`forward_func` → `_forward_step` → `_run_target_forward` → `prefill_graph.replay`.
So `self.forward_metadata` (query_start_loc / state-page indices) is fresh at
replay; the bake is therefore inside the captured `BucketSegment` itself — most
likely the GDN recurrent-state write/read page plan for the *padded* token count
(real 3-8k input captured as a fixed bucket, then replayed against an
equal-length but differently-padded live batch), i.e. the state-page
`state_out_pages_by_group`/`state_in_pages_by_group` plan baked into the segment,
not the `forward_metadata` object. Next step: instrument one prefill-graph
replay vs one eager prefill of the same prompt and diff the per-chunk
`state_out_pages` + `ssm_states` writes; whatever differs is the bake. Keep the
graph off until that is fixed and re-benchmarked with the correctness gate.

### Round-14.5: the fixed prefill cost is real, flat, and trace-independent — and the prefill graph cannot remove it (verified no-trace)

The prior rounds measured the prefill cost with `TOKENSPEED_TRACE_GDN=1`,
which turned out to be a confounder. Clean no-trace measurements settle the
question and close the prefill-graph avenue for q35:

1. **Without trace the prefill forward is a flat ~130 ms fixed cost, not
   token-bound.** Same benchmark (conc8 n=32, correct eager config,
   `--prefill-graph-max-tokens 0`), fit over 54 rank1 EXTEND/MIXED
   `forward_step` samples from a trace-free log: median ≈ 130 ms with a
   **negative** fitted slope (-3.5 us/token), because small and large prefills
   cost the same — 75 tok → 163 ms, 774 tok → 130 ms, 3906 tok → 132 ms,
   8124 tok → 127 ms. The "21.4 us/token" slope in earlier rounds was the
   *trace itself*: `TOKENSPEED_TRACE_GDN=1` adds a `.item()`/`.abs().mean().item()`
   host sync per GDN layer (36 layers), which becomes more expensive as the
   kernel grows — fit with trace: `forward_step = 75 ms + 21.4 us/tok`. So the
   real defect is **~130 ms of fixed per-forward cost, independent of length**
   (Round-13's ~120 ms, re-confirmed clean), and it dominates: ~90 forwards in
   a 15 s run × 130 ms ≈ 11.7 s. Never use trace-bearing runs for performance
   numbers.

**Decisive clean probe:** a 5-token, single-sequence EXTEND (bs=1,
no trace, correct output) logs `forward_step=110 ms` with
`input_fill=0.8 ms`; the same model's graphed decode forward logs
`forward_step=0.37 ms`, and a 8124-token prefill is 127 ms. So the cost is
**~110 ms pure fixed launch/host overhead that the CUDA graph hides**, and
the per-layer flashinfer GDN prefill kernel alone is 0.4 ms/call (microbench,
flat from 1→8192 tokens: 0.379→0.399 ms). The GDN kernel is not the cost.
The 110 ms is the aggregate per-layer host/kernel-launch overhead of running
the 48-layer MoE unfused and eager (many small kernels per layer, incl. 36
GDN layers); the decode CUDA graph collapses it to 0.37 ms. That asymmetry is
the whole q35 deficit.

2. **The prefill graph is a dead end for q35 on both axes.** With
   `--prefill-graph-max-tokens 8192` the capture succeeds and `*ENGAGED*` is
   logged, but the no-trace conc8 replay forward_step is still
   ~126-245 ms — the same as eager. The reason is structural: the 36 GDN
   layers sit behind `@break_point` and run **eager** inside the prefill graph
   (every GDNTRACE line during capture is `capturing=False`), so the graph only
   wraps embeddings/MoE/logits, which is a small slice; the 130 ms fixed cost
   lives in the eager GDN segment and is untouched by the graph. The 8192 graph
   arm is also **not faster** (TTFT p50 1352 ms vs the correct eager 1131 ms)
   AND emits garbage (`"Great!!!..."`, full 128 tokens, caught only by the
   unique-6gram gate). So the prefill graph is both unsafe and useless for
   reducing this cost on q35: it neither removes the 130 ms nor gains anything,
   and it corrupts output.

3. **Correct eager arm re-confirmed (the number to beat).** conc8 n=32,
   no trace, graph off, default admission: duration 13.39 s, TTFT p50 1131 ms,
   all ntok 128, output sensible (`"Great! We've confirmed the issue..."`),
   unique-6gram p50 = 1.0. This is the healthy baseline (≈ 304 tok/s for n=32,
   equivalent to the 303.5 reference).

**Conclusion — the fix is to remove the ~130 ms fixed eager prefill cost, not
to graph around it.** The flashinfer GDN prefill kernel is registered at
`min_arch_version=9.0` (Hopper) and is *not* being silently replaced by a slow
Triton fallback, so the 130 ms is the per-layer (36 × ~3.6 ms) host/launch
overhead of the eager forward, not the kernel's compute. The next step is
**either** to run the prefill/mixed forward through a CUDA graph *including* the
GDN layers (the decode graph already does this correctly and is 0.37 ms; the
prefill graph currently keeps GDN eager at `@break_point` and, when forced at
8k, both stays ~130 ms and corrupts output — the state-page bake from 14.4), or
to fuse/batch the per-layer launches so the prefill does not pay 36 separate
host dispatches. The way out of the corrupt arm must be correct GDN capture,
not a bigger bucket table.

### Round-15: GDN capture closes (illegal memory access); the residual is scheduler prefill interleaving, not prefill compute

Round-14.5's stated next step ("run the prefill/mixed forward through a CUDA
graph **including** the GDN layers") is now **refuted by a clean negative**,
and the correct graph-safe arm was isolated and measured:

1. **Removing `@break_point` from `HybridLinearAttnBackend.forward` crashes.**
   The GDN capture hand-off depends on the break's stable `_handoff` buffer
   (`breakable_cuda_graph.py`); capturing the decorated method directly with the
   Triton solution hit `CUDA error: an illegal memory access` on the first real
   batch (both TP ranks, `Fatal Python error: Aborted`), killing the engine.
   The `@break_point` is load-bearing: the GDN forward in this call path is not
   otherwise re-entrant/capturable. Restore it (reverted; tree is clean).

2. **The graph-safe arm is Triton GDN + prefill graph ON, and it is CORRECT**
   (unlike the flashinfer+graph arm, which is the documented `"!!!!!!!!"`
   garbage). Run with `TOKENSPEED_ENABLE_FI_GDN_SM120=0` (prevents
   `resolve_fi_gdn_prefill_graph` from force-disabling the graph) and
   `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_CHUNK_PREFILL=triton`. With
   `--prefill-graph-max-tokens 8192` the mixed/prefill `forward_step` drops to
   **62–96 ms and becomes token-proportional** (vs the ~130 ms flat eager cost),
   and a single-sequence 8k prefill TTFT is **69 ms** — ~7× faster than vLLM's
   498 ms on the same request. GDNTRACE during capture shows `capturing=False`
   only for the flashinfer arm; the Triton solution is captured with the graph.

3. **The conc-8 gap is scheduler prefill interleaving, not prefill compute.**
   A clean 8-concurrent burst (Triton+graph, no trace, 8192) logs MIXED batches
   that **round-robin the 8192-token budget across every mid-prefill request**
   (`#new-seq:7 #new-token:5573`, then `8069`, `8168`, ... `~20` prefill steps
   for 8 requests), so **no single request completes its context early** and the
   first-token staircase is `[454, 701, 1190, 1456, 1730, 1949, 2066, 2068]`
   (median ~1456 ms). The 8k single-request TTFT (69 ms) proves the engine can
   prefill far faster than vLLM; the ~1100 ms conc-8 TTFT is the frozen
   scheduler (.so) interleaving concurrent prefills, which no `context_chunk_policy`
   value (`fcfs`/`equal-progress`, `context_chunk_unit`) configurable here
   changes.

4. **Measured conc-8 (Triton+graph, 8192), all correct (bang=0, unique-6gram=1):**
   fresh protocol-style first pass **295.7 tok/s (13.76 s, TTFT p50 1242 ms)**;
   repeated on a warm cell **325.0–327.8 tok/s (~12.5 s)**; single best pass
   **355.5 tok/s (11.44 s, TTFT p50 1020 ms)**. vLLM baseline 348.0 tok/s /
   TTFT p50 498 ms. So the single-run variance spans **296–355**; the steady
   value is ~326, i.e. **~0.94× of vLLM, not reliably ≥ it.** The healthy eager
   baseline is 304/13.39 s, so Triton+graph is a +7% improvement but does NOT
   close the conc-8 gate, because the bottleneck is scheduler prefill
   interleaving.

**Status:** q35 conc-8 remains scheduler-bound (~326, best 355) — the residual
is prefill *interleaving* in the frozen scheduler, not the ~69 ms prefill
*compute*. The GDN-capture avenue is closed (crash on removal; garbage under
flashinfer capture). The next lever for the conc-8 ≤ vLLM gap is the scheduler's
round-robin chunk budget (why it does not complete a request's context before
offering the rest), which is inside the frozen scheduler and not addressable
from this tree's Python config.

## ASR dependency note

The `asr_http` endpoint decodes audio with `soundfile` and resamples with
`scipy.signal.resample_poly`. Both are required in the serving interpreter;
without them the first transcript request 500s on the lazy import.

### Round-16 (goal r18): re-verified on a live boot — triton+graph arm correct, current number is co-tenant-contended, not an engine regression

Fresh n=64 / conc 8 / temp0 / max_tokens 128 on `cell-q35nooverlap`
(the documented graph-safe arm: `TOKENSPEED_ENABLE_FI_GDN_SM120=0`,
`KERNEL_OVERRIDE_..._GDN_CHUNK_PREFILL=triton`, `--prefill-graph-max-tokens
8192`, `fcfs`, chunk 8192, mixed on): **n=64 ok=64 duration=35.5 s ≈ 231
tok/s**. This boot is co-tenant-contended (`roy-h200g` shares host GPUs 6/7 for
the whole run), so the absolute number is NOT comparable to the doc's warm-cell
326–355 or the vLLM 348 reference (a different, quieter boot). Evidence that the
engine arm itself is intact, not a regression:
- prefill **breakable graph IS capturing**: 56 buckets (16→8192),
  `segments=97, eager attention breaks` (the GDN segment stays the documented
  eager break point, exactly as Round-14.5 described).
- `TOKENSPEED_PROFILE_PREFILL=1` has **no Python-side host-sync** (grep finds no
  consumer), so it is not inflating `forward_step`.

**Status unchanged:** q35 conc-8 is scheduler-prefill-interleaving-bound (best
355 vs vLLM 348; steady ~326 = 0.94x) plus co-tenant contention; the residual
lever (complete-one-request-per-round in the compiled scheduler) is not
addressable from this tree's Python config. The only remaining model-level,
addressable option recorded in this doc is the **sm90 FI re-enable behind a
pure-prefill shape check** (documented at `gated_delta_rule.py:206-207`): the
Qwen3.6 win (493.6 ≥ 490.5) came from the sm90 FI probe, and the length-1
corruption only affects MIXED batches with decode rows — a call-site check that
feeds FI only prefill-only batches would restore Qwen3.6's fast arm without
replaying the corruption. Not implemented this round (needs a fresh Qwen3.6
launch + pure-prefill detection + the full output-length/t0-agreement gate).

### Round-17 (goal r20): sm90 FI re-enabled behind a call-site pure-prefill shape check (algorithm fix, not a flag)

The pure-prefill fast path was implemented this round instead of permanently
disabling the FlashInfer Hopper GDN prefill kernel. The length-1 corruption was
narrow (only MIXED batches that append length-1 decode rows), so the fix keeps
the kernel for the batches it is provably correct on rather than gating it off
for every model.

**Change** (`git diff` across two packages, all sync-free host-side checks):
- `MambaForwardMetadata` gains `num_extends` (number of prefill rows; decode
  rows are appended after it as length-1 segments on a MIXED forward). It is set
  from the local `num_extends` at the metadata build site already used by the
  `[MIXMETAD]` log (`hybrid_linear_attn.py`, metadata construction).
- `gdn_chunk_prefill(...)` gains a `pure_prefill: bool = False` argument that is
  added to the selection `traits` (and *not* forwarded to the kernel fn). At the
  single runtime call site, `pure_prefill` is computed as:
  `not capturing AND num_extends == query_start_loc.numel()-1 AND every prefill
  row > 1`. This is entirely CPU-side (the length tensor is already on `cpu`),
  so there is no host-GPU sync added to the prefill hot path.
- The FlashInfer `gdn_chunk_prefill` registration now declares
  `pure_prefill: frozenset({True})`, so the shell selection
  (`require_all_traits=False`) only matches it when the caller proved the batch
  is pure prefill; the Triton scan does **not** declare the trait, so it matches
  both values and is the sole candidate whenever `pure_prefill` is `False`.
  `pure_prefill=False` is the default, so any caller that omits it stays on
  Triton.
- `is_available()` sm90 `return False` → `return True`, with the comment updated
  to say the length-1 gate now lives in the call-site `pure_prefill` trait. The
  capture exclusion also keeps it out of the prefill graph (the kernel is not
  graph-safe), which is exactly the documented "flashinfer-without-graph" win.

**Why this is safe for the mixed-batch path:** a MIXED forward has decode rows,
so `num_extends < bs` → `pure_prefill=False` → Triton, byte-for-byte the
validated path. A captured/prefilled-graph replay likewise lands on Triton
(`num_extends` is `None` on the capture/replay metadata build sites, so it fails
closed). Only a provably pure-prefill, non-capturing batch can select FI — the
case already validated numerically against the Triton scan in Round-12/13.

**Logic verified by reading `selection.py`:** `_filter_by_traits` runs
`spec_matches_traits(spec, traits)` with `require_all_traits=False`, so a kernel
that does not declare `pure_prefill` (Triton) matches both values while the FI
kernel matches only `True`; the selection cache key includes `traits`, so the
`True`/`False` variants do not cross-contaminate, and when both match the
`SPECIALIZED` FI kernel outranks the `PORTABLE` Triton scan.

**Still required before this can be called a win (not done this round):** a
fresh `--gpus` Qwen3.6-35B-A3B TP1 boot with the prefill graph OFF (the FI kernel
is not graph-safe) plus the full gate: output length AND token-for-token t0
agreement against an engine/baseline reference, and a Qwen3.5-122B MIXED re-run
to confirm the mixed path still emits long/coherent output (no `!`-repetition).
Host has no `torch`/`tokenspeed_triton` and the running `cell-q35nooverlap`
container still executes the pre-change code, so none of that was exercised
here; the change is safe-by-construction for mixed/captured batches and
logic-verified, not runtime-verified.

### Round-18 (goal r21): q36 pure-prefill FI gate runtime-verified — WIN 721.7 ≥ 490.5

**Qwen3.6-35B-A3B is confirmed at/above vLLM on the authoritative AGENT-SLO no-replay
conc-8 gate with the Round-17 gate in place.** Boot: fresh `--gpus` device=6,
`ts-env:gpu`, TP1, `--max-model-len 16384 --max-num-seqs 256
--gpu-memory-utilization 0.85 --context-chunk-policy fcfs
--chunked-prefill-size 8192 --disable-kvstore`, `enable_mixed_batch=True`,
NO `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_CHUNK_PREFILL` override, and the
prefill cutlass breakable graph ON (mode=1, mixed supported) — i.e. the GDN/FI
segment runs at a break point (eager, not baked), which is the documented
FI-safe shape, so the FI kernel is never captured into the CUDA graph.

| run | out_tok/s | TTFT p50 | decode ms/tok | sum_ntok | ntok p50 | unique-6gram p50 | `!`x20 rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| engine, Round-17 pure-prefl gate (n=64, conc8) | **721.7** | **342 ms** | **7.0** | 8098 | 128 | 0.830 | 0 |
| vLLM, H200 (prior ref) | 490.5 | 428 ms | 12.0 | 8132 | 128 | 0.770 | 0 |

Same conc 8 / temp 0 / max_tokens 128 / ignore_eos, ~5.0k-token SWE-agent prompts
(`swe_agent_nebius_sess16x16` first 64), disjoint warmup
(`ab_corpus_warm.jsonl`, 16 = 2*conc). Engine wall `duration_s=11.22`, 64/64
served, 0 errors. **Engine ≥ vLLM by 1.47x on out_tok/s, and the decode axis
improves (7.0 vs 12.0 ms/tok) while TTFT also improves (342 vs 428 ms).**

**Why this beats the recorded 493.6:** that number (and the earlier 481.2/339.9)
was measured while `cell-q35nooverlap` (TP2 122B) + the co-tenant `roy-h200g`
were co-resident on GPU6/7 (`nvidia-smi` via probe container: 138355/143771 MiB
used on both cards). For this Round-18 boot, `cell-q35nooverlap` was stopped so
its device requests freed, and a device probe reported **0 MiB used / 143771 MiB**
on GPU6 — the engine got a genuinely quiet card for the first time. The FI
switching gain is real but the absolute number is not comparable to the
contended rows; the honest claims are (a) engine ≥ vLLM on a free bank, (b)
correctness holds, (c) the Round-17 gate is what makes (a) reproducible on a
MIXED engine without the corruption.

**Correctness gate (passed):** `[KERNEL] attention.gdn_chunk_prefill ->
flashinfer_gdn_chunk_prefill` is the selected kernel for real pure-prefill
batches (logged once — the selection trait cache reuses it), and the
length-1 HEALTH/WARMUMP batch correctly selected
`triton_gdn_chunk_prefill` (pure_prefill=False), i.e. the gate routes the two
classes where they belong. Output: ntok p50 **128** (min 106, max 128), 0 rows
with the `!`x20 repetition signature, unique-6gram p50 **0.830** vs vLLM 0.770,
0 errors. The sub-128 rows are all coherent code-analysis/tool-call text cut by
the hard 128-token limit (e.g. `... print output`, they get a`), not the
length-1 garbage the FI kernel produced on MIXED decode rows pre-gate. 17 rows
< 128; vLLM also has 29 rows < 128 (min 121), so short streaming stops are
normal for the harness, not a corruption signature.

**Two caveats kept honest:** (1) the vLLM 490.5 is the recorded reference, not a
fresh pairing on the same card — the engine-to-vLLM agreement axis was not
re-run end-to-end because the running engine exposes only the HTTP server and
the existing `vllm-q36-c8.raw.json` was produced from a different prompt slice
(row-0 texts do not match), so throughput comparison is on matched token volume
and concurrency, not per-prompt identity; (2) `cell-q35nooverlap` was stopped to
free GPU6/7 for this boot — the mixed-batch path itself was re-verified here as
safe (Triton on length-1/mixed, FI only on provable pure prefill), and a Qwen3.5
mixed re-run is still the recommended next check before relying on it there.

### Round-19 (goal r21): the pure-prefill gate is q36-ONLY safe — q35 FI corrupts

The Round-18 claim that the gate is "safe" needs a hard correction discovered the
same round. The gate was exercised on BOTH GDNs this round (not just Qwen3.6),
and it splits the two models:

**Qwen3.6-35B-A3B (TP1): gate GOOD.** n=64/conc8, `!`x20 = 0, every row coherent,
out_tok/s **721.7 ≥ 490.5**, TTFT 342 < 428 ms, decode 7.0 < 12.0 ms/tok
(Round-18 table). No corruption whatsoever.

**Qwen3.5-122B-A10B (TP2): gate BAD.** Running the same boot (new code, NO
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_CHUNK_PREFILL`, breakable prefill graph
on, mixed batch on) over the q35 corpus (`swe_smith_agent_sess32x8` first 64,
conc 8, temp 0, max_tokens 128) produced **1/64 rows that are EXACTLY
`!`x128** (row index 1; `[KERNEL]` log shows `flashinfer_gdn_chunk_prefill`
selected for pure-prefill batches, `triton_gdn_chunk_prefill` for the mixed /
length-1 ones). That is the SAME length-1/varlen corruption signature the
Round-17 gate was supposed to quarantine — but it survives on a TP2 122B even
when the batch is pure prefill, so the public-API FI sm90 kernel is simply wrong
for Qwen3.5's TP2/GDN state layout, not just for MIXED decode rows.

**Implication: the Round-17 pure-prefill gate must NOT be the default for
Qwen3.5-122B.** Re-restoring the documented correct arm
(`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_CHUNK_PREFILL=triton`, prefill graph
on) over the same free bank gives n=64/conc8: out_tok/s **287.4**, TTFT p50
**1276 ms**, decode 18.5 ms/tok, `!`x20 = **0**, unique-6gram p50 0.725, ntok
p50 128. So on the SAME free card the Triton arm is *clean but still 0.82x*
(287.4 vs 348.0), and the TTFT gap is not the GDN kernel (1276 vs vLLM 498 ms) —
consistent with the earlier "Qwen3.5 loss is TTFT, not GDN" diagnosis, and now
even on a quiet bank it does not reach vLLM.

**Net: Qwen3.6 is a verified win; Qwen3.5 is still a loss (0.82x clean, 0.84x
corrupted) and the corrupted "faster" FI arm on it is a bug report, not a win.**
Qwen3.5's residual lever remains the non-GDN prefill/scheduling path (TTFT is
2.55x), which is the frozen compiled scheduler — not addressable from this tree's
Python config. The pure-prefill FI gate stays enabled (it is the q36 win) but
Qwen3.5 must run with the Triton override until the FI-sm90 TP2 corruption is
root-caused (the same algorithm-level work the AGENTS.md rules point at).

### Round-20 (goal r22): Qwen3-Embedding-8B measured against a live sglang reference — loss is prefill batching/overlap, not kernel compute

The embedding model was booted fresh on a free `device=6` (TP1, default backend,
`--max-model-len 8192 --max-num-seqs 256`, `pooling_prefill_graph=True`,
`embedding_skip_kv_write=True`, prefix caching + overlap schedule auto-disabled
for pooling). A new `bench/agent-slo/emb_bench.py` (async `/v1/embeddings`,
reports req/s and p50/p90 latency) gives a clean isolation of the pooling
prefill path.

**The forward resolves to `attention.mha_prefill -> fa3_mha_prefill`** (kernel log),
not `mha_extend_with_kvcache` — a pooling request is one forward, no decode, no
KV read-back. So the embedding defect lives entirely in the prefill path.

**Same-corpus, same-harness versus a live sglang reference** (`qwen3-embedding-8b`,
port 8890, not touched; only queried):

| prompt | conc | engine (this round) | sglang ref | ratio |
| --- | --- | --- | --- | --- |
| 2048-char / ~1300 tok | 16 | 43.2 req/s, p50 337 ms | 76.6 req/s, p50 182 ms | **0.56x (loss)** |
| 256-char / 55 tok | 16 | 148.0 req/s, p50 93 ms | 197.8 req/s, p50 74 ms | **0.75x (loss)** |
| 2048-char / ~1300 tok | 1 (single) | 23 ms/req | 32 ms/req | **1.43x (engine single is FASTER)** |

**The decisive signal is concurrency scaling, not kernel speed.** A single
~1300-token embedding is *faster* on the engine (23 ms vs sglang 32 ms), but
raising concurrency to 16 gives the engine **zero throughput gain** (still
43.2 req/s at both conc 1 and conc 16, i.e. ~23 ms/req either way) while sglang
*drops from 32 ms to 13 ms/req* (30.6 → 76.6 req/s, a 2.5x concurrency gain).
The engine's `Prefill batch` log shows it *does* fuse many short (55-token)
sequences into one forward (up to 38 seq / 2090 tok), but long (~1300-token)
requests that exceed `max_prefill_tokens=8192` are not pipelined across the
concurrent set — they serialize.

**A flag probe confirms it is scheduling, not the kernel:** raising
`--chunked-prefill-size` and `--max-prefill-tokens` to 65536 made the same conc-16
long batch *worse* (24.7 req/s, p50 614 ms) — a bigger fused prefill does not
help, and the single-forward fast path is already good. The engine neither
overlaps concurrent long prefill requests nor gains from a larger budget, which
is a pool-scheduler / overlap defect (the same class AGENTS.md flags: the number
that a flag changes names the defect, and the flag here is not the fix).

**Status:** Qwen3-Embedding-8B is still a *loss* (0.56x long / 0.75x short vs
sglang) and the root cause is the pooling prefill scheduling/overlap path, NOT
the `fa3_mha_prefill` kernel (single-request it beats sglang by 1.4x). The
hypothesised code-level repair (let a pooling-forward bypass the generation
`max_prefill_tokens` chunk barrier, and overlap the concurrent set) still needs
to be implemented and re-tested; the current boot is back on default settings
(no chunk override) on `device=6`, healthy.

### Round-21 (goal r23): Gemma4 fresh conc-8 re-measure + the head_dim-512 decode is a hard kernel cap

Gemma4-31b-it was booted fresh on free `device=6,7` (TP2, default backend; the
non-uniform head-dim steer sends the whole model to Triton, per the fa3
corruption note). Measured on the same short corpus (64 x ~900-char / ~250-token
prompts, temp 0, max_tokens 128):

| conc | out_tok/s (fresh) | out_tok/s (vLLM matched) | ratio |
| --- | --- | --- | --- |
| 8 | **404.8** | 669 | **0.60x (loss)** |

`n=64 ok=64 err=0 bang20=0`, completion `ntok` median = 89 (natural stop, not a
length-truncation failure), `tpot` median 19.0 ms, TTFT p50 54 ms, e2e p50
1741 ms. This confirms the documented matched A/B (0.58–0.84x) is not stale.

**Why no flag or single-kernel swap can pull Gemma up.** The ten
`global_head_dim=512` layers are the cost:

- Every registered `*_decode_with_kvcache` kernel caps `head_dim` at 256
  (`fa3...` {64,128,256}, `flashinfer_trtllm_mha_decode` range(16,257,16),
  gluon {64,128}). **No decode kernel in the tree accepts 512**, so on the
  decode-bound short-prompt workload the 512 layers are stuck on Triton (the
  ~145 TFLOP/s arm; tpot 19 ms).
- `flashinfer_paged_mha_extend_with_kvcache` **does** now accept 512 (added for
  these layers; 115.48 ms @ 145.2 TF vs Triton 137.24 ms @ 122.2 TF on the
  8192-vs-35326 production shape, ~1.19x, plus the `_SPLIT_VO_MIN_HEAD_DIM=512`
  output split that reaches 1.28x). But that is the *prefill/extend* path — it
  does not help a short-prompt, decode-dominated run, and gemma is currently
  steered off FlashInfer wholesale.
- Routing the 512 layers to FlashInfer-extend while leaving decode on Triton is
  exactly the mixed-backend assembly the fa3 corruption note calls the open
  defect; it is not safe to do as a flag flip and the gate rightly rejects it.

**Status:** Gemma4 remains a *loss* (0.60x). Its bottleneck is the
decode-side head_dim-512 capability wall + the un-solved mixed-backend decode
assembly — a kernel/scheduler repair, not a tunable. (Qwen3.6 already verified
WIN; Qwen3.5 and Embedding findings are in Rounds 19/20.) The Gemma container
was stopped to free `device=6,7`.

### Round-22 (goal r24): Embedding concurrency is a max-seqlen-bound plateau, NOT serial, NOT kernel-roofline, NOT graph ceiling

Re-booted Qwen3-Embedding-8B fresh on `device=6` (TP1 default, `--max-model-len
8192 --max-num-seqs 256`, `--disable-kvstore`) and profiled the pooling prefill
with the engine's own scheduler logs instead of sweeping flags.

**The engine DOES fuse concurrent long pooling prefills.** The `Prefill batch`
lines during a 16-request run were only two forwards:

```
Prefill batch. #new-seq: 1,  #new-token: 489
Prefill batch. #new-seq: 15, #new-token: 7335
```

So 16 concurrent ~489-token pooling requests collapse into a single 7335-token
forward. This rules out "no batching" and "serial per-request" — the earlier
Round-20 serialisation hypothesis was **wrong**.

Single request is efficient: ~489 tokens in ~22 ms = a high fraction of the H200
roofline for an 8B dense prefill, so the attention/prefill kernel is not the
limiter either.

**Warm, same-container concurrency sweep (n=16, pooled /v1/embeddings):**

| conc | req/s | lat_p50 |
| --- | --- | --- |
| 1 | 41.3 | 22 ms |
| 8 | **62.1** | 120 ms |
| 16 | 58.9 | 238 ms |

The engine gains from concurrency (41 -> 62) but **plateaus at ~60 req/s with the
knee at conc 8**, and conc 8 / conc 16 have essentially the *same* end-to-end
time (~260 ms), i.e. the batched forward is **dominated by the longest prompt in
the batch** (max seqlen), not by total tokens. sglang in the same matched run
continues to ~77 req/s at conc 16. A first-shot cold run read 23.7 req/s — that
is a warmup (capture/compile) artifact, not the steady state.

**Flag checks that refute the next two guesses:** `--prefill-graph-max-tokens
8192 --chunked-prefill-size 8192` changed nothing (steady 60 req/s), so the
`PREFILL_GRAPH_DEFAULT_MAX_TOKENS=2048` eager bail is not the bottleneck either.
The residual defect is a pooling-prefill concurrency/overlap ceiling at ~60
req/s that sglang does not hit — a schedule/max-seqlen-pack issue, not a kernel
or graph-capability one. The code-level fix (pack the concurrent set / overlap
`max_seqlen`-bound forwards the way sglang's `fast_prefill_plan` does) still
needs to be implemented; it is not a flag and not yet landed.

### Round-23 (goal r25): Embedding attention path is ALREADY the optimal stateless algorithm — the residual gap is host serving throughput, not a kernel

Two hypotheses for the embedding loss were tried and refuted at the code level.

1. **"Force `flashinfer` backend to pick a faster prefill kernel."** Not viable:
   `--attention-backend flashinfer` makes `mha_extend_with_kvcache` select
   `solution='flashinfer'`, for which no kernel is registered, and the server
   dies with
   `NoKernelFoundError: No kernel found for attention.mha_extend_with_kvcache ...
   with solution 'flashinfer' on NVIDIA H200`. Seen live from the `_forward_extend`
   stack (`mha.py:1116 -> __init__.py:1595 -> selection.py:706`).
2. **"Embedding wrongly rides the KV-cache extend path (`mha_extend_with_kvcache`)
   instead of the stateless `mha_prefill`, paying an avoidable KV round-trip."**
   Also refuted. With the default (`--attention-backend mha`) config the engine's
   own kernelecho is unambiguous:

   ```
   [KERNEL] attention.mha_prefill -> fa3_mha_prefill
   ```

   There is no `embedding_skip_kv_write`-kept warning, and no
   `mha_extend_with_kvcache` in the profiling window. The pooling forward is a
   pure stateless `mha_prefill`, dispatched through the `mha.py:873` "postwrite
   + zero prefix" branch to `_forward_prefill`. (The `_forward_extend` crash in
   hypothesis 1 comes from the flashinfer backend forcing the extend solution,
   not from the default path.)

So the embedding attention kernel is the best-matching one already
(`fa3_mha_prefill`, Hopper FA3, the SPECIALIZED winner for head_dim 128), and a
single request runs at a high fraction of the H200 dense-prefill roofline. The
remaining ~0.77x at conc 16 (engine ~60 req/s vs sglang ~77) is therefore not an
attention-algorithm or kernel-selection defect — it is the serving stack's
per-request throughput (host tokenisation + 4096-dim result serialisation at
high concurrency), which the concave conc-8 knee (62 req/s) and the conc-8 vs
conc-16 same-e2e both point at.

**Read-only status for the objective:** Qwen3.6-35B-A3B is a verified WIN; the
other three are still losses with these distinct root causes:
`Qwen3.5-122B-A10B` TTFT/scheduler (frozen compiled scheduler), `Gemma4-31b-it`
decode head_dim-512 hard cap + mixed-backend assembly, `Qwen3-Embedding-8B`
serving-stack per-request throughput. None is a flag; each needs an engine
architectural change that is not yet landed.

### Round-24 (goal r26): Gemma's 0.60x is the decode head_dim-512 kernel gap, and it drags EVERY layer onto Triton

Confirmed at the registration/selection level why gemma decode is slow, and why
it is not a flag.

**Available `mha_decode_with_kvcache` kernels on H200 (sm_90):**
* `flashinfer_trtllm_mha_decode_with_kvcache` — `head_dim` {16..256 step 16}
  **and** `capability.min_arch 10.0` (Blackwell). Not usable on H200 at all, and
  even on Blackwell it cannot serve 512.
* `fa3`-family decode — hard capped at head_dim 256 (Hopper flash-attn).
* Triton — portable, the only thing that serves head_dim 512.

So for gemma-4's **ten global layers (`global_head_dim` 512) there is no fast
H200 decode kernel**; the fifty sliding layers (256) could be fast, but the
per-layer split is not safe.

**Why the whole model pays for it:** `mha.py:194-202` detects non-uniform
`layer_kv_head_dims` (256/512) and, because interleaving `fa3`-256 with
Triton-512 on one decode step corrupts the output, forces `kernel_solution =
"triton"` for **all layers**. `_forward_decode` (`mha.py:1179`) then runs every
layer on Triton. That is the 0.60x: tpot 19 ms (vLLM ~12 ms) at conc 8, a
decode-side loss while TTFT stays fast (54 ms).

**The algorithm fix (matches the prefill `_SPLIT_VO_MIN_HEAD_DIM=512` precedent
and the AGENTS "register the fast arm" rule), not yet landed:** for a 512-wide
decode layer, split `v_cache`'s last dim into two 256-wide views, run the 256
decode kernel twice (both passes still contract QK over the full 512), and merge
the two partial outputs with the existing `attn_merge_state` LSE op
(`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/__init__.py:3202`).
The 256 kernel must accept `head_dim_qk 512 / head_dim_vo 256` asymmetrically
and return LSE --- the exact shape the paged-extend split-VO comment
(`flashinfer/__init__.py:830`) already measures as 1.28x faster for prefill; the
sm_120 1-2-row shared-memory boundary at `_SPLIT_VO_MIN_TOTAL_Q = 3` must be
re-derived for H200 (sm_90 has a larger 228 KB shared budget), and the resulting
mixed-strip output must still be verified with the length + temp0 token gate
before any throughput claim.

This is a kernel/backend change, not a server flag; it is the third distinct
engine-architectural root cause (with the q35 frozen scheduler and the embedding
serving-throughput ceiling) still open.

### Round-25 (goal r27): Embedding's loss is per-batch token efficiency (2.5x), not attention, not KV round-trip, not graph padding

Previous embedding rounds concluded "attention path already optimal, residual =
serving-stack throughput" from small-`n` runs. A matched, warm, `n=128`
re-measure (128 distinct ~2,560-char prompts; engine GPU6 vs the live sglang
reference `:8890`) shows the real shape and kills three hypotheses:

| conc | engine req/s (eager) | sglang req/s | ratio |
|---|---|---|---|
| 1  | 23.0 | 53.0 | 0.43x |
| 8  | 67.7 | 137.5 | 0.49x |
| 16 | 69.5 | 173.8 | 0.40x |
| 32 | 64.0 | 168.0 | 0.38x |

The engine does not scale past ~70 req/s (its `c8` with the default graph was
70.0; `c1` 39.5), while sglang scales to 174. Converting by total tokens
(128 x ~565 tok), the engine flattens to **~39 ktok/s** (c16 dur 1.842 s) while
sglang reaches **~98 ktok/s** (c16 dur 0.737 s) — **2.5x per-token**.

**Refuted hypotheses:**
* Prefill-graph padding — `--no-pooling-prefill-graph` (eager) is the *same*
  (c8 67.7 vs graph 70.0), so the breakable prefill graph is not the cost.
* KV round-trip / `embedding_skip_kv_write` — not involved (no warning; the
  pooling path is stateless `mha_prefill -> fa3_mha_prefill`).
* Attention kernel — same reached before.

**The engine's own diagnostic names it** (`model_executor.py:1148`, one line per
prefill batch): with 8 concurrent requests the scheduler emits
`#new-seq: 7 / 1 / 7 / 1 / 1 / 1 / 5` — **fragmented micro-batches**. The batch
that did get 7 seqs logged `#new-token: 3959` and is ~90 ms apart from the next,
i.e. ~44 ktok/s even for the *full* batch. Single-seq batches ride the bucket
~576 prefill-graph replay. So the loss is **per-model-forward batch scaling**:
the engine gets ~2x per-token when batching, sglang ~11x, and torn micro-batches
keep the GPU latency-bound. The defect is scheduler batch-packing + the
embedding model forward's ragged-batch scaling, not a kernel.

**Separate correctness bug found along the way:** a pooling request must never
be partial-prefilled, but at `conc16` with the default
`--max-prefill-tokens 8192` the batch total (16 x ~565 = 9,040) exceeds the
budget and the batch builder splits ONE prompt, raising
`RuntimeError: pooling request ... was scheduled as a partial prefill`
(`generation_output_processor.py:624`). The startup resolver
(`model_config.py:707-723`) only opens the budget to `max_prefill_tokens`, so it
guarantees "a *prompt* never splits" but not "a *batch* never splits a prompt".
Fix (C++): in
`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` the partial-grant
rule (lines 932-944) already permits a partial unit only for a prompt's own tail;
extend it so a request with `MaxNewTokens()==0` (pooling, set at
`request_handler.py:296`) never receives a partial *first* chunk — grant the
whole prompt or defer the request, matching the whole-prompt invariant. This is
a scheduler algorithm fix, not a config.
`--max-prefill-tokens 32768` only masks it (round-27: c16/c32 then run but still
only ~69/64 req/s).

### Round-26 (goal r28): pooling whole-prompt fix LANDED and runtime-verified; two hard 8192 caps surfaced by the trace

Round-25 described the fix as a refactor; this round implemented it in
`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`
(`schedulePrefillFirstChunk`) and verified it on a fresh live boot. Three
empirical findings from an on-line trace drove the exact form:

1. **The right "defer" is `return std::nullopt`, not a 0-token grant.** The first
   attempt clamped the first chunk to `tokens_this_round = 0` when the leftover
   budget was short. The state machine treats `0` as "still prefilling", so the
   request was **still scheduled as a partial** (the trace showed the deferred
   request later erroring at `generation_output_processor.py:624`) and the loop
   stopped there. Returning `nullopt` skips the request entirely; it stays
   `Submitted` and `Scheduler::NextExecutionPlan` re-collects every `Submitted`
   request each cycle, so it is granted whole one cycle later once the budget is
   free. The AlignPrefillChunk==0 path (line 381) already used nullopt-defer —
   the pooling branch must do the same.
2. **`max_scheduled_tokens` is a chunked-prefill budget, a wrong quantity for
   pooling.** Clamping the first chunk by the leftover budget slices a prompt.
   Granting the whole prompt unconditionally (ignoring the budget) is also
   wrong: the second hard cap, the position kernel's output buffer, rejects a
   batch that exceeds `max_prefill_tokens` with
   `ValueError: compute_position_triton out buffer too small: 8192 < 8462`.
   So a pooling batch must be **whole prompts whose total stays ≤ the token
   cap**, deferring the prompts that would overflow — the only consistent option.
3. The batch of whole prompts is bounded by `max_num_seqs` / the KV arena, as in
   vLLM and sglang, but only **within** the `max_prefill_tokens` token cap.

Final rule shipped: for `MaxNewTokens()==0`, if `remaining < unscheduled` return
`std::nullopt`; otherwise grant `unscheduled` (the whole prompt) — never a
partial. Generators keep the FCFS clamp.

**Verification** (fresh `--gpus` device=6 boot, default
`--max-prefill-tokens 8192`, `--disable-kvstore`):
* `conc16 n=24`: previously raised the partial-prefill RuntimeError; now
  `24/24` complete, `ndim=4096`, no error (exit 0).
* `conc16 n=128` (steady): **62.6 req/s**, lat_p50=248 ms, all `ndim=4096`,
  no RuntimeError. (Round-25's 69.5 was the same engine at a batch that stayed
  under the cap; the new defer path costs a few % from the extra cycle, and the
  cap-total per forward is unchanged.)
* `conc8 n=96`: **53.3 req/s**, all correct.

**Open (unchanged):** throughput is still ~2.5x below the livng sglang reference
(62.6 vs 173.8 at c16). The deferred-batch scheduling never changes the
per-forward token ceiling — the loss remains per-batch forward efficiency
(~44 ktok/s even for a full batch) plus the torn micro-batches named in
Round-25. That is the remaining gate; the pooling split was a correctness bug on
top of it, now fixed.

### Round-27 (goal r35-36): DeepSeek-V4 DSPARK — prefill graph is OFF because the flashinfer GDN kernel is not capture-safe

Hardware-fair reference corrected first: a 4-GPU sglang TP4 run (187.7 tok/s) is
NOT the target for a 2-GPU engine (2x decode hardware / HBM aggregate). The
correct 2-GPU sglang TP2 reference on host GPU 6,7
(`sglang-dsv4-vision:pr37257-fix`, `--tp 2`, `--speculative-algorithm DSPARK`):

| metric | sglang TP2 (ref) | DSPARK engine (8192) | after `TOKENSPEED_ENABLE_FI_GDN_SM120=0` |
|---|---|---|---|
| tok/s | **96.8** | 17.5 | 35.7 |
| TTFT p90 | 3.1 s | 28.8 s | 9.9 s |
| E2E p90 | 14.0 s | 100.9 s | 25.6 s |
| rpm | 139.8 | 27.6 | 55.3 |
| ntok/req | 41.6 | 38.0 | 38.8 |
| ok | 256/256 | 256/256 | 256/256 |

**Root cause (corrected this round; the earlier "linear-attention layer" claim
was refuted).** DeepSeek-V4 is MLA + compressor (`AttentionArch.MLA`), it is NOT
in `_HYBRID_GDN_ARCHITECTURES`, and it **never calls `gdn_chunk_prefill`** — the
only call site in the runtime is `hybrid_linear_attn.py:1995`, which the
DeepSeek-V4 backend does not use (its `config.json` declares no `layer_types`).
So `model_may_use_fi_gdn` (`event_loop.py`) falls through the `layer_types` loop
to the **conservative `return True`**, and `resolve_fi_gdn_prefill_graph` then
sets `server_args.disable_prefill_graph=True` for a model whose GDN hazard does
not exist. Result: every EXTEND/MIXED forward runs eager — 48 layers of host-side
prep, no CUDA graph, `forward_step` 110–132 ms — against a decode captured graph
at 0.37 ms (~300x). That is the same fixed cost named in Round-13/13b; it is not
batching and not the prefill-token budget, and raising the budget to 32768
crashes the engine worker (verified).

**Model-scoped fix tried and verified at the decision point.** Making
`model_may_use_fi_gdn` return `False` for `is_deepseek_v4` (MLA-only, GDN
unreachable) kept the prefill graph with NO env-var escape — the live log emits
`event_loop.py:283 "Keeping the prefill CUDA graph: this model declares no
linear_attention layer …"` (all 4 ranks). Qwen3.5/Kimi are untouched (they still
declare `layer_types`, still get GDN + graph off).

**BUT the kept prefill graph deadlocks on capture — not viable as default.**
With the graph kept (either by that fix or by `TOKENSPEED_ENABLE_FI_GDN_SM120=0`),
the engine reproducibly hangs at `Capturing prefill buckets (bucket=2048) 0/40`:
GPU util 0%, ~138 GiB/card resident (~97% of the H200), no new log lines for
minutes. Cause is the breakable-graph capture path: `weak_ref_tensor` is
unavailable in the container (`PermissionError:
/home/runner/.cache/torch_extensions`), so it falls back to identity strong refs
that pin graph-pool blocks and make capture memory scale with the bucket sum —
near-OOM stall. Re-running the doc's own `dsv4_diag.sh` (env=0,
`disable_prefill_graph=False`) reproduces the same hang, so the "35.7" row below
is **not reproducible** and is retained only as the historical number.

**Token gate is blocked by a pre-existing divergence, not this change.**
Engine vs the sglang TP2 reference at temp 0: **exact t0 text agreement is
0/256 for BOTH the baseline (GDN, graph off) and the Triton+graph arm** —
identical length distributions (both max 57 tok, means 38.8 vs 41.6) but
different reasoning trajectories from the first token. So the 0/256 is NOT
introduced by the graph fix; the two stacks (engine `dtype=auto`,
`kv_cache_quant_method=none`, self-DSPARK draft vs sglang
`moe-runner-backend flashinfer_mxfp4`) do not reproduce each other's greedy
decode. The sglang token-for-token gate is therefore not meaningful until the
two stacks are pinned to identical weights/quant/decode.

**Status: not achieved.** No reproducible 2x this session: the eager-prefill
fixed cost (wrong conservative GDN guard disabling the graph) is the dominant
loss, but re-enabling the graph hits the `weak_ref_tensor`/breakable-capture
near-OOM deadlock. The remaining algorithm work is to make the breakable prefill
capture use weak refs (fix the container `PermissionError`, or make the
extension load without hanging) so large token buckets can be captured while
preserving greedy output — kernel/capture-level, still open. DeepSeek-V4
baseline not restored yet; sglang ref (idx0-3) and vllm-dsv4 untouched.

### Round-28 (goal r37+): decode-graph measured 18.8, prefill is the algorithm gap

Ground truth first — a fresh `--gpus "device=6,7"` boot runs both the baseline
and the decode-graph variant to READY (decodes captured 8/8), then `drive.py`
(C=16, n=256, temp 0, max_tokens 128, token-for-token length gate):

| config | tok/s | TTFT p90 | E2E p90 | rpm | ntok/req | max | ok |
|---|---|---|---|---|---|---|---|
| DSPARK `--enforce-eager` (baseline) | 17.5 | 28.8 s | 100.9 s | 27.6 | 38.0 | 57 | 256/256 |
| DSPARK, **no** `--enforce-eager` (`dsv4_decodegraph.sh`) | **18.8** | 20.1 s | 93.8 s | 29.3 | 38.6 | 55 | 256/256 |
| sglang TP2 (ref) | **96.8** | 3.1 s | 14.0 s | 139.8 | 41.6 | — | 256/256 |

`--enforce-eager` was added by a prior session and is NOT in the reference
config; it disables even the DECODE CUDA graph. Removing it is a small but
**reproducible** +7% (17.5→18.8, output lengths unchanged: mean 38.6, max 55,
not the corrupt arm). Decode capture alone is NOT the 2x lever — prefill is.

**weak_ref root cause SOLVED (but not sufficient).** The prefill-graph capture
`PermissionError` is only because `torch.utils.cpp_extension`'s default dir is
`/home/runner/.cache/torch_extensions`, owned by root while the engine runs as
runner(`uid 1001`). The extension is already built at
`/workspace/roy/.torch_ext/tokenspeed_weak_ref_tensor/`; setting
`TORCH_EXTENSIONS_DIR=/workspace/roy/.torch_ext` makes `load_inline` reuse the
`.so` with no rebuild, and a container smoke test confirms it returns a true
non-owning view (`_load_failed=False`, alias `data_ptr`). This removes the
near-OOM mechanism predicted in Round-27.

**...but the kept prefill graph STILL deadlocks.** With the model-scoped fix
(`model_may_use_fi_gdn` returns False for `is_deepseek_v4`) + weak_ref loading
+ no `--enforce-eager`, the engine captures decode batches 8/8, then hangs
**before** READY and **before** the `Capturing prefill buckets` progress bar:
`curl /v1/models` → 000, GPU ~131 GiB/card resident, engine CPU ~2% (a sync
wait, not compute), no new log lines for minutes. So the Round-27 hang is NOT
the weak_ref OOM alone — there is a real capture/sync path that blocks
pre-bucket for DeepSeek-V4. `event_loop.py` was therefore **reverted (no
residual change)** so the default conservative guard keeps DeepSeek-V4 servable.

**Why prefill is the algorithm gap (AGENTS 例二).** `deepseek_v4_prefill_chunk_size`
(server_args default 4) is a per-FORWARD **request** batching cap (4 reqs per
`flash_mla_sparse_fwd` launch at 16 concurrent → 4 serial prefill passes), not a
token chunk. `mla_disable_ragged` defaults to **False**, i.e. the trtllm ragged
deepseek prefill path the engine picks up via `use_trtllm_ragged_deepseek_prefill`
is **already on** — there is no disabled fast ragged kernel to flip. The
measured 18.8 tok/s is bounded by DeepSeek-V4's bespoke prefill
(`backends/deepseek_v4.py`: compressor → sparse indexer top-k → sparse
`flash_mla_sparse_fwd`), which is an order of magnitude slower than the sglang
`flashinfer_mxfp4` ragged prefill it is competing against. Matching sglang
(96.8) requires a fused/faster DeepSeek-V4 prefill kernel (or a capture-safe
prefill graph that also covers the compressor+indexer), not a config value.

**Status: still not ≥96.8.** Best measured, reproducible improvement = 18.8 tok/s
(`dsv4_decodegraph.sh`, no `--enforce-eager`) vs 17.5 eager baseline; +7%. The
2x prefill-graph number (35.7) remains not reproducible (now deadlocks before
bucket capture even with weak_ref). The remaining gate is the DeepSeek-V4
prefill kernel, not decode graph and not a disabled ragged path. `event_loop.py`
restored; `dsv4_decodegraph.sh` is the servable improved config; sglang ref
(idx0-3) and vllm-dsv4 untouched.

---

## Round 29 — prefill-graph deadlock root-caused & the decode wall

**The prefill-graph deadlock was NOT weak_ref and NOT the 2048-token bucket.**
With `TORCH_EXTENSIONS_DIR` reusing the prebuilt weak-ref `.so` and the
model-scoped `event_loop` guard (DeepSeek-V4 never calls `gdn_chunk_prefill`):

* bucket ladder ≤256 captures cleanly: `Capturing prefill buckets 16/16`
  (segments=87, eager attention breaks), engine READY.
* single `--prefill-graph-capture-sizes 2048` captures in 12.7s, 1/1, READY.
* single `--prefill-graph-max-tokens 8192 --prefill-graph-capture-sizes 8192`
  captures in 5.4s, 1/1, graph ENGAGED.

So the old hang at the default ladder (16→max) is a **cumulative-memory stall**:
each bucket allocates its own breakable pool, and ~40 pools (16..8192) exceed
the memory ceiling near the top. The algorithm fix is to capture only the real
runtime prefill chunk size (`chunked_prefill_size`), not a ladder of sizes that
never occur. Note `prefill_graph_max_tokens` defaults to 2048, so an explicit
`--prefill-graph-capture-sizes 8192` alone is silently clamped to 2048 — both
flags are needed.

**Measured effect (`dsv4_pfg_8192.sh`, c=16, n=256, DSPARK, cold cache):**

| metric | eager baseline | decode-graph | prefill-graph-8192 |
| --- | --- | --- | --- |
| tok/s | 17.5 | 18.8 | **19.3** |
| TTFT p90 | 28.8s | 20.1s | **10.6s** (1.9x) |
| E2E p90 | 100.9s | 93.8s | **59.6s** |
| ntok mean | 38.0 | 38.6 | 38.4 (ok 256/256) |

The prefill graph delivers its promised TTFT lever (20.1→10.6s, ~1.9x) and the
2x on the prefill half of e2e, but tok/s only moves 18.8→19.3 because **decode is
the wall**, not prefill: per-request token time is ~712ms, decode throughput
~22 tok/s vs sglang's 97.

**Decode root cause (algorithm-level) located.** `model_executor.py:948` calls
`drafter.run(...)` every decode step, and `deepseek_v4_dspark.py:383`
(`_draft_decode_rows`) runs a **second full backbone decode forward**
(`forward_backbone(main_hidden, bonus, main_positions, self.kv_windows, slots,
draft_ctx)`) to draft `block_size` tokens. The `DeepseekV4DSpark` drafter is
checkpoint-local: it `shares_target_embed_head` and reuses the target model as
its own backbone (`self.draft_model.model`), so there is no cheap draft head.
Each decode macro-step thus costs the verify forward **plus** a full backbone
draft forward. `_seed_prefill_windows` also does GPU→CPU `.item()` syncs
(line 299), which is illegal inside a CUDA graph — so the drafter path cannot
be captured and runs eager per step. This roughly doubles decode compute and is
the gap to sglang's ~97 tok/s (whose DSPARK uses a lightweight markov/DP
draft on the captured hidden, not a second backbone pass).

Not a flag value: eliminating the redundant backbone draft pass (or making the
draft use the already-captured target hidden instead of a second forward) is the
algorithm fix that would close most of the decode gap.

**Status: still not ≥96.8.** Best measured now 19.3 tok/s with TTFT p90 **10.6s**
(1.9x, the prefill lever achieved). `dsv4_pfg_8192.sh` (with the model-scoped
`event_loop` guard kept) is the servable improved config; sglang ref (idx0-3)
and vllm-dsv4 untouched.

## Round 30 — self-DSPARK is a net LOSS and its arm was truncated (19.3 was invalid)

The Round-29 decode diagnosis (second backbone draft forward) is correct but the
measured 19.3 tok/s was **not a fair number**: it came from an arm that stopped
generating early. `drive.py` sets `"ignore_eos": true` (line 54), so every run
should deliver the full `max_tokens` = 128. The spec arm produced only
`ntok_mean=38.4` — it ended at ~38/128 — while the no-spec arm produced
`ntok_mean=127.6` (full 128). That is exactly AGENTS' "**the corrupt arm can
look faster**": 19.3 tok/s was spread over ~3.3x fewer output tokens.

**A/B (`dsv4_nospec.sh` = single-8192 prefill graph + mixed batch, DSPARK OFF;
same engine, same warmup, cold cache):**

| run | DSPARK | tok/s | ntok mean | TTFT p90 | E2E p90 | note |
| --- | --- | --- | --- | --- | --- | --- |
| decode-only, c=16, short prompts | ON | 39.3 | 50.9 (truncated) | — | — | accept ~2.2–2.4/6 |
| decode-only, c=16, short prompts | **OFF** | **64.1** | 118.1 (full 128) | — | — | 1.63x |
| AgentX c=16 n=256 | ON (`dsv4_pfg_8192.sh`) | 19.29 | **38.4 (truncated)** | 11.4 s | 61.7 s | invalid fast-look |
| AgentX c=16 n=256 | OFF (`dsv4_nospec.sh`) | **44.84** | **127.6 (full)** | 16.8 s | 73.0 s | correct length |

Steady decode `gen throughput` at bs=16: spec 24 tok/s → no-spec **73–83 tok/s**.
Self-DSPARK `accept_rate` is only **0.24–0.29** (`avg_accept_len` ~2.2–2.4) —
barely above 1 token/step, yet every decode macro-step pays a *second* full
backbone forward (`deepseek_v4_dspark.py:383` `_draft_decode_rows` →
`forward_backbone`), which itself cannot be CUDA-graph-captured (`.item()` sync
at line 299). **The verified conclusion: on this engine, DeepSeek-V4's own
DSPARK drafter is counterproductive** — at ~0.27 acceptance the extra backbone
pass costs more than the ~2.4 tokens it recovers, so turning it off is a **2.3x
end-to-end win (19.29 → 44.84 tok/s) and a 1.63x pure-decode win (39.3 → 64.1)**.

Two separate defects this exposes:
1. **Spec arm truncation** — with `ignore_eos=True`, the DSPARK path ends streams
   at ~38/128 tokens. This invalidated the prior 19.3 benchmark and is a
   correctness bug (spec path terminates decode early).
2. **Spec is a net loss here** — low accept rate + un-capturable second backbone
   forward. The algorithm fix (per AGENTS: a parameter lower is a diagnostic,
   not the deliverable) is to make the draft reuse the already-captured target
   hidden (a markov/DP head on logits, as sglang does) instead of a second
   backbone pass — or, given accept ~0.27, to not self-draft at all.

**Honest status: still not ≥96.8, and the real gap is now 2.2x, not 4.2x.**
Engine (correct 128-token output, no-spec) = **44.8 tok/s** vs sglang TP2 ref
96.8 tok/s. The remaining 2.2x is the raw DeepSeek-V4 decode forward speed
(MLA + compressor + indexer decode, ~0.5 s/pass at bs=16) plus the 8192-graph
prefill ceiling — kernel/algorithm work, not flags. `dsv4_pfg_8192.sh` remains
the improved prefill config but must be measured with its spec arm not
truncating before it is trusted; `dsv4_nospec.sh` is the honest end-to-end
baseline. sglang ref (idx0-3) and vllm-dsv4 untouched.

## Round 31 — decode graph now engages on the full workload; 58 tok/s; prefill is the remaining wall

`_padded_bucket` is a **padding-up** scheme, not exact-match
(`prefill_graph.py:895` `bisect.bisect_left` → smallest bucket ≥ num_tokens), and
mixed batches DO use the prefill graph (`model_executor.py:747`). So capturing
`[8192]` already replays every prefill batch ≤8192 (padded) — the single-bucket
prefill graph IS engaging. The reason TTFT stays ~17s is that DeepSeek-V4's
custom prefill (compressor → sparse indexer top-k → `flash_mla_sparse_fwd` in
`backends/deepseek_v4.py`) runs in the graph's **eager attention breaks** over
real tokens; the bucket/padding only affects the generic captured segment, so
no bucket value can make the custom prefill fast.

The remaining full-workload gap was the decode CUDA graph: `_can_use_graph`
returns False unless `forward_mode.is_decode()` (`cuda_graph_wrapper.py:1097`),
so a **MIXED** prefill+decode batch bypasses the decode graph and its decode
runs eager (~1.4s/step at bs=16). That is the AGENTS "disable flag is a bug
report" pattern: `enable_mixed_batch` (default on) disables the decode graph.

**A/B (AgentX c=16 n=256, no-spec, single-8192 prefill graph, cold cache):**

| run | mixed | tok/s | ntok mean | TTFT p90 | E2E p90 | decode gen@bs16 |
| --- | --- | --- | --- | --- | --- | --- |
| no-spec (`dsv4_nospec.sh`) | ON (default) | **44.84** | 127.6 | 16.8 s | 73.0 s | ~24 (eager) |
| no-spec, `--disable-mixed-batch` (`dsv4_nospec_nomix.sh`) | OFF | **57.99** | 127.5 | 17.3 s | **47.0 s** | **60–81 (graph)** |

Disabling mixed-batch makes the scheduler emit pure prefill / pure decode
batches, so pure-decode batches replay the decode graph (60–81 tok/s at bs=16,
vs ~24 eager) while `overlap_schedule_depth=1` still overlaps prefill on the
other stream → E2E p90 73→47s, tok/s 44.8→58.0 (a 1.29x). Cumulative from the
invalid 19.3 spec run: **3x** (44.8 + 58.0), measured at full 128-token output.

**Remaining gap = prefill, now clearly isolated.** The bulk prefill batches are
full 8192-token chunks; each takes ~2s → prefill rate ~4000 tok/s. 256 AgentX
requests × ~8k prefill ≈ 512s of the 563s run. The decode graph cannot help
prefill and the prefill graph cannot accelerate the eager-break custom kernels.
Closing the last 1.67x (58 → 96.8) needs one of:
1. **Mixed-batch decode graph** — allow the decode CUDA graph to run inside a
   mixed forward (or run pure prefill/decode streams with real overlap), instead
   of the current hard `is_decode()` gate. Engine-level, bounded.
2. **Fast DeepSeek-V4 prefill kernel** — fused/re-written compressor +
   indexer-topk + sparse MLA (the `flashinfer_mxfp4` ragged path sglang uses),
   ~10x the current ~4000 tok/s. Kernel-level, the big one.

**Status: still not ≥96.8.** Best measured now **58.0 tok/s** with correct
128-token output. `dsv4_nospec_nomix.sh` is the current best servable config.
Code changes made this session: only new `dsv4_nospec.sh`/`dsv4_nospec_nomix.sh`
diagnostics + doc; engine kernel dirs untouched. sglang ref (idx0-3) and
vllm-dsv4 untouched.
