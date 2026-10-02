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

`tokenspeed` suites (`4096` / `8192`-token prompts from `ts64.jsonl`,
`temperature=0`, `max_tokens=128` unless noted), run from inside the container
(the engine HTTP port is container-bound). "knee" = concurrency where
throughput flattens.

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

## ASR dependency note

The `asr_http` endpoint decodes audio with `soundfile` and resamples with
`scipy.signal.resample_poly`. Both are required in the serving interpreter;
without them the first transcript request 500s on the lazy import.
