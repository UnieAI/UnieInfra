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

## ASR dependency note

The `asr_http` endpoint decodes audio with `soundfile` and resamples with
`scipy.signal.resample_poly`. Both are required in the serving interpreter;
without them the first transcript request 500s on the lazy import.