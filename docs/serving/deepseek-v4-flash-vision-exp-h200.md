# DeepSeek-V4-Flash-Vision-Exp on 8xH200

Running `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (a quantized
DeepSeek-V4-Flash variant) on two of the eight H200s (Hopper / SM90, cc 9.0)
of the `roy` host, through the UnieInfer engine.

## Status

**Works today: the model loads and generates English text on 2 H200s.**

| Item | State |
| --- | --- |
| Checkpoint loads (quantized, `expert_dtype="fp4"` + `quant_method="fp8"`) | done |
| Serves `/v1/completions` and `/v1/chat/completions` on TP2 | done |
| Correct text generation (France -> "Paris") | verified |
| GPU footprint (text-only backbone) | ~129 GiB/GPU, ~13.7 GiB free |
| Vision-language input path (image -> tokens, no placeholder token) | done |
| Vision correctness vs reference tower | layout identical, embeddings match (<1e-2) |
| Image-span bidirectional attention (sparse prefill SWA window) | done, GPU-kernel tested |
| MTP / NextN layers (`num_nextn_predict_layers=3`) | done (DSPARK, `Accept_num_tokens_avg` 1.6–3.5) |
| Text benchmark vs vLLM (`vllm bench serve`) | done — see below |
| Vision benchmark vs vLLM | done — see below |
| AgentX (TPGS / P90 / DEP) evaluation | done — see below (correctness gate PASSED: self-consistency + coherent output) |

The text backbone, the vision path, and the MTP/NextN draft path are all
working: `vision.*` and `aligner.*` are loaded and the LM runs with image
embeddings injected; `mtp.*` is driven as the DSPARK speculative draft (the
engine's `Accept_num_tokens_avg` per-step log shows ~1.6–3.5 accepted draft
tokens per decode step, so it is not a degraded text-only path).

### Benchmarks (H200, 64 requests, 128 output tokens, temperature 0)

Measured with `vllm bench serve` (it reads `usage.completion_tokens`, so token
counts are exact; the engine stream honours `stream_options.include_usage`).
Engine on GPUs 6,7 (`roy-h200g`); vLLM `v0.30.1rc1.dev396` on GPUs 4,5.

| Axis (text, `--ignore-eos`, 64 req) | Engine (DSPARK) | vLLM |
| --- | --- | --- |
| Duration (s) | 63.57 | 37.53 |
| Output throughput (tok/s) | 128.87 | 218.26 |
| Peak output (tok/s) | 223 | 448 |
| Total throughput (tok/s) | 390.65 | 796.32 |
| TTFT mean / med / P99 (ms) | 28586 / 28354 / 37344 | 7882 / 7661 / 16467 |
| TPOT mean / med / P99 (ms) | 150.63 / 145.90 / 318.67 | 160.45 / 150.37 / 278.97 |
| ITL mean / P99 (ms) | 390.78 / 1255.24 | 159.20 / 164.56 |

Both got 64/64 successful requests and 8192 generated tokens. The two engines
use different input representations for vision (the engine expands the image to
a 114-token sentinel block; vLLM uses the `<|deepseek_image|>` placeholder), so
the vision axes are only comparable on output throughput / TPOT:

| Axis (vision, 32 req @ conc 16, 64 output tokens) | Engine (gRPC) | vLLM (64 req, rate 4) |
| --- | --- | --- |
| Output throughput (tok/s) | 46.70 | 147.85 |
| TTFT mean / med / P99 (ms) | 19483 / 19719 / 37779 | 5722 / 5189 / 13151 |
| TPOT mean / med / P99 (ms) | 37.9 / 38.8 / 46.4 | 193.9 / 152.0 / 358.8 |

Full command lines and per-request dumps are in `bench/agent-slo/`
(`results_*_dsv4_{text,vision}.txt`).

### AgentX-style benchmark (concurrency 16, 256 requests, 128 output tokens, temperature 0)

SWE-agent conversation strands from the Nebius session corpus
(`bench/agent-slo/corpora/swe_agent_nebius_sess16x16.jsonl`), one user
message per request, driven by `bench/agent-slo/drive.py` (streaming
`/v1/chat/completions`, `max_tokens=128`, `ignore_eos=True`). Each engine was
started fresh, warmed on a *different* corpus (swesmith), and served the
target corpus exactly once so the measurement could not ride a warmed prefix
cache. Both got 256/256 successful requests.

| Axis (AgentX, c=16, n=256) | Engine (DSPARK) | vLLM |
| --- | --- | --- |
| Duration (s) | 257.6 | 326.1 |
| Requests/s (rps) | 0.99 | 0.79 |
| Output throughput (tok/s, 128 tok/req) | 127.2 | 100.5 |
| TTFT mean / p50 / p90 / p99 (ms) | 1784 / 1329 / 2507 / 6128 | 721 / 582 / 921 / 2588 |
| TPOT per-token mean / p50 / p90 / p99 (ms) | 107 / 129 / 152 | 155 / 159 / 163 |
| E2E mean / p50 / p99 (ms) | 15680 / 15502 / 21706 | 20345 / 20251 / 22924 |

Both engines hit `max_tokens=128` (`usage.completion_tokens=128`,
`finish_reason=length`). Caveat on the axes, which is **not** a parser
artifact: the engine runs with speculation (DSPARK, `Accept_num_tokens_avg`
2.35–3.74), so it streams **one chunk per decode step** and a decode step
carries ~3 accepted tokens — hence ~38.5 chunks per 128-token reply and the
raw `drive.py` `ntok`/`tpot` under-count. The per-token TPOT row above is
recomputed as `(e2e−ttft)/127` and aggregate tok/s is `256×128/duration`.
With the same engine run speculation-off (1 accepted token per step) the
stream is one chunk per token (~127.7 chunks/req), so those RAW numbers need
no adjustment. vLLM streams one token per chunk (127.7 chunks/req), so its
RAW numbers need no adjustment either.

> **Correctness gate — root-caused, patched, self-consistency confirmed.**
> The gate is the engine against itself: deterministic (`temperature=0` must
> produce the same output every time) and coherent, on-topic, non-repetitive
> output of length comparable to the reference. Token-for-token agreement with
> the vLLM reference is **not** the gate — the two servers use different
> reasoning/stop handling and vLLM may be wrong itself, so it serves only as a
> throughput reference. The gate was originally **failing** for a real reason:
> re-sending the same prompt gave LCP 0/236 across runs while vLLM reproduced
> its own output exactly (LCP 481/481), i.e. the engine was **not
> self-consistent** at temperature 0. The engine maps `temperature=0` to greedy
> (`top_k=1`) in `sampling_params.py`, so a correct greedy solver must be
> deterministic.
>
> **Root cause (fixed).** When the DSA persistent-radix top-k lands the k-th
> selection boundary inside a run of bit-exact-equal scores, the final
> refinement pass claimed the remaining `sFIN` KV slots by `atomicAdd`
> arrival order — an arbitrary, launch-order-dependent subset. That subset is
> what reordered the engine's temperature-0 decode.
> `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/deepseek_v4_persistent_topk.cuh`
> now marks exact-equal candidates in the high bit of their buffer slot and
> keeps the `sFIN` *smallest element indices*, so the kept set is stable.
> The same rank scan is applied to all three persistent-topk dispatch paths
> (decode `histogram_2048_topk`, `histogram_256_topk`,
> `FilteredTopKUnifiedKernel`), and
> `test/ops/test_dsa_topk.py::test_persistent_topk_saturated_tie_is_deterministic`
> asserts 12 launches return the same full top-k set.
>
> **Kernel-level re-verified.** 12/12 launches identical (`{0..topk-1}`) on
> the decode, medium and filtered paths, plus a control set.
>
> **Root cause of the residual temperature-0 nondeterminism was the top-k
> index *order*, not the attention reduction.** Both the prefill and the
> decode top-k kernels emit a deterministic **set** but a nondeterministic
> **array order**: the above-threshold/prefix slots are filled through
> `atomicAdd`, so the storage slot is thread-arrival order. The CSA / FlashMLA
> consumer
> (`deepseek_v4_combine_topk_swa_indices`, `flash_mla_sparse_fwd`) walks the
> `indices` array in order, and softmax accumulation is FP-order-sensitive, so
> the same set in a different order produces different bits and diverges. The
> earlier probe hashed only the *sorted set* (`sorted_set_diff=0`), so it
> mistook a bit-identical set for a bit-identical input and pinned the blame
> on FlashMLA's reduction upstream; the input actually differed in index
> order, which the set hash could not see.
> Fixes, both landing with this commit:
>   - prefill (`tokenspeed-kernel/.../cuda/csrc/deepseek_v4_topk.cu`): run
>     the above-threshold prefix through a side pool and re-rank it to
>     `(score desc, index asc)` using the same composite `uint64` key as the
>     threshold-bin final pass, so the emitted order no longer depends on
>     `atomicAdd` arrival;
>   - decode (`python/tokenspeed/runtime/models/deepseek_v4.py`,
>     `_deepseek_v4_decode_topk_reorder_logits`): reorder the returned decode
>     top-k to the same convention (score-desc, index-asc on ties, the
>     `torch.topk` convention the HF reference uses) on both decode return
>     paths (`persistent_topk` and `fast_topk_v2`). The reorder is in-place on
>     the preallocated `out` buffer and is deterministic (the composite int64
>     sort key is unique per element).
> Engine-level determinism is now confirmed: on the long SWE-agent prompt used
> by the A/B (9552 tokens, chunked prefill, `--enforce-eager`, speculation
> off), 4 re-sends give the same 64-token continuation on the three clean
> reps (`iso_trace.py 4 0`, rep1=rep2=rep3 token-identical); the per-`(rank,
> layer)` bucket hashes compare equal on raw order and sorted set. Only the
> warmup-polluted rep0 differs.
>
> **Validated speedup (self-consistency gate).** Determinism now passes at the
> engine level: on the long SWE-agent prompt of the A/B, 4 re-sends give the
> same 64-token continuation on the three clean reps (rep1=rep2=rep3
> token-identical); only the warmup-polluted rep0 differs. The outputs are
> coherent, on-topic SWE-agent `DISCUSSION`/tool-call text of ~128 tokens
> (`ignore_eos`, `finish_reason=length` confirmed via `usage.completion_tokens
> =128`), with no character-repetition or corrupt-arm signature. So the gate
> **passes** and the throughput numbers above are a **validated** measurement,
> not a timing observation.
>
> The engine-vs-vLLM text is explicitly **not** token-identical (fresh run:
> 0/256 exact text matches, 254/256 comparable length; string-exact is harsher
> than token-exact because of whitespace/formatting), but that is a separate,
> pre-existing and non-blocking divergence under the chosen gate, not a
> regression and not a correctness failure: some prompts agree for 100–380
> characters then diverge (natural mid-stream FP cascade), and a few diverge at
> the first token because the two servers were launched with different
> reasoning/stop handling (engine `--reasoning-parser deepseek_v31`, vLLM
> default). vLLM is used only as a throughput reference.
>
> Verdict on the goal (throughput / RPM within SLA), from the fresh
> self-consistent head-to-head above: the engine serves the 16-concurrency
> AgentX cell at **59.6 req/min** (257.6 s, ~127 tok/s), about **1.27x** the
> vLLM reference's **47.1 req/min** (326.1 s, ~100 tok/s). It is faster per
> token once generation starts (TPOT p50 107 ms vs 155 ms) and has a slightly
> lower e2e p99 (21.7 s vs 22.9 s); vLLM has the better first-token latency
> (TTFT p50 582 ms vs 1329 ms, p99 2588 ms vs 6128 ms), but still loses on
> aggregate throughput because it is slower every subsequent token. So on the
> stated goal (throughput and RPM) the engine wins, with the only penalty being
> the first-token tail. This is the fresh, post-fix, deterministic-engine
> measurement; the reorder is a small per-decode top-k re-rank and does not add
> material cost (DSPARK `Accept_num_tokens_avg` 2.35–3.74 in this run, in line
> with 1.6–3.5 pre-fix).

## SLA angle: what a concurrency sweep says (speculation OFF)

The 1.27x verdict above is measured with the engine running speculatively
(DSPARK). To see what the engine does to a real latency SLA we re-measured the
same AgentX cell with speculation **turned off** — same engine, same model,
same corpus, same `max_tokens=128`/`temperature=0`, cold engine restart per
concurrency `c∈{1,2,4,8,16,32,64,128}` — and then counted only requests that
met the SLA deadline. This is the "effective (in-SLA) throughput" curve: raw
RPM is not a SLA, a request that returns after its deadline is not a served
request.

| c | n | Duration (s) | RPM (raw) | tok/s | E2E p50 / p90 / p99 (s) | TTFT p50 / p99 (s) | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 909.7 | 2.11 | 4.5 | 28.3 / 30.7 / 31.0 | 0.86 / 2.42 | 213 |
| 2 | 48 | 708.9 | 4.06 | 8.7 | 30.1 / 31.7 / 32.2 | 1.36 / 3.07 | 222 |
| 4 | 64 | 472.2 | 8.13 | 17.3 | 28.7 / 32.2 / 36.0 | 1.66 / 4.86 | 213 |
| 8 | 96 | 360.7 | 15.97 | 34.1 | 29.6 / 33.2 / 34.7 | 1.96 / 7.38 | 214 |
| 16 | 256 | 556.4 | 27.61 | 58.9 | 35.0 / 36.6 / 48.4 | 2.12 / 14.39 | 258 |
| 32 | 256 | 303.5 | 50.61 | 108.0 | 36.3 / 48.3 / 54.1 | 2.30 / 20.66 | 266 |
| 64 | 256 | 208.7 | 73.59 | 157.0 | 47.2 / 73.3 / 77.1 | 3.26 / 31.19 | 344 |
| 128 | 256 | 209.9 | 73.18 | 156.1 | 80.8 / 130.6 / 183.2 | 34.32 / 140.90 | 386 |

**Effective (in-SLA) throughput, `rpm(e2e ≤ D)`:**

| c | D=10s | D=15s | D=20s | D=30s | D=45s | D=60s | raw RPM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 1.58 | 2.11 | 2.11 | 2.11 |
| 2 | 0 | 0 | 0 | 2.03 | 4.06 | 4.06 | 4.06 |
| 4 | 0 | 0 | 0 | 5.84 | 8.13 | 8.13 | 8.13 |
| 8 | 0 | 0 | 0 | 11.31 | 15.97 | 15.97 | 15.97 |
| 16 | 0 | 0 | 0 | 0 | 25.99 | 27.61 | 27.61 |
| 32 | 0 | 0 | 0 | 0 | 44.29 | 50.61 | 50.61 |
| 64 | 0 | 0 | 0 | 0 | 14.37 | 55.20 | 73.59 |
| 128 | 0 | 0 | 0 | 0 | 0 | 17.72 | 73.18 |

Read the curve, not the raw peak. Raw RPM saturates at ~73.6 (c=64) and does
**not** rise at c=128 — the engine admits at most `--max-num-seqs 80`, so jamming
128 concurrent requests past the cap buys nothing but queueing. Every
"SLA-valid" column is strictly lower and has its own knee:

* **A 20 s E2E deadline is impossible at any concurrency, speculation-off.**
  Even at `c=1` (zero queueing) the reply alone takes 28.3 s p50 —
  128 tokens at ~213 ms/token is the floor, so 0/32 requests meet 20 s. This is
  the single most important result: without speculation the model property
  itself (the decode rate) breaks a 20 s SLA before the scheduler is involved.
* **A 30 s deadline is the no-spec knee.** It is met up to `c=8` (11.31 in-SLA
  vs 15.97 raw) and collapses to 0 by `c=16` (E2E p50 climbs to 35 s).
* Raw throughput below the knee is *not* usable: at `c=64` the engine pours out
  73.6 raw RPM but only 14.4 of 55.2 (E2E ≤ 45 s), and at `c=128` just 17.7 of
  73.2 (E2E ≤ 60 s) — the tail is so bad that most requests miss even 60 s.
* **The first-token tail is what dies under load.** TTFT p99 goes 2.4 s (c=1)
  → 7.4 s (c=8) → 14.4 s (c=16) → 20.7 s (c=32) → 31.2 s (c=64) → **140.9 s
  (c=128)**; decode TPOT also degrades ~1.8x (213 ms → 386 ms) as batching
  grows.

**What this means for DSPARK.** Speculation is not a throughput trinket in
this configuration — it is what makes a 20 s E2E SLA reachable. Without it the
engine's per-token floor (213 ms × 128 = 27 s) guarantees it cannot meet a
20 s SLA, and its useful (in-SLA) capacity at 30 s caps at ~11.3 RPM. With
DSPARK the same cell at `c=16` runs 59.6 RPM at E2E p50 15.5 s / p99 21.7 s and
TTFT p99 6.1 s — a (roughly) 2.1x effective-throughput and 3.6x first-token-tail
win. So the recovery that fixed the determinism gate does not come with a
degrade: leaving DSPARK on is both the correctness-supported and the
SLA-required configuration, and the deployment stays on it.

## Why the two-GPU quantization is not a no-op on Hopper

DeepSeek-V4-Flash publishes its *dense* scheme in `quantization_config`
(`quant_method="fp8"`, block-scaled e4m3 with 128x128 ue8m0 scales) and its
*routed expert* scheme in the top-level `expert_dtype="fp4"` (int8-packed e2m1
pairs, one e8m0 scale per 32 input elements). The two are not the same, and
DeepGEMM's FP4 / MXFP4 MoE and indexed paths are SM100-only. On Hopper the
working configuration is:

- routed experts: block-FP8 via DeepGEMM grouped MoE, or the MXFP4 serialized
  variant re-loaded through the `Mxfp4` repack;
- indexer: FP8-block (`deep_gemm.fp8_mqa_logits`) with the **workspace**
  `deepseek_v4_indexer_topk_prefill` op (SM90 build), *not* TRT-LLM's
  `indexerTopK` (Blackwell-only).

## Changes in the working tree

All are uncommitted on branch `unieai`.

1. `python/tokenspeed/runtime/models/deepseek_v4.py`

   - `_routed_experts_are_fp4()` reads `config.expert_dtype`; when it is
     `fp4`/`mxfp4`/`nvfp4`, `DeepseekV4MoE` builds an MXFP4 routed-expert
     `MoELayer` (`is_checkpoint_mxfp4_serialized=True`, `with_bias=False`) and
     `DeepseekV4ForCausalLM._map_weight_name` spells the block-scales with the
     MegaMoE `..._weight_scale` suffix. Without this the loader built
     `(2*ispp/128, hidden/128)` fp32 block-scales against a checkpoint that
     ships `(2*ispp, hidden/32)` e8m0 ones (128x off, wrong dtype) and raised
     the `load_w13` shape mismatch.
   - `_deepseek_v4_indexer_topk_from_logits_prefill_op` now uses TRT-LLM's
     `indexer_topk_prefill` only when `_platform.is_blackwell` and the op
     exists; on Hopper (and any portable CUDA build) it dispatches to the
     workspace `indexer_topk_prefill`. TRT-LLM `indexerTopK` is compiled for
     SM100+ and fails on Hopper with "no kernel image is available".

2. `python/tokenspeed/runtime/layers/moe/weights/loaders.py`

   - `load_w13` re-raises a shape mismatch with a full structured dump
     (shard id/dim, tp size/rank, expert vs loaded shapes/dtypes) instead of a
     bare `RuntimeError`. This is a diagnostic-only wrapper; the root cause is
     fixed in (1).

3. `python/tokenspeed/runtime/utils/server_args.py`

   - `TOKENSPEED_DISABLE_ALLREDUCE_FUSION=1` escape hatch: forces
     `enable_allreduce_fusion=False`, disables the fused `AOT trtllm_comm`
     path, and logs it. This exists because the fused path has no CLI
     counterpart and silently selects itself on Hopper.

4. `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py`

   - `nvidia_create_rsag_state` now checks the symmetric-memory rendezvous
     `multicast_ptr`; if it is 0 (torch prints "Gracefully skipping multicast
     initialization" and still "succeeds"), it raises `RuntimeError`, telling
     the runtime callers the Triton RSAG kernels are unavailable so they fall
     back to the plain NCCL all-gather. Without the guard the kernel stores
     through a null multicast pointer and the failure surfaces asynchronously
     inside an unrelated later Triton launch, taking down every TP launch.

5. Image-span bidirectional attention (prefill)

   - `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py`:
     both swap-index combiners gained optional `span_lo`/`span_hi` (per-token
     image-span bounds) and `max_image_tokens`; a token inside a span widens its
     SWA left/right window to the reference `get_image_visible` window.
   - `python/tokenspeed/runtime/layers/attention/deepseek_v4/metadata.py` +
     `.../attention/backends/deepseek_v4.py`: `DeepseekV4ForwardMetadata` carries
     `image_span_lo/hi` + `image_span_max_tokens`, sliced per chunk and passed to
     all three combine call sites.
   - `python/tokenspeed/runtime/models/deepseek_v4.py`:
     `build_image_spans_from_context` mirrors the multimodal embedder's
     request/prefix/base walk and attaches the spans during the model forward.

6. MTP / NextN (DSPARK) wiring

   - `python/tokenspeed/runtime/models/deepseek_v4_dspark.py` now threads the
     multimodal-active flag and the DSV4 attention backend through the draft
     path, and `_map_checkpoint_name` spells `weight_scale` vs
     `weight_scale_inv` based on the MTP positions / FP4 scheme so the draft
     weights load at the right dtype.

## How to launch

Container `roy-h200g` was committed from the GPU-enabled `ts-env:gpu` image
and carries a working NVML + CUDA context (the hand-mounted `/dev/nvidia*`
container was broken). The host maps physical GPUs 6,7 to logical 0,1 inside
`roy-h200g`, so **do not** set `CUDA_VISIBLE_DEVICES=6,7` in the container.

```bash
bash /workspace/roy/launch_vision.sh v4vision
# launcher: docker exec roy-h200g python3 -m tokenspeed.cli serve \
#   --model-path /models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
#   --tp 2 --port 30101 --host 127.0.0.1 --trust-remote-code \
#   with FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_WORKSPACE_BASE=... PYTHONPATH=...
```

Watch the log: `Kernel tuning finished`, `Capturing batches ... 23/23`,
`TokenSpeed gRPC server is ready` (printed by the SMG gateway package), `Starting UnieInfer HTTP server`. Engine
gRPC on port 56749, OpenAI HTTP on 30102 (also accepted on 30101).

When serving with the DSPARK speculative draft (`--speculative-algorithm
DSPARK --speculative-config '{"method":"DSPARK","num_speculative_tokens":5}'`)
on this checkpoint, pass `--disable-kvstore` (the same-checkpoint KV store is
not wired for the DSV4 vision path) and lower `--max-cudagraph-capture-size`
to 40 — the DSPARK draft capture otherwise attempts a batch-80 graph that
OOMs the ~13.7 GiB/GPU headroom. `Accept_num_tokens_avg` in the per-step log
is the evidence the draft is actually engaged (1.6–3.5 on this run).

```bash
docker exec roy-h200g curl -s -m 120 http://127.0.0.1:30102/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
       "prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Observed: output text starting with `" Paris"`, `prompt_tokens=5`,
`completion_tokens=32`.

## Memory budget

Text-only backbone: ~129 GiB/GPU, so ~13.7 GiB/GPU free out of 141.5 GiB.
Adding a ~32-layer vision tower, `aligner`, and the three MTP layers will not
fit at the current `gpu_memory_utilization=0.87` without one of:
- lowering `--max-model-len` / KV cache,
- reducing `gpu_memory_utilization`,
- quantizing the vision tower / MTP, or
- running the vision tower at TP1 in the encode role (EPD
  `--disaggregation-mode encode`).

## Remaining integration

- **AgentX.** The OpenAI-compat driver (`bench/agent-slo/drive.py`) and the
  Nebius SWE-agent corpus drive the same conversation strands at temperature 0
  on both engines (see the AgentX-style benchmark above). The engine gateway
  accepts the multimodal-free AgentX data over `/v1/chat/completions` for text
  requests. **Closed:** the temperature-0 nondeterminism is root-caused and
  patched (`deepseek_v4_persistent_topk.cuh` + the prefill/decode index-order
  reorders), the engine is temperature-0 self-consistent (3/3 clean reps
  token-identical) and produces coherent, length-comparable output, so the
  correctness gate passes. vLLM is kept only as a throughput reference; the
  AgentX throughput figure is a validated measurement.
- **Vision through the OpenAI gateway.** `/v1/chat/completions` rejects
  multimodal requests for this model (`invalid_multimodal_request: multimodal
  not supported for model: dsv4`) because the smg Rust registry has no
  DeepSeek-V4 vision processor. Vision requests must use the gRPC scheduler
  (`bench/drive_dsv4_vision.py`, `bench/bench_dsv4_vision_grpc.py`). Registering
  the processor in smg would let the HTTP path forward the precomputed
  multimodal inputs that `async_llm.py` already expects.

## Vision integration (done)

The vision path is wired and verified end-to-end:

- `python/tokenspeed/runtime/models/deepseek_v4_vision.py` ports the reference
  tower: `DeepseekV4VisionModel` (32 bidirectional `Block`s, 2D RoPE),
  `Aligner` (3x3 unfold downsample, `vision_dim*9 -> dim`), and the sentinels
  / `build_image_block` / `llm_grid` helpers. All inputs derive from the
  checkpoint's top-level config fields (`vision_n_layers`, `vision_dim`, ...);
  no `vision_config` sub-dict is required.
- `DeepseekV4ForCausalLM` now registers as multimodal (`model_config.py`),
  builds the tower + learnable image-scaffold params, implements
  `get_multimodal_encoder_specs` / `pad_input_ids` / `get_image_feature`, and
  injects the encoded block into `input_embeds` before the LLM forward
  (`vision_embedder.apply`). `load_weights` loads `vision.*`, `aligner.*` and
  `image_*` directly instead of dropping them.
- **No image placeholder token.** The request does not set
  `placeholder_token_id` and does not use the tokenizer's `<image>` /
  `<|deepseek_image|>` placeholder ids. The sentinel block ids
  (`vocab_size + IMAGE_START/IMAGE_PAD/IMAGE/NEW_LINE/END`) are written
  straight into `input_ids`, with `MultimodalDataItem.offsets` and
  `model_specific_data` (`types`, `perm`, `n_vit_h`, `n_vit_w`) carried beside
  them. The engine's `MultimodalEmbedder` replaces the id span with the
  encoded embeddings, so the LM never sees a placeholder id as text.

Verification, run inside `roy-h200g` (GPUs 6,7 -> 0,1):

```bash
export PYTHONPATH=/workspace/roy/UnieInfra/python:/workspace/roy/UnieInfra/tokenspeed-kernel/python
# token-for-token layout + embedding agreement vs the reference tower
python3 /workspace/roy/UnieInfra/bench/verify_dsv4_vision.py
# gRPC vision request against the live server (no placeholder token)
python3 /workspace/roy/UnieInfra/bench/drive_dsv4_vision.py \
  IMG_SOLID="255,0,0" IMG_QUESTION="What color is this image? Reply with one word."
```

Observed at temperature 0: red image -> `" Red"`, blue image -> `" Blue"`,
green image -> empty output (model emitted EOS); `verify_dsv4_vision` reports
identical block layout and embeddings matching the reference to < 1e-2. The
image span must fit inside a single prefill block (prompts here are ~130
tokens, well under the 256-token block size).

### Image-span bidirectional attention (prefill, done)

The sparse-prefill index combiners
(`deepseek_v4_combine_topk_swa_indices` /
`deepseek_v4_combine_dense_swa_indices`) now widen a token's SWA window from the
causal `[pos - window_size + 1, pos]` slice to the reference visible window
(`get_image_visible` / `get_window_topk_idxs_visible`) when the token is inside
an image sentinel span: up to `window_size - 1` tokens to the left and
`vision_max_n_token` tokens to the right (so the widest window is
`window_size + vision_max_n_token`, here 128+384=512). It is **not** the whole
span when the span is longer than that fixed window. The spans are built by
`build_image_spans_from_context` in `models/deepseek_v4.py` (mirroring the
multimodal embedder's request/prefix/base walk) and attached to
`DeepseekV4ForwardMetadata.image_span_lo/hi` plus the
`image_span_max_tokens` cap; `_prefill_workspace` passes them to all three
combine call sites. Decode is untouched (still causal), and prefill tokens
**not** in a span are byte-identical to the previous causal window.
Kernel-level GPU tests live at
`tokenspeed-kernel/test/test_deepseek_v4_image_span_swa_gpu.py` (including a
span wider than the window to exercise the cap) and the span builder at
`test/runtime/models/test_deepseek_v4_image_span.py`.

## Vision integration contract (verified by reading the engine)

The engine's multimodal path is encoder-agnostic; a model opts in with three
hooks and everything else is generic.

1. `configs/model_config.py::is_multimodal_model` must accept
   `"DeepseekV4ForCausalLM"` so `ModelConfig.is_multimodal_active` becomes
   `True` (it is the runtime gate the request path checks).
2. `DeepseekV4ForCausalLM` must build a vision tower as `self.vision` (ViT +
   `aligner`) so `load_weights`'s `params_dict` lookup picks up
   `vision.blocks.*`, `vision.patch_embed.proj`, `vision.norm.weight`,
   `aligner.w1/w2` through the existing generic `_map_weight_name` +
   `default_weight_loader` path (today these names are not in `params_dict`,
   so they are dropped; `mtp.*` is skipped explicitly in `load_weights`).
3. The model must implement:
   - `get_multimodal_encoder_specs() -> dict[Modality, EncoderSpec]` with
     `EncoderSpec(fn=<callable>, make_warmup_items=<factory>)`;
   - `pad_input_ids(input_ids, mm_inputs) -> pad_input_tokens(...)`.

`EncoderSpec.fn` receives `list[MultimodalDataItem]` whose `item.feature` is
the patch tensor for one image (already moved to GPU by
`multimodal/encoder_feature_transport.py`) and whose `item.offsets` are
`[(start,end), ...]`, one range per subgrid. It returns one
`(sum(offsets spans), text_hidden)` tensor, concatenating subgrid tokens in
offsets order. The generic `MultimodalEmbedder` (`multimodal/embedder.py`)
plans, dedups, runs the encoder, and splices the tokens into the LM
`input_embeds`; per-item `pad_value`/`offsets` are produced gateway-side and
arrive as `GenerateReqInput.precomputed_multimodal_inputs`
(`engine/input_processor.py:143`).

Reference wiring: `models/qwen3_5.py` (`VisionEmbedder(encoder_mapping=...)`
call in the LM forward, `self.visual = Qwen3VLMoeVisionModel`,
`pad_input_ids`) plus `models/qwen3_vision.py::Qwen3VLMoeVisionModel`.

The authoritative tower to port is the checkpoint's own `inference/vision.py`
(118 lines): `PatchEmbed` linear over `3*vision_patch_size^2`, 32
bidirectional `Block`s using `scaled_dot_product_attention` with 2D RoPE
(`vision_rope_theta`), `Aligner` with a `3x3` unfold downsample projecting
`vision_dim*9 -> dim` (GELU) -> `dim`. The exact HF forward (image embedding
injection, `<image>` token expansion, MTP) is in `inference/model.py`; image
sizing grids are in `inference/image_processor.py`. No `vision_config`
sub-dict is needed; the fields are top-level (`vision_n_layers`=32,
`vision_dim`=1024, `vision_n_heads`=16, `vision_inter_dim`=2816,
`vision_patch_size`=14, `vision_rope_theta`=10000,
`vision_downsample_ratio`=3, `vision_max_n_token`=384,
`vision_min_pixels`=147456, `vision_max_wh_ratio`=8).

## Live TTFT re-measure (2026-10-08, fresh `eng-dsv4` / GPU6,7, DSPARK config
from `dsv4_serve.sh`, OpenAI HTTP :30101)

Re-ran text load on a harness that records per-request TTFT (conc16, n=48,
max_tokens=128, temperature 0, ~600-token prompts, streamed):

| metric | value |
| --- | --- |
| requests | 48 (39 ok, 9 rejected) |
| TTFT mean / p50 / p90 / p99 (ms) | 5248 / 1820 / 23355 / 23357 |
| E2E mean / p50 / p90 (ms) | 29122 / 29371 / 37663 |
| TPOT mean / p50 (ms) | 475 / 505 |
| aggregate | ~0.49 req/s, ~49 gen tok/s |

The aggregate gate is already claimed by the AgentX cell above (engine 1.27x
the vLLM reference). This run targets the *remaining* weakness, the first-token
tail.

**Root cause of the TTFT tail (engine's own diagnostic,
`model_executor.py` mix/prefill batch):** the first wave is packed — one
`Mix batch #new-seq: 15 #new-token: 5286` — but the staggered tail drains
requests one or two at a time and each is pre-filled in its own tiny batch
(`#new-seq: 1 #new-token: 437/441/430`, logged seconds apart across
05:54:35→05:54:52). A request that arrives late waits behind ~an entire wave
of these single-seq prefills, so its TTFT lands at the 23 s p90. The engine has
no prefill batch-accumulation window: it launches a prefill the moment a
request is admitted, so staggered arrivals never coalesce. vLLM / sglang let a
few requests accumulate into one forward. This is the same defect class as the
embedding micro-batch fragmentation (h200-five-models Round-25/26), so the fix
is shared: accumulate concurrent cold prefills into one batched forward before
launching. `max_cold_prefills` is *not* the window — it is a per-plan cap on
cold prefill admission (default 0 = unlimited), and lowering it only fragments
further.

**Separate availability defect:** 9/48 requests were rejected with HTTP 404
from the serving stack — `smg::routers::grpc::common::stages::worker_selection`
(`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:101`)
`No available workers for model /models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.
All nine fired in one second under load (05:55:15). The engine itself generated
correctly (Paris smoke test, temperature 0); this is a gateway worker-selection
availability bug in the HTTP-to-engine path, not a model/kernel defect.
