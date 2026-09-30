# DeepSeek-V4-Flash-Vision-Exp on 8xH200

Running `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (a quantized
DeepSeek-V4-Flash variant) on two of the eight H200s (Hopper / SM90, cc 9.0)
of the `roy` host, through the token-speed / UnieInfra engine.

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
| AgentX (TPGS / P90 / DEP) evaluation | pending |

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
`TokenSpeed gRPC server is ready`, `Starting TokenSpeed HTTP server`. Engine
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

- **AgentX.** Run `SemiAnalysis/agentx-conversation-explorer` with HF traces
  (`semianalysisai/cc-traces-*`) on both engines to compare TPGS, P90
  interactivity and DEP disaggregation. The OpenAI-compat driver
  (`bench/agent-slo/drive.py`) and corpus (`ts64.jsonl`) are in place; the
  engine gateway accepts the multimodal-free AgentX data over
  `/v1/chat/completions` for text requests.
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
