# Model coverage — TokenSpeed vs the checkpoints we hold

Surveyed 2026-08-02 against `tokenspeed` @ `3f88dcc2` and `/workspace/models`.
Configs read directly; nothing here has been run yet.

## Qwen3.5 and Qwen3.6 are the same architecture

| checkpoint | `model_type` | architecture | TokenSpeed module |
|---|---|---|---|
| `Qwen/Qwen3.5-27B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | `models/qwen3_5.py` |
| `Qwen/Qwen3.6-35B-A3B` | **`qwen3_5_moe`** | `Qwen3_5MoeForConditionalGeneration` | `models/qwen3_5_moe.py` |

Qwen3.6 is not a new architecture — it declares `qwen3_5_moe`, which TokenSpeed
already implements. "Support Qwen3.6" is therefore a **validation** task, not a
port. Both are VLMs (`vision_config` + `text_config`), so `qwen3_vision.py` is
on the path too.

Shared traits: hybrid attention, `layer_types` = three `linear_attention` then
one `full_attention` (`full_attention_interval: 4`), `head_dim` 256,
`attn_output_gate: true`. TokenSpeed has `backends/hybrid_linear_attn.py`.

This is the family where prefix caching was found silently disabled on the vLLM
side — check `--enable-prefix-caching` is actually engaged here rather than
assumed.

## Gemma 4 is a real port, and heavier than the model list suggests

| checkpoint | layers | heads (q/kv) | MoE | sliding/full |
|---|---|---|---|---|
| `google/gemma-4-31B-it` | 60 | 32/16 | no | 50/10 |
| `google/gemma-4-26B-A4B-it` | 30 | 16/8 | 128 experts | 25/5 |

Both are `model_type: gemma4`, `Gemma4ForConditionalGeneration`, with
`text_config` + `vision_config` + **`audio_config`** — omni, not text-only.

Four config fields drive the port cost:

1. **`attention_k_eq_v: true`** — K and V are the same tensor. TokenSpeed has no
   concept for this. It is both a porting obstacle and the single largest KV
   capacity win available on these checkpoints (halves KV bytes per token), so
   it should not be worked around by materialising V.
2. **`global_head_dim: 512` alongside `head_dim: 256`** — the 10 full-attention
   layers use twice the head dim of the 50 sliding layers. Two problems: the KV
   cache must carry per-layer-type geometry, and many attention kernels cap at
   head_dim 256. Which of `triton` / `flashinfer` / `fa3` accepts 512 on sm_120
   is unverified and is the first thing to test.
3. **`final_logit_softcapping: 30.0`** — `logits_processor.py` already has
   softcap handling.
4. **`sliding_window: 1024`** — supported: `kv_cache/mha.py` takes
   `sliding_window_tokens` as a per-layer tuple.

So of the four, two are already served and two are new work. The per-layer
heterogeneity in (2) has a plausible home: `flat_memory_plan.py`'s
`components_from_layers` already builds per-layer `ComponentSpec`s with
differing byte counts for the mamba/attention hybrids.

`use_bidirectional_attention: vision` — image tokens attend bidirectionally,
needed for output parity but not for a throughput number.

## Consequence for phase ordering

The Phase 0 bring-up should not start on Gemma 4. Qwen3.5-27B and Qwen3.6-35B-A3B
need no model work, so they isolate the sm_120 question from the porting
question. Gemma 4 belongs after the sm_120 build is known good.
