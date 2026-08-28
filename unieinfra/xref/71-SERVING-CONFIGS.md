# Per-model serving configs (measured, not guessed)

Every line here was read off a running server's resolved `ServerArgs` or its
`Cache profile:` line, not copied from a script. Re-derive rather than trust if
the model, the tree, or the card changes.

**These configs are NOT interchangeable between models.** The two models below
disagree about KV geometry, page size, and what the engine auto-disables; a flag
that is load-bearing for one is a no-op or a regression for the other. The
engine has **no per-model default layer at the `ServerArgs` level** — measured
2026-08-28, the resolved args for gemma-4-31B and Qwen3.5-27B differ only in
`model`, `tokenizer`, `port`, `rl_control_port` and `seed`. Everything else,
including `chunked_prefill_size`, `admission_reserve` and `block_size`, is
identical for a 60-layer SWA dense model and a 64-layer GDN hybrid.

---

## gemma-4-31B-it — verified good, 2026-08-28

```bash
CUDA_VISIBLE_DEVICES=$GPU \
TOKENSPEED_ENABLE_FI_PAGED_EXTEND=1 \
PYTHONPATH=$W/python:$W/tokenspeed-kernel/python \
bash -c 'cd $W && source $W/legacy/benchmark/scripts/sm120_env.sh; \
  exec python3 -m tokenspeed.cli serve /workspace/models/google/gemma-4-31B-it \
    --port $PORT --served-model-name q27 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 452 \
    --enable-metrics --enable-cache-report'
```

**No `--disable-mixed-batch`.** That flag existed only to dodge the corruption
fixed by `c277b35c`; keeping it now costs throughput and hides nothing.
Verified after the fix: token-identical replies against upstream vLLM
6/256 → 208/256, mean output length 58.02 → 43.31 tokens, throughput +3.9%.

**No `--chunked-prefill-size`.** The resolved default is already 8192
(`chunked_prefill_size=8192 max_prefill_tokens=8192`). Passing it restates the
default and reads as tuning that is not happening.

### What the engine derives on its own
| | value | where |
|---|---|---|
| cache page size | `P=64` | `Cache profile:` |
| LCM parent | `10,485,760` B, `parents=2702` | after `21c4f936` (was 2x that) |
| token capacity | `172,928` | `Cache profile:` |
| geometry families | 2 (10 global 4x512, 50 sliding 16x256), overlaid | all 40 planes are 524,288 B, so overlay is legal — see `873b731a` |
| prefill CUDA graph | kept | no linear_attention layer, so the GDN capture hazard does not apply |

### Measured under the no-replay protocol (ours / upstream vLLM, wall time)
| c | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| agent | 1.10x | 1.27x | 1.73x | 2.22x | **2.33x** | 1.60x |
| RAG | 0.95x | 0.94x | 0.96x | 0.97x | 0.87x | 0.80x |

Known bad shape: **c=64 on agent is worse than c=32** (127.8s vs 87.8s, TTFT p50
3.6s → 13.9s, TPOT p99 706 ms). `token_capacity / max_model_len` is 5.3, so the
cache holds about five full-length requests; the engine accepts 64 anyway and
they evict each other. Do not read c=64 as this model's operating point.

---

## Qwen — different family, different config

Qwen3.5-27B ran with the same args above plus `--chunked-prefill-size 8192`
(a no-op) and is **not** a template for the newer models. What differs
structurally:

| | gemma-4-31B | Qwen3.5-27B |
|---|---|---|
| cache page size | 64 | **128** |
| LCM parent | 10,485,760 B | **100,663,296 B** |
| token capacity | 172,928 | **531,456** |
| cache groups | 6 paged (2 geometry families) | 1 paged `full_attention` + 3 `linear_attention` state groups |
| `binding_utilization` | 1.0 | **0.5098** on all three recurrent groups |
| prefill CUDA graph | kept | **auto-disabled** — "the flashinfer GDN chunk-prefill kernel is active and silently produces garbage under graph capture" |

The 0.5098 is structural: the SSM state is 3,145,728 B and exactly fills its
plane, which pins the recurrent groups at packing 1; the exact-page-stride
constraint then drives `full_attention` to packing 12, which sizes the `.b`
planes at 12 x 262,144 while the conv state puts 61,440 there. Measured cost at
the operating points tested: **none** — `#running-req` reached 64 with parents
peaking at 316/347 and zero admission refusals. Recorded, not fixed.

Splitting the plane namespaces to recover it corrupts: the two families then
have different per-plane sizes and the arena overlay writes into another
family's parent. `873b731a` now refuses that overlay and falls back to a union
layout.
