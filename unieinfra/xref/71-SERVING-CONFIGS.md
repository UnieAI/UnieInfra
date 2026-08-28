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

### gemma-4-31B: where the remaining decode time goes

Measured 2026-08-28 on a batch-32 pure-decode window (short prompts, so the KV
is nearly empty and attention is at its cheapest), device kernel time only —
this box's profiler CPU times run ~3.6x high and cannot be used for this.

| | ms/step | share |
|---|---|---|
| GEMM | 45.7 | 87.4% |
| attention | 5.15 | 9.8% |
| everything else | 1.4 | 2.8% |
| **total** | **52.3** | measured step 55.7 ms, vLLM 50.1 ms |

**The GEMM half is already at the weight-streaming floor.** A standalone timing
of the model's own eight projection shapes sums to 832.5 us per layer at M=16,
i.e. 49.95 ms across 60 layers — the measured GEMM time is 45.7 ms. There is no
kernel to swap in: the remaining gap to vLLM lives in the 6.6 ms that is not
GEMM, and attention is 5.15 ms of it (`kernel_mha`, 77 us per layer per step
against an almost-empty KV).

Three explanations were tested and refuted; do not re-try them without new
evidence:

- **"A bad kernel takes 41% of decode."** 41% of decode kernel time is
  `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128`, and cuBLAS does
  switch to it at M=16 for every projection except gate_up. But it is the
  *right* choice there: padding M from 16 to 32 to escape it makes 7 of the 8
  shapes slower and the per-layer total worse (832.5 -> 857.6 us). It holds 41%
  because it does most of the projections, not because it is slow.
- **The `out=` call form.** `torch_mm` uses `torch.mm(A, B.T, out=out)` while a
  microbench used `F.linear`. Measured: `F.linear`, `torch.mm`, and
  `torch.mm(out=)` select the same kernel and the same time for every shape.
- **QKV fusion.** Implemented with `MergedColumnParallelLinear` (vLLM has
  `QKVParallelLinear`; we issue q/k/v separately). Numerically better — 196/256
  token-identical replies against vLLM versus 187/256 — and the microbench says
  the merged GEMM is 15% cheaper at M=32. Integrated it is **worse**: agent
  c=32 −9.6%, c=14 −5.6%, because `qkv.split()` yields non-contiguous slices and
  the following `reshape(-1, head_dim)` copies ~410 MB per layer on a
  12.5k-token prefill. Reverted.

`TOKENSPEED_MM_SHAPES=1` prints the dense-GEMM shape histogram at the dispatch
point. It exists because the trace could not answer this: correlation ids repeat
across a long capture, and `record_shapes` does not survive the `/start_profile`
proxy.

### Operating point

**c=32 on agent, not c=64.** The engine's effective concurrency is 24.6 at both
c=32 and c=64 (`token_capacity / mean prompt` = 172,928 / ~7,000 = 24.7), so
c=64 only adds 23.7 queued requests whose prefixes are evicted while they wait:
in-window prefill tokens go 16,384 -> 53,218 for the same 76% hit rate, and rpm
drops 76.6 -> 52.6. Holding those prefixes would need twice the arena, which
does not exist. This is a serving recommendation, not a defect to fix.
