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
including `chunked_prefill_size`, `max_num_seqs` and `block_size`, is
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
3.6s → 13.9s, TPOT p99 706 ms). Do not read c=64 as this model's operating point.

How many requests actually fit -- and note that `token_capacity / max_model_len`
is NOT the way to compute it, because `token_capacity = parents * P` assumes
every parent serves one group while a request needs pages in ALL six
simultaneously:

    per request, prompt T tokens
      full_attention   ceil(T / 64) / 2 parents      (packing 2, grows with T)
      5x sliding       (ceil(1023 / 64) + 1) = 17 parents each = 85 total
                       (constant -- ReclaimExpired nulls pages that slid out)

At the corpus median of ~7.5k tokens that is 60 + 85 = 145 parents, so
2,702 / 145 = 18.6 concurrent before sharing. Observed `#running-req` plateaus
at 24.6 at both c=32 and c=64 -- higher than 18.6 because the full-attention
pages are shared across sessions by the prefix cache, and flat across the two
concurrencies because the arena, not `max_num_seqs` (452), is the bound.

The damage at c=64 is not eviction of the reusable prefix -- the hit rate holds
at 76% -- it is that the 23.7 queued requests lose their pages while they wait
and re-prefill on admission: in-window prefill tokens go 16,384 -> 53,218 for
the same hit rate, and rpm drops 76.6 -> 52.6. Holding those prefixes would need
twice the arena, which does not exist.

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

---

## The A/B parameter record (2026-08-28)

Both engines, every cell, same GPU, back to back. Command lines are
authoritative; resolved values are read off each server's own startup output.

### Passed on the command line

| | ours (`tokenspeed.cli serve`) | upstream vLLM |
|---|---|---|
| `--max-model-len` | 32768 (agent/RAG short) / 73728 (long agent) | same |
| `--gpu-memory-utilization` | 0.90 | 0.90 |
| `--max-num-seqs` | 452 | 452 |
| prefix caching | on by default | `--enable-prefix-caching` |
| extra | `--enable-metrics --enable-cache-report` (reporting only) | — |
| env | `TOKENSPEED_ENABLE_FI_PAGED_EXTEND=1`, `sm120_env.sh` | `FLASHINFER_DISABLE_VERSION_CHECK=1` |

No `--disable-mixed-batch` (obsolete after `c277b35c`) and no
`--chunked-prefill-size` (8192 is already the resolved default; passing it
restates the default and reads as tuning that is not happening).

### Resolved by each engine

| | ours | vLLM |
|---|---|---|
| per-step batched tokens | `chunked_prefill_size=8192` / `max_prefill_tokens=8192` | `max_num_batched_tokens=8192` |
| chunked prefill | `enable_mixed_batch=True` | `enable_chunked_prefill=True` |
| prefix caching | `enable_prefix_caching=True` | `enable_prefix_caching=True` |
| dtype / KV dtype | `auto` / `auto` (bf16) | `torch.bfloat16` / `auto` |
| quantization | `None` | none |
| scheduler block size | `block_size=64` (gemma) — the cache recipe derives P=64 for gemma, P=128 for Qwen | — |
| admission | `admission_reserve='generation-full'` before the 2026-10-18 upstream merge; the merged scheduler reserves per cache group from each request's own demand and the flag no longer exists (see `72-PARITY-STRATEGY.md` §2) | — |
| recurrent checkpoints | `state_checkpoint_stride_blocks=0` (derive; `_MAX_STATE_CHECKPOINTS=2`) | — |
| host KV tier | `kvstore_ratio=2.0` | — |
| CUDA graph | `max_cudagraph_capture_size=160`, `enforce_eager=False` | default |
| sampling | `sampling_backend='flashinfer'` | default |
| **KV capacity, gemma-4-31B** | **token_capacity 172,928** | **85,470 tokens, "Maximum concurrency 2.61x"** |
| **KV capacity, Qwen3.8-27B** | **token_capacity 531,456, parents 346** | — |

The KV capacity row is not a setting — it is what each engine derives from the
same 0.90 utilization, and it is the single largest structural difference on
gemma-4.

### Workload

| corpus | n | input median | input max | output |
|---|---|---|---|---|
| `claude_v3_sess14x8` (ours, not citable) | 112 | 7,562 | 15,131 | 128 fixed |
| `claude_root_sess8x8` (ours, not citable) | 64 | 64,548 | 68,526 | 128 fixed |
| `swe_smith_agent_sess32x8` (MIT) | 256 | 10,077 | 12,492 | 128 fixed |
| `swe_agent_nebius_sess16x16` (CC-BY-4.0) | 256 | 8,413 | 12,489 | 128 fixed |
| `rag_custom` | 1024 | 1,088 | 1,387 | 128 fixed |

Output is pinned at 128 tokens with `ignore_eos` so both engines do identical
work. That choice has a cost worth stating: at ~48 ms TPOT it makes decode 58%
of the wall time on the agent shapes, and decode is at parity (both engines sit
on the weight-streaming floor), which caps the achievable ratio. A shape with
longer prompts and shorter outputs moves the measurement back onto prefix reuse,
where the engines actually differ.

### Protocol

Fresh server per cell. Engine warmed with 2xC requests from a corpus that shares
**0 characters of prefix** with the measured one (verified, not assumed:
`swe_agent_nebius` against `claude_v3` / `claude_root` / `rag_custom` = 0 chars;
against `swe_smith` = 9 chars). Target corpus run **once** -- a ladder that
reruns the same prompts on one server measures replay, not reuse: the hit rate
climbs 0% -> 95% across three passes and the same cell's wall time moves 51.7%.

## Host DRAM KV tier (L2) — measured

`--kvstore-ratio R` allocates `R × device_lcm_blocks` LCM parents of **pinned**
host memory. Sizes on this box (gemma-4-31B, GMU 0.90): R=2 → 56.67 GB / 5,404
blocks; R=8 → 226.66 GB / 21,616 blocks.

**Ceiling is between R=8 and R=12**, not host RAM exhaustion. R=12 (340 GB) and
R=16 (453 GB) both die in `HostCacheStorage.__init__` at the single
`torch.empty(..., pin_memory=True)` with `CUDA error: out of memory`, while the
box shows 703 GB available. It is the pinned/registered-memory path that refuses,
not the allocator — so the usable knob today is R ≤ 8.

`swe_smith_agent_sess32x8`, c=32, 256 requests, no-replay protocol:

| arm | host GB | dur | rpm | tps | tps/req | TTFT p50 | TTFT p99 | TPOT p50 | hit |
|---|---|---|---|---|---|---|---|---|---|
| device only | 0 | 514.8 s | 29.8 | 57.4 | 2.41 | 27,143 | 81,382 | 200 | 40.8% |
| host ×2 | 56.67 | 470.9 s | 32.6 | 63.6 | 2.68 | 21,217 | 90,747 | 191 | 47.9% |
| host ×8 | 226.66 | **247.8 s** | 62.0 | 120.1 | 6.27 | **2,995** | 155,774 | 97 | 81.0% |
| vLLM (no KV offload in this build) | — | 811.6 s | 18.9 | 36.5 | 1.29 | 69,613 | 94,530 | 253 | — |

**Noise floor first.** The device-only c=32 cell was run twice, 9 hours apart,
with identical flags, corpus, warm-up and no engine commits in between: 467.4 s
and 514.8 s — a **10.2% spread** on nominally the same cell (this box has two
other tenants' servers on GPU 0 and GPU 2 sharing host RAM bandwidth, PCIe and
CPU). So **host ×2's -8.5% is inside the noise and is not a result.** Only host
×8, at -52%, is far enough outside it to claim. Every arm comparison below is run
back-to-back in one session for this reason.

**The tier buys the median and sells the tail.** TTFT p50 falls 27.1 s → 3.0 s
and the mean 36.4 s → 17.0 s, but p99 rises 81.4 s → 155.8 s (max 211.7 s). Most
requests become hits and start almost at once; the minority that still miss now
queue behind a much denser admitted stream. An SLO written on **p99 TTFT is worse**
with the tier on; one written on p50, on TPOT, or on throughput is much better.
Establish which one the customer has before recommending it.

Ratios against vLLM in the same cell: device-only 1.58x, host ×8 **3.28x**.
