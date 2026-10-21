# Level 2 results (Qwen3.8-27B, GB10, `run.sh window`)

Summaries copied from the GB10 logs (`~/unieinfra-main/logs/<run>/metrics.json`). Full per-request
JSON from `vllm bench serve` stays on the box. Shapes: `unique` 1024 in / 128 out, `shared` 1024-token
shared prefix + 256 new / 128 out, `long` 6144 in / 128 out, `longshared` 6144-token shared prefix + 256 new.
`util` is `--gpu-memory-utilization 0.6`; the other serve flags are in each `metrics.json`.

| run | rev | what it tests | ready | KV | GSM8K |
|---|---|---|---|---|---|
| baseline-bf16 | 4d5ed6d0 | BF16 KV, no plugin | 482 s | 36.48 GB, 597504 slots, bfloat16 | 0.945 (n=200) |
| fp8-kv | 4d5ed6d0 | FP8 KV (`--kv-cache-dtype fp8_e4m3`), no plugin | 497 s | 36.66 GB, 1201152 slots, float8_e4m3fn | 0.970 (n=200) |
| plugin-fp8kv | cf5d6b15 | FP8 KV, plugin installed (inert for attention: `--attention-backend triton` locks the solution, kernel_zoo C4) | 497 s | 36.84 GB, 1207296 slots, float8_e4m3fn | 0.965 (n=200) |
| prefill-proof | 07696f3d | FP8 KV; `long` vs `longshared`: same 6144-token decode context, with and without its prefill | 497 s | 36.66 GB, 1201152 slots, float8_e4m3fn | - |

Readings (details in `../plan.md` §2 and `../kernel_zoo.md` §E):

- c=1 TPOT 223-225 ms = 52 GB of BF16 weights at ~233 GB/s, 85% of the 273 GB/s bus: BF16 decode is at its ceiling.
- FP8 KV: +0-7% throughput, 2x KV slots, GSM8K 0.97 vs 0.945; no downside measured.
- The plugin changed nothing here because `--attention-backend triton` keeps its FlashInfer decode from being selected.
- `prefill-proof`: at c=64 the same 6144-token decode context runs at TPOT 435 ms without its prefill and 1998 ms with it;
  the prefill stalls (`enable_mixed_batch` off) are ~80% of the long-prompt TPOT, not the KV reads.

## baseline-bf16

| shape | c | out tok/s | TTFT p50 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|
| long | 1 | 3.7 | 4.75 s | 233 ms | 234 ms |
| long | 4 | 11.0 | 12.61 s | 247 ms | 337 ms |
| long | 16 | 21.4 | 30.30 s | 426 ms | 820 ms |
| long | 64 | 22.8 | 107.84 s | 1934 ms | 2294 ms |
| shared | 1 | 4.4 | 0.61 s | 225 ms | 226 ms |
| shared | 4 | 16.3 | 1.45 s | 234 ms | 246 ms |
| shared | 16 | 54.4 | 4.95 s | 256 ms | 291 ms |
| shared | 64 | 122.4 | 9.44 s | 441 ms | 490 ms |
| unique | 1 | 4.3 | 1.04 s | 223 ms | 225 ms |
| unique | 4 | 15.7 | 3.02 s | 234 ms | 248 ms |
| unique | 16 | 48.5 | 6.21 s | 260 ms | 318 ms |
| unique | 64 | 90.0 | 15.44 s | 597 ms | 692 ms |

## fp8-kv

| shape | c | out tok/s | TTFT p50 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|
| long | 1 | 3.8 | 4.64 s | 232 ms | 234 ms |
| long | 4 | 11.2 | 11.34 s | 243 ms | 349 ms |
| long | 16 | 22.2 | 29.35 s | 412 ms | 791 ms |
| long | 64 | 24.4 | 76.16 s | 2043 ms | 2598 ms |
| shared | 1 | 4.3 | 0.60 s | 226 ms | 227 ms |
| shared | 4 | 16.3 | 1.45 s | 235 ms | 246 ms |
| shared | 16 | 52.1 | 4.95 s | 256 ms | 337 ms |
| shared | 64 | 124.5 | 7.37 s | 439 ms | 491 ms |
| unique | 1 | 4.3 | 1.03 s | 224 ms | 227 ms |
| unique | 4 | 15.7 | 3.03 s | 234 ms | 248 ms |
| unique | 16 | 48.7 | 6.22 s | 262 ms | 315 ms |
| unique | 64 | 92.0 | 15.57 s | 582 ms | 694 ms |

## plugin-fp8kv

| shape | c | out tok/s | TTFT p50 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|
| long | 1 | 3.8 | 4.65 s | 232 ms | 233 ms |
| long | 4 | 11.3 | 12.21 s | 260 ms | 330 ms |
| long | 16 | 22.3 | 29.14 s | 410 ms | 792 ms |
| long | 64 | 24.4 | 80.46 s | 1986 ms | 2588 ms |
| shared | 1 | 4.3 | 0.60 s | 225 ms | 227 ms |
| shared | 4 | 16.2 | 1.41 s | 235 ms | 246 ms |
| shared | 16 | 52.3 | 4.80 s | 256 ms | 337 ms |
| shared | 64 | 125.2 | 7.54 s | 437 ms | 487 ms |
| unique | 1 | 4.3 | 1.01 s | 225 ms | 226 ms |
| unique | 4 | 15.5 | 3.10 s | 234 ms | 255 ms |
| unique | 16 | 49.0 | 6.05 s | 257 ms | 301 ms |
| unique | 64 | 92.6 | 12.78 s | 577 ms | 694 ms |

## prefill-proof

| shape | c | out tok/s | TTFT p50 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|
| long | 16 | 19.6 | 40.63 s | 501 ms | 787 ms |
| long | 64 | 24.5 | 78.01 s | 1998 ms | 2581 ms |
| longshared | 16 | 46.5 | 4.70 s | 263 ms | 423 ms |
| longshared | 64 | 115.9 | 10.01 s | 435 ms | 501 ms |
