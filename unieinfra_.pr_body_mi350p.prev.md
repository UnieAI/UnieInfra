## Summary

DeepSeek-V4.1-Flash performance work for AMD MI350P (gfx950, 128 CUs, 4 XCDs, PCIe-only TP4), plus probabilistic DSpark verification. Every behavioural change that is not a pure kernel speed-up is behind an opt-in env flag.

**Kernels (gfx950)**
- CSA2 indexer: 512 MiB score budget (fixes the long-context crash), vectorized key loads, 16-row one-wave tiles, row-block logits for speculative rows, candidate-mode logits on the 32x32x64 MFMA tile, two-pass LDS-compacted top-k select, no `-inf` logits pre-fill on the rows path.
- Selected attention: dim-split decode kernel, rows kernel above one WG per CU, ROCm decode queries padded to 16 heads instead of 64.
- Block32 MXFP8 GEMM for decode-sized dense linears; grid sizes and XCD swizzles read from the device instead of MI355X constants.
- DSpark draft attention routed to the sparse kernel.

**Communication**
- Iris all-reduce for prefill sizes on PCIe-only hosts; size-dependent Iris program cap; opt-in INT8/FP8 pull two-shot (`TOKENSPEED_IRIS_Q8_ALLREDUCE`, default off).

**Scheduler / prefill**
- Optional decode round after N prefill rounds (`TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS`), prefill coalescing, batched prefill attention, shared SWA workspace, captured prefill projections, 32-row compressor-tail pages on ROCm.

**Speculative decoding (DSpark, `--sampling-backend triton_full`)**
- Coupled drafting (`TOKENSPEED_DSPARK_COUPLED_DRAFT=1`): the draft samples with the verifier's Gumbel noise; output stays exactly the target's sample.
- Block verification of the draft chain (`TOKENSPEED_DSPARK_BLOCK_VERIFY=1`, Sun et al. 2024): lossless joint acceptance. It processes the batch in chunks of `TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK` (16) requests, so its full-vocab temporaries stay small. It is only used up to `TOKENSPEED_DSPARK_BLOCK_VERIFY_MAX_BS` (128) requests per step; larger steps keep exact-match verify.
- Optional first-k draft verification (`TOKENSPEED_DSPARK_VERIFY_DRAFTS`).

## Results (4x MI350P, AgentX corpus, think time removed, 600 s per cell, T=1.0 sampling, GPU memory utilization 0.9 for both engines, each engine at its better running cap)

Requests/min, UnieInfra with coupled drafting + block verification vs the better vLLM cap:

| Concurrency | UnieInfra | vLLM | Ratio |
|---|---|---|---|
| c1 | 18.9 | 11.0 | 1.72x |
| c4 | 38.4 | 37.3 | 1.03x |
| c16 | 67.0 | 63.9 | 1.05x |
| c32 | 115.0 | 111.7 | 1.03x |
| c64 | 140.1 | 135.6 | 1.03x |
| c128 | 127.0 | 23.2 | 5.47x |

All cells 0 errors. At c128 the budget-sized KV cache (39.6M tokens vs vLLM's 18.6M) keeps the prefix-cache hit rate at 94%, against 24-64% for vLLM. Block verification alone adds +19% (c1), +2-8% (c4-c64) and is neutral at c128 (-4%, within run-to-run noise). GSM8K, 500 questions at T=1.0: 95.8% with block verification, 96.2% without.

## Test plan
- [x] Kernel unit tests for the changed gfx950 kernels; needle-in-haystack 6/6; GSM8K (T=0) 92.6–93.6% across the kernel commits.
- [x] Block verification: chunked result equals the unchunked one bit for bit at chunk = batch; accept-index layout checked for chunk < batch (CPU).
- [x] Block-verification AgentX stress sweep c1–c128 (table above).
