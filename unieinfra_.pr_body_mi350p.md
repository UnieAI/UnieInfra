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
- Shard-local block verification across TP ranks (`TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1`): each rank scores its own vocab shard and two small all-gathers replace the full-vocab FP32 gather.

**Prefill overlap and MoE (gfx950)**
- Optional two-micro-batch prefill (`TOKENSPEED_PREFILL_TBO=1`): the all-reduce of one half runs on a side stream while the other half computes.
- Package-prefill MXFP4 MoE, on by default (set the variable to `0` for the previous kernels; output is bit-identical): pipelined stage 1 with a three-slot A ring and dword E8M0 scale loads (`TOKENSPEED_MOE_PREFILL_S1_V2`), dword scale loads in stage 2 (`TOKENSPEED_MOE_PREFILL_S2_V2`), M-major stage-2 tiles and `.cg` partial reads in the reduce combine (`TOKENSPEED_MOE_PREFILL_S2_REDUCE_V2`). Per layer: 1111→1021 µs at M=4096, 2239→1570 µs at M=8192.

**Tooling**
- `TOKENSPEED_GPU_STEP_TIMING=1`: profiler-free GPU timing of forwards (decode/extend GPU time, launch host time, idle between forwards). Use it instead of a torch-profiler trace for host/launch questions: under the profiler ROCm attaches a completion signal to every graph packet and changes how graphs are submitted.
- `.skills/unieinfra-hw-optimize`: step-by-step playbook and scripts for bringing up and tuning UnieInfra on new GPUs.

## Results (4x MI350P, AgentX corpus, think time removed, 600 s per cell, T=1.0 sampling, each engine at its better running cap; UnieInfra at GPU memory utilization 0.86, vLLM at 0.9)

Requests/min, UnieInfra (coupled drafting + shard-local block verification + prefill TBO) vs the better vLLM cap:

| Concurrency | UnieInfra | vLLM | Ratio |
|---|---|---|---|
| c1 | 18.4 | 11.0 | 1.67x |
| c4 | 44.3 | 37.3 | 1.19x |
| c16 | 81.7 | 63.9 | 1.28x |
| c32 | 124.3 | 111.7 | 1.11x |
| c64 | 165.4 | 135.6 | 1.22x |
| c128 | 136.4 | 23.2 | 5.88x |

All cells 0 errors and no HSA `OUT_OF_RESOURCES`. At c128 the budget-sized KV cache keeps the prefix-cache hit rate above 90%, against 24-64% for vLLM. Run-to-run spread on this corpus is about ±10% per cell, so single-flag A/B results below that are not resolvable end to end; the MoE kernels are kept on their per-layer measurements.

## Test plan
- [x] Kernel unit tests for the changed gfx950 kernels; needle-in-haystack 6/6; GSM8K (T=0) 92.6–93.6% across the kernel commits.
- [x] Block verification: chunked result equals the unchunked one bit for bit at chunk = batch; accept-index layout checked for chunk < batch (CPU).
- [x] AgentX stress sweep c1–c128 on the final build (table above).
- [x] Final build, production flags: needle-in-haystack 6/6; GSM8K (T=0, 1319 questions) 90.45% (prefill TBO baseline 90.67%, CI threshold 0.90).
- [x] Prefill MoE: `test_gluon_mxfp4_prefill_v2_gfx950.py` + `test_gluon_mxfp4_situ_gfx950.py`, 257 passed; reduce mode bit-exact with the previous kernels; no register spills.
- [x] Shard-local block verification: CPU statistical test, total variation 0.03–0.04 (noise floor).
