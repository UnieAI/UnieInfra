# DeepSeek-V4.1 on H200 TP4: AgentX against vLLM 0.31

What the `dsv41` branch measures on the AgentX agent workload after the
upstream merge (see `REPORT-upstream-merge-2.md`), which commits produced it,
and what was tried and dropped. Target was 2x vLLM 0.31 (c16 ≤ 23.2 s).

## Result

Median of three arms per engine, each arm running the cells in a different
order (vLLM: 8-16-32, 32-16-8, 16-8-32; ours: 8-16-32, 16-32-8, 32-8-16),
seconds for 256 AgentX requests, [range]:

| engine | c8 | c16 | c32 |
|---|---|---|---|
| vLLM 0.31.0 | 57.7 [56.5-59.5] | 47.3 [46.4-56.2] | 41.0 [39.6-42.3] |
| tokenspeed `dsv41` (795db10c) | 34.4 [33.9-35.9] | 26.2 [26.0-27.0] | 21.2 [21.1-22.2] |
| + `--prefill-coalesce-ms 100` (04e27531) | 32.8 [32.7-34.7] | 24.2 [23.8-25.4] | 19.1 [18.8-20.0] |
| **speedup** (with coalescing) | **1.76x** | **1.95x** | **2.15x** |

GSM8K (evalscope, 1319 problems, greedy): 0.9340 on 795db10c, 0.9356 with
coalescing (CI gate 0.90).

Low and high concurrency, two arms per engine (orders 1-64-128 / 128-1-64
for ours, 128-64-1 / 64-1-128 for vLLM), both engines at 128 sequences:

| engine | c1 | c64 | c128 |
|---|---|---|---|
| vLLM 0.31.0 | 136.4 [134.3-138.5] | 36.1 [35.2-37.0] | 37.7 [37.5-37.8] |
| tokenspeed `dsv41` (795db10c) | 106.1 [105.7-106.5] | 19.4 [18.9-19.8] | 20.7 [20.6-20.9] |
| + `--prefill-coalesce-ms 100` | 106.9 [105.5-108.4] | 18.5 [18.4-18.6] | 20.4 [19.6-21.1] |
| **speedup** (with coalescing) | **1.28x** | **1.95x** | **1.85x** |

Both engines saturate around c64; c128 only queues more. At c1 decode is at
parity (TPOT 2.6 vs 2.7 ms) and the lead is TTFT (84 vs 186 ms).

### Serving metrics

Each request sends 9.7k input tokens on average (p50 10.1k, p90 11.9k,
2.48M per cell; about 86% hit the prefix cache inside a session) and
generates exactly 128 output tokens (`min_tokens = max_tokens = 128`).
RPM and output TPS come from the cell's wall time. TTFT is measured at the
client, so it includes queueing. TPOT is (E2E - TTFT) / 127. Every value is
the median over the arms. dsv41 rows are 04e27531 with
`--prefill-coalesce-ms 100` (c64/c128 with 128 sequences).

| c | engine | RPM | output TPS | TTFT mean / p50 / p90 (ms) | TPOT (ms) | E2E p50 / p90 (ms) |
|---|---|---|---|---|---|---|
| 1 | vLLM 0.31 | 113 | 240 | 186 / 185 / 395 | 2.7 | 512 / 729 |
| 1 | **dsv41** | **144** | **307** | **84 / 66 / 155** | **2.6** | **405 / 497** |
| 8 | vLLM 0.31 | 266 | 568 | 338 / 221 / 509 | 11.5 | 1458 / 3121 |
| 8 | **dsv41** | **469** | **1000** | **262 / 209 / 343** | **5.9** | **916 / 1711** |
| 16 | vLLM 0.31 | 325 | 693 | 576 / 282 / 1131 | 18.7 | 2347 / 4842 |
| 16 | **dsv41** | **634** | **1354** | **354 / 238 / 608** | **9.1** | **1291 / 2911** |
| 32 | vLLM 0.31 | 375 | 799 | 1207 / 410 / 2899 | 30.3 | 3834 / 13588 |
| 32 | **dsv41** | **805** | **1717** | **597 / 307 / 1043** | **14.0** | **1868 / 5414** |
| 64 | vLLM 0.31 | 426 | 908 | 3093 / 655 / 13620 | 45.2 | 6329 / 19048 |
| 64 | **dsv41** | **830** | **1770** | **1415 / 514 / 5127** | **24.4** | **3143 / 9539** |
| 128 | vLLM 0.31 | 408 | 870 | 9265 / 4877 / 20753 | 68.6 | 17402 / 30222 |
| 128 | **dsv41** | **755** | **1610** | **4894 / 3320 / 12963** | **37.4** | **12547 / 17011** |

Coalescing trades a little TTFT at low load (c8 mean 151 -> 262 ms, still
below vLLM's 338 ms) for throughput: a request that arrives while others
decode waits up to 100 ms to share a prefill step.

Counting input tokens as well, total throughput at c16 is
(2.48M + 33k) / 24.2 s ≈ 103.7k tok/s for dsv41, against 53.0k tok/s for
vLLM 0.31.

The 2x target (c16 ≤ 23.7 s against this median) is reached at c32 (2.15x)
but not at c16 (1.95x, 0.5 s short). Every flush before a cell returned 200 (`run_cells.sh` now
aborts otherwise: one vLLM repeat without `VLLM_SERVER_DEV_MODE=1` got 404
flushes and replayed a warm cache, and was discarded).

## Setup

- 4x H200 (GPUs 4-7), TP4 + EP4, DeepSeek-V4.1-Flash-1017, DSpark speculative
  decoding with 5 draft steps (the checkpoint's block size; vLLM uses the same).
- AgentX: 32 SWE-smith agent sessions x 8 turns (256 requests,
  `bench/agent-slo/corpora/swe_smith_agent_sess32x8.jsonl`), at least 128
  output tokens, greedy. The prefix cache is flushed before every
  concurrency cell, and a short-decode warmup that shares no prefix runs first.
  Duration is wall time for all 256 requests.
- vLLM 0.31.0 (`vllm/vllm-openai:v0.31.0`) with the same model, TP4 + EP,
  prefix caching, 64 sequences, DSpark with 5 speculative tokens
  (`unieinfra/bench/agentx/arm_vllm.sh`).
- Accuracy gate: evalscope GSM8K (1319 problems, greedy, 512 tokens) on the
  same server after the AgentX cells; CI's threshold is 0.90.

### Serving recipe

```
ts serve --model DeepSeek-V4.1-Flash-1017 --tensor-parallel-size 4 --enable-expert-parallel \
  --speculative-algorithm DSPARK --speculative-num-steps 5 --speculative-num-draft-tokens 6 \
  --moe-mxfp4-fp8-activation --chunked-prefill-size 8192 --prefill-graph-max-tokens 8192 \
  --gpu-memory-utilization 0.90 --engram-host-table --max-num-seqs 64 --max-cudagraph-capture-size 64 \
  --prefill-coalesce-ms 100
```

For c64 and above use `--max-num-seqs 128 --max-cudagraph-capture-size 128`.
`--prefill-graph-max-tokens 8192` needs `--gpu-memory-utilization 0.90`: at
0.95 the decoder-graph capture runs out of memory.

## Where the time went, commit by commit

AgentX duration in seconds (c8 / c16 / c32), one run per row unless noted.
Rows marked *recipe* are flag changes on the same build.

| step | commit | c8 | c16 | c32 | GSM8K |
|---|---|---|---|---|---|
| upstream main, as merged | 975fa69b | 87.2 | 51.4 | 41.4 | |
| batch prefill attention, fuse Hopper index top-k glue | e3cc26b3 | 66.7 | 45.3 | 37.9 | |
| capture the attention projections in prefill graphs | 3a2b67eb | 52.2 | 36.5 | 33.0 | |
| *recipe:* prefill graphs up to 4096 tokens | | 49.0 | 33.5 | 27.5 | |
| *recipe:* FP8 activations into the MXFP4 experts (W4A8) | | 43.3 | 32.8 | 27.7 | 0.9287 |
| capture the CED narrowing layer around its row selection | f57f6f27 | 43.5 | 29.8 | 23.6 | 0.9318 |
| *recipe:* 5 DSpark draft steps (vLLM's width) | | 42.6 | 28.7 | 22.0 | 0.9272 |
| memoize kernel selection on the eager path | 52b91676 | 38.5 | 27.4 | 21.8 | |
| query pad + compressor projection in graph | 110f60fb | 40.3 | 27.5 | 22.3 | 0.9227 |
| graph the prefill `index_topk` spans, shared SWA workspace | 70a4f893 | 40.0 | 27.1 | 21.9 | 0.9265 |
| *recipe:* 8192-token chunks and prefill graphs | | 35.9 / 34.4 | 26.2 / 26.1 | 21.4 / 22.4 | 0.9303 |
| narrowing layer's ratio-1 compressor in graph | 795db10c | 35.9 | 26.0 | 21.2 | 0.9340 |
| coalesce prefills arriving during decode, 50 ms | 04e27531 | 35.2 | 24.2 | 19.4 | |
| *recipe:* coalescing 100 ms (median of 3) | 04e27531 | 32.8 | 24.2 | 19.1 | 0.9356 |
| *recipe:* coalescing 150 ms | 04e27531 | 34.5 | 24.1 | 19.0 | |

Cells of about 1 s are within run-to-run noise (the repeated runs in the
results table show the spread); the large steps are host-bound prefill
fixes (graph capture of everything token-local, batching, memoized kernel
selection) and the serving recipe. Coalescing addresses the prefill step's
fixed cost: before it, 73% of AgentX prefill steps at c16 carried a single
sequence and each paid the full ~24 ms weight stream.

## Where a step's time goes now

Profiled at the recipe (c16, 120 steps after the cache flush, TP0):

- **Decode** (22.6 ms/step): routed MoE 9.8 ms, dense projections 4.2 ms,
  all-reduce 2.7 ms (0.7 ms transfer, the rest waiting for the slowest rank),
  attention 2.2 ms, mHC 1.8 ms. The host launches about 17 ms ahead; the
  step is GPU-bound.
- **Prefill** (52 ms/step at ~1.7k new tokens): GPU busy 46-48 ms. Routed MoE
  14.4 ms, all-reduce 5.2 ms, sparse attention 4.1 ms.
- Both are bound by streaming the routed experts' weights: a prefill step
  touches every local expert (2 x 180 µs per layer, about the H200's
  bandwidth for 96 MXFP4 experts), and decode is at the same floor for the
  experts it touches.

## Tried and dropped (measured)

| idea | result | why |
|---|---|---|
| all per-layer prefill attention inside the graph (41 -> 9 breaks; static fixed-capacity metadata) | 36.1 / 26.2 / 21.0, GSM8K 0.9363 | correct, neutral: at 8192-token chunks prefill is GPU-bound. Patch kept off the branch |
| mixed prefill+decode batches | 38.6 / 28.4 / 26.7 | slower |
| MoE TP4 instead of EP4 (intermediate 576 -> 640 for the W4A8 kernel) | 38.5 / 27.7 / 22.6 (4096 recipe) | padding and narrower GEMMs cost more than balance gains; 8192 graphs crash in capture |
| EPLB / redundant experts | not built | systematic EP imbalance is 0.62 ms of a 22.6 ms decode step (< 2%); V4.1's router has no placement hook |
| FP8 (W8A16 Marlin MXFP8) dense projections, as vLLM does | vLLM's kernel is 1.7-4x slower than cuBLAS BF16 at V4.1 shapes | BF16 widening is right on Hopper |
| GIL switch interval 100 / 20 µs | 35.2 / 25.9 / 20.8 and 34.5 / 25.9 / 21.3 | within noise |
| pinning all ranks to one NUMA node | 42.4 / 28.6 / 23.6 | worse; keep per-GPU NUMA-local affinity |

## Reproduce

`unieinfra/bench/agentx/` (see its README): `arm_ts.sh TAG IMAGE <recipe flags>`
(`GSM=1` adds the GSM8K gate, `CONCS="16 32 8"` reorders the cells),
`arm_vllm.sh TAG` (`MAX_SEQS=128` for c64/c128), and
`summarize.py results vllm=TAGS ours=TAGS` for the medians above
(`CONCS="1 64 128"` for the other cells).
