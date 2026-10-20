# UnieInfra

UnieInfra is UnieAI's LLM inference engine for **agent workloads**: a fixed
system prompt and tool schemas shared by every request, conversations that grow
turn by turn, and many agents running at once. It is built on
[TokenSpeed](https://github.com/lightseekorg/tokenspeed) and tracks it
closely, adding UnieAI's model support, scheduling and kernel work on top.

- **Scheduler**: C++ control plane and Python execution plane. Request
  lifecycle and KV cache ownership are a finite-state machine, with safe KV
  reuse enforced by the type system at compile time.
- **Kernels**: a pluggable kernel system with a portable public API and a
  central registry, so each platform selects its own implementation.
- **Entrypoint**: an OpenAI-compatible server with low-overhead request
  handling.

## Performance

### DeepSeek-V4.1-Flash on 4x H200, against vLLM 0.31

AgentX agent traces: 256 requests per cell, about 9.7k input tokens each
(most of it a prefix shared within a session) and exactly 128 output tokens.
The prefix cache is flushed before every cell. Each value is the wall time for
the cell, median over repeated arms run in swapped orders; lower is better.

| concurrency | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| vLLM 0.31 (s) | 136.4 | 57.7 | 47.3 | 41.0 | 36.1 | 37.7 |
| **UnieInfra (s)** | **106.9** | **32.8** | **24.2** | **19.1** | **18.5** | **20.4** |
| **speedup** | **1.28x** | **1.76x** | **1.95x** | **2.15x** | **1.95x** | **1.85x** |

At concurrency 16, UnieInfra serves 634 requests per minute at a mean TTFT of
354 ms; vLLM serves 325 at 576 ms. GSM8K accuracy is 0.9356 on the same
deployment.

Serving recipe (TP4 with expert parallelism, DSpark speculative decoding):

```bash
ts serve --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --trust-remote-code \
  --tensor-parallel-size 4 --enable-expert-parallel --dtype bfloat16 \
  --speculative-algorithm DSPARK --speculative-num-steps 5 --speculative-num-draft-tokens 6 \
  --moe-mxfp4-fp8-activation --engram-host-table \
  --chunked-prefill-size 8192 --prefill-graph-max-tokens 8192 \
  --max-num-seqs 64 --max-cudagraph-capture-size 64 \
  --prefill-coalesce-ms 100 --gpu-memory-utilization 0.90 \
  --reasoning-parser deepseek_v31
```

`--prefill-coalesce-ms` holds a newly arrived request for up to that many
milliseconds while others are decoding, so requests that arrive close together
share one prefill step. A request that arrives while the engine is idle is
never held. The concurrency 64 and 128 cells ran with `--max-num-seqs 128
--max-cudagraph-capture-size 128`.

## Getting started

UnieInfra installs, launches and is configured the same way as TokenSpeed, and
its documentation applies:

- [Getting Started](https://lightseek.org/tokenspeed/guides/getting-started)
- [Launching a Server](https://lightseek.org/tokenspeed/guides/launching)
- [Model Recipes](https://lightseek.org/tokenspeed/recipes/models)
- [Server Parameters](https://lightseek.org/tokenspeed/configuration/server)
- [Parallelism](https://lightseek.org/tokenspeed/serving/parallelism)

## Built on TokenSpeed

UnieInfra regularly merges upstream TokenSpeed, which is developed by the
[LightSeek Foundation](https://lightseek.org/) and released under the MIT
license (see [LICENSE](LICENSE)). Fixes that belong upstream are sent back
upstream. When citing the engine design, please cite TokenSpeed:

```bibtex
@misc{tokenspeed2026,
  author       = {{TokenSpeed Team}},
  title        = {{TokenSpeed}: A Speed-of-Light {LLM} Inference Engine},
  year         = {2026},
  howpublished = {\url{https://github.com/lightseekorg/tokenspeed}}
}
```
