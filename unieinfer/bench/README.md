# bench — how this engine is measured against the reference ones

Two goals, one axis: **each user as fast as possible** at one end, **as many
users as possible** at the other. `e2e_concurrency.py` sweeps between them and
is the benchmark to run first; everything else exists to explain a number it
produces.

Every script prints both engines' numbers the same way and is meant to be run
once per engine:

```sh
# this engine
source legacy/benchmark/scripts/sm120_env.sh
TOKENSPEED_ENABLE_FI_PAGED_EXTEND=1 PYTHONPATH=python \
  python3 unieinfer/bench/e2e_concurrency.py

# the reference
source /home/ubuntu/workspace/roy/vllm/.venv/bin/activate
BACKEND=vllm VLLM_USE_FLASHINFER_SAMPLER=0 \
  python3 unieinfer/bench/e2e_concurrency.py
```

## What is here

| file | answers |
|---|---|
| `e2e_concurrency.py` | TTFT and inter-token latency, p50/p99, per concurrency level. The knee in TTFT p99 is how many users this can serve. |
| `e2e_shapes.py` | TTFT by workload shape: cold, exact repeat, shared prefix + new tail (RAG), growing conversation (agent). |
| `e2e_throughput.py` | Offline batch throughput, unique vs shared prompts. |
| `kernel_*.py` | One kernel family each, both engines, device and host time separately. |
| `harness.py` | The measurement primitives, and why each is shaped the way it is. |

## Rules, each of which was learned by publishing a wrong number

**Two instruments, never one.** Device time from the profiler, host time from an
enqueue loop with no events in it. For anything short the host cannot enqueue as
fast as the device runs, so a CUDA-event delta measures host stall. RoPE read
16.3 µs that way and looked 1.28x behind vLLM; the kernel is 1.47 µs against
vLLM's 2.51, i.e. 1.7x ahead. That single mistake inverted four kernels at once.

**Print GB/s.** Above 1792 GB/s on this box the working set is in L2 (128 MB),
so the row is a cache-hit rate, not a kernel speed.

**Print launch counts.** They catch a reference engine that has fallen onto its
own slow path — vLLM's RMSNorm runs an eager decomposition under a bare
`VllmConfig()`, 8-11 launches and 194 µs against 14 for its real kernel, which
would have handed us a 13x win that evaporates in any real deployment.

**A timing delta is not evidence that a mechanism engaged.** Two consecutive
attempts at hoisting a flashinfer plan measured 1-2% better while the fast path
was never taken once — the first planned the wrong page table, the second missed
on the last bit of a float. Make the slow path raise and count. `harness.py`
has `assert_engaged` for this.

**Vary the workload shape.** A unique random prompt per request is the one shape
in which a completely dead prefix cache costs nothing. It is also the shape a
throughput harness naturally reaches for. Run `e2e_shapes.py` before believing
any prefix-cache claim.

**Watch the ordering between shapes.** Running the unique-prompt set before the
shared-prefix set fills the KV pool and changes the second result by 17% here
and 0% on vLLM. That difference is a real finding, but it is a property of the
sequence, not of the shape — report which one you measured.

**Three runs minimum after changing a binary or a config.** The first run
recompiles (flashinfer and Triton JIT) and its output can differ from the
steady state. A single post-change run that looks like a regression has twice
been a cold cache here.

**Check whether a failure predates you.** `git stash` the change and re-run.
Several suites in this tree hang or fail on their own; attributing those to a
change under test wastes a day.

## What is deliberately not measured here

SGLang and TRT-LLM are read for implementation approach (see `../xref/`) but not
benchmarked; only vLLM is run as a live reference. The hybrid checkpoint cannot
start in this container at all — flashinfer's xqa kernel fails to JIT — so every
number here is a dense model.
