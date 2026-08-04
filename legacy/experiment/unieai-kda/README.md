# UnieAI KDA

An AutoML-shaped loop for kernel and serving-path optimization on UnieAI's
hardware: propose a candidate, evaluate it behind hard gates, keep it only if
the evidence survives, remember why.

Built on top of [Kernel Design Agents](../../third_party/kernel-design-agents),
which supplies the workflow; this project supplies the parts a KDA task
workspace is supposed to own — the search space, the strategies, the evaluator,
and the memory.

## The thesis: the evaluator is the hard part

The obvious reading of the LLM-kernel-generation literature is that the
interesting work is in the search — evolutionary operators, multi-turn RL,
agent teams. That is where most of the named systems put their effort
(EvoEngineer, Kevin, CUDA-L1, Astra, STARK, cuPilot...).

Our own experience says otherwise. In one session of manual optimization work on
sm_120, **three separate headline conclusions were wrong**, and none of them was
wrong because of a bad idea:

| claim | reality | cause |
|---|---|---|
| "speculative decoding loses 42-66% everywhere" | it **gains 27-45%** | the harness counted SSE chunks as tokens |
| "UnieInfra's kernel layers cost 5.19%" | they cost **0.03%** | one arm had not reached steady state |
| "gemma-4 has no working attention backend" | it **works fine** | correctness judged against expectation, not a reference |

All three passed a significance test. Two were flagged by instrumentation that
already existed and was not read. A search loop running unattended would have
optimized hard against every one of them.

So the design centre of gravity here is `kda/gates.py`, not `kda/search/`. A
strategy that proposes mediocre candidates wastes GPU hours. An evaluator that
returns wrong numbers wastes the whole project and produces confident,
well-documented, false results.

## Loop

```
   contract ──> propose ──> build ──> CORRECTNESS GATE ──> measure
                   ^                        │                 │
                   │                        ✗ reject          ▼
                   │                                    STABILITY GATES
                   │                                          │
                   └────────── memory <── promote/reject <─────┘
```

## Gates

Every gate exists because something got past its absence.

| gate | rejects | origin |
|---|---|---|
| `reference_match` | output judged against expectation rather than a known-good path | gemma-4 retraction |
| `stationary` | either arm still drifting; comparison void regardless of sigma | kernel-layer retraction |
| `period_blocked` | window means on a signal with a period-2 limit cycle | 50% swing at 90% KV utilisation |
| `counter_agreement` | client-side token counts that disagree with the server's | speculative-decoding retraction |
| `switch_engaged` | a flag that was silently ignored | 3 separate flags in one session |
| `attribution` | an effect quoted without model, concurrency and shape | none of ours transfer across all three |

## Strategies

Pluggable, selected by a bandit over observed yield per GPU-hour. See
[`docs/strategies.md`](docs/strategies.md).

- `refine` — profile, diagnose against the ncu playbook, apply one change,
  re-measure. Serial, high hit rate, low ceiling.
- `evolve` — mutate a population of variants over a parameterised space
  (tiling, launch geometry, fusion boundaries, backend selection).
- `sweep` — exhaustive over a small discrete space. Cheap, and the right answer
  more often than it looks: several of our wins were "this table has no entry
  for sm_120".
- `port` — take a kernel qualified on another architecture and re-qualify it
  here. Most of UnieInfra's existing kernels were qualified on batch-1 8B
  decode and are inert at 30B; re-qualification is a large, unexplored space.

## Status

Skeleton. The evaluator and gates are the parts to trust first; strategies are
deliberately simple until the evaluator has been shown to reject known-bad
candidates. See [`docs/design.md`](docs/design.md).
