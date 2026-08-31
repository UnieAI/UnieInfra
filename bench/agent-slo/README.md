# Agent / SLO benchmark — reproduce it yourself

Everything here runs on public datasets. Nothing depends on our traffic.

## What this measures

A multi-turn agent client resends the whole conversation each turn, so turn N+1's
prompt **contains turn N's prompt as an exact prefix**. Serving that well is a
prefix-cache problem, and the engines differ there far more than they differ on
decode. This harness measures that difference without letting the benchmark
hand either engine a free hit.

## Why the protocol is shaped like this

**A concurrency ladder run on one server measures replay, not reuse.** If c=1,
4, 8, 16… all send the same prompts to the same process, every cell after the
first is serving a prompt it has *already served verbatim*. Measured here: three
passes over the same 112 prompts moved the prefix hit rate

    0% → 59.5% → 71.3% → 81.5% → 90.7% → 93.0% → 94.4% → 95.2%

and the same cell's wall time by **51.7%**. So:

1. **One fresh server per cell.** No state crosses cells.
2. **Warm the engine on a different corpus.** `run_matrix.sh` refuses to start
   if the warmup and measurement corpora share more than 256 characters of
   prefix — the two shipped corpora share 9. Warmup covers autotune, CUDA graph
   capture and first-touch allocation, which a real server pays once at startup.
   It must not preload the cache with the prompts about to be timed.
3. **Run the target corpus once.** The only reuse left is the corpus's own: the
   same session's earlier turns, plus whatever preamble sessions share.

## The result depends on the corpus, and by a lot

The ratio between two engines is a function of **working set ÷ cache capacity**,
and a corpus sets the working set through two properties: how many sessions
interleave, and how much they share. gemma-4-31B on one RTX PRO 6000, same
protocol, same parameters, ours against upstream vLLM:

| corpus | sessions | shared across sessions | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| `swe_smith_agent_sess32x8` | 32 | 12.5% | 1.00x | 1.13x | 1.74x |

At c=8 both engines sit at a ~9% prefix hit rate — 32 sessions round-robin means
about 310k tokens are written between one session's consecutive turns, past both
engines' capacity, and neither gets to reuse anything. As concurrency rises the
same session's turns overlap in flight instead of queueing behind 31 others, the
effective pressure drops, and the engine with more cache converts that into hits
(ours 9.2% → 15.9% → 49.3%; vLLM stalls at 1.0%).

**Report the corpus and the concurrency with any number from this harness.** A
single ratio without them is not a claim about anything.

## Build the corpora

```
pip install pyarrow transformers
hf download nebius/SWE-agent-trajectories --repo-type dataset \
   --include "data/train-00000-of-00012.parquet" --local-dir ./raw/nebius
hf download SWE-bench/SWE-smith-trajectories --repo-type dataset \
   --include "data/train-00000-of-00008.parquet" --local-dir ./raw/swesmith

mkdir -p corpora
python3 build_corpus.py nebius   ./raw/nebius/data/train-00000-of-00012.parquet 16 16 12500 \
        corpora/swe_agent_nebius_sess16x16.jsonl
python3 build_corpus.py swesmith ./raw/swesmith/data/train-00000-of-00008.parquet 32 8 12500 \
        corpora/swe_smith_agent_sess32x8.jsonl
python3 corpus_stats.py corpora/*.jsonl
```

Licences: `nebius/SWE-agent-trajectories` CC-BY-4.0, `SWE-bench/SWE-smith-trajectories` MIT.

Length is capped by **choosing an earlier turn**, never by truncating text.
Truncation makes different sessions collide on identical strings and hands the
cache free hits. Sessions are deduplicated by repo, which also drops the
same-instance rollouts whose first user message is byte-identical.

## Run

```
MODEL=/path/to/model ENGINE=ours GPU=0 CONCS="8 16 32" ./run_matrix.sh
MODEL=/path/to/model ENGINE=vllm GPU=0 CONCS="8 16 32" ./run_matrix.sh
```

Output is one JSON per cell in `results/`, with per-request TTFT, TPOT, token
count and **the generated text**. Keep the text: an engine that corrupts output
under load looks healthy in latency alone. We shipped a defect where the corrupt
arm scored **+25.9% tok/s** because it emitted longer garbage.

`drive.py` pins output at 128 tokens with `ignore_eos` so both engines do
identical work. That has a cost worth knowing: at ~50 ms TPOT it makes decode
~58% of the wall time on these shapes, and decode is at parity between engines,
which caps the ratio any agent benchmark of this shape can show.

## Isolate the reuse path

```
python3 reuse_probe.py <port> <tag> [words] [gap_seconds]
```

Sends one prompt cold, then the same prompt three times, and reports **each
re-send separately**. The first re-send is the one that matters — in an agent
workload every reuse is a first reuse, because turn N+1 resumes turn N's tail
exactly once. A median over the hot sends hides this: it is how a defect that
cost 800 ms on every single turn stayed invisible while the prefix hit rate
reported 93–95%. Hit rate counts tokens marked as hits; it does not count work
avoided.
