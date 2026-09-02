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

## What each number measures

Every cell drives `N` requests at concurrency `c` against one fresh server and
records per request: time to first token, time per output token, chunk count,
end-to-end time, and the generated text.

| column | definition | what it answers |
|---|---|---|
| `dur` | wall-clock seconds to finish all `N` requests | **the primary quantity.** Every ratio in this README is `vLLM dur / ours dur`, because it is the only number that cannot be gamed by counting differently |
| `rpm` | `60 × N_ok / dur` | service rate — how many requests per minute this GPU retires |
| `tps` | `Σ ntok / dur` | server-side output throughput, summed over all requests in flight |
| `tps/req` | `mean(ntok / e2e)` | what **one** user feels — a server at 600 tps with 64 users gives each of them ~9 |
| `TTFT p50 / p99` | ms from request sent to first token | the wait before anything appears. p99 is the one that shows queueing: a server can hold a fine median while a tail waits 60 s |
| `TPOT p50 / p99` | ms per output token after the first | typing speed. `1000 / TPOT` ≈ `tps/req` once generation dominates |
| `hit` | cumulative prefix-cache hit rate, `cached_tokens / prompt_tokens` | how much prompt was reused rather than recomputed. **Reports tokens marked as hits, not work avoided** — see the reuse probe below |
| `ntok` | SSE chunks carrying non-empty text | a proxy for output tokens. Our engine sometimes packs more than one token into a chunk (measured: 506 chars in 114 chunks against vLLM's 429 in 128), so `tps` and `tps/req` are **understated for us**. `dur`, `rpm`, TTFT and TPOT are unaffected. **This is the column that catches silent corruption**: a KV-transfer defect on gemma-4 held every file-backed arm at 123-126 while every RAM-backed arm sat at 115-117, and nothing else in the run differed |
| `bad` | character-repetition OR repeated 6-gram within 40 words | a broken engine looks healthy in latency alone — but **under `ignore_eos` this counter decides nothing, not even between engines**. Both are forced past their natural stop, so both loop, and the counts differ by which one starts first. Measured on Qwen3.8-27B: 101 flagged for us against 88 for vLLM, but **42 of vLLM's flags were requests we did not flag at all**, and one of the 55 we flagged alone had output identical to vLLM's over its first 180 characters. Treat a count difference as noise; only a *split* pattern means anything — see `ntok` |

**Which are SLO axes and which are capacity axes.** TTFT and TPOT are what a user
experiences and what an SLO is written against. `rpm` and `tps` are what the box
delivers. They trade off: pushing concurrency raises `rpm` and lowers `tps/req`,
and past the point where the cache stops holding the working set it lowers both.

**Reading a row honestly.** `dur` alone can hide a corrupt engine, a shorter
answer, or a different token count, so always read `ntok` from both engines in
the same cell before comparing anything else. `bad` is not usable for that under
`ignore_eos`; use the byte round-trip in `tests/l2/` when correctness is the
question.

**Do not use a cold/hot model comparison to clear a transfer path.** This engine
is not deterministic at `temperature: 0` even with one request in flight and no
cache at all: gemma-4-31B returned **1.40 distinct outputs per prompt** over 5
prompts × 4 repeats, and Qwen3.8-27B matched on only 8 of 12 with zero cached
tokens on the second pass. A cold/hot check therefore has a floor of 30-40%
disagreement, and a passing score is a draw from that population rather than
evidence. The byte round-trip does not run the model, and it was shown to fail
without the fix it guards.

## Before reading any hit rate: what reuse does the corpus contain?

A hit rate has no meaning without a denominator. `reuse_ceiling.py` computes it
from the corpus alone, no server involved:

```
python3 reuse_ceiling.py corpus.jsonl
```

It reports two numbers:

- **reuse ceiling** — the fraction of prompt tokens that are a page-aligned
  prefix of an earlier turn in the same session. No cache can beat it.
- **reuse distance** — how many other requests are issued between a session's
  consecutive turns. This is what the cache has to survive. If the KV pool holds
  fewer sessions than the distance spans, the reuse in the corpus is unreachable
  and the hit rate collapses no matter how good the matching logic is.

The two corpora used here:

| corpus | sessions × turns | prompt tokens | ceiling | reuse distance |
|---|---|---|---|---|
| `swe_agent_nebius_sess16x16` | 16 × 16 | 2,066,695 | 91.0% | 16 |
| `swe_smith_agent_sess32x8` | 32 × 8 | 2,503,295 | 84.8% | 32 |

Both are round-robin interleaved — every session's turn *k* is issued before any
session's turn *k+1* — which is what a coding-agent provider actually sees, and
it is the shape that makes the working set, not the individual prompt, the thing
the cache has to hold.
