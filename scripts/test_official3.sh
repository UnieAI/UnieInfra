#!/bin/bash
# Official-recipe sweep, part 3: two single-knob A/Bs for our engine first, then
# the remaining reference pairs from part 2.
#
# Why the A/Bs lead: read the c64 cap16 official cell's scheduler log and the
# hour splits into ~2.7-4.2K prefill rounds (~150 ms each, ~10-20% of the time)
# and ~28.7K decode rounds at ~110 ms each (~80-90%). The decode lines report
# #running-req 14.5 of 16 and the queue averages 45, so the slots are full and
# the engine is decode-bound -- the 1.2 s "closing extend" a throttled log made
# look like a fixed per-request cost is really the decode rounds that fit before
# the next log line. What still needs measuring is how much of the prefill share
# can be folded into decode: outside mixed mode a prefill round runs instead of
# decode, never beside it, and this config never emits a "Mix batch." line, so
# interleaving has to come from the scheduler:
# TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 forces a decode round between prefills.
#
# The second A/B targets the cost inside those rounds. Round duration splits on
# the extend size, not on context: extends of exactly 4096 tokens (1263 rounds,
# 31K cached tokens on average) take 157 ms, while every smaller extend takes
# ~1.45 s whether it is 128 or 2985 tokens, at 30K or 100K cached tokens. The
# prefill CUDA graph is what can differ across that line: get_prefill_token_buckets
# ladders up to prefill_graph_max_tokens, which defaults to 2048
# (model_executor.py:183), so 4096-token extends run eager and every smaller
# extend replays a captured bucket. Turning the graph off makes all of them eager.
#
# Remaining cells are exactly part 2's queue minus the vLLM c64 cell, which part 2
# finished before this script took over.
set -uo pipefail
ROOT=/home/smc/workspace/roy
LOG=$ROOT/logs/official.log
mkdir -p "$ROOT/logs"

note() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

ab() { # tag conc cap  (scheduling/serve env comes from the caller's environment)
  local tag=$1 conc=$2 cap=$3
  note "OFFICIAL A/B $tag c$conc cap$cap (chunk=${CHUNKED_PREFILL_SIZE:-4096} DECODE_EVERY_PREFILL_ROUNDS=${TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS:-0} EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-})"
  MAX_NUM_SEQS=$cap "$ROOT/scripts/run_round.sh" "$tag" official unieinfra "$conc" >> "$LOG" 2>&1
  note "OFFICIAL A/B $tag c$conc done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"
}

cell() { # engine conc cap
  local eng=$1 conc=$2 cap=$3
  local tag=ref
  [[ "$eng" == unieinfra ]] && tag=off$cap
  note "OFFICIAL $eng c$conc cap$cap"
  MAX_NUM_SEQS=$cap TAG=$tag "$ROOT/scripts/run_round.sh" "$tag" official "$eng" "$conc" >> "$LOG" 2>&1
  note "OFFICIAL $eng c$conc cap$cap done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"
}

# Attribution check, and it comes first because everything else leans on it.
# "Decode batch." lines are throttled to one per 40 rounds (--decode-log-interval),
# so a prefill round followed by a decode batch shows that batch inside its own
# measured gap: a closing extend looks like ~1.2s while a full 4096 chunk looks
# like 152ms, purely because the small extends are followed by a decode line. Log
# every decode round and the split is exact, with no censoring left to argue about.
EXTRA_SERVE_ARGS="--decode-log-interval 1" ab declogoff16 64 16

# Round cost, part 3, and it runs second because it is the cheap version of
# packing that needs no rebuild. The scheduler stops admitting behind a request
# that is still prefilling (holdsHeadOfLine), and the only pass that packs several
# prompts into one round (scheduleShortPromptsAhead) needs a Submitted prompt
# that fits in one chunk whole -- 4096 at most, while this dataset's prompts
# are 50K-750K. So every round carries one extend, and a round of 1.5K tokens
# costs a full 1.45s. chunked_prefill_size also sets max_scheduled_tokens
# (event_loop.py:326), so 8192 lets a round carry two chunks' worth.
CHUNKED_PREFILL_SIZE=8192 ab chunk8off16 64 16

EXTRA_SERVE_ARGS="--disable-prefill-graph" ab nopfgoff16 64 16
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 ab decoff16 64 16

cell vllm 32 8
cell unieinfra 32 8
cell vllm 128 128
# c128 next, out of order on purpose: it is the decisive row for the goal and the
# most expensive one (~2.5-3 h: 67 + 10x128 warmup replays, each a long-context
# prefill), so it should not wait behind eleven 1 h cells. cap is the recipe's
# CONC/4 = 32, the same slot budget vLLM gets at c128, not the 128 that the c64
# screening used. The KV pool gives back ~8 GiB for the prefill workspace here
# because c32 already proved 33.5M cache tokens leaves no room (see below).
note "OFFICIAL unieinfra c128 cap32 (MAX_TOTAL_TOKENS=25165824)"
MAX_TOTAL_TOKENS=25165824 cell unieinfra 128 32
note "OFFICIAL c128 done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"

# The c128 row decides the headline, but every scheduling lever is closed and the
# only measured fact about a long-context decode step is its wall time, while the
# static traffic estimate for the sparse path is 1-2 orders of magnitude smaller.
# A per-kernel trace of the losing cell's shape (c64 cap16, 150 K prefix) costs
# ~20 min of GPU and names the gap, so it takes a window before the 1 h ladder
# resumes.
note "TRACE window: rocprofv3 per-kernel decode profile at c64 cap16 / 150K"
"$ROOT/scripts/trace_round.sh" 16 150000 10 >> "$LOG" 2>&1 || true
note "TRACE window done: $(ls -dt "$ROOT"/results/trace/decode-* 2>/dev/null | head -1)"
# Order revised 21:25Z. The vLLM c128 cell still running needed 1347 warmup
# replays (67 primers + 10 per lane over 128 lanes, each a full long-context
# prefill, max_tokens=1) and was still in warmup 65 min in at ~11 req/min, so a
# c128 cell costs ~2.5-3 h per engine rather than the ~1 h of every other rung.
# The cheap rungs and the c32 rerun therefore go first and c128 goes last, so the
# c1-c32 ladder closes before the GPU disappears into a 4 h cell.
cell vllm 16 4
cell unieinfra 16 4
cell unieinfra 16 8

# c8 was the one concurrency with no vLLM point at its own recipe (cap 2), so
# the ladder c1-c128 was missing both halves of one row.
cell vllm 8 2
cell unieinfra 8 2
cell vllm 4 2
cell unieinfra 4 2
cell vllm 1 1
cell unieinfra 1 1
note "OFFICIAL SWEEP c1-c32 COMPLETE"

# c32 was lost at 19:39:57Z to an engine OOM, not to the harness: the scheduler
# exited with "Tried to allocate 952.00 MiB. GPU 0 has a total capacity of 143.98
# GiB of which 0 bytes is free" inside _forward_prefill_batched, sizing one bf16
# workspace for the ~950K-token global history of a single long prompt. Its
# leftovers then held port 8000 (smg::router, a name the stop pattern did not
# match) and 127.7 GB of HBM (an unreaped tokenspeed::scheduler_tp0), so the vLLM
# c128 cell that followed could not bind and the round waited out its 3000s
# readiness budget. Both causes are fixed (bracketed stop patterns plus a
# kill-by-pid sweep in stop_servers.sh); the KV pool also gives back ~8 GiB here.
note "RERUN c32 cap8 with MAX_TOTAL_TOKENS=25165824 (OOM fix)"
MAX_TOTAL_TOKENS=25165824 TAG=off8 "$ROOT/scripts/run_round.sh" off8 official unieinfra 32 >> "$LOG" 2>&1
note "RERUN c32 done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"

# Fairness control: the reference cells above run the recipe's compressed sparse
# indexer (--attention-config with indexer_kv_dtype=mxfp4 and mxfp4 sparse logits).
# UnieInfra cannot match that without quantizing, and the goal forbids
# quantization, so the reference is re-run at c64 with the indexer at the engine's
# own dtypes. The difference between the two vLLM cells prices how much of its lead
# comes from that compression alone.
note "OFFICIAL A/B noiqref c64 cap16 (vLLM c64 with VLLM_NO_INDEXER_QUANT=1)"
VLLM_NO_INDEXER_QUANT=1 TAG=noiqref "$ROOT/scripts/run_round.sh" noiqref official vllm 64 >> "$LOG" 2>&1
note "OFFICIAL A/B noiqref c64 done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"
note "OFFICIAL SWEEP + FAIRNESS CONTROL COMPLETE"

# decoff16 was the only positive lever of 2026-11-10 (+2.9% RPM, +5.3% output
# tok/s, TTFT p90 -8% vs off16) yet the day's cells drift by +-3%, so repeat it
# once to separate signal from drift.
note "REPEAT decoff16b c64 cap16 (noise check)"
TAG=decoff16b DECODE_EVERY_PREFILL_ROUNDS=1 "$ROOT/scripts/run_round.sh" decoff16b official unieinfra 64 >> "$LOG" 2>&1
note "REPEAT decoff16b done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"

note "ALL DONE"
