#!/bin/bash
# Official-recipe sweep, part 4 -- the same queue as the tail of part 3, recreated
# as a fresh process.
#
# Why a new file instead of letting part 3 run on: part 3 was mid-cell when its
# tail was rewritten (c128 moved to the front, cap 128 -> 32), and bash reads a
# script by byte offset, so the rewrite made the running process execute a stale
# line -- it launched c128 with cap128, four times the recipe's slot budget. That
# cell was killed 9 s in. Rewriting a script under a running bash is not safe;
# this file is the correction.
#
# Order: c128 first because it is the decisive row for the goal (vLLM fell from
# 66.6 RPM at c64 to 20.2 RPM at c128, so the pair has to be measured at the same
# c128/cap32 recipe), then the kernel trace that names the decode gap, then the
# cheap rungs, then the c32 rerun (OOM fix), the fairness control, the noise check.
set -uo pipefail
ROOT=/home/smc/workspace/roy
LOG=$ROOT/logs/official4.log
mkdir -p "$ROOT/logs"

note() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

cell() { # engine conc cap
  local eng=$1 conc=$2 cap=$3
  local tag=ref
  [[ "$eng" == unieinfra ]] && tag=off$cap
  note "OFFICIAL $eng c$conc cap$cap"
  MAX_NUM_SEQS=$cap TAG=$tag "$ROOT/scripts/run_round.sh" "$tag" official "$eng" "$conc" >> "$LOG" 2>&1
  note "OFFICIAL $eng c$conc cap$cap done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"
}

# c128 is decisive and expensive: 67 + 10x128 warmup replays (each a long-context
# prefill) plus the 3600 s profile puts it at ~2.5-3 h, so it is not queued behind
# eleven 1 h cells. cap is the recipe's CONC/4 = 32, the same slot budget vLLM
# gets at c128; 128 slots would not be a fair row. The KV pool gives back ~8 GiB
# because c32 at 33.5M cache tokens ran out of room for the prefill workspace.
note "OFFICIAL unieinfra c128 cap32 (MAX_TOTAL_TOKENS=25165824)"
MAX_TOTAL_TOKENS=25165824 cell unieinfra 128 32
note "OFFICIAL c128 done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"

# Every scheduling lever is closed and the only measured fact about a 150 K-context
# decode step is its wall time (ITL p90 62.1 ms at cap16 vs vLLM's 18.0 ms with the
# same 3.5-3.7 spec acceptance), while the static traffic estimate for the sparse
# path is 1-2 orders of magnitude smaller. ~20 min of GPU buys the kernel names.
note "TRACE window: rocprofv3 per-kernel decode profile at c64 cap16 / 150K"
"$ROOT/scripts/trace_round.sh" 16 150000 10 >> "$LOG" 2>&1 || true
note "TRACE window done: $(ls -dt "$ROOT"/results/trace/decode-* 2>/dev/null | head -1)"

cell vllm 16 4
cell unieinfra 16 4
cell unieinfra 16 8

# c8 was the one concurrency with no vLLM point at its own recipe (cap 2).
cell vllm 8 2
cell unieinfra 8 2
cell vllm 4 2
cell unieinfra 4 2
cell vllm 1 1
cell unieinfra 1 1
note "OFFICIAL SWEEP c1-c32 COMPLETE"

# c32 was lost at 19:39:57Z to an engine OOM: the scheduler exited with "Tried to
# allocate 952.00 MiB" inside _forward_prefill_batched, sizing one bf16 workspace
# for the ~950K-token global history of a single long prompt, with 0 bytes free.
# Its leftovers then held port 8000 (smg::router) and 127.7 GB of HBM (an unreaped
# tokenspeed::scheduler_tp0), which cost the following vLLM c128 cell two attempts.
# The stop pattern fix is in place; this rerun also hands ~8 GiB back to the pool.
note "RERUN c32 cap8 with MAX_TOTAL_TOKENS=25165824 (OOM fix)"
MAX_TOTAL_TOKENS=25165824 TAG=off8 "$ROOT/scripts/run_round.sh" off8 official unieinfra 32 >> "$LOG" 2>&1
note "RERUN c32 done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"

# Fairness control: the reference cells above run the recipe's compressed sparse
# indexer (indexer_kv_dtype=mxfp4 with mxfp4 sparse logits). UnieInfra cannot match
# that without quantizing, and the goal forbids quantization, so the reference is
# re-run at c64 with the indexer at the engine's own dtypes; the difference prices
# how much of vLLM's lead comes from that compression alone.
note "OFFICIAL A/B noiqref c64 cap16 (vLLM c64 with VLLM_NO_INDEXER_QUANT=1)"
VLLM_NO_INDEXER_QUANT=1 TAG=noiqref "$ROOT/scripts/run_round.sh" noiqref official vllm 64 >> "$LOG" 2>&1
note "OFFICIAL A/B noiqref c64 done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"
note "OFFICIAL SWEEP + FAIRNESS CONTROL COMPLETE"

# decoff16 was the only positive lever of 2026-11-10 (+2.9% RPM, +5.3% output
# tok/s, -8% TTFT p90) but the day's cells drift by +-3%, so repeat it once to
# separate signal from drift.
note "REPEAT decoff16b c64 cap16 (noise check)"
TAG=decoff16b DECODE_EVERY_PREFILL_ROUNDS=1 "$ROOT/scripts/run_round.sh" decoff16b official unieinfra 64 >> "$LOG" 2>&1
note "REPEAT decoff16b done: $(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)"

note "ALL DONE"
