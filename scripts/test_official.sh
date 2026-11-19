#!/bin/bash
# Official-recipe sweep: every cell runs the agentx preset exactly as documented
# in inferencex-e2e/docs/agentx-standalone.md -- one hour of profiling and ten
# warmup requests per lane (aiperf preset: default_benchmark_duration_seconds
# 3600, warmup_requests_per_lane 10, random_seed 42, streaming). No
# --benchmark-duration and no --warmup-requests-per-lane override is passed, and
# --ignore-trace-delays is never used, so every cell is a submission-valid run.
#
# Each engine gets the same window, dataset and seed, and the server is
# restarted for every concurrency point (the docs forbid reusing a live server).
# Our cap per concurrency is the one the 900 s screening sweep identified as the
# production frontier point; vLLM keeps its own recipe cap (CONC/4).
set -uo pipefail
ROOT=/home/smc/workspace/roy
LOG=$ROOT/logs/official.log
mkdir -p "$ROOT/logs"

cell() { # engine conc cap
  local eng=$1 conc=$2 cap=$3
  local tag=ref
  [[ "$eng" == unieinfra ]] && tag=off$cap
  echo "=== $(date -u +%FT%TZ) OFFICIAL $eng c$conc cap$cap" | tee -a "$LOG"
  MAX_NUM_SEQS=$cap TAG=$tag "$ROOT/scripts/run_round.sh" "$tag" official "$eng" "$conc" >> "$LOG" 2>&1
  local out
  out=$(grep -o "$ROOT/results/agentx/[^ ]*-dofficial-[0-9TZ]*" "$LOG" | tail -1)
  echo "=== $(date -u +%FT%TZ) OFFICIAL $eng c$conc cap$cap done: $out" | tee -a "$LOG"
}

# Decisive rows first: c64 is where the 900 s sweep said we trail most, c32 next,
# then the wide-concurrency row whose reference is starved of interactivity.
cell vllm 64 16
cell unieinfra 64 16
cell vllm 32 8
cell unieinfra 32 8
cell vllm 128 128
cell unieinfra 128 128
cell vllm 16 4
cell unieinfra 16 4
cell unieinfra 16 8
cell vllm 4 2
cell unieinfra 4 2
cell vllm 1 1
cell unieinfra 1 1
echo "=== $(date -u +%FT%TZ) OFFICIAL SWEEP COMPLETE" | tee -a "$LOG"
