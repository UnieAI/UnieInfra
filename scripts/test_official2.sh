#!/bin/bash
# Official-recipe sweep, part 2. The first attempt lost its opening vLLM c64 cell
# to a scripting bug (`local` at script level plus `set -u` arithmetic on the
# literal "official"), so this script picks up exactly where the surviving queue
# left off: our c64 half runs under the previous queue, and vLLM's c64 half leads
# here so the decisive pair is complete before the cheaper rows.
#
# Every cell runs the agentx preset exactly as documented in
# inferencex-e2e/docs/agentx-standalone.md: one hour of profiling, ten warmup
# requests per lane, seed 42, recorded inter-turn delays kept. A fresh server per
# concurrency point, and no reuse of a live server across points.
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

cell vllm 64 16
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
