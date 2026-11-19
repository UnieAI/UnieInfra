#!/usr/bin/env bash
# One unattended chain for the GPU once the peer session's pilot releases it:
#   1. kernel-level decode profile at the official c64 shape (bs 16, ~170K)
#   2. the three official 3600 s cells still missing (c32, c4, c1)
# Gated on 'PILOT DONE' in logs/aligned_pilot.log, not on file timestamps.
set -u
cd /home/smc/workspace/roy
LOG=logs/after_pilot.log
SAMPLER=${SAMPLER:-triton_full}
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
unset TOKENSPEED_KERNEL_VERBOSE

for i in $(seq 1 1900); do
  grep -q "PILOT DONE" logs/aligned_pilot.log 2>/dev/null && break
  sleep 15
done
if ! grep -q "PILOT DONE" logs/aligned_pilot.log 2>/dev/null; then say "pilot never finished; abort"; exit 1; fi
for i in $(seq 1 60); do
  [ -e logs/GPU_LOCK ] || break
  sleep 10
done
[ -e logs/GPU_LOCK ] && { say "GPU_LOCK still held: $(cat logs/GPU_LOCK)"; exit 1; }
say "gpu free"

# 1. profile
say "profile: 16 slots, 170K ctx, 400 prof tokens"
bash scripts/prof_decode_c64.sh 16 170000 400
n=$(ls logs/prof-dec-b16/*.trace.json.gz 2>/dev/null | wc -l)
say "profile traces=$n"
[ "$n" -eq 0 ] && say "WARNING: no trace captured; continuing to official cells"

# 2. official cells
rm -f logs/skip-o32b logs/skip-o4 logs/skip-o1
export GPU_MEMORY_UTILIZATION=0.95
export MAX_TOTAL_TOKENS=33554432
run_cell(){ # tag conc slots
  local tag=$1 conc=$2 slots=$3
  export MAX_NUM_SEQS=$slots
  export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend $SAMPLER"
  say "cell $tag: c$conc, $slots slots, d3600, $SAMPLER"
  bash scripts/run_round.sh "$tag" 3600 unieinfra "$conc" >> $LOG 2>&1
  say "cell $tag finished rc=$?"
}
run_cell o32b 32 8
run_cell o4   4  4
run_cell o1   1  2
say AFTER_PILOT_DONE
