#!/usr/bin/env bash
# Official 3600 s sweep, c8..c128, slot budget = CONC/4 (floor 2), one sampler for all.
# Waits for the v16 probe (queue5) so the sampler choice is decided by data.
set -u
cd /home/smc/workspace/roy
LOG=logs/queue6.log
SAMPLER=${SAMPLER:-triton_full}
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
unset TOKENSPEED_KERNEL_VERBOSE

# gate on the probe
for i in $(seq 1 200); do
  grep -q QUEUE5B_DONE logs/queue5b.log 2>/dev/null && break
  sleep 15
done
if ! grep -q QUEUE5B_DONE logs/queue5b.log 2>/dev/null; then say "probe never finished; abort"; exit 1; fi

export GPU_MEMORY_UTILIZATION=0.95
export MAX_TOTAL_TOKENS=33554432

run_cell(){ # tag conc slots
  local tag=$1 conc=$2 slots=$3
  export MAX_NUM_SEQS=$slots
  export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend $SAMPLER"
  say "cell $tag: c$conc, $slots slots, d3600, $SAMPLER"
  bash scripts/run_round.sh "$tag" 3600 unieinfra "$conc"
  say "cell $tag finished rc=$?"
}

run_cell o64 64  16
export MAX_TOTAL_TOKENS=25165824
run_cell o128 128 32
export MAX_TOTAL_TOKENS=33554432
run_cell o32 32  8
run_cell o16 16  8
run_cell o8  8   4
run_cell o4  4   4
run_cell o1  1   2
say QUEUE6_DONE
