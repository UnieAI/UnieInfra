#!/usr/bin/env bash
# Fill the two remaining holes in the official 3600 s sweep: c32 (o32 died in
# warmup, o32b never ran) and c4 (o4 was killed by the profiler's
# stop_servers.sh), plus c1 (skipped while the GPU was needed for profiling).
# Starts only after the long-context decode profile has released the GPU.
set -u
cd /home/smc/workspace/roy
LOG=logs/queue8.log
SAMPLER=${SAMPLER:-triton_full}
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
unset TOKENSPEED_KERNEL_VERBOSE

# The profiler script waits for logs/GPU_LOCK, so it may finish long after its
# own START line. Follow the marker it writes last.
for i in $(seq 1 720); do
  grep -q PROFDEC_DONE logs/profdec.log 2>/dev/null && break
  sleep 15
done
if ! grep -q PROFDEC_DONE logs/profdec.log 2>/dev/null; then say "profile never finished; abort"; exit 1; fi
# Do not start while a foreign chain still owns the GPU.
for i in $(seq 1 60); do
  [ -e logs/GPU_LOCK ] || break
  sleep 10
done
if [ -e logs/GPU_LOCK ]; then say "GPU_LOCK still held by another chain; abort"; exit 1; fi

# These two cells were deferred, not retired.
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
say QUEUE8_DONE
