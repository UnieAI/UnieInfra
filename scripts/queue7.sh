#!/usr/bin/env bash
# Fill the holes left by queue6: o32 died of "server disconnected" during warmup,
# so c32 has no official 3600 s row. Runs after queue6 finishes.
set -u
cd /home/smc/workspace/roy
LOG=logs/queue7.log
SAMPLER=${SAMPLER:-triton_full}
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
unset TOKENSPEED_KERNEL_VERBOSE

for i in $(seq 1 400); do
  grep -q QUEUE6_DONE logs/queue6.log 2>/dev/null && break
  sleep 15
done
if ! grep -q QUEUE6_DONE logs/queue6.log 2>/dev/null; then say "queue6 never finished; abort"; exit 1; fi
if [ -f logs/skip-o32b ]; then say "skip-o32b -> SKIP o32b"; exit 1; fi

export GPU_MEMORY_UTILIZATION=0.95
export MAX_TOTAL_TOKENS=33554432
export MAX_NUM_SEQS=8
export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend $SAMPLER"
say "cell o32b: c32, 8 slots, d3600, $SAMPLER"
bash scripts/run_round.sh o32b 3600 unieinfra 32 >> $LOG 2>&1
say "cell o32b finished rc=$?"
say QUEUE7_DONE
