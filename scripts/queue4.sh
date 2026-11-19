#!/usr/bin/env bash
# triton_full is the fair sampler (AgentX sends temperature=1.0; the ROCm
# default 'greedy' backend ignores it and, empirically, costs ~35% RPM).
# Sweep slot count with the vLLM-identical aiter MoE at that sampler.
set -u
cd /home/smc/workspace/roy
LOG=logs/queue4.log
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
for _ in $(seq 1 400); do
  grep -q QUEUE3_DONE logs/queue3.log 2>/dev/null && break
  sleep 15
done

export GPU_MEMORY_UTILIZATION=0.95
export EXTRA_SERVE_ARGS="--moe-backend aiter --prefill-coalesce-ms 100 --sampling-backend triton_full"
export TOKENSPEED_KERNEL_VERBOSE=1

export MAX_NUM_SEQS=16
say "cell tf16: c64, 16 slots, aiter + coalesce + triton_full"
bash scripts/run_round.sh tf16 900 unieinfra 64

export MAX_NUM_SEQS=32
say "cell tf32: c64, 32 slots"
bash scripts/run_round.sh tf32 900 unieinfra 64

export MAX_NUM_SEQS=64
say "cell tf64: c64, 64 slots"
bash scripts/run_round.sh tf64 900 unieinfra 64

export MAX_NUM_SEQS=32
say "cell tf128: c128, 32 slots"
bash scripts/run_round.sh tf128 900 unieinfra 128

say QUEUE4_DONE
