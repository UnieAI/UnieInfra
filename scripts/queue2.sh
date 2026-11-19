#!/usr/bin/env bash
# Coalesce isolation queue. Baseline to beat is repait-c64 (aiter, cap16,
# coalesce 0, same 900 s agentx profile): 39.5 RPM / 576 out/s.
# The 11-10 04:15 c64 cell logged --prefill-coalesce-ms 100 (+triton_full) and
# reached 60.5 RPM, so this isolates which of the two does the work before
# spending slots on it.
set -u
cd /home/smc/workspace/roy
LOG=logs/queue2.log
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }

export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=16
export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100"
say "cell co16a: c64, 16 slots, coalesce 100 only"
bash scripts/run_round.sh co16a 900 unieinfra 64

export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full"
say "cell co16b: c64, 16 slots, coalesce 100 + triton_full"
bash scripts/run_round.sh co16b 900 unieinfra 64

export MAX_NUM_SEQS=32
say "cell co32: c64, 32 slots, coalesce 100 + triton_full"
bash scripts/run_round.sh co32 900 unieinfra 64

say QUEUE2_DONE
