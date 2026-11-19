#!/usr/bin/env bash
# One clean probe: same config as the unreproducible 60.5 RPM cell of 11-10,
# the only difference being TOKENSPEED_KERNEL_VERBOSE unset (it wrote ~15K
# synchronous log lines inside every recent cell).
set -u
cd /home/smc/workspace/roy
LOG=logs/queue5.log
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
unset TOKENSPEED_KERNEL_VERBOSE
export GPU_MEMORY_UTILIZATION=0.95
export MAX_NUM_SEQS=16
export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full"
say "cell v16: c64, 16 slots, auto moe + coalesce 100 + triton_full, VERBOSE off"
bash scripts/run_round.sh v16 900 unieinfra 64
say QUEUE5_DONE
