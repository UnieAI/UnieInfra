#!/usr/bin/env bash
# The 2026-11-10 04:15 c64 cell ran at 60.5 RPM with --prefill-coalesce-ms 100
# (+ --sampling-backend triton_full) while every agentx-recipe cell at the same
# concurrency sits at 38-40 RPM. Coalescing lets the scheduler merge the many
# small (96% prefix-hit) extends into fewer, fuller batches, which is exactly
# the shape of this workload. Isolate the flag, then the pair.
set -u
cd /home/smc/workspace/roy
echo "== $(date -u +%FT%TZ) waiting for SLOT_CELLS_DONE"
for _ in $(seq 1 240); do
  grep -q SLOT_CELLS_DONE logs/slot_cells.log 2>/dev/null && break
  sleep 15
done

export GPU_MEMORY_UTILIZATION=0.95
export MAX_NUM_SEQS=16
export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100"
echo "== $(date -u +%FT%TZ) cell co16a: c64, 16 slots, coalesce only"
bash scripts/run_round.sh co16a 900 "unieinfra" "64"

export EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full"
echo "== $(date -u +%FT%TZ) cell co16b: c64, 16 slots, coalesce + triton_full"
bash scripts/run_round.sh co16b 900 "unieinfra" "64"

export MAX_NUM_SEQS=32
echo "== $(date -u +%FT%TZ) cell co32: c64, 32 slots, coalesce + triton_full"
bash scripts/run_round.sh co32 900 "unieinfra" "64"

echo COALESCE_CELLS_DONE
