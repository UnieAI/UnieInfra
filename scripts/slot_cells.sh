#!/usr/bin/env bash
# Chained behind scripts/qcp_cells.sh. Slot scaling is the gap that matters:
# per-slot throughput is already ~3.6x vLLM's (60.5 RPM / 16 slots vs 66.8 / 64
# slots at c64), so every extra concurrent slot is nearly free RPM. These cells
# raise MAX_NUM_SEQS at fixed concurrency and keep the known-good Triton MoE so
# the only variable is slot count.
set -u
cd /home/smc/workspace/roy
echo "== $(date -u +%FT%TZ) waiting for QCP_CELLS_DONE"
for _ in $(seq 1 480); do
  grep -q QCP_CELLS_DONE logs/qcp_cells.log 2>/dev/null && break
  sleep 15
done
grep -q QCP_CELLS_DONE logs/qcp_cells.log || echo "!! QCP chain never finished; starting anyway"

export TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY=triton
export GPU_MEMORY_UTILIZATION=0.95

export MAX_NUM_SEQS=32
echo "== $(date -u +%FT%TZ) cell s64c32: c64, 32 slots, triton"
bash scripts/run_round.sh s64c32 900 "unieinfra" "64"

export MAX_NUM_SEQS=64
echo "== $(date -u +%FT%TZ) cell s64c64: c64, 64 slots, triton"
bash scripts/run_round.sh s64c64 900 "unieinfra" "64"

# c128 keeps the fair slot budget of CONC/4 = 32.
export MAX_NUM_SEQS=32
echo "== $(date -u +%FT%TZ) cell s128c32: c128, 32 slots, triton"
bash scripts/run_round.sh s128c32 900 "unieinfra" "128"

echo SLOT_CELLS_DONE
