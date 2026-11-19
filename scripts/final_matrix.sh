#!/bin/bash
# Fill every missing AgentX cell: c1/c4/c16/c32/c64/c128 for all three engines.
cd /home/smc/workspace/roy
scripts/run_round.sh final 900 "sglang" "32 64 128"
scripts/run_round.sh final 900 "vllm" "64 128"
scripts/run_round.sh final 900 "unieinfra" "1 4 16 32 64 128"
echo "FINAL MATRIX DONE $(date -u +%FT%TZ)" >> logs/round-final.log
