#!/bin/bash
# After the final matrix: rerun SGLang c128 (first run hit the AiPerf timeout), then clean up.
cd /home/smc/workspace/roy
until grep -q "FINAL MATRIX DONE" logs/round-final.log 2>/dev/null; do sleep 30; done
scripts/run_round.sh final 900 "sglang" "128"
scripts/post_matrix_cleanup.sh
