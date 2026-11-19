#!/bin/bash
# After the baseline round finishes, rerun points that failed for config reasons.
until grep -q "ROUND baseline COMPLETE" /home/smc/workspace/roy/logs/round-baseline.log; do sleep 30; done
cd /home/smc/workspace/roy && scripts/run_round.sh baseline-rerun 900 "vllm" "4"
