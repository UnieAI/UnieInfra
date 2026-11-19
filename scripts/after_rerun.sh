#!/bin/bash
# Queue 2: rerun SGLang c1 (its AgentX client exited instantly with an empty log).
until grep -q "ROUND baseline-rerun COMPLETE" /home/smc/workspace/roy/logs/round-baseline-rerun.log 2>/dev/null; do sleep 30; done
cd /home/smc/workspace/roy && scripts/run_round.sh baseline-rerun2 900 "sglang" "1"
