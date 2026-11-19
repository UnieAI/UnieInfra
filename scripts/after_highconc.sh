#!/bin/bash
# Rerun vLLM c32 (failed: first cold load from the SATA /data copy exceeded the 25 min readiness timeout).
until grep -q "ROUND highconc COMPLETE" /home/smc/workspace/roy/logs/round-highconc.log 2>/dev/null; do sleep 30; done
cd /home/smc/workspace/roy && scripts/run_round.sh highconc-rerun 900 "vllm" "32"
