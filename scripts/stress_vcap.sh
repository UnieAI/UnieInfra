#!/bin/bash
# Fairness check: vLLM c4 with MAX_NUM_SEQS 32 (UnieInfra c4 cap32 51.9 RPM vs cap8 44.3).
cd /home/smc/workspace/roy; M=logs/stress_vcap.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_vcap) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
echo "$(date -u +%T) start vllm c4 cap32" >> $M
STRESS=1 VLLM_MAX_NUM_SEQS=32 scripts/run_round.sh SVC4c32 600 vllm 4
echo "$(date -u +%T) done; STRESS VCAP DONE" >> $M
