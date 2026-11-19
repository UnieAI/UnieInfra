#!/bin/bash
# GPU+CPU profile of UnieInfra (BV + prefill TBO, 6361be94) under AgentX stress load at c$1, 6 s window ~4 min into the run.
cd /home/smc/workspace/roy; C=${1:-32}; LOG=logs/prof_agentx.log; LOCK=logs/GPU_LOCK; TAG=$(date -u +%m%d%H%M)
until ( set -o noclobber; echo "owner=roy-d8(prof_agentx c$C) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
ENGINE_PYTHONPATH=/workspace/.wt-tbo2/python TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*C)) \
MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
scripts/run_round.sh PRT$C 300 unieinfra "$C" > logs/prof_agentx.round.out 2>&1 &
RR=$!
until grep -q "running agentx" logs/prof_agentx.round.out; do sleep 5; done
sleep 240
echo "$(date -u +%T) profile c$C tag=$TAG" >> $LOG
curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\": \"/logs/prof-agentx-tbo-c$C-$TAG\", \"activities\": [\"GPU\"]}" >> $LOG; echo >> $LOG
sleep 6; curl -s -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
wait $RR
echo "PROF DONE c$C $(date -u +%FT%TZ) dir=logs/prof-agentx-tbo-c$C-$TAG" >> $LOG
