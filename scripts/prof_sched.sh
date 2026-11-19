#!/bin/bash
# sched-host agent: GPU-only trace of UnieInfra (.wt-sched build, BV-TP + prefill TBO production env) under AgentX stress at c$1.
# Usage: prof_sched.sh <C> <label> [ENV=VAL ...]   extra env applied to the server. Trace -> logs/prof-sched-<label>-c<C>-<tag>
cd /home/smc/workspace/roy; C=${1:-32}; LABEL=${2:?label}; shift 2
LOG=logs/prof_sched.log; LOCK=logs/GPU_LOCK; TAG=$(date -u +%m%d%H%M)
for kv in "$@"; do export "$kv"; done
until ( set -o noclobber; echo "owner=sched-agent(prof_sched $LABEL c$C) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 30; done
trap 'rm -f $LOCK' EXIT
OUT=logs/prof_sched.round.$LABEL.out
STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
ENGINE_PYTHONPATH=/workspace/.wt-sched/tokenspeed-kernel-amd/python:/workspace/.wt-sched/tokenspeed-kernel/python:/workspace/.wt-sched/python \
CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*C)) MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
scripts/run_round_sched.sh PSC$LABEL$C 300 unieinfra "$C" > $OUT 2>&1 &
RR=$!
until grep -q "running agentx\|FAILED" $OUT; do sleep 5; done
if grep -q FAILED $OUT; then echo "$(date -u +%T) $LABEL c$C SERVER FAILED" >> $LOG; wait $RR; exit 1; fi
sleep 240
echo "$(date -u +%T) profile $LABEL c$C tag=$TAG env=[$*]" >> $LOG
[ -z "$NOPROF" ] && curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\": \"/logs/prof-sched-$LABEL-c$C-$TAG\", ${PROF_EXTRA:-\"activities\": [\"GPU\"]}}" >> $LOG; echo >> $LOG
sleep ${PROF_SECS:-6}; [ -z "$NOPROF" ] && curl -s -X POST localhost:8000/stop_profile >> $LOG; echo >> $LOG
wait $RR
cp logs/unieinfra-server.log logs/prof_sched.server.$LABEL.log 2>/dev/null
echo "PROF DONE $LABEL c$C $(date -u +%FT%TZ) dir=logs/prof-sched-$LABEL-c$C-$TAG" >> $LOG
