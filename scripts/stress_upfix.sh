#!/bin/bash
# Upstream correctness fixes (PR #52) on production: A = mi350p-final (.wt-final), B = + fixes (.wt-upb). ABBA per concurrency.
cd /home/smc/workspace/roy; M=logs/stress_upfix.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy(stress_upfix) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() { # $1 A|B $2 conc $3 cap $4 rep
  local WT=/workspace/.wt-final; [ "$1" = B ] && WT=/workspace/.wt-upb
  echo "$(date -u +%T) start $1 c$2 r$4" >> $M
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$3 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh UPF$1c$2r$4 600 unieinfra "$2"
  echo "$(date -u +%T) done $1 c$2 r$4" >> $M
}
for c in 4 64; do cap=$((c==4?32:128)); run A $c $cap 1; run B $c $cap 1; run B $c $cap 2; run A $c $cap 2; done
echo "UPFIX DONE $(date -u +%FT%TZ)" >> $M
