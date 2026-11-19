#!/bin/bash
# Mixed batch + prefill TBO in mixed rounds (.wt-mtbo): GSM8K under constant long-prefill pressure, then stress A/B vs production.
cd /home/smc/workspace/roy; M=logs/verify_mtbo.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(verify_mtbo) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK; kill $PP 2>/dev/null' EXIT
WT=/workspace/.wt-mtbo
common() {
  export TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
    HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
    CHUNKED_PREFILL_SIZE=8192 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
    TOKENSPEED_MIXED_REF_CHUNK=8192 TOKENSPEED_MIXED_LONG_CHUNK=8192 TOKENSPEED_MIXED_DECODE_REF=100000
}
# 1) correctness: GSM8K (CI threshold 0.90) while 40K-token prefills keep mixed rounds splitting.
echo "$(date -u +%T) gsm8k start" >> $M
( common; export ENGINE_PYTHONPATH=$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python MAX_NUM_SEQS=64 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats --enable-mixed-batch"; scripts/serve_unieinfra.sh 32 )
scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $M 2>&1 || { echo "server failed" >> $M; exit 1; }
python3 scripts/prefill_pressure.py --tokens 40000 > logs/verify_mtbo_pressure.log 2>&1 & PP=$!
scripts/run_gsm8k.sh unieinfra-mtbo >> $M 2>&1; kill $PP
grep -iE "AverageAccuracy|score|acc" results/gsm8k/unieinfra-mtbo-*/eval.log 2>/dev/null | tail -3 >> $M
echo "$(date -u +%T) gsm8k done; pressure prefills: $(wc -l < logs/verify_mtbo_pressure.log)" >> $M
grep -c "first mixed window split" logs/unieinfra-server.log >> $M
# 2) perf: stress mixed+TBO, c4/c16/c64 (production numbers come from the same-day B runs of stress_r10).
run() { # conc cap
  echo "$(date -u +%T) start MT c$1" >> $M
  ( common; STRESS=1 ENGINE_PYTHONPATH=$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python MAX_NUM_SEQS=$2 \
    EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats --enable-mixed-batch" \
    scripts/run_round.sh MTBO$1 600 unieinfra "$1" )
  echo "$(date -u +%T) done MT c$1" >> $M
}
run 16 32; run 64 128; run 4 32
echo "MTBO DONE $(date -u +%FT%TZ)" >> $M
