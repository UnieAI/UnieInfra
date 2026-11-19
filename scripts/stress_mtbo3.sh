#!/bin/bash
# mixed+TBO (.wt-mtbo) vs production (.wt-final), interleaved same-day pairs: c16 repeat, c32, c128, c1.
cd /home/smc/workspace/roy; M=logs/stress_mtbo3.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(stress_mtbo3) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
run() { # variant(MT|B) conc cap tag
  echo "$(date -u +%T) start $1 c$2" >> $M
  if [ "$1" = MT ]; then WT=/workspace/.wt-mtbo; X=" --enable-mixed-batch"; else WT=/workspace/.wt-final; X=; fi
  ( export STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
      HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
      CHUNKED_PREFILL_SIZE=8192 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 MAX_NUM_SEQS=$3 \
      TOKENSPEED_MIXED_REF_CHUNK=8192 TOKENSPEED_MIXED_LONG_CHUNK=8192 TOKENSPEED_MIXED_DECODE_REF=100000 \
      ENGINE_PYTHONPATH=$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python \
      EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats$X"
    scripts/run_round.sh M2$1c$2$4 600 unieinfra "$2" )
  echo "$(date -u +%T) done $1 c$2" >> $M
}
run B 128 256 r2
echo "MTBO3 DONE $(date -u +%FT%TZ)" >> $M
