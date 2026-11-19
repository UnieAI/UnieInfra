#!/bin/bash
# ROCm 10 user-space (HIP 7.15 / RCCL / hipBLASLt / comgr) overlay vs ROCm 7.2.4, interleaved A/B, AgentX stress 600 s.
cd /home/smc/workspace/roy; M=logs/stress_r10.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(stress_r10b) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
WT=/workspace/.wt-final
run() { # $1 variant(R10|B) $2 conc $3 cap
  echo "$(date -u +%T) start $1 c$2" >> $M
  if [ "$1" = R10 ]; then PP=/workspace/.rocm10/site:; LD=/workspace/.rocm10/lib:/workspace/.rocm10/sysdeps:/workspace/.rocm10/llvm:/opt/rocm/lib; else PP=; LD=/opt/rocm/lib:/opt/rocm/lib; fi
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=$PP$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python LD_LIBRARY_PATH=$LD \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$3 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh R10$1c$2b 600 unieinfra "$2"
  echo "$(date -u +%T) done $1 c$2" >> $M
}
run R10 16 32
echo "R10B DONE $(date -u +%FT%TZ)" >> $M
