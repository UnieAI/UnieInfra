#!/bin/bash
# Cold long-prompt prefill latency per ROCm runtime, prefill TBO on/off: does the TBO overlap survive HIP 7.14 / 7.15?
cd /home/smc/workspace/roy; M=logs/rocm_prefill.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(rocm_prefill) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK; scripts/stop_servers.sh' EXIT
WT=/workspace/.wt-final
cfg() { local rt=$1 tbo=$2 name="$1-TBO$2" PP= LD=/opt/rocm/lib:/opt/rocm/lib
  case $rt in
    R714) PP=/workspace/.rocm714/site:; LD=/workspace/.rocm714/lib:/workspace/.rocm714/lib/rocm_sysdeps/lib:/workspace/.rocm714/lib/llvm/lib:/opt/rocm/lib;;
    R10)  PP=/workspace/.rocm10/site:; LD=/workspace/.rocm10/lib:/workspace/.rocm10/sysdeps:/workspace/.rocm10/llvm:/opt/rocm/lib;;
  esac
  echo "=== $(date -u +%T) $name" >> $M
  scripts/stop_servers.sh
  TOKENSPEED_PREFILL_TBO=$tbo TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=$PP$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python LD_LIBRARY_PATH=$LD \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=8 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/serve_unieinfra.sh 4 >> $M.server 2>&1 || { echo "$name LAUNCH FAIL" >> $M; return; }
  if ! scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $M 2>&1; then echo "$name SERVER FAIL" >> $M; return; fi
  echo "$name log $(readlink -f logs/unieinfra-server.log)" >> $M
  python3 scripts/repro_longprefill.py --n 1 --tokens 100000 --rounds 3 2>&1 | grep round | sed "s/^/$name 150K /" >> $M
  python3 scripts/repro_longprefill.py --n 1 --tokens 400000 --rounds 2 2>&1 | grep round | sed "s/^/$name 600K /" >> $M
}
cfg B 1; cfg B 0; cfg R714 1; cfg R714 0; cfg R10 1; cfg R10 0
echo "ROCM PREFILL DONE $(date -u +%FT%TZ)" >> $M
