#!/bin/bash
# Cold long-prompt prefill with profiler-free GPU step timing, ROCm 7.2.4 vs 7.14.1 (TBO on, production flags).
cd /home/smc/workspace/roy; M=logs/rocm_prefill_t.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(rocm_prefill_t) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK; scripts/stop_servers.sh' EXIT
WT=/workspace/.wt-final
cfg() { local rt=$1 PP= LD=/opt/rocm/lib:/opt/rocm/lib
  [ $rt = R714 ] && { PP=/workspace/.rocm714/site:; LD=/workspace/.rocm714/lib:/workspace/.rocm714/lib/rocm_sysdeps/lib:/workspace/.rocm714/lib/llvm/lib:/opt/rocm/lib; }
  echo "=== $(date -u +%T) $rt" >> $M
  scripts/stop_servers.sh
  TOKENSPEED_GPU_STEP_TIMING=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=$PP$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python LD_LIBRARY_PATH=$LD \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=8 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/serve_unieinfra_sched.sh 4 >> $M.server 2>&1 || { echo "$rt LAUNCH FAIL" >> $M; return; }
  if ! scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $M 2>&1; then echo "$rt SERVER FAIL" >> $M; return; fi
  echo "$rt log $(readlink -f logs/unieinfra-server.log)" >> $M
  for i in 1 2 3; do
    echo "$rt mark 150K-$i $(date -u +%T.%N)" >> $M
    python3 scripts/repro_longprefill.py --n 1 --tokens 100000 --rounds 1 2>&1 | grep round | sed "s/^/$rt 150K-$i /" >> $M; sleep 8
  done
  for i in 1 2 3; do
    echo "$rt mark 600K-$i $(date -u +%T.%N)" >> $M
    python3 scripts/repro_longprefill.py --n 1 --tokens 400000 --rounds 1 2>&1 | grep round | sed "s/^/$rt 600K-$i /" >> $M; sleep 8
  done
}
cfg B; cfg R714; cfg B; cfg R714
echo "ROCM PREFILL T DONE $(date -u +%FT%TZ)" >> $M
