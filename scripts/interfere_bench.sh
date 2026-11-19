#!/bin/bash
# Cold long prefill alone vs beside 6 long-context decoders (production mi350p-final flags, GPU step timing on).
cd /home/smc/workspace/roy; M=logs/interfere_bench.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(interfere_bench) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK; scripts/stop_servers.sh' EXIT
scripts/stop_servers.sh
TOKENSPEED_GPU_STEP_TIMING=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
ENGINE_PYTHONPATH=/workspace/.wt-final/tokenspeed-kernel-amd/python:/workspace/.wt-final/tokenspeed-kernel/python:/workspace/.wt-final/python \
TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=32 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
scripts/serve_unieinfra.sh 4 >> $M.server 2>&1 || { echo LAUNCH FAIL >> $M; exit 1; }
scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $M 2>&1 || { echo SERVER FAIL >> $M; exit 1; }
echo "log $(readlink -f logs/unieinfra-server.log)" >> $M
python3 scripts/repro_longprefill.py --n 1 --tokens 300000 --rounds 2 2>&1 | grep round | sed 's/^/alone /' >> $M
echo "$(date -u +%T) beside start" >> $M
python3 scripts/prefill_interfere.py --decoders 6 --ctx 100000 --dec-tokens 4000 --cold 300000 --wait 40 >> $M 2>&1
python3 scripts/prefill_interfere.py --decoders 6 --ctx 100000 --dec-tokens 4000 --cold 300000 --wait 40 >> $M 2>&1
echo "INTERFERE DONE $(date -u +%FT%TZ)" >> $M
