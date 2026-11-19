#!/bin/bash
# Long-context cold prefill: V4.1 indexer/attention query-parallel (never tested, #158 lever).
# Per config: production env of stress_final (mi350p-final), one 400K-token cold prompt warm + 2 measured, plus 100K.
cd /home/smc/workspace/roy; M=logs/qp_bench.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(qp_bench) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK; scripts/stop_servers.sh' EXIT
cfg() { local name=$1; shift
  echo "=== $(date -u +%T) $name: $*" >> $M
  scripts/stop_servers.sh
  TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-final/tokenspeed-kernel-amd/python:/workspace/.wt-final/tokenspeed-kernel/python:/workspace/.wt-final/python \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=8 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats $*" \
  scripts/serve_unieinfra.sh 4 >> $M.server 2>&1 || { echo "$name LAUNCH FAIL" >> $M; return; }
  if ! scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $M 2>&1; then echo "$name SERVER FAIL" >> $M; return; fi
  echo "$name log $(readlink -f logs/unieinfra-server.log)" >> $M
  python3 scripts/repro_longprefill.py --n 1 --tokens 400000 --rounds 3 2>&1 | grep round | sed "s/^/$name 400K /" >> $M
  python3 scripts/repro_longprefill.py --n 1 --tokens 100000 --rounds 2 2>&1 | grep round | sed "s/^/$name 100K /" >> $M
}
cfg BASE
cfg IQP4 --deepseek-v41-indexer-query-parallel-size 4
cfg NOPG --disable-prefill-graph
cfg IAQP4 --deepseek-v41-indexer-query-parallel-size 4 --deepseek-v41-attention-query-parallel-size 4 --disable-prefill-graph
echo "QP BENCH DONE $(date -u +%FT%TZ)" >> $M
