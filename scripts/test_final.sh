#!/bin/bash
# Correctness of mi350p-final (production env): needle-in-haystack + GSM8K (T=0).
cd /home/smc/workspace/roy; T=final; LOG=logs/$T.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(test_final) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
scripts/stop_servers.sh
export TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  ENGINE_PYTHONPATH=/workspace/.wt-final/tokenspeed-kernel-amd/python:/workspace/.wt-final/tokenspeed-kernel/python:/workspace/.wt-final/python \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 GPU_MEMORY_UTILIZATION=0.86 EXTRA_SERVE_ARGS="--sampling-backend triton_full"
scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if scripts/wait_ready.sh 3000 >> $LOG.server 2>&1; then
  python3 scripts/needle_test.py 2>&1 | tail -1 | sed "s/^/$T /" >> $LOG
  scripts/run_gsm8k.sh unieinfra-$T >> $LOG.server 2>&1
  python3 -c "import json,glob;f=sorted(glob.glob('results/gsm8k/unieinfra-$T-*/*/reports/*/gsm8k.json'))[-1];j=json.load(open(f));print('$T gsm8k',[m.get('score') for m in j.get('metrics',[])][:1])" >> $LOG
else
  echo "$T SERVER FAILED" >> $LOG
fi
scripts/stop_servers.sh
echo "FINAL TEST DONE $(date -u +%FT%TZ)" >> $LOG
