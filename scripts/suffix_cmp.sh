#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/suffix-prefill.log
until grep -q "PROF DONE" logs/head2head-long.log 2>/dev/null; do sleep 20; done
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
scripts/wait_ready.sh 3000 >> $LOG.server 2>&1 && python3 scripts/suffix_prefill.py 2>&1 | sed 's/^/unie /' >> $LOG
scripts/stop_servers.sh
VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG.server 2>&1
scripts/wait_ready.sh 3000 >> $LOG.server 2>&1 && python3 scripts/suffix_prefill.py 2>&1 | sed 's/^/vllm /' >> $LOG
scripts/stop_servers.sh
echo "SUFFIX DONE $(date -u +%FT%TZ)" >> $LOG
