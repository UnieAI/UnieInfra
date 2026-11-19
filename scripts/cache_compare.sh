#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/multiturn-cache.log
until grep -q "H2HLONG DONE" logs/head2head-long.log 2>/dev/null; do sleep 30; done
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=64 scripts/serve_unieinfra.sh 16 >> $LOG.server 2>&1
scripts/wait_ready.sh 3000 >> $LOG.server 2>&1 && python3 scripts/multiturn_cache.py 2>&1 | sed 's/^/unie /' >> $LOG
scripts/stop_servers.sh
VLLM_MAX_NUM_SEQS=32 scripts/serve_vllm.sh 16 >> $LOG.server 2>&1
scripts/wait_ready.sh 3000 >> $LOG.server 2>&1 && python3 scripts/multiturn_cache.py 2>&1 | sed 's/^/vllm /' >> $LOG
scripts/stop_servers.sh
echo "CACHECMP DONE $(date -u +%FT%TZ)" >> $LOG
