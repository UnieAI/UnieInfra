#!/bin/bash
cd /home/smc/workspace/roy
LOG=logs/vllm-accept.log
until grep -q "IRIS32 DONE" logs/iris32.log 2>/dev/null; do sleep 20; done
scripts/stop_servers.sh
VLLM_MAX_NUM_SEQS=128 scripts/serve_vllm.sh 64 >> $LOG 2>&1
scripts/wait_ready.sh 3000 >> $LOG 2>&1 && python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/vllm /" >> $LOG
docker logs bench-server 2>&1 | grep -iE "accept|SpecDecod|draft" | tail -20 >> $LOG
scripts/stop_servers.sh
TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> $LOG 2>&1
scripts/wait_ready.sh 3000 >> $LOG 2>&1 && python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1 | sed "s/^/unie /" >> $LOG
sed 's/\x1b\[[0-9;]*m//g' $(readlink -f logs/unieinfra-server.log) | grep "Decode batch" | tail -8 | grep -oE "#running-req: [0-9]+|gen throughput \(token/s\): [0-9.]+|avg_accept_len: [0-9.]+|accept_rate: [0-9.]+" | paste - - - - >> $LOG
scripts/stop_servers.sh
echo "ACCEPT DONE $(date -u +%FT%TZ)" >> $LOG
