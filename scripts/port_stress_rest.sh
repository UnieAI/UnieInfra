#!/bin/bash
# Finish the ported build's stress sweep (no official runs until it wins).
cd /home/smc/workspace/roy
ROUND_PID=${1:?}
while kill -0 "$ROUND_PID" 2>/dev/null; do sleep 20; done
seqs_for() { (( $1 >= 64 )) && echo 128 || echo 64; }
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1
for c in 32 16 4 1; do
  STRESS=1 MAX_NUM_SEQS=$(seqs_for $c) EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" \
    scripts/run_round.sh port 300 unieinfra "$c"
done
STRESS=1 scripts/run_round.sh stress 300 vllm "4"
STRESS=1 VLLM_MAX_NUM_SEQS=8 scripts/run_round.sh stressbig 300 vllm "4"
echo "PORT STRESS DONE $(date -u +%FT%TZ)" >> logs/validate-port.log
