#!/usr/bin/env bash
# Chained behind scripts/rep_ab.sh: test DeepSeek-V4.1 long-prefill QCP, which
# splits an extend chunk's rows (indexer scoring / attention) over attention TP.
# Motivation: measured prefill chunk latency 136 ms -> 2500 ms as contexts grew
# past ~200K (logs/unieinfra-server-c64-20261111T033341Z.log), i.e. the
# long-context extend, not decode, is what saturates at c64/c128.
set -u
cd /home/smc/workspace/roy
TAGMARK=REP_AB_DONE
echo "== $(date -u +%FT%TZ) waiting for $TAGMARK"
for _ in $(seq 1 240); do
  grep -q "$TAGMARK" logs/rep_ab.log 2>/dev/null && break
  sleep 15
done
grep -q "$TAGMARK" logs/rep_ab.log || { echo "AB never finished"; exit 1; }
echo "== $(date -u +%FT%TZ) A/B done, waiting for port 8000 to free"
for _ in $(seq 1 80); do
  ss -ltn 2>/dev/null | grep -q ':8000 ' || break
  sleep 5
done

export MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.90

export EXTRA_SERVE_ARGS="--deepseek-v41-indexer-query-parallel-size 4"
echo "== $(date -u +%FT%TZ) cell qcp64: indexer QCP 4"
bash scripts/run_round.sh qcp64 900 "unieinfra" "64"

export EXTRA_SERVE_ARGS="--deepseek-v41-indexer-query-parallel-size 4 --deepseek-v41-attention-query-parallel-size 4 --disable-prefill-graph"
echo "== $(date -u +%FT%TZ) cell qcp64b: indexer+attention QCP 4, no prefill graph"
bash scripts/run_round.sh qcp64b 900 "unieinfra" "64"

echo QCP_CELLS_DONE
