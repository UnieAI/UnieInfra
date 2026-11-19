#!/bin/bash
# After the decode-interleave stress pair: move the main checkout to the
# ported branch (lost DSv4.1 prefill work + coalescing), check correctness and
# 104K-prefill speed, then stress c64/c128 with the H200 dsv41 recipe knobs.
set -u
cd /home/smc/workspace/roy
LOG=logs/validate-port.log
until grep -q "DEC1 DONE" logs/round-dec1.log; do sleep 30; done
git -C unieinfra merge --ff-only mi350p-port >> $LOG 2>&1 || { echo "FF FAILED" >> $LOG; exit 1; }
echo "=== $(date -u +%FT%TZ) main checkout at $(git -C unieinfra log --oneline -1)" >> $LOG
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1
EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/serve_unieinfra.sh 16 >> $LOG 2>&1
if ! scripts/wait_ready.sh 3000 >> $LOG 2>&1; then
  echo "=== SERVER FAILED (port)" >> $LOG; cp logs/unieinfra-server.log logs/fail-port.log; exit 1
fi
python3 scripts/repro_longprefill.py --n 1 --tokens 104000 --rounds 3 >> $LOG 2>&1
python3 scripts/needle_test.py >> $LOG 2>&1
scripts/run_gsm8k.sh unieinfra-port >> $LOG 2>&1
grep -h "AverageAccuracy\|mean_acc\|score" $(ls -td results/gsm8k/unieinfra-port-* | head -1)/outputs/*/reports/*/*.json 2>/dev/null | head -3 >> $LOG
echo "segfaults=$(grep -ac Segfault $(readlink -f logs/unieinfra-server.log))" >> $LOG
echo "=== $(date -u +%FT%TZ) correctness done" >> $LOG
# H200 dsv41 recipe: max-num-seqs 64 up to c32, 128 from c64.
seqs_for() { (( $1 >= 64 )) && echo 128 || echo 64; }
for c in 64 128 32 16 4 1; do
  STRESS=1 MAX_NUM_SEQS=$(seqs_for $c) EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" \
    scripts/run_round.sh port 300 unieinfra "$c"
done
# Fill vLLM's missing stress c4 (both settings) so every column has c1-c128.
STRESS=1 scripts/run_round.sh stress 300 vllm "4"
STRESS=1 VLLM_MAX_NUM_SEQS=8 scripts/run_round.sh stressbig 300 vllm "4"
echo "PORT STRESS DONE $(date -u +%FT%TZ)" >> $LOG
# Official AgentX c1-c128 on the ported build.
for c in 64 128 32 16 4 1; do
  MAX_NUM_SEQS=$(seqs_for $c) EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" \
    scripts/run_round.sh portofficial 900 unieinfra "$c"
done
echo "PORT DONE $(date -u +%FT%TZ)" >> $LOG
