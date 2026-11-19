#!/bin/bash
# Kernel-level breakdown of a steady-state decode step at c64 shape
# (batch 16, ~170K context) with no aiperf overhead.
# Usage: prof_decode_c64.sh [slots] [tokens] [prof_tokens]
set -uo pipefail
cd /home/smc/workspace/roy
SLOTS=${1:-16}
TOKENS=${2:-170000}
PROFTOK=${3:-400}
LOG=logs/profdec.log
OUT=/logs/prof-dec-b$SLOTS
echo "== $(date -u +%FT%TZ) START slots=$SLOTS tokens=$TOKENS proftok=$PROFTOK" >> $LOG

# Another session (roy-d8) serializes GPU work with logs/GPU_LOCK and takes the
# GPUs for its own pilot; never start a server while a lock or its chain lives.
LOCK=logs/GPU_LOCK
for _ in $(seq 1 900); do [ -e $LOCK ] || break; sleep 10; done
if [ -e $LOCK ]; then echo "== $(date -u +%FT%TZ) GPU_LOCK still held: $(cat $LOCK)" >> $LOG; exit 1; fi
echo "owner=roy(prof-dec-b$SLOTS) since=$(date -u +%FT%TZ) release=when PROFDEC_DONE is written" > $LOCK
trap 'rm -f $LOCK' EXIT
echo "== $(date -u +%FT%TZ) lock taken" >> $LOG

scripts/stop_servers.sh >> $LOG 2>&1
MAX_NUM_SEQS=$SLOTS MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.95 \
  EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
  scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
if ! scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1; then
  echo "== $(date -u +%FT%TZ) SERVER FAILED" >> $LOG
  scripts/stop_servers.sh >> $LOG 2>&1
  exit 1
fi
# Port 8000 must be *our* engine: another session's vLLM answers /v1/completions
# just as well, so warm the wrong server and the profile silently 404s at the end.
# Our routes exist, so a wrong method is 405 there and 404 on vLLM.
code=$(curl -s -o /dev/null -w '%{http_code}' -X OPTIONS localhost:8000/start_profile)
if [ "$code" = 404 ]; then
  echo "== $(date -u +%FT%TZ) port 8000 is not unieinfra (/start_profile -> $code); aborting" >> $LOG
  scripts/stop_servers.sh >> $LOG 2>&1
  exit 1
fi
python3 scripts/decode_profile.py --n "$SLOTS" --tokens "$TOKENS" \
  --prof-tokens "$PROFTOK" --out "$OUT" >> $LOG 2>&1
echo "== $(date -u +%FT%TZ) profile rc=$? out=$OUT" >> $LOG
scripts/stop_servers.sh >> $LOG 2>&1
echo "== $(date -u +%FT%TZ) PROFDEC_DONE" >> $LOG
