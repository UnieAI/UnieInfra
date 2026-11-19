#!/bin/bash
# After stress c1/c4 (decode-bound, KV size irrelevant) finish, stop stress_bvb.sh and run c32/c64/c128 with the
# UnieInfra KV token cap lifted (MAX_TOTAL_TOKENS=2^30: arena sized by the gpu util 0.9 budget instead of 33.5M).
# Falls back to the 33.5M cap if the uncapped server fails to start. Same vLLM cells as stress_bvb.sh.
cd /home/smc/workspace/roy
M=logs/stress_bvb.log; LOCK=logs/GPU_LOCK
until grep -q "done vllm c4 r" $M; do sleep 2; done
S=$(pgrep -f "scripts/stress_bvb[.]sh")
for p in $S; do for c in $(pgrep -P $p); do kill $c; done; kill $p; done
sleep 5
for i in $(seq 120); do pgrep -f "serve_vllm[.]sh|serve_unieinfra[.]sh|run_round[.]sh" >/dev/null || break; sleep 2; done
for p in $(pgrep -f "serve_vllm[.]sh|serve_unieinfra[.]sh|run_round[.]sh|aiperf"); do kill $p; done
scripts/stop_servers.sh >/dev/null 2>&1; sleep 10; rm -f $LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress kv-uncapped) since=$(date -u +%FT%TZ) release=STRESS DONE in logs/stress_bvb.log" > $LOCK ) 2>/dev/null; do sleep 3; done
trap 'rm -f $LOCK' EXIT
echo "$(date -u +%T) stress_bvb stopped after c4; c32+ with UnieInfra KV uncapped (MAX_TOTAL_TOKENS=2^30)" >> $M
declare -A RCAP=([32]=8 [64]=16 [128]=32)
KV=1073741824
vcell() { echo "$(date -u +%T) start vllm c$1 cap=$2" >> $M
  if [ $3 = x ]; then STRESS=1 VLLM_MAX_NUM_SEQS=$2 scripts/run_round.sh SB$1$3v 600 vllm "$1"; else STRESS=1 scripts/run_round.sh SB$1$3v 600 vllm "$1"; fi
  echo "$(date -u +%T) done vllm c$1 $3" >> $M; }
ucell() { echo "$(date -u +%T) start unieinfra c$1 cap=$2 kv=$KV" >> $M
  STRESS=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 \
  MAX_NUM_SEQS=$2 MAX_TOTAL_TOKENS=$KV GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SB$1xu$3 600 unieinfra "$1"
  local cap=$(grep -oE "token_capacity=[0-9]+" logs/unieinfra-server.log | head -1)
  echo "$(date -u +%T) done unieinfra c$1 $cap" >> $M
  if [ -z "$cap" ] && [ $KV != 33554432 ]; then echo "$(date -u +%T) uncapped start failed; fallback 33.5M" >> $M; KV=33554432; ucell $1 $2 $3; fi; }
o=v
for c in 32 64 128; do x=$((2*c))
  if [ $o = u ]; then ucell $c $x k; vcell $c $x x; o=v; else vcell $c $x x; ucell $c $x k; o=u; fi
  vcell $c ${RCAP[$c]} r
done
echo "STRESS DONE $(date -u +%FT%TZ)" >> $M
