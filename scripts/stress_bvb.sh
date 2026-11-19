#!/bin/bash
# Capacity sweep: --ignore-trace-delays (STRESS=1), each client fires its next request as soon as the previous ends.
# Unofficial (no recorded gaps) but the only mode that loads the engine at c<=64 (official mode: 34-63% of client time idle).
# Same engine configs as bvb_sweep2.sh. 600 s per cell. Waits for sweep2 to finish c64, then stops it.
cd /home/smc/workspace/roy
M=logs/stress_bvb.log; LOCK=logs/GPU_LOCK
: > $M
until grep -q "done unieinfra c64 x" logs/bvb_sweep2.log; do sleep 2; done
S=$(pgrep -f "scripts/bvb_sweep2[.]sh"); G=$(pgrep -f "scripts/bvb_guard2[.]py")
for p in $S; do for c in $(pgrep -P $p); do kill $c; done; kill $p; done; [ -n "$G" ] && kill $G
sleep 5; echo "BVB2 STOPPED after c64 $(date -u +%FT%TZ)" >> logs/bvb_sweep2.log
for i in $(seq 120); do pgrep -f "serve_vllm[.]sh|serve_unieinfra[.]sh|run_round[.]sh" >/dev/null || break; sleep 2; done
for p in $(pgrep -f "serve_vllm[.]sh|serve_unieinfra[.]sh|run_round[.]sh|aiperf"); do kill $p; done; sleep 20
rm -f $LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress best-vs-best) since=$(date -u +%FT%TZ) release=STRESS DONE in logs/stress_bvb.log" > $LOCK ) 2>/dev/null; do sleep 3; done
trap 'rm -f $LOCK' EXIT
echo "$(date -u +%T) lock taken; unieinfra HEAD $(git -C unieinfra rev-parse --short HEAD)" >> $M
declare -A RCAP=([1]=1 [4]=1 [16]=4 [32]=8 [64]=16 [128]=32)
vcell() { echo "$(date -u +%T) start vllm c$1 cap=$2" >> $M
  if [ $3 = x ]; then STRESS=1 VLLM_MAX_NUM_SEQS=$2 scripts/run_round.sh SB$1$3v 600 vllm "$1"; else STRESS=1 scripts/run_round.sh SB$1$3v 600 vllm "$1"; fi
  echo "$(date -u +%T) done vllm c$1 $3" >> $M; }
ucell() { echo "$(date -u +%T) start unieinfra c$1 cap=$2" >> $M
  STRESS=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/nonexistent TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 \
  MAX_NUM_SEQS=$2 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SB$1xu 600 unieinfra "$1"
  echo "$(date -u +%T) done unieinfra c$1" >> $M; }
o=u
for c in 16 1 4 32 64 128; do x=$((2*c))
  if [ $o = u ]; then ucell $c $x; vcell $c $x x; o=v; else vcell $c $x x; ucell $c $x; o=u; fi
  vcell $c ${RCAP[$c]} r
done
echo "STRESS DONE $(date -u +%FT%TZ)" >> $M
