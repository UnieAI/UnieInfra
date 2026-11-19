#!/bin/bash
# AgentX stress 600 s: UnieInfra block verify + GIL switch interval 500 us (combo-bv-gil 714877e6, worktree .wt-combo).
# Same config as stress_bv.sh otherwise; compare with the SBV* (block verify only) and vLLM SB* cells.
cd /home/smc/workspace/roy; M=logs/stress_combo.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(stress_combo) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
for c in 16 4 32 64; do
  echo "$(date -u +%T) start unieinfra BV+GIL500 c$c" >> $M
  STRESS=1 TOKENSPEED_GIL_SWITCH_INTERVAL_US=500 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-combo/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh SBG$c 600 unieinfra "$c"
  echo "$(date -u +%T) done unieinfra BV+GIL500 c$c $(grep -m1 -o 'GIL switch interval set to [0-9]* us' logs/unieinfra-server.log)" >> $M
done
echo "STRESS COMBO DONE $(date -u +%FT%TZ)" >> $M
