#!/bin/bash
# Decode A/B on branch decode-host-gap2 (b7c56f1c, worktree .wt-hgap2; includes HOST_PREP 537d92ef), all arms same code.
#   SEG = DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING (2 = force packet-batch graph launch, skip clr "complex graph" fallback)
#   HC  = TOKENSPEED_DECODE_HC_INLINE (mHC mixes on main stream in decode)
#   HP  = TOKENSPEED_DECODE_HOST_PREP (GIL fix)
# Runs after the stress sweep. Bench: shared 130K prefix, T=1.0, n=1/16/32.
cd /home/smc/workspace/roy; LOG=logs/ab_decode.log; LOCK=logs/GPU_LOCK
until grep -q "STRESS DONE" logs/stress_bvb.log 2>/dev/null; do sleep 20; done
until ( set -o noclobber; echo "owner=roy-d8(ab_decode) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
srv() {  # name SEG HC HP [extra env...]
  local name=$1 seg=$2 hc=$3 hp=$4; shift 4
  scripts/stop_servers.sh >/dev/null 2>&1
  env "$@" DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=$seg TOKENSPEED_DECODE_HC_INLINE=$hc TOKENSPEED_DECODE_HOST_PREP=$hp \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-hgap2/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 || { echo "$name server failed: $(grep -m3 -iE 'error|exception' logs/unieinfra-server.log | cut -c1-200)" >> $LOG; return 1; }
}
arm() {  # name SEG HC HP
  srv "$@" || return
  for n in 1 16 32; do
    python3 scripts/bench_decode_shared.py --n $n --max-tokens 1024 --seed 7 --temperature 1.0 2>&1 | tail -1 | sed "s/^/$1 /" >> $LOG
    echo "$1 n=$n $(grep -oE 'accept_len: *[0-9.]+' logs/unieinfra-server.log | tail -5 | tr '\n' ' ')" >> $LOG
  done
}
# verification: does clr fall back to classic RunNodes for our decode graph?
srv V 1 0 0 AMD_LOG_LEVEL=3 AMD_LOG_MASK=0x4000 && echo "V(default): complex=$(grep -c 'Complex graph detected' logs/unieinfra-server.log) classic=$(grep -c 'Falling back to classic' logs/unieinfra-server.log) aqlfull=$(grep -c 'reached the AQL queue limit' logs/unieinfra-server.log)" >> $LOG
arm A0 1 0 0
arm A1 2 0 0
arm A2 2 1 0
arm A3 2 1 1
arm A0b 1 0 0
arm A3b 2 1 1
scripts/stop_servers.sh >/dev/null 2>&1
echo "AB DONE $(date -u +%FT%TZ)" >> $LOG
