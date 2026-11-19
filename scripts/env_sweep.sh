#!/bin/bash
# Decode speed (n=1 twice, n=16 once) for ROCm launch env-var combinations.
cd /home/smc/workspace/roy
OUT=logs/env-sweep2.log
run() {
  local label=$1; shift
  scripts/stop_servers.sh
  env "$@" scripts/serve_unieinfra.sh 16 >/dev/null
  if ! timeout 900 scripts/wait_ready.sh 880 >/dev/null; then echo "$label: NOT READY" >> $OUT; return; fi
  for n in 1 1 1 16; do
    echo "$label: $(timeout 1200 python3 scripts/bench_decode.py --n $n --tokens 2000 --max-tokens 1024 2>&1 | tail -1)" >> $OUT
  done
}
: > $OUT
run A1-default X=1
run B1-devkernarg HIP_FORCE_DEV_KERNARG=1
run A2-default X=1
run B2-devkernarg HIP_FORCE_DEV_KERNARG=1
echo DONE >> $OUT
