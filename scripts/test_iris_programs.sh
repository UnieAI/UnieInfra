#!/bin/bash
# Iris pull-kernel program count, decode side only.
# The 1.5 MB decode all-reduce runs on the small-message path, whose program
# count has only ever been compared at 84 and 32: 32 won at n=16 (87.0 vs 95.0
# tok/s/user) and 84 won at n=64 (44.7 vs 42.0). Two points cannot tell whether
# the optimum tracks batch size, so this sweeps 16/24/32/48/64/84 at both
# batch sizes with the same server args otherwise unchanged.
# Usage: scripts/test_iris_programs.sh   (logs to logs/iris_programs.log)
cd /home/smc/workspace/roy || exit 1
L=logs/iris_programs.log
: > "$L"
echo "IRIS PROGRAMS start $(date -u +%FT%TZ)" >> "$L"
for P in 16 24 32 48 64 84; do
  scripts/stop_servers.sh
  TOKENSPEED_IRIS_PULL_SMALL_PROGRAMS=$P \
  TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 \
  TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100 --sampling-backend triton_full" \
  MAX_NUM_SEQS=128 scripts/serve_unieinfra.sh 64 >> "$L.server" 2>&1
  # serve_unieinfra.sh writes one log per launch; handing it to wait_ready is
  # what tells a dead launch (bad flags, port held) from one still coming up,
  # instead of burning the whole timeout on a process that is already gone.
  srv=$(ls -t logs/unieinfra-server-c64-*.log | head -1)
  if scripts/wait_ready.sh 1500 "$srv" >> "$L.server" 2>&1; then
    # Two passes at each size: the first absorbs any lazy kernel compile, the
    # second is the measurement.
    python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 >/dev/null 2>&1
    python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 >/dev/null 2>&1
    for i in 1 2; do
      echo "P$P n16 $(python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 2>&1 | tail -1)" >> "$L"
      echo "P$P n64 $(python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 2>&1 | tail -1)" >> "$L"
    done
  else
    echo "P$P SERVER FAILED" >> "$L"
  fi
done
scripts/stop_servers.sh
echo "IRIS PROGRAMS DONE $(date -u +%FT%TZ)" >> "$L"
