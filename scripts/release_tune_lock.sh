#!/bin/bash
# Release the aiter-tune GPU lock when its tuning finishes, or at 08:36 UTC at the latest.
cd /home/smc/workspace/roy
end=$(date -u -d "2026-11-13 08:36" +%s)
while [ $(date -u +%s) -lt $end ]; do
  docker exec ts-dev bash -c 'ps -eo args | grep -E "[r]un_tune2|[t]une_wrap"' >/dev/null || break
  sleep 30
done
for p in $(docker exec ts-dev bash -c 'ps -eo pid,args | grep -E "[r]un_tune2|[t]une_wrap" | awk "{print \$1}"'); do docker exec ts-dev kill $p; done
grep -q "owner=aiter-tune" logs/GPU_LOCK 2>/dev/null && rm -f logs/GPU_LOCK
echo "tune lock released $(date -u +%T)" >> logs/aiter_tune/run_progress.log
