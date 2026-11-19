#!/bin/bash
# Pruned pass: each arg is token:regex, regex filters candidate kernel names
# by tile_m (TUNE_MOE_KERNEL_REGEX). Waits for a previous tuner pid ($WAITPID).
set -u
W=/logs/aiter_tune
export PYTHONPATH=/workspace/.vendor/aiter_meta:/workspace/.wt-aitfix/python:/workspace/.wt-aitfix/tokenspeed-kernel/python:/workspace/.vendor
export HIP_VISIBLE_DEVICES=0 TUNE_ONLY=flydsl AITER_BF16_FP8_MOE_BOUND=0
export FLYDSL_RUNTIME_CACHE_DIR=$W/flydsl_cache
while [ -n "${WAITPID:-}" ] && kill -0 $WAITPID 2>/dev/null; do sleep 5; done
echo "=== prev done $(date -u +%T)"
cd /workspace/.vendor/aiter_meta
for spec in "$@"; do
  t=${spec%%:*}; rx=${spec#*:}
  head -1 $W/untuned_dsv41_tp4_a8w4.csv > $W/untuned_t$t.csv
  grep "^$t,5120" $W/untuned_dsv41_tp4_a8w4.csv >> $W/untuned_t$t.csv
  echo "=== token $t start $(date -u +%T) rx=$rx"
  TUNE_MOE_KERNEL_REGEX="$rx" FAKE_ROOT=$W/fakeroot python $W/tune_wrap.py \
    -i $W/untuned_t$t.csv -o $W/tuned_dsv41_tp4_a8w4_cu128.csv -o2 $W/profile_dsv41_tp4_a8w4_cu128.csv \
    --mp 1 > $W/tune_t$t.log 2>&1
  echo "=== token $t done rc=$? $(date -u +%T)"
done
