#!/bin/bash
# Runs inside ts-dev. Tunes each token shape in its own process so a crash
# loses one shape only. Usage: run_tune.sh <token> [<token> ...]
set -u
W=/logs/aiter_tune
export PYTHONPATH=/workspace/.wt-aitfix/python:/workspace/.wt-aitfix/tokenspeed-kernel/python:/workspace/.vendor
export HIP_VISIBLE_DEVICES=0
export TUNE_ONLY=flydsl
export AITER_BF16_FP8_MOE_BOUND=0
export PYTHONPATH=/workspace/.vendor/aiter_meta:$PYTHONPATH
export FLYDSL_RUNTIME_CACHE_DIR=$W/flydsl_cache
cd /workspace/.vendor/aiter_meta
for t in "$@"; do
  head -1 $W/untuned_dsv41_tp4_a8w4.csv > $W/untuned_t$t.csv
  grep "^$t,5120" $W/untuned_dsv41_tp4_a8w4.csv >> $W/untuned_t$t.csv
  echo "=== token $t start $(date -u +%T)"
  FAKE_ROOT=$W/fakeroot python $W/tune_wrap.py \
    -i $W/untuned_t$t.csv -o $W/tuned_dsv41_tp4_a8w4_cu128.csv -o2 $W/profile_dsv41_tp4_a8w4_cu128.csv \
    --mp 1 > $W/tune_t$t.log 2>&1
  echo "=== token $t done rc=$? $(date -u +%T)"
done
