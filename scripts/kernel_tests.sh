#!/bin/bash
# GPU window for unit tests + kernel microbenchmarks of the new decode branches, between ab_decode and ab_bv.
# Runs inside the ts-dev container (server stopped), one GPU.
cd /home/smc/workspace/roy; LOG=logs/kernel_tests.log; LOCK=logs/GPU_LOCK
until grep -q "AB DONE" logs/ab_decode.log 2>/dev/null; do sleep 20; done
until ( set -o noclobber; echo "owner=roy-d8(kernel_tests) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
scripts/stop_servers.sh >/dev/null 2>&1
echo "=== $(date -u +%T) start" >> $LOG
run() {  # label pythonpath workdir cmd
  echo "=== $(date -u +%T) $1" >> $LOG
  timeout 1800 docker exec -u root -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH="$2" -w "$3" ts-dev bash -c "$4" >> $LOG 2>&1
  echo "=== $1 rc=$?" >> $LOG
}
ATTN=/workspace/.wt-attn2/tokenspeed-kernel-amd/python:/workspace/.wt-attn2/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor
run attn2-kernel-tests $ATTN /workspace/.wt-attn2 "python -m pytest -q -x tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsv41_selected.py 2>&1 | tail -25"
run attn2-dspark-test $ATTN /workspace/.wt-attn2 "python -m pytest -q test/runtime/test_deepseek_v41_dspark.py -k 'folded_window or window_attention_kernel_matches_reference' 2>&1 | tail -15"
cp /tmp/claude-1000/-home-smc-workspace-roy/4ac055d4-cd27-41ff-96a7-d0aebbe01373/scratchpad/bench.py unieinfra/.wt-attn2/.attn2_bench.py 2>/dev/null
run attn2-bench $ATTN /workspace/.wt-attn2 "python .attn2_bench.py 2>&1 | tail -60"
HG=/workspace/.wt-hgap2/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor
run hgap2-test $HG /workspace/.wt-hgap2 "python -m pytest -q test/runtime/test_deepseek_v41_model.py -k 'hc_mixes or single_pass' 2>&1 | tail -15"
# decode-kernels2 (radix topk / MoE e4m3) tests: run if the branch has them
if [ -f unieinfra/.wt-k2/.kt_cmd ]; then
  K2=/workspace/.wt-k2/tokenspeed-kernel-amd/python:/workspace/.wt-k2/tokenspeed-kernel/python:/workspace/.wt-k2/python:/workspace/.vendor
  [ -f unieinfra/.wt-k2/.kt_pythonpath ] && K2=$(cat unieinfra/.wt-k2/.kt_pythonpath)
  run k2-tests $K2 /workspace/.wt-k2 "$(cat unieinfra/.wt-k2/.kt_cmd)"
fi
echo "KT DONE $(date -u +%FT%TZ)" >> $LOG
