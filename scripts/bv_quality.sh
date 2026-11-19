#!/bin/bash
# Block-verify correctness: GSM8K (500 questions) sampled at T=1.0, BV=1 vs BV=0, same code (.wt-bv e966e7a5).
# Block verification is only active when the request temperature is 1.0, so the CI's T=0 GSM8K cannot test it.
cd /home/smc/workspace/roy; LOG=logs/bv_quality.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-d8(bv_quality) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 10; done
trap 'rm -f $LOCK' EXIT
for F in 1 0; do
  scripts/stop_servers.sh >/dev/null 2>&1
  TOKENSPEED_DSPARK_BLOCK_VERIFY=$F HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=/workspace/.wt-bv/python:/workspace/.wt-ait/tokenspeed-kernel/python:/workspace/.vendor TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
  CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=128 MAX_TOTAL_TOKENS=33554432 GPU_MEMORY_UTILIZATION=0.9 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full" scripts/serve_unieinfra.sh 64 >> $LOG.server 2>&1
  scripts/wait_ready.sh 3000 logs/unieinfra-server.log >> $LOG.server 2>&1 || { echo "F=$F server failed" >> $LOG; continue; }
  OUT=results/gsm8k/bvT1-F$F-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
  ( cd $OUT && HF_TOKEN=$(cat ~/.cache/huggingface/token) /home/smc/workspace/roy/.venv-eval/bin/evalscope eval \
    --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --api-url http://127.0.0.1:8000/v1 --api-key EMPTY_TOKEN \
    --datasets gsm8k --dataset-hub huggingface --dataset-args '{"gsm8k":{"dataset_id":"openai/gsm8k"}}' \
    --eval-batch-size 32 --limit 500 \
    --generation-config '{"do_sample":true,"temperature":1.0,"max_tokens":1024}' --work-dir . > eval.log 2>&1 )
  echo "F=$F $(grep -iE 'mean_acc|AverageAccuracy|\| gsm8k' $OUT/eval.log | tail -3 | tr '\n' ' ')" >> $LOG
  echo "F=$F accept: $(grep -oE 'accept_len: *[0-9.]+' logs/unieinfra-server.log | tail -8 | tr '\n' ' ')" >> $LOG
done
scripts/stop_servers.sh >/dev/null 2>&1
echo "BVQ DONE $(date -u +%FT%TZ)" >> $LOG
