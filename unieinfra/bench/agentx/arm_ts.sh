#!/bin/bash
# One tokenspeed arm: launch, wait, run the AgentX cells, optionally GSM8K, save the log, tear down.
#   arm_ts.sh TAG IMAGE [extra serve args...]
# Env: GPUS (4,5,6,7), PORT, MODEL_DIR (/models), CACHE_DIR, CONCS, GSM=1, EVALSCOPE,
#      ARM_ENV / ARM_MOUNTS (extra docker args), ARM_EP (default --enable-expert-parallel).
set -u
TAG=$1; IMG=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-4,5,6,7}; PORT=${PORT:-30204}; MODELS=${MODEL_DIR:-/models}
R=${RESULTS:-$HERE/results}; mkdir -p "$R"
NAME=bench-ts-$TAG-$(date +%H%M%S)
# Wait until the arm's GPUs are free (named by memory use, not by containers:
# an idle dev container holding a device must not block the queue).
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPUS | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
docker run -d --name $NAME --gpus "\"device=$GPUS\"" --ipc host --init --ulimit memlock=-1 --shm-size 64g -p $PORT:$PORT \
  ${ARM_ENV:-} -v $MODELS:/models:ro ${CACHE_DIR:+-v $CACHE_DIR:/root/.cache} ${ARM_MOUNTS:-} \
  $IMG ts serve --model /models/deepseek-ai/DeepSeek-V4.1-Flash-1017 --served-model-name dsv41 --trust-remote-code \
  --speculative-algorithm DSPARK --tensor-parallel-size 4 ${ARM_EP---enable-expert-parallel} --dtype bfloat16 --engram-host-table \
  --max-model-len 262144 --max-num-seqs 64 --chunked-prefill-size 8192 --max-cudagraph-capture-size 64 \
  --reasoning-parser deepseek_v31 --host 0.0.0.0 --port $PORT "$@" >/dev/null || exit 1
echo "launched $NAME $(date +%T)"
until curl -sf -m 5 localhost:$PORT/readiness >/dev/null; do
  sleep 20
  docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo died; docker logs $NAME > "$R/${TAG}_server.log" 2>&1; grep -E Error "$R/${TAG}_server.log" | tail -5 | cut -c1-300; exit 1; }
done
echo "ready $(date +%T)"
RESULTS=$R "$HERE/run_cells.sh" http://localhost:$PORT dsv41 $TAG /flush_cache "${CONCS:-8 16 32}"
if [ -n "${GSM:-}" ]; then
  ${EVALSCOPE:-evalscope} eval --model dsv41 --api-url http://127.0.0.1:$PORT/v1 --api-key EMPTY_TOKEN \
    --datasets gsm8k --dataset-hub huggingface --dataset-args '{"gsm8k":{"dataset_id":"openai/gsm8k"}}' \
    --eval-batch-size 32 --generation-config '{"do_sample":false,"temperature":0.0,"max_tokens":512}' \
    --work-dir "$R/${TAG}_gsm8k" > "$R/${TAG}_gsm8k.log" 2>&1
  grep -ho '"score": [0-9.]*' $(find "$R/${TAG}_gsm8k" -name '*.json' -path '*report*') | head -1
fi
docker logs $NAME > "$R/${TAG}_server.log" 2>&1
docker rm -f $NAME >/dev/null 2>&1
sleep 30
echo "done $(date +%T)"
