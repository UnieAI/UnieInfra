#!/bin/bash
# One vLLM 0.31 arm with the baseline's arguments: launch, wait, run AgentX cells, save log, tear down.
# VLLM_SERVER_DEV_MODE exposes /reset_prefix_cache; without it every cell after
# the first replays a warm prefix cache.
#   arm_vllm.sh TAG [IMAGE]
set -u
TAG=$1; IMG=${2:-vllm/vllm-openai:v0.31.0}
HERE=$(cd "$(dirname "$0")" && pwd); GPUS=${GPUS:-4,5,6,7}; R=${RESULTS:-$HERE/results}; mkdir -p "$R"
NAME=bench-vllm-$TAG-$(date +%H%M%S); PORT=${PORT:-30203}
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPUS | sort -n | tail -1)" -lt 1000 ]; do sleep 5; done
docker run -d --name $NAME --gpus "\"device=$GPUS\"" --ipc host --init --ulimit memlock=-1 --shm-size 64g -p $PORT:$PORT \
  -e VLLM_SERVER_DEV_MODE=1 \
  -v ${MODEL_DIR:-/models}:/models:ro $IMG /models/deepseek-ai/DeepSeek-V4.1-Flash-1017 --host 0.0.0.0 --port $PORT \
  --tokenizer-mode deepseek_v41 --trust-remote-code --max-model-len 262144 --served-model-name dsv41 \
  --reasoning-parser deepseek_v41 --tensor-parallel-size 4 --enable-expert-parallel --enable-prefix-caching \
  --max-num-seqs 64 --speculative-config '{"method":"dspark","model":"/models/deepseek-ai/DeepSeek-V4.1-Flash-1017","num_speculative_tokens":5}' >/dev/null || exit 1
echo "launched $NAME $(date +%T)"
until curl -sf -m 5 localhost:$PORT/health >/dev/null; do
  sleep 20
  docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo died; docker logs $NAME > "$R/${TAG}_server.log" 2>&1; exit 1; }
done
echo "ready $(date +%T)"
RESULTS=$R "$HERE/run_cells.sh" http://localhost:$PORT dsv41 $TAG /reset_prefix_cache "${CONCS:-8 16 32}"
docker logs $NAME > "$R/${TAG}_server.log" 2>&1
docker rm -f $NAME >/dev/null 2>&1
sleep 30
echo "done $(date +%T)"
