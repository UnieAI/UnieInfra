#!/usr/bin/env bash
# Ask the engine itself whether the fused FP8 indexer insert matches its own
# eager reference, on real inputs.
#
# The module ships this instrument already: TOKENSPEED_DSV4_FP8_INDEXER_FUSED_VERIFY=1
# makes `deepseek_v4_csa_indexer_cache_insert` replay the eager chain on a cache
# copy after the fused kernel has written the real cache, and log the differing
# byte count together with the slot ids either side considered eligible.  It is
# one-shot and skipped while capturing, so it fires on the first non-captured
# forward -- which is why a real request has to be sent, not just a boot.
#
# Usage: verify_fused_indexer.sh CONC DATA
set -u

CONC=${1:?conc}
DATA=${2:?data}

HERE=/workspace/roy
HERE_SLO=$HERE/UnieInfra/bench/agent-slo
MODEL=/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
IMG_ENG=ts-env:gpu
NAME=fsverify
PORT=8231
LOG=$HERE/logs/fsverify.log

docker rm -f "$NAME" >/dev/null 2>&1 || true
: >"$LOG"

docker run -d --name "$NAME" --gpus '"device=6,7"' --shm-size 32g \
  -p "$PORT:8015" -v /models:/models -v /workspace:/workspace \
  -v "$HERE/UnieInfra/tokenspeed-scheduler/python/tokenspeed_scheduler:/home/runner/.local/lib/python3.12/site-packages/tokenspeed_scheduler:ro" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e FLASHINFER_WORKSPACE_BASE=/workspace/roy/.fiws \
  -e PYTHONPATH=/workspace/roy/UnieInfra/python:/workspace/roy/UnieInfra/tokenspeed-kernel/python \
  -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED=1 \
  -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED_VERIFY=1 \
  -e TOKENSPEED_DSV4_FP8_INDEXER_DUMP=/workspace/roy/logs/live_indexer_inputs.pt \
  -e TORCH_EXTENSIONS_DIR=/workspace/roy/.torch_ext \
  "$IMG_ENG" bash -lc "python3 -m tokenspeed.cli serve \
    --model-path $MODEL --tp 2 --port 8015 --host 0.0.0.0 --trust-remote-code \
    --reasoning-parser deepseek_v31 --disable-kvstore \
    --gpu-memory-utilization 0.87 --max-num-seqs 80 --max-model-len 32768 \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --admission-reserve generation-full --block-size 64 \
    --prefill-graph-max-tokens 8192" >/dev/null || true

docker logs -f "$NAME" >>"$LOG" 2>&1 &
LOGPID=$!

up=0
for i in $(seq 1 900); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then up=1; echo "[verify] UP after $((i * 2))s"; break; fi
  sleep 2
done
if [ "$up" -ne 1 ]; then
  echo "[verify] FAILED to come up"; docker logs --tail 80 "$NAME" 2>&1
  kill "$LOGPID" 2>/dev/null
  docker rm -f "$NAME" >/dev/null 2>&1
  exit 1
fi

# A real request is what makes the eager reference run on a live batch shape.
"$HERE/UnieInfra/.venv/bin/python" "$HERE_SLO/decode_only.py" \
  "http://127.0.0.1:$PORT" "$MODEL" "$DATA" "$CONC" \
  "$HERE/ab_real/agentx-sweep/fsverify-c${CONC}.json" || echo "[verify] drive failed"

sleep 3
echo
echo "================ [FSVERIFY] ================"
grep -a "FSVERIFY" "$LOG" | tail -20 || echo "(no FSVERIFY line)"
echo
echo "--- any diff detail ---"
grep -a -A 20 "diff_bytes=[1-9]" "$LOG" | tail -40 || true

kill "$LOGPID" 2>/dev/null
docker rm -f "$NAME" >/dev/null 2>&1
echo "[verify] torn down"