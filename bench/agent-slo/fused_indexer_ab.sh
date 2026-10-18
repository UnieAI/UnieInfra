#!/bin/bash
# A/B of the fused CSA indexer FP8 cache write (TOKENSPEED_DSV4_FP8_INDEXER_FUSED).
#
# Boots a FRESH serving cell per (concurrency, arm), drives the same decode-only
# workload at it, tears it down. One arm per container so a warm prefix cache or
# a leftover graph can never leak from the other arm into the measurement.
#
# Usage:
#   fused_indexer_ab.sh CONCS DATA [ARMS]
#     CONCS : comma-separated concurrency list, e.g. "16,64"
#     DATA  : measured corpus (.jsonl, one {"prompt":...} per line)
#     ARMS  : space-separated label:flag pairs, default "0:0 1:1".
#             "0:0 1:1 0b:0" adds a second baseline boot, which is the control:
#             an engine that is not bit-reproducible across runs disagrees with
#             ITSELF at some rate, and a difference between arms only means
#             something once that rate is known.
#
# Writes ab_real/agentx-sweep/fused-c<CONC>-arm<label>.json per cell and prints
# the agreement-gated comparison at the end.
set -u

CONCS=${1:?concs}
DATA=${2:?data}
ARMS=${3:-"0:0 1:1"}

HERE=/workspace/roy
HERE_SLO=$HERE/UnieInfra/bench/agent-slo
MODEL=/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
IMG_ENG=ts-env:gpu-sched39
OUTDIR=$HERE/ab_real/agentx-sweep
mkdir -p "$HERE/logs" "$OUTDIR"

if [ ! -f "$DATA" ]; then echo "no such corpus: $DATA" >&2; exit 1; fi

for CONC in $(echo "$CONCS" | tr ',' ' '); do
  IDX=0
  for SPEC in $ARMS; do
    ARM=${SPEC%%:*}
    FLAG=${SPEC##*:}
    IDX=$((IDX + 1))
    NAME="fused-ab-c${CONC}-arm${ARM}"
    PORT=$((8210 + CONC + IDX))
    LOG=$HERE/logs/fused-ab-c${CONC}-arm${ARM}.log
    OUT=$OUTDIR/fused-c${CONC}-arm${ARM}.json

    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "[c$CONC arm$ARM] booting on GPUs 6,7 port $PORT (log: $LOG)"
    : >"$LOG"

    # A boot costs ~5 minutes, so a cell that already produced its raw JSON is
    # not repeated: a retry after a later failure resumes instead of redoing
    # the whole sweep.
    if [ -s "$OUT" ]; then
      echo "[c$CONC arm$ARM] $OUT already present, skipping"
      continue
    fi

    docker run -d --name "$NAME" --gpus '"device=6,7"' --shm-size 32g \
      -p "$PORT:8015" -v /models:/models -v /workspace:/workspace \
      -v "$HERE/UnieInfra/tokenspeed-scheduler/python/tokenspeed_scheduler:/home/runner/.local/lib/python3.12/site-packages/tokenspeed_scheduler:ro" \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -e FLASHINFER_WORKSPACE_BASE=/workspace/roy/.fiws \
      -e PYTHONPATH=/workspace/roy/UnieInfra/python:/workspace/roy/UnieInfra/tokenspeed-kernel/python \
      -e TOKENSPEED_DSV4_FP8_INDEXER_FUSED="$FLAG" \
      -e TORCH_EXTENSIONS_DIR=/workspace/roy/.torch_ext \
      "$IMG_ENG" bash -lc "python3 -m tokenspeed.cli serve \
        --model-path $MODEL --tp 2 --port 8015 --host 0.0.0.0 --trust-remote-code \
        --reasoning-parser deepseek_v31 \
        --disable-kvstore \
        --gpu-memory-utilization 0.87 --max-num-seqs 80 --max-model-len 32768 \
        --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
        --block-size 64 \
        --prefill-graph-max-tokens 8192" >/dev/null

    # `docker run -d` prints only the container id on its own stdout; the
    # engine's output lives in the container log, so follow it into $LOG
    # rather than redirecting the run itself.
    docker logs -f "$NAME" >>"$LOG" 2>&1 &
    LOGPID=$!

    st=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo missing)
    if [ "$st" != "true" ]; then
      echo "[c$CONC arm$ARM] FAILED: container not running"; docker logs --tail 60 "$NAME" 2>&1; exit 1
    fi

    up=0
    for i in $(seq 1 1200); do
      code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" || true)
      if [ "$code" = "200" ]; then up=1; echo "[c$CONC arm$ARM] UP after ${i}s"; break; fi
      if [ $((i % 60)) -eq 0 ]; then echo "[c$CONC arm$ARM] waiting ${i}s (code=$code)"; fi
      sleep 2
    done
    if [ "$up" -ne 1 ]; then
      echo "[c$CONC arm$ARM] FAILED to come up"; docker logs --tail 120 "$NAME" 2>&1
      kill "$LOGPID" 2>/dev/null
      docker rm -f "$NAME" >/dev/null 2>&1
      exit 1
    fi

    # The served name defaults to the model path, so the gateway only resolves
    # the tokenizer when it is given the path -- the short HF name returns
    # 500 "Tokenizer not found for model".
    echo "[c$CONC arm$ARM] driving conc=$CONC -> $OUT"
    if ! "$HERE/UnieInfra/.venv/bin/python" "$HERE_SLO/decode_only.py" \
        "http://127.0.0.1:$PORT" "$MODEL" "$DATA" "$CONC" "$OUT"; then
      echo "[c$CONC arm$ARM] drive failed"; docker logs --tail 60 "$NAME" 2>&1
      kill "$LOGPID" 2>/dev/null
      docker rm -f "$NAME" >/dev/null 2>&1
      exit 1
    fi

    kill "$LOGPID" 2>/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1
    echo "[c$CONC arm$ARM] cell torn down"
  done

  echo
  echo "================ c$CONC comparison ================"
  BASE=$OUTDIR/fused-c${CONC}-arm0.json
  if [ ! -s "$BASE" ]; then
    echo "no baseline $BASE; nothing to compare"
    continue
  fi
  # Any second boot of the SAME configuration is the control: it quantifies how
  # much the engine disagrees with itself, which is the floor under the
  # cross-arm agreement number.
  for SPEC in $ARMS; do
    CAND=$OUTDIR/fused-c${CONC}-arm${SPEC%%:*}.json
    [ "$CAND" = "$BASE" ] && continue
    if [ -s "$CAND" ]; then
      echo "--- arm0 vs arm${SPEC%%:*} (flag=${SPEC##*:}) ---"
      "$HERE/UnieInfra/.venv/bin/python" "$HERE_SLO/compare_arms.py" "$BASE" "$CAND"
      echo
    fi
  done
done
