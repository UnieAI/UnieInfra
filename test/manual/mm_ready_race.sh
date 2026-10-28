#!/usr/bin/env bash
# Repro for the multimodal "first request after ready" race.
#
# Starts a multimodal model script N times. On each start it fires one image
# request the moment the chosen readiness signal turns green and records the
# status, then keeps polling until an image request succeeds to measure the
# window during which multimodal requests would fail.
#
# Usage (from the repo root):
#   RUNS=3 READY=readiness CUDA_DEVICES=6 ./test/manual/mm_ready_race.sh scripts/gemma3.sh
#   READY=models ...   # treat GET /v1/models == 200 as ready (old smoke harness)
#
# Env: RUNS (3), READY (readiness|models), HOST_PORT (18790), TIMEOUT_S (1800),
#      LOG_DIR (/tmp/mm_ready_race). Other env is passed to the model script.
set -uo pipefail
script=${1:?model script, e.g. scripts/gemma3.sh}
RUNS=${RUNS:-3}; READY=${READY:-readiness}; PORT=${HOST_PORT:-18790}
TIMEOUT_S=${TIMEOUT_S:-1800}; LOG_DIR=${LOG_DIR:-/tmp/mm_ready_race}
PY=${PYTHON:-python3}
mkdir -p "$LOG_DIR"
base=http://127.0.0.1:$PORT

mm_req() {
    "$PY" - "$base" <<'PY'
import base64, io, json, sys, urllib.request, urllib.error, zlib, struct
def png(w=64, h=64, rgb=(230, 20, 20)):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
url = "data:image/png;base64," + base64.b64encode(png()).decode()
body = {"model": "test", "max_tokens": 4, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Color? One word."}]}]}
req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r: print(r.status)
except urllib.error.HTTPError as e: print(e.code)
except Exception: print("ERR")
PY
}

ready() {
    case "$READY" in
        models) curl -sf -m 5 "$base/v1/models" >/dev/null ;;
        *) curl -sf -m 5 "$base/readiness" >/dev/null ;;
    esac
}

fail=0
for i in $(seq 1 "$RUNS"); do
    log=$LOG_DIR/run_$i.log
    HOST_PORT=$PORT setsid bash "$script" >"$log" 2>&1 </dev/null &
    pid=$!
    t0=$(date +%s); ok=0
    while (( $(date +%s) - t0 < TIMEOUT_S )); do
        kill -0 $pid 2>/dev/null || break
        if ready; then ok=1; break; fi
        sleep 0.5
    done
    if (( ok )); then
        first=$(mm_req); t_ready=$(( $(date +%s) - t0 ))
        window=0
        if [[ "$first" != 200 ]]; then
            fail=1; t1=$(date +%s)
            while (( $(date +%s) - t1 < 300 )); do
                [[ "$(mm_req)" == 200 ]] && break; sleep 1
            done
            window=$(( $(date +%s) - t1 ))
        fi
        echo "run $i: ready(${READY}) after ${t_ready}s first_mm=$first mm_fail_window=${window}s"
    else
        fail=1; echo "run $i: server not ready (see $log)"
    fi
    kill -TERM -$pid 2>/dev/null; sleep 10; kill -KILL -$pid 2>/dev/null; wait $pid 2>/dev/null
    sleep 5
done
exit $fail
