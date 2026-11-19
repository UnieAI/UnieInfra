#!/bin/bash
# After the final AgentX matrix: drop the duplicate NVMe model copy and serve
# every engine from /data/models (SATA), keeping ts-dev's installed packages.
# Aborts before deleting anything if the /data copy differs from the NVMe one.
set -euo pipefail
ROOT=/home/smc/workspace/roy
NVME=$ROOT/models
DATA=/data/models
LOG=$ROOT/logs/post-matrix-cleanup.log
exec >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) waiting for the final matrix"
until grep -q "FINAL MATRIX DONE" "$ROOT/logs/round-final.log" 2>/dev/null; do sleep 30; done
echo "=== $(date -u +%FT%TZ) matrix done; stopping servers"
"$ROOT/scripts/stop_servers.sh" || true

# 1. The /data copy must hold every NVMe file at the same size.
if [[ ! -L "$NVME" ]]; then
  diff <(cd "$NVME" && find . -path ./DeepSeek-V4.1-Flash-1017/.cache -prune -o -type f -printf '%P %s\n' | sort) \
       <(cd "$DATA" && find . -path ./DeepSeek-V4.1-Flash-1017/.cache -prune -o -type f -printf '%P %s\n' | sort) \
    || { echo "ABORT: /data copy differs from NVMe copy; nothing deleted"; exit 1; }
  echo "verified: /data copy matches NVMe file list and sizes"
  # 2. Drop the NVMe copy; the old path now points at /data.
  rm -rf "$NVME"
  ln -s "$DATA" "$NVME"
  echo "removed NVMe copy; $NVME -> $DATA"
fi
df -h /

# 3. Scripts read /data directly; slower SATA cold loads get more time.
sed -i "s|-v $ROOT/models:/models:ro|-v $DATA:/models:ro|" "$ROOT/scripts/serve_vllm.sh" "$ROOT/scripts/serve_sglang.sh" "$ROOT/scripts/start_dev_container.sh"
sed -i 's|wait_ready.sh" 1500|wait_ready.sh" 3000|' "$ROOT/scripts/run_round.sh"
grep -q VLLM_ENGINE_READY_TIMEOUT_S "$ROOT/scripts/serve_vllm.sh" || \
  sed -i 's|  -e HF_HUB_OFFLINE=1 \\|  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \\|' "$ROOT/scripts/serve_vllm.sh"
grep -h -- "-v .*:/models" "$ROOT/scripts/serve_vllm.sh" "$ROOT/scripts/serve_sglang.sh" "$ROOT/scripts/start_dev_container.sh"

# 4. Recreate ts-dev on /data, keeping its installed packages.
docker commit ts-dev ts-dev:snap >/dev/null
sed -i 's|lightseekorg/tokenspeed-runner-amd:rocm7.2.4-pytorch-2.13.0-gfx950 sleep infinity|ts-dev:snap sleep infinity|' "$ROOT/scripts/start_dev_container.sh"
"$ROOT/scripts/start_dev_container.sh"
docker exec ts-dev bash -c 'ls /models/DeepSeek-V4.1-Flash-1017/config.json && python3 -c "import tokenspeed, tokenspeed_kernel, tokenspeed_kernel_amd; print(\"ts-dev ok\")"'

# 5. Warm the page cache so the next server start skips the SATA cold read.
echo "=== $(date -u +%FT%TZ) warming page cache"
find "$DATA/DeepSeek-V4.1-Flash-1017" -name '*.safetensors' -print0 | xargs -0 -P 4 -n 4 cat > /dev/null
echo "=== $(date -u +%FT%TZ) CLEANUP DONE"
df -h /
