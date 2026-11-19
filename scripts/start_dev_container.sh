#!/bin/bash
# Long-lived unieinfra dev container: source mounted at /workspace (editable install).
docker rm -f ts-dev >/dev/null 2>&1
docker run -d --name ts-dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 993 \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --network host --ipc host --shm-size 64g --ulimit memlock=-1 \
  -v /home/smc/workspace/roy/unieinfra:/workspace \
  -v /data/models:/models:ro \
  -v /home/smc/workspace/roy/logs:/logs \
  -v /home/smc/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -w /workspace \
  ts-dev:snap sleep infinity
