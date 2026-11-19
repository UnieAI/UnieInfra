#!/bin/bash
# RL env step 1: separate container rl-dev (reuses local tokenspeed-runner-amd image, no pull), install libaio-dev + DeepSpeed, check async_io (NVMe offload op).
# No GPU work: only builds/imports. Offload dir on the NVMe system disk: /home/smc/workspace/roy/rl/offload.
set -x
docker rm -f rl-dev 2>/dev/null
docker run -d --name rl-dev --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g \
  --security-opt seccomp=unconfined -v /home/smc/workspace/roy/rl:/rl -v /models:/models:ro \
  lightseekorg/tokenspeed-runner-amd:rocm7.2.4-pytorch-2.13.0-gfx950 sleep infinity
docker exec rl-dev bash -c 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libaio-dev >/dev/null && echo LIBAIO_OK; ls /usr/include/libaio.h'
docker exec rl-dev bash -c 'python -c "import torch;print(torch.__version__, torch.version.hip)"; pip install -q deepspeed 2>&1 | tail -3; ds_report 2>&1 | grep -iE "async_io|cpu_adam|torch version|deepspeed info|hip version|compatible" | head -20'
docker exec rl-dev bash -c 'cd /tmp && python - <<PY
import time,torch
from deepspeed.ops.op_builder import AsyncIOBuilder
t=time.time(); m=AsyncIOBuilder().load(); print("async_io JIT build ok %.0fs"%(time.time()-t))
h=m.aio_handle(1048576,128,True,True,8)
buf=torch.empty(2*1024**3,dtype=torch.uint8).pin_memory() if False else torch.empty(2*1024**3,dtype=torch.uint8)
f="/rl/offload/aio_test.bin"
t=time.time(); h.sync_pwrite(buf,f); w=time.time()-t
t=time.time(); h.sync_pread(buf,f); r=time.time()-t
print("aio 2GiB write %.2f GB/s read %.2f GB/s"%(2.147/w,2.147/r))
import os; os.remove(f)
PY'
echo RL_ENV_DONE
