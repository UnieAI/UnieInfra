#!/bin/bash
# Stop every benchmark server (vLLM/SGLang container and unieinfra in ts-dev), wait for GPUs to free.
docker rm -f bench-server >/dev/null 2>&1
# The bracket tricks below keep this pattern from matching the very shell that runs
# it: a literal `pkill -f "tokenspeed::|smg::"` also matches its own `bash -c`
# command line, so the shell killed itself and the trailing `pkill -9` never ran
# (2026-11-10 19:39Z, a scheduler that had died of an OOM kept 127.7 GB on GPU 1
# through the next cell's start).
docker exec -u root ts-dev bash -c '
  P="smg[_]grpc_servicer|smg[:][:]|tokenspeed[:][:]|tokenspeed[ ]serve|deepseek-v4[.]1-flash-1017[.]sh|multiprocessing[.]spawn"
  pkill -f "$P"; sleep 6; pkill -9 -f "$P"; sleep 2
  # Whatever survived pkill (an unreaped rank can keep its KFD allocation) is
  # listed here and killed by pid instead.
  stray=$(ps -eo pid,args --no-headers | grep -E "$P" | grep -v "grep -E" | awk "{print \$1}" | tr "\n" " ")
  [ -n "$stray" ] && { echo "stop_servers: leftover pid(s) $stray"; kill -9 $stray 2>/dev/null; sleep 3; }
  exit 0' >/dev/null 2>&1
for _ in $(seq 1 60); do
  used=$(amd-smi metric --mem-usage 2>/dev/null | awk '/USED_VRAM/ {s+=$2} END {print s+0}')
  (( used < 8000 )) && exit 0
  sleep 2
done
# amd-smi is unprivileged here and can report a stale total after a hard kill, so
# confirm against the KFD table before giving up: no KFD process means the memory
# is free and the next server may start.
stray=$(docker exec -u root ts-dev /opt/rocm/bin/rocm-smi --showpids 2>/dev/null |
  awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -u | tr "\n" " ")
if [[ -z "$stray" ]]; then
  echo "stop_servers: amd-smi still reports ${used} MB but KFD lists no process" >&2
  exit 0
fi
echo "WARNING: GPU memory still in use: ${used} MB (KFD pid(s): $stray)" >&2
