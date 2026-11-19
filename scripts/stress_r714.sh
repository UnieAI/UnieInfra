#!/bin/bash
# ROCm 7.14.1 user-space (host /opt/rocm/core-7.14: HIP 7.14, RCCL, hipBLASLt/rocBLAS, comgr, kpack) overlay vs ROCm 7.2.4, interleaved A/B.
cd /home/smc/workspace/roy; M=logs/stress_r714.log; LOCK=logs/GPU_LOCK
until ( set -o noclobber; echo "owner=roy-a7(stress_r714) since=$(date -u +%FT%TZ)" > $LOCK ) 2>/dev/null; do sleep 15; done
trap 'rm -f $LOCK' EXIT
R714=/workspace/.rocm714
# microbenchmarks first (single GPU + 4-GPU RCCL), 2 rounds each runtime
docker exec ts-dev bash -lc "cd /tmp; B=/workspace/.rocm10/bench
for v in base r714 base r714; do
  if [ \$v = r714 ]; then export LD_LIBRARY_PATH=$R714/lib:$R714/lib/rocm_sysdeps/lib:$R714/lib/llvm/lib:/opt/rocm/lib PYTHONPATH=$R714/site; else export LD_LIBRARY_PATH=/opt/rocm/lib; unset PYTHONPATH; fi
  python \$B/micro2.py 2>/dev/null >> /tmp/m714_\$v.jsonl
  HIP_VISIBLE_DEVICES=0,1,2,3 timeout 300 torchrun --nproc-per-node 4 --master-port 29640 \$B/ar4.py 2>/dev/null | grep -v '^\[' >> /tmp/ar714_\$v.txt
done; cp /tmp/m714_*.jsonl /tmp/ar714_*.txt \$B/"
echo "$(date -u +%T) micro done" >> $M
WT=/workspace/.wt-final
run() { # $1 variant(R714|B) $2 conc $3 cap
  echo "$(date -u +%T) start $1 c$2" >> $M
  if [ "$1" = R714 ]; then PP=$R714/site:; LD=$R714/lib:$R714/lib/rocm_sysdeps/lib:$R714/lib/llvm/lib:/opt/rocm/lib; else PP=; LD=/opt/rocm/lib:/opt/rocm/lib; fi
  STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
  HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 \
  ENGINE_PYTHONPATH=$PP$WT/tokenspeed-kernel-amd/python:$WT/tokenspeed-kernel/python:$WT/python LD_LIBRARY_PATH=$LD \
  TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$3 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86 \
  EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
  scripts/run_round.sh R714$1c$2 600 unieinfra "$2"
  echo "$(date -u +%T) done $1 c$2" >> $M
}
run R714 16 32; run B 16 32; run R714 64 128; run B 64 128; run R714 4 32; run B 4 32
echo "R714 DONE $(date -u +%FT%TZ)" >> $M
