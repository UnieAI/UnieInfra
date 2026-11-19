#!/usr/bin/env bash
# co16a (aiter + coalesce 100) came in at 42.4 RPM / 569 out/s, i.e. coalesce
# is worth ~0: the 11-10 60.5 RPM cell at the same 900 s agentx profile must
# differ in the MoE backend (it predates the aiter registration, so it ran the
# heuristic gluon path). Reproduce it through the supported CLI knob
# --moe-backend gluon rather than the blind kernel override.
set -u
cd /home/smc/workspace/roy
LOG=logs/queue3.log
say(){ echo "== $(date -u +%FT%TZ) $*" >> $LOG; }
for _ in $(seq 1 120); do
  grep -q QUEUE2_DONE logs/queue2.log 2>/dev/null && break
  sleep 15
done

export GPU_MEMORY_UTILIZATION=0.95
export MAX_NUM_SEQS=16
export TOKENSPEED_KERNEL_VERBOSE=1
export EXTRA_SERVE_ARGS="--moe-backend gluon --prefill-coalesce-ms 100"
say "cell gl16: c64, 16 slots, gluon + coalesce 100"
bash scripts/run_round.sh gl16 900 unieinfra 64

export EXTRA_SERVE_ARGS="--moe-backend gluon --prefill-coalesce-ms 100 --sampling-backend triton_full"
say "cell gl16f: c64, 16 slots, gluon + coalesce 100 + triton_full"
bash scripts/run_round.sh gl16f 900 unieinfra 64

export MAX_NUM_SEQS=32
say "cell gl32: c64, 32 slots, gluon + coalesce 100 + triton_full"
bash scripts/run_round.sh gl32 900 unieinfra 64

say QUEUE3_DONE
