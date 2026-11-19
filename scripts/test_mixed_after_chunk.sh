#!/bin/bash
# Run the mixed-batch official cells AFTER the chunk sweep, not before it.
#
# Reordered 2026-11-10 09:45Z on new evidence: the c64 cap8 ladder cell collapsed
# to 32.4 RPM @ interactivity 39.7 (ITL p50/p90 10.8/25.2 ms, TTFT p90 305 s),
# i.e. BELOW the cap16 cell it was supposed to bracket (60.3 @ 49), and its tail
# ratio went 1.50 -> 2.33. A short decode step with a small batch is exactly what
# a 4096-token prefill pass punishes, so the prefill-chunk size is now the first
# thing to test at c64 cap16; mixed batching targets the same interference and can
# be measured afterwards. test_chunk_sweep.sh already waits for the ladder queue,
# so killing that queue starts the chunk cells immediately.
cd /home/smc/workspace/roy || exit 1
L=logs/mixed_after_chunk.log
: > "$L"
echo "MIXED-AFTER-CHUNK start $(date -u +%FT%TZ), waiting for the chunk sweep" >> "$L"
while pgrep -f "[t]est_chunk_sweep.sh" >/dev/null; do sleep 60; done
echo "chunk sweep finished $(date -u +%FT%TZ)" >> "$L"
exec scripts/test_mixed_official.sh
