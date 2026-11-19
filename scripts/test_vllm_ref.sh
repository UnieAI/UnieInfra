#!/bin/bash
# vLLM reference rerun at the two concurrencies our headline ratio comes from.
# The reference so far is one 900 s cell per concurrency with the recipe's own
# max-num-seqs (CONC/4 above c32), and its c128 cell (355 out tok/s, ITL p90
# 104.5 ms) looks degenerated next to its own c64 cell (1028 tok/s), so a 4.28x
# ratio there would credit our engine for the reference's cap choice. These
# cells give vLLM the cap freedom we give ourselves; fair.reference() picks the
# fastest valid cell per concurrency, so a better vLLM point replaces the old one
# automatically rather than being argued about.
# Usage: scripts/test_vllm_ref.sh   (logs to logs/vllm_ref.log)
cd /home/smc/workspace/roy || exit 1
L=logs/vllm_ref.log
: > "$L"
echo "VLLM REF start $(date -u +%FT%TZ)" >> "$L"
# conc:cap - the cap is passed through VLLM_MAX_NUM_SEQS, which serve_vllm.sh
# appends after the recipe's per-concurrency max-num-seqs.
for cell in 128:64 128:128 64:32; do
  conc=${cell%%:*}
  cap=${cell##*:}
  echo "start conc=$conc cap=$cap $(date -u +%T)" >> "$L"
  VLLM_MAX_NUM_SEQS=$cap scripts/run_round.sh "off${cap}" 900 vllm "$conc" >> "$L" 2>&1
  echo "done conc=$conc cap=$cap rc=$? $(date -u +%T)" >> "$L"
done
echo "VLLM REF DONE $(date -u +%FT%TZ)" >> "$L"
