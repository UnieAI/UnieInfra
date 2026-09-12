# Sparse state ownership and host restoration

Two scheduler defects affect generic checkpoint-sparse state groups. Model
arithmetic, dtype, kernels, and checkpoint publication interval are unchanged.

## Expired request ownership

Sparse allocation inserts logical holes between periodic checkpoints and backs
each live chunk tail. A backwards reclamation scan cannot infer from one null
slot that every older slot is null. That inference retained old request-owned
snapshots and also underestimated reclaimable capacity.

BlockTable now tracks an explicitly reclaimed prefix. Reclamation traverses only
newly expired slots, including holes; both forecasts use the same boundary.
Cached and external owners survive request-reference release. Free and prefix
replacement reset the cursor. Historical slots are not rescanned each decode.
Forecasts may still revisit an uncommitted range during admission attempts.

## Host restore must honor the selected snapshot

The old host-extension allocator and estimator discarded non-periodic hits.
But chunk tails and endpoints can be valid resumable snapshots off that grid.
Discarding the matched state left a hole subsequently mapped to the newly
allocated output page: a valid historical state was replaced by fresh state.

The host-extension estimator and materializer now reserve/load every non-null
hit selected by the existing prefix probe. Publication spacing constrains NEW
snapshots, not restoration of an existing one. Logical holes remain holes.

Reference source comparison was performed across vLLM, SGLang, TRT-LLM and
TurboMind; tileRT source was unavailable. SGLang explicitly allocates/restores a
matched host Mamba state independently of publication. vLLM converges offload
coverage against state requirements; TRT bounds matching by SSM snapshots;
TurboMind separates checkpoint resume selection from publication. No performance
of these references is inferred from source inspection.

## Validation and limits

- Sparse reclamation: old 66,560-token illustrative chunk trace retains11
  request-owned state pages; corrected trace retains1. This is a CPU ownership
  result, not a measured GPU memory reduction.
- Host restoration: both small and P128/stride512/551-page cases reproduce the
  missing transfer and aliased input/output state in the original.
- Coordinator admission: a13-parent demand is rejected at12 without table
  mutation and accepted at13 with all11 required host transfers. The original
  wrongly admitted at12 by dropping the state destination.
- Lifecycle tests cover cached snapshots, external owners, shared tables,
  prefix replacement, materialized suffix and one-token decode across pages.
- Integrated C++ suite passes452/452 tests; the core and Python extension build.
  A pre-existing misplaced binding argument was corrected identically to the
  frozen diagnostic source so the baseline checkout can build the extension.
- The pre-existing promotion test delivered ExtendResult before the final
  aligned-prefill tail was scheduled. Its schedule steps were corrected while
  preserving the promotion and reuse assertions; production FSM is unchanged.
- SM120 Qwen3.6-35B-A3B diagnostic, unchanged frozen Python runtime: the combined
  prototype completes32/32 long inventory answers correctly and all32 native
  sequences match the saved vLLM reference. Two retracted requests are among
  those exact matches. The prior reclaim-only candidate scored31/32; its failure
  began exactly after retraction. A551-block failing host load becomes554 after
  restoration includes the three state snapshots.

Artifacts are in the sibling `agentx-diagnosis-20260912` directory, especially
`SPARSE-RECLAMATION.md`, `SPARSE-HOST-RESTORE.md`, the original failing tests,
`sparse-restore-r0-recovery-vllm.json`, and the per-arm pressure manifests/raw
responses/scores. Failed artifacts were not removed or reclassified as passing.
The historical vLLM comparison is not a contemporaneous throughput comparison.

This is not general quality or2x AgentX acceptance. Same-run original results,
reversed repeats, broader task/model validation, commit-built full SLO curves,
and target-hardware checks remain required. Main user worktree changes were
left untouched; integration is on a separate worktree/branch.
