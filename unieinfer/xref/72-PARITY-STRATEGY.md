# Decode parity strategy

Rebuilt 2026-10-18, after merging `upstream/main` into `unieai`, on the
`engine-decode-parity` skill. It supersedes the flat port list that grew out of
the 2026-10-16 measurement round: that list was written against a tree that no
longer exists, and it ordered work by intuition instead of by predicted
milliseconds.

## 0. Why the previous plan is void

The merge (`ac2f43a8`) replaced the DeepSeek V4 decode path wholesale. 26 files
the old plan named were deleted upstream (`models/deepseek_v4.py`,
`layers/deepseek_v4_mhc.py`, `ops/attention/triton/deepseek_v4.py`, the old
`csrc/cache/manager/*`), and the model is now
`models/deepseek_v41{,_moe,_engram,_dspark,_vision}.py`. Every per-family
attribution in `DECODE_PARITY.md` describes kernels that are no longer on the
decode path, so it cannot size the work. It remains valid only as a record of
the *method* that produced it.

Three items on the old list upstream now implements itself:

| item | old plan | merged tree | reachable on H200? |
| --- | --- | --- | --- |
| mHC seam fusion | our hand-written Triton `mhc_fused_hc` (2.13 ms of seams) | `thirdparty/cuda/mhc.py::mhc_big_fuse`, used as `pre_reduce_apply_impl` by `ops/residual/deep_gemm.py::deep_gemm_mhc_pre` | **yes** — `min_arch_version=ArchVersion(9, 0)`, `supported_hc_mults={4}`, `supported_n_splits={1..64}`, `supports_fused_norm=True`; our model is hc_mult 4 / sinkhorn 20 |
| MoE routing weights | our bf16-routing patch | upstream #1770 | yes |
| decode GEMV expert count | "widen the gate to 8 experts" | upstream decode GEMV | yes |

Still absent upstream, so still candidates to port: `fused_compress_quant_cache`,
`indexer_k_store`, `fused_wo_a` (vLLM 0.31). Absence is not an argument —
§3 gates them like everything else.

## 1. The quantities that decide the work

Skill §1–2. Fill this table from the merged tree before acting; the numbers
below are the pre-merge ones and are shown only so the post-merge delta is
visible.

| quantity | how to read it | pre-merge ours | vLLM 0.31 c16 |
| --- | --- | --- | --- |
| ms/step | wall clock between decode steps, bs=16 | 23.59 | 16.66 |
| kernels/step | trace kernel count ÷ steps | 4 271 | — |
| µs/kernel | ms/step ÷ kernels | 5.5 | — |
| device-busy | busy span ÷ wall | 21.53 (91%) | 22.70 ⚠ |
| MoE bandwidth floor | 41 GiB ÷ 4.8 TB/s | 8.6 | 8.6 |
| elementwise total | sum of kernel-node time | 2.88 | 0.14 |
| mHC seam total | 2 seams/layer × 43 layers | 2.13 | 0.96 |

How these are read now: `bench/agent-slo/profile_decode.sh PORT MODEL CONC DATA
OUTDIR` drives a decode-only load against a live server, calls the engine's own
`/start_profile` on the control port (`main + 1`), and runs
`bench/agent-slo/trace_stats.py` over the resulting chrome trace to print
ms/step, kernels/step, µs/kernel, device-busy and host remainder -- all on one
span, from one file. The tooling is measured, not scaffolding: nothing has to be
added to the hot path, and `--kernel mhc_big_fuse` answers G1's "is the fast path
selected, or only registered" from the same trace.

⚠ vLLM's device-busy exceeds its own wall clock, which is impossible: those two
figures come from different spans. **Do not use device-busy to argue "GPU-bound"
until it is re-measured on the same window as ms/step.** The skill's 0.94
busy-ratio heuristic is only as good as the span.

## 2. Gate G0: re-baseline (no optimization work before this is done)

The merge changed the algorithm, the model file, and the C++ cache manager at
once. Three things must be true before any item in §4 is started.

G0 also runs in a different image. `ts-env:gpu` carried
`tokenspeed-scheduler` 0.1.6 while the tree requires `>=0.1.39`, so its C++
scheduler predated the merge; the numbers in §1 were produced that way. The
wheel is rebuilt from `tokenspeed-scheduler/` and baked into
`ts-env:gpu-sched39`, which is what the sweep scripts now use -- so G0 is a
re-baseline in the literal sense, not a comparison against §1.

1. **It runs.** Container boot on GPUs 6,7 reaches `/health`; `--help` parses;
   one c16 AgentX cell completes and produces a JSON.
2. **It is numerically intact.** The kernel ops tests for `mhc_big_fuse` /
   `dsv4_topk` pass, and a temperature-0 response is token-for-token comparable
   to the pre-merge engine on a fixed prompt (skill §4: the corrupt arm can look
   faster). Our own mHC equivalence test was removed with the merge -- it
   targeted the layer module that no longer exists, and upstream does not test
   its replacement -- so the token-agreement check is currently the *only* guard
   on the fused mHC seam (`73-MERGE-LEDGER.md` §5).
3. **No fix went missing in the merge.** Our three landed fixes were in files we
   resolved to upstream. Flag audit against
   `python/tokenspeed/runtime/utils/server_args.py`:

   | flag the harness uses | merged tree |
   | --- | --- |
   | `--disable-kvstore`, `--reasoning-parser`, `--gpu-memory-utilization`, `--max-num-seqs`, `--max-model-len`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--block-size`, `--prefill-graph-max-tokens`, `--enable-mixed-batch` | present |
   | `--admission-reserve` | **gone (0 hits)** |

   The harness used to pass `--admission-reserve generation-full` on every
   `ours` cell and would now fail to boot; the flag was dropped from
   `agentx_sweep.sh`, `fused_indexer_ab.sh` and `verify_fused_indexer.sh`, and
   the two survey pages that still described it are annotated. Upstream changed the admission
   mechanism rather than the flag (`server_args.py` now describes "the shortest
   L3 prefix worth one" in prefix pages). Resolve by re-deriving what the flag
   was buying — it was a workaround for a wrong admission quantity, which is
   exactly the class of defect AGENTS.md says to fix at the source — and drop it
   from the harness with a comment recording the replacement. Re-run the two
   guard tests that pinned the old behaviour
   (`--max-cudagraph-capture-size`, prefill ladder) since both defaults now come
   from upstream, not from us.

G0 produces `ab_real/agentx-sweep/post_merge_baseline.json` plus a decode trace
at bs=16. Nothing else starts until it exists.

## 3. How the work list is built (skill §3, applied)

Every candidate must carry four things before it is scheduled:

1. **the quantity it changes** — ms/step, kernels/step, TTFT tail, or memory;
2. **predicted ms saved**, as arithmetic on measured numbers, not a range;
3. **the diagnostic that proves the defect exists in the merged tree** — not in
   the pre-merge trace;
4. **a kill criterion.** Run-to-run noise floor measured on this rig is 1.008×
   on a long cell (skill §4). An item predicted below **2% of ms/step (~0.2 ms)**
   is not scheduled; below 2× that, it is batched with a neighbour that touches
   the same seam, never run as its own A/B.

The five questions from the skill's catalogue that this engine can still answer:

- **A. fold the epilogue into the producer.** mHC seams were the case; upstream
  did it. Next candidate: the residual-add + quant epilogue after the MoE
  combine.
- **B. one launch per seam.** Count launches per seam in the new trace. A seam
  that costs 4 launches × 10 µs × 2 seams × 43 layers is 3.4 ms of pure launch
  floor, and that is a target no parameter can reach.
- **C. recompute instead of materialize.** Candidate: indexer/top-k metadata that
  is materialised per step and consumed once.
- **D. compute over the visible set, not capacity.** Candidate: indexer rows
  allotted per page-table capacity (the old trace showed 41 GiB of MoE traffic
  and a 524 288-column row elsewhere; check whether any surviving kernel does
  the same).
- **E. select instead of sort.** The old path paid 4 CUB `torch.nonzero`
  launches plus an 86-node reorder chain per step. The merged path has no
  `torch.nonzero` in `layers/attention`, so this must be re-derived or dropped —
  do not carry it forward on faith.
- **F–J.** dtype production, per-step hoisting, reuse of an existing kernel,
  skipping work that cannot produce output, shape specialisation with declared
  capability attributes. J is where the three unported vLLM ops land, and each
  must still pass §3.2.

`--admission-reserve` is the standing example of what not to do here: it moved a
number, never explained it, and the merge deleted it without a single test
failing.

## 4. Phase table

Order is by predicted saving, and each phase ends with a re-measured ms/step.
A phase that does not move the number by more than the noise floor is reverted
rather than kept "because it should help".

| phase | change | evidence needed first | predicted | stop rule |
| --- | --- | --- | --- | --- |
| G0 | re-baseline | — | — | blocking |
| G1 | verify `mhc_big_fuse` is actually selected at bs=16 (kernel name in trace) | mHC seams ≥ 0.2 ms in new trace | up to −1.1 ms vs our old seams | if already selected, close |
| G2 | launch-count reduction at the two worst seams | launches/step > 1 500 | −(excess/2) × 10 µs | < 2% saved → stop |
| G3 | hoist per-step metadata out of the layer loop | per-step glue > 0.2 ms | 0.2–0.5 ms | inert hoist = revert (an unused `memo` argument in the now-deleted `models/deepseek_v4.py` survived a whole round doing nothing) |
| G4 | select-not-sort on the surviving reorder site | site exists | 0.1–0.3 ms | not scheduled below 0.2 ms |
| G5 | port one of the three absent vLLM ops | that op's node time in new trace | node time | one op per phase, measured alone |
| G6 | TTFT tail | c32/c64 cell | p95/p99 only | corpus effect must be excluded first (cold cohort) |

Parity bar (skill §0): total tps, rpm, tokens/req and TTFT p50/p90/p95/p99
within 5–10% of vLLM at the same concurrency, and tokens/req within 10% or the
comparison is void.

## 5. Out of scope, with reasons

- **MoE grouped GEMM.** At the HBM floor (8.6 ms of 16.66 ms for vLLM). No
  algorithm reaches below it; only a smaller working set would.
- **Flag sweeps.** AGENTS.md; and the one flag this plan inherited has now been
  deleted by upstream, which is the proof.
- **sglang arm.** Previous cells ran at 41 tokens/req against 127 for ours, with
  `No HuggingFace chat template found`. Until it runs the same corpus to the same
  length, its p95/p99 are measuring a different workload.
- **`mega_mhc` (family 100) and `all_reduce_mhc` (TP4 / 5120 / MNNVL).** Not
  reachable at TP2 / hidden 4096 / sm_90.

## 6. Verification ladder, per change

1. the unit test for the kernel or layer;
2. `assert_no_triton_compile`-style check where a new parameter could take
   per-request values (skill §4: JIT cache key);
3. numerics against the previous engine on a fixed prompt at temperature 0 —
   bf16 1 ULP is 0.008 near 1.0 and 0.031 near 4.0, so compare against the
   pre-change output, not against a hand-written constant;
4. ms/step at bs=16, then the c8/c16/c32 sweep, then the parity table.

## 7. Risk register

| risk | signal | action |
| --- | --- | --- |
| merge silently dropped a fix | flag/behaviour audit in G0 | restore at the new location, with a test |
| upstream's fused path is inert on sm_90 | no `mhc_big_fuse` kernel in trace | treat as G1 work, not a config change |
| the +36% seen in `mhcfuse2` c16 was cohort noise | c8 reproduced the old numbers | re-run c16 three times before attributing anything |
| a dropped behaviour is mistaken for an obsolete one | `73-MERGE-LEDGER.md` audit | every dropped delta carries a verdict and its evidence; `prefill_defer_window` is the only behaviour still unported, with a kill criterion |
| verification debt: re-ported fixes are unverified until the GPUs free up | `weak_ref.py` restore, capture-size warning | both are exercised by the first G0 boot; nothing is claimed before that |
