# General Agent Guidelines

> If a `AGENTS.local.md` file exists alongside this file, read and respect it--
> it contains developer-specific overrides that supplement this shared guidance.

## Development environment

* Before any work, check local Python venv and activate if one exists.
* Don't install pip packages outside the local Python venv if one exists.

## Code changes

* Add tests and update docs for the changed code.
* Use absolute imports instead of relative imports.
* Use the repository's full MIT license header for copyright notices; do not use
  an abbreviated copyright-only header.
* Before creating commits, run `pre-commit run --all-files` to format.
* Do not substitute a narrower lint command for the repository hook before
  committing. Always run the exact `pre-commit run --all-files` command and
  commit any formatter changes it makes.
* When creating commits, perform sign off on behalf of the author.

## Dependency boundaries

* `tokenspeed` runtime dependencies should stay vendor-neutral.
* Runtime code should use `tokenspeed-kernel` as its only kernel package
  boundary.
* Third-party kernel libraries belong under `tokenspeed-kernel`; avoid direct
  runtime dependencies or imports that bypass it.
* If a dependency repeatedly breaks during version upgrades or slows project
  progress, consider removing it entirely or at least making it optional.

## tokenspeed-kernel

Inside the root `tokenspeed-kernel/` directory:

* All direct tokenspeed-triton imports should happen in `_triton.py` and then
  re-import to other places.
* All direct third-party code should be placed in `thirdparty/` and imported
  into `ops/` then registered via `register_kernel`.
* Prefer CuteDSL for NVIDIA GPU kernels and Triton Gluon for AMD GPU kernels.
  Use Triton for portable solutions across vendors. Vendor libraries should
  stay optional, and other solutions may be used as temporary transitions, but
  new work should consolidate toward these backend choices.
* Files under `ops/` should follow `<family>/<solution>` structure, like
  `gemm/trtllm.py` or `attention/triton/`.
* When defining new public APIs, explain arguments and returns in docstring.

## tokenspeed-kernel-amd

Inside the root `tokenspeed-kernel-amd/` directory:

* There should be no dependency on `tokenspeed-kernel`.
* AMD Gluon Kernel tests should live in `tokenspeed-kernel/test/` to reuse
  common platform utilities and reference computations.

## Performance work: go to the algorithm, not the flags

**A parameter sweep is a diagnostic, never a deliverable.** Set parameters to
their measured optimum once, then stop touching them and go find what makes that
optimum necessary. If a flag changes the number, the flag is telling you where
the defect is; it is not the fix.

The pattern that keeps costing days, with the 2026-08-26 gemma-4 case as the
worked example:

1. A metric looks wrong (TTFT tail: p50 3.8 s, max 41 s, against vLLM's max
   12.7 s at the same mean).
2. Flags get swept -- `admission-reserve`, `preempt-on-blocked-request`,
   `disable-mixed-batch`, `chunked-prefill-size`, `max-num-seqs`. Some move the
   number. None explains it, and the ones that "worked" were later refuted
   (`preempt_on_blocked_request` measured -36%, then -61% when paired).
3. The actual defect was one missing virtual override:
   `SwaManager` never overrode `AdmissionBlocksNeededFor`, so all five sliding
   groups were charged the whole prompt span (172 pages) instead of their true
   peak (window + one step = 145). Admission demanded 947 parents against 925
   free and refused -- 6,136 times across 92 requests in one 112-request run.

**Rules that follow from it:**

* **Ask what quantity the code computes before asking what value to give it.**
  The gemma defect was a wrong quantity, and no value of any flag could fix a
  wrong quantity.
* **Read the engine's own diagnostics first.** `parents_needed=947
  empty_parents=925` with a per-group breakdown was already being printed behind
  `.plan_trace.on`. It named the defect in one line. Several days of flag
  sweeping preceded anyone reading it.
* **A capability-gated kernel is an algorithm problem, not a config problem.**
  `mha_extend_with_kvcache` registered only a trtllm-gen kernel (needs tcgen05,
  <= sm_103) and a Triton fallback, so sm_120 silently ran the fallback at 1.4x
  the TTFT. The fix was to register the flashinfer FA2 path -- and it still sits
  behind an env var, which is the same defect one level up.
* **When a survey exists, read it before measuring.** `unieinfra/xref/` already
  placed all five engines on the admission axis and named the recommended cell.
  Re-deriving it from A/B runs cost hours and produced a worse answer.

### Worked example 2: a flag that existed to hide a defect

`.matrix.sh` served gemma-4 with `--disable-mixed-batch`. The flag was not a
tuning choice -- it was there because mixed batching corrupted gemma's output.
So the RAG cell's 0.71x versus vLLM was not "our scheduler is slower"; it was
"we are running with prefill/decode overlap switched off to avoid a bug".

What found it: making the engine print the quantity it was actually computing.
One line at the mixed KV write site reported
`locs=8191 total=8192 prefill=8191 decode=1`. The per-group write locations were
built from `extend_seq_lens_cpu`, which on a MIXED step covers the prefill rows
only, and `_trim_kv_to_locs` then silently trimmed k/v to that count -- so every
decode row's newest token never entered the cache.

Three rules this adds:

1. **A disable flag is a bug report.** Before treating a disabled feature as a
   performance ceiling, find out why it is disabled. Read the commit or the
   comment that introduced it.
2. **A silent trim is a correctness bug.** Any helper that reconciles two sizes
   by shrinking one of them must say so, or be replaced by a check. This one was
   documented as a padded-replay convenience and was load-bearing for a defect.
3. **The corrupt arm can look faster.** It scored +25.9% tok/s because the broken
   KV made it emit longer garbage. Gate every A/B on output length AND on
   token-for-token agreement with a reference engine at temperature 0 -- 6/256
   agreement versus 205/256 is what actually separated them, and neither
   character-repetition nor "well-formed JSON" would have caught it.
