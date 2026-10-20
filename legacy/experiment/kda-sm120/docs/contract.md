# Task Contract — vLLM/UnieInfra serving performance on sm_120

## Objective

Find and land throughput improvements for UnieInfra's serving stack on
RTX PRO 6000 Blackwell Server Edition (sm_120, 97 GB, 4x, PCIe only, no NVLink),
for the Qwen3.x and gemma-4 model families at 30B scale.

Throughput is the primary metric. Per-user token rate is secondary and only
binds where an interactive SLO is stated.

## Inputs and outputs

- Input: the vendored `unieinfra-vllm` tree, `tokenspeed`, `tokenspeed-kernel`,
  and the checkpoints under `/workspace/models`.
- Output: code changes in those trees, each with a measurement that justifies
  it, plus recorded rejections.

## Correctness requirements

1. Any change that alters numerics must state how it alters them. Bit-identical
   is not required; a stated and bounded difference is.
2. Model output must be checked against a **reference path**, not against
   expectation. HF transformers with eager/sdpa is the reference. Reproducing
   the reference's *bad* output is a pass.
3. `-it` checkpoints must be exercised through their chat template at least
   once before any claim about output quality.

## Constraints

- GPUs 1, 2, 3 only. GPU 0 belongs to another tenant.
- `unieinfra-vllm-build` is the only working build; do not rebuild it without
  explicit confirmation.
- Do not modify shared containers' pip state.
- Only one vLLM-container experiment at a time: the harnesses use container-wide
  `pkill -f "VLLM::"`, which kills siblings. TokenSpeed runs in
  `unieinfra-dev` and is isolated by PID namespace.

## Validation command

Correctness: `scratchpad/gemma_hf_ref.py` pattern — same prompts through HF and
through the engine, compared as text.

## Evaluation command

`scratchpad/steady_bench.py` for serving throughput,
`scratchpad/ladder.py` for the comparison,
`scratchpad/stationarity.py` as an admission gate.

## Promotion criteria

A candidate is promoted only when **all** hold:

1. Both arms read `settled` from `stationarity.py`. A comparison against a
   non-converged arm is void regardless of its sigma.
2. The comparison uses **period-blocked means** (`ladder.py`), not window means.
   Throughput here settles into a period-2 limit cycle whose swing can reach 50%
   of the mean.
3. `client_captures_frac` is within 1.0 +/- 2% for both arms, or the comparison
   uses `server_tok_s`.
4. The effect is stated with the model, the concurrency, and the prompt shape.
   None of the effects measured here transfer across all three.
5. Switches are verified to have engaged, from the run's own log — not assumed
   from the flag being passed.

## Rejection rule

Record why. Three of this session's largest "findings" were rejected on
re-examination and the reasons are more useful than the original claims.
