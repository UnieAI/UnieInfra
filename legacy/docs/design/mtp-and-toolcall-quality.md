<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# Where tokenspeed's MTP Acceptance and Tool-Call Correctness Actually Come From

Status: **Study — read-only code recon, no GPU, nothing implemented** · Date: 2026-07-25 · Owner: royshih (drafted by agent)

## Method and honesty rules

Read-only study of two trees at fixed pins. No source modified, no GPU used (four
agents were working, two holding GPUs 2/3). Trees:

- Old base: `python/tokenspeed/` (SGLang-derived), working tree at `unieinfra-base`.
- New base: `third_party/vllm/` @ `d02df748b`, byte-identical to `unieinfra-vllm/`
  (verified: `diff -rq unieinfra-vllm/vllm/v1/spec_decode third_party/vllm/vllm/v1/spec_decode`
  and the same for `structured_output/` both return empty). Citations below use
  `third_party/vllm/`; they apply unchanged to the vendored fork.

Every claim is labelled:

- **[verified]** — read directly from source at the cited `file:line`.
- **[inferred]** — a conclusion drawn from verified code, but not stated by the code.
- **[measured]** — cites `docs/experiments-log.md`.
- **[unverifiable]** — could not be settled from this checkout; the reason is given.

Where a claimed advantage has **no mechanism visible in the code**, this document
says so instead of inventing one. That happens once, in Part 2, and it is the most
important sentence in the document.

---

## Verdict, in three sentences

1. **MTP acceptance: the advantage is real and has exactly one dominant cause — a
   capability gap, not an algorithm subtlety.** vLLM's Inkling MTP is hard-capped at
   **one** speculative token and builds **only depth layer 0**
   (`third_party/vllm/vllm/models/inkling/nvidia/mtp.py:97-100`, `:103-106`, `:204-206`);
   tokenspeed builds all depth layers and drives them with a genuine multi-depth
   drafter (`python/tokenspeed/runtime/models/inkling_nextn.py:156-164`,
   `python/tokenspeed/runtime/execution/drafter/mtp.py`). Capped at K=1 the best
   possible mean accepted length is 2.0; we measured **3.72/4** at K=4 (E32). The
   accept/reject *rule* is at parity — both lossless, both target-only, both greedy
   drafting — so this is a draft-quality/capability story end to end.
2. **Tool calling: there is no in-engine mechanism in tokenspeed at all.** The
   engine has zero tool parsers, zero `tools`/`tool_choice` handling, and ships with
   `grammar_backend="none"` (`python/tokenspeed/runtime/utils/server_args.py:239`).
   Whatever produced Toolcall-15 100% lives in the closed-source Rust `smg` gateway,
   which is not in this repo and not installed here. **The claimed tool-calling
   advantage is not attributable to any code we own.** What *is* ours and worth
   keeping is `capturable_grammar.py` — but that is a throughput mechanism, not a
   correctness one.
3. **The real portable tool-call lever is on the vLLM side and is a policy default,
   not missing machinery:** vLLM has excellent constrained decoding and simply does
   not turn it on for the common case
   (`third_party/vllm/vllm/tool_parsers/structural_tag_registry.py:106-110`).

---

# Part 1 — MTP acceptance rate

## 1.0 A measurement caveat that must be stated first

Acceptance *rate* (accepted ÷ drafted) is not comparable across engines unless the
draft length K matches, and mean accepted length is bounded above by K+1. Any
cross-engine acceptance comparison that does not pin K is measuring K, not quality.

- tokenspeed defaults to `speculative_num_steps=3`, hence
  `speculative_num_draft_tokens = steps + 1 = 4`
  (`python/tokenspeed/runtime/utils/server_args.py:257`, `:392-393`). **[verified]**
- vLLM has no default; `num_speculative_tokens` must be supplied
  (`third_party/vllm/vllm/config/speculative.py:995-999` raises if absent).
  **[verified]**
- For Inkling specifically vLLM *forbids* anything but K=1
  (`third_party/vllm/vllm/config/speculative.py:1002-1007`). **[verified]**

So for the Inkling family the two engines cannot even be run at the same K. That is
the finding, not a confound to be corrected away.

## 1.1 The accept/reject rule — parity, with one asterisk each

Separating the two explanations as required: **the accept/reject rule explains none
of the gap.** Both engines implement the same lossless chain rejection sampler.

| Property | tokenspeed | vLLM | Verdict |
|---|---|---|---|
| Greedy (T=0) rule | argmax equality, `verify_chain_greedy` (`python/tokenspeed/runtime/sampling/backends/flashinfer.py:468-479`, `.../greedy.py:223-233`) | argmax equality, `rejection_greedy_sample_kernel` (`third_party/vllm/vllm/v1/sample/rejection_sampler.py:715-769`) | **parity** |
| Stochastic rule | `chain_speculative_sampling_target_only`, `draft_probs=None` (`.../flashinfer.py:519-532`) | `rejection_random_sample_kernel` with `NO_DRAFT_PROBS` branch `draft_prob = 1` (`rejection_sampler.py:805-834`) | **parity** |
| `q` used | one-hot (target-only) | one-hot by default; true `q` only under opt-in `draft_sample_method="probabilistic"` (`third_party/vllm/vllm/config/speculative.py:283-286`) | **vLLM has a knob we lack** |
| Target `p` conditioning | temp → top-k → top-p renorm *before* the test (`.../flashinfer.py:497-518`) | same order (`rejection_sampler.py:510-565`, `:472`) | **parity** |
| Relaxed / typical acceptance | thresholds exist but are hard-set to 1.0 = no relaxation (`python/tokenspeed/runtime/sampling/backends/base.py:40-41`) | none in the standard path; `synthetic` mode is a benchmark harness only (`third_party/vllm/vllm/config/speculative.py:216-236`) | **parity — both lossless** |
| Tree vs chain | chain only; `speculative_eagle_topk > 1` raises at startup (`python/tokenspeed/runtime/utils/server_args.py:622-626`) | chain only; `FIXME: when using tree-based specdec…` (`third_party/vllm/vllm/v1/spec_decode/llm_base_proposer.py:1633-1634`) | **parity** |
| Draft sampling | greedy argmax, fused into the draft's `LogitsProcessor` via `do_argmax=True` (`python/tokenspeed/runtime/models/deepseek_nextn.py:276`, `.../glm5_nextn.py:221`, `.../llama_eagle3.py:578`); drafter falls back to `sampling_argmax` (`.../drafter/eagle.py:506-509`) | greedy argmax by default (`speculative.py:283`), `_greedy_sample` (`llm_base_proposer.py:428-438`) | **parity** |

**[verified]** for every row above.

One honest asterisk on tokenspeed's side: `SPECULATIVE_ACCEPT_THRESHOLD_SINGLE/_ACC`
are module constants, not server args
(`python/tokenspeed/runtime/sampling/backends/base.py:40-41`). The SGLang lineage
that these come from exposes them as tunables; ours are frozen at the lossless
value. Good default, but it means nobody can accidentally buy acceptance with
distribution drift — worth preserving as a property when porting. **[verified]**

One honest asterisk on vLLM's side: when the per-request cached draft probs go
missing, vLLM silently falls back to the one-hot rule with only a warning
(`third_party/vllm/vllm/v1/worker/gpu_model_runner.py:4979-4986`). Still lossless,
strictly lower acceptance. **[verified]**

## 1.2 Draft quality — the chained/Eagle path is near-parity

For single-MTP-layer models (DeepSeek NextN, Qwen3.5 NextN, GLM5 NextN) tokenspeed
routes to the `Eagle` drafter, not `Mtp`: only `InklingForConditionalGenerationNextN`
gets the multi-depth path
(`python/tokenspeed/runtime/execution/model_executor.py:93-101`). **[verified]**
E32's 3.716/4 was measured on Qwen3.5-9B, i.e. on the **Eagle** path. **[measured]**

On that path the two implementations are structurally the same:

| Mechanism | tokenspeed (`drafter/eagle.py`, `models/*_nextn.py`) | vLLM (`llm_base_proposer.py`, `models/*_mtp.py`) |
|---|---|---|
| Hidden-state hand-off | fused, passed as `captured_hidden_states` (`eagle.py:330`) | fused, `model_kwargs["hidden_states"]` (`llm_base_proposer.py:988-989`) |
| Fusion in the head | `pre_fc_norm_embedding`/`pre_fc_norm_hidden` → `fc(2H→H)` (`qwen3_5_nextn.py:174-179`, `:257-262`); `enorm`/`hnorm` → `eh_proj` (`deepseek_nextn.py:151-154`, `:185-190`) | identical (`qwen3_next_mtp.py:117-124`, `deepseek_mtp.py:126-133`) |
| Steps 2..K hidden | drafter's own output recycled (`eagle.py:426`) | drafter's own output recycled (`llm_base_proposer.py:713-714`, `:747-754`) |
| Depth-layer selection | `layers[spec_step_idx % num_mtp_layers]` (`inkling_nextn.py:197`) | `layers[spec_step_idx % num_mtp_layers]` (`deepseek_mtp.py:229-230`) |
| Draft KV write position | `valid_cache_lengths + accept_lengths`, i.e. the committed position (`eagle.py:355-364`) | rejected slots masked to `PADDING_SLOT_ID`, `seq_lens -= num_rejected` (`spec_decode/utils.py:265-269`, `llm_base_proposer.py:674-678`) |
| Attention kept off the rejected tail | `_advance_draft_forward_metadata_if_supported` + the comment at `eagle.py:413-418` | `seq_lens` rollback, same effect |
| Precision | target dtype throughout | target dtype throughout (`llm_base_proposer.py:85`, `:202-204`) |

**[verified]** for all rows. **[inferred]:** on single-layer MTP heads there is no
structural reason for tokenspeed to accept more than vLLM, and I found none in the
code. If a benchmark shows one on such a model, suspect K, the prompt corpus, or
weight sharing (§1.4) before suspecting the algorithm.

## 1.3 The one decisive difference: multi-depth MTP

This is the mechanism.

**vLLM builds one depth and refuses the rest.** `InklingMultiTokenPredictor`
constructs `nn.ModuleDict({"0": InklingMTPDepthLayer(...)})` — a single entry
(`third_party/vllm/vllm/models/inkling/nvidia/mtp.py:103-106`); the constructor
raises unless `num_speculative_tokens == 1` (`:97-100`); the forward raises on
`spec_step_idx != 0` with `"Inkling MTP only supports spec_step_idx=0"` (`:204-206`);
and `SpeculativeConfig` enforces the same at config time
(`third_party/vllm/vllm/config/speculative.py:1002-1007`). **[verified]**

**tokenspeed builds every depth and runs them.**
`InklingMultiTokenPredictor.layers` is an `nn.ModuleList` over
`range(self.num_mtp_layers)` where `num_mtp_layers = config.num_hidden_layers`
(`python/tokenspeed/runtime/models/inkling_nextn.py:146`, `:156-164`), and the
`Mtp` drafter drives depth `d` at step `d`
(`python/tokenspeed/runtime/execution/drafter/mtp.py:612-646`). **[verified]**

The `Mtp` drafter is not just "the Eagle loop with more layers". Three things in it
are absent from vLLM entirely:

1. **Full-window re-run per depth.** Depth `d` re-runs over the *same k-row verify
   window* the first step processed — same positions, same `out_cache_loc` — because
   multi-depth heads are trained as full-sequence layers whose attention at position
   `p` must see depth-`d` activations of all positions `≤ p`
   (`mtp.py:576-601` docstring, loop at `:612-646`). vLLM's steps 2..K run **one row
   per request** (`llm_base_proposer.py:682-761`, `input_num_tokens = batch_size`),
   which for a genuinely multi-depth head means depth-`d` queries attend over KV that
   depth `d` mostly never wrote. **[verified]**
2. **Lookback KV repair.** `_run_multi_step_lookback` (`mtp.py:648-771`) prepends
   `D = steps-1` rows covering the last committed positions, whose KV and conv state
   were first written while their input tokens were still *unverified drafts*, and
   overwrites them from now-committed tokens (`mtp.py:659-668`). Cross-round stashes
   `_lookback_tokens_buf` / `_lookback_hidden_buf` are allocated eagerly because the
   decode rounds that fill them run inside CUDA-graph capture (`mtp.py:386-402`).
   **vLLM has no analogue** — its draft KV for rejected positions is simply left
   stale and overwritten next round. **[verified]**
3. **Per-depth prompt coverage at EXTEND.** `_run_extend_depth_catchup`
   (`mtp.py:843-953`) re-runs each depth over the prompt chunk, including *mid*
   chunks, with the explicit reason that skipping them "would leave every chunk but
   the last uncovered for depths ≥ 1 on long prompts" (`mtp.py:1024-1033`).
   **[verified]** vLLM does not need this because it only ever has depth 0 — which is
   precisely the point.

**[inferred] Consequence.** On Inkling-family checkpoints vLLM's ceiling is a mean
accepted length of 2.0 tokens/step. tokenspeed measured **3.716/4 (acceptance 0.905)
at C=1** and **3.575/4 (0.858) at C=16** on the benchmark corpus, and **3.37/4
(0.79)** on realistic prose/code prompts (E32, `docs/experiments-log.md:21`).
**[measured]** That is the whole reported MTP acceptance advantage, and it is a
capability gap, not a tuning difference.

## 1.4 Two smaller, cheaper draft-quality levers tokenspeed gets right

- **The draft head is never quantized.**
  `speculative_draft_model_quantization` defaults to `"unquant"` and is resolved to
  `None` (`python/tokenspeed/runtime/utils/server_args.py:256`, `:601-602`), and
  `qwen3_5_nextn.py:166-169` additionally force-drops nvfp4 for the MTP head with the
  comment "The MTP model is unquantized in the nvfp4 checkpoint." **[verified]**
  vLLM by contrast builds an *independent* draft config with its own `quant_config`,
  `moe_backend` and `kv_cache_dtype`
  (`third_party/vllm/vllm/v1/spec_decode/llm_base_proposer.py:1277-1313`;
  separate draft KV quant at `vllm/config/speculative.py:1304-1311`). A separately
  quantized drafter is a direct acceptance tax with no diagnostic. **[inferred]**
- **Embedding + LM head sharing is unconditional for EAGLE3/MTP.**
  `model_executor.py:390-394` calls `get_embed_and_head()` / `set_embed_and_head()`
  with no equality test. **[verified]** vLLM decides heuristically, by a **bitwise**
  `torch.equal` on CPU copies (`llm_base_proposer.py:1443-1479`) and disables sharing
  entirely under PP > 1 (`:1509-1513`); a near-identical-but-not-equal head silently
  keeps separate weights. **[verified]** **[inferred]** a silent sharing failure
  costs acceptance and produces no log line.

Related and worth recording because vLLM's own comment quantifies it: feeding the
Inkling MTP head raw embeddings instead of backbone-normed ones "drops MTP1
acceptance from ~0.85 to ~0.70"
(`third_party/vllm/vllm/models/inkling/nvidia/mtp.py:118-124`). tokenspeed applies
the same base `embed_norm` before each depth's own norm
(`python/tokenspeed/runtime/models/inkling_nextn.py:173-179`, `:194-197`).
**[verified]** Both sides already get this right; it is here as a regression tripwire.

## 1.5 Where vLLM is already equal or better — say it plainly

The ledger has to be correct, so:

1. **Dynamic speculation length.** vLLM ships a batch-size → K schedule
   (`third_party/vllm/vllm/v1/spec_decode/dynamic/utils.py:76-148`, consumed at
   `vllm/v1/core/sched/scheduler.py:1122-1126`), with K=0 still running the drafter
   forward to keep its KV in sync (`llm_base_proposer.py:607-616`). tokenspeed has
   **nothing** here — `docs/design/adaptive-speculative-decoding.md` is explicitly
   "Design — NOT implemented". vLLM's version is a static lookup, not
   acceptance-feedback, but it exists and ours does not. **[verified]**
2. **Coupled draft/target Gumbel noise.** The V2 runner deliberately offsets the
   draft's noise to match the target's
   (`third_party/vllm/vllm/v1/worker/gpu/spec_decode/speculator.py:292-303`) — this
   raises acceptance for free under stochastic sampling. tokenspeed's draft is
   unseeded argmax. **[verified]**
3. **Block verification** (Sun et al., arXiv:2403.10444) —
   `rejection_sample_method="block"`, V2 only
   (`third_party/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:338-367`,
   `:535-562`): strictly higher acceptance than token-wise rejection while remaining
   distribution-preserving. We have no equivalent. **[verified]**
4. **True probabilistic draft probs** as an opt-in (`speculative.py:283-286`).
   **[verified]**
5. **Acceptance metrics per position**, Prometheus-exported
   (`third_party/vllm/vllm/v1/spec_decode/metrics.py:41-49`, `:105-135`). tokenspeed
   has scalar `avg_accept_len` only (E32 item 2). **[verified]**
6. **Breadth of proposers**: ngram (CPU + GPU), suffix decoding, medusa, standalone
   draft models with heterogeneous vocabularies, dflash/dspark parallel drafting
   (`third_party/vllm/vllm/v1/spec_decode/`). tokenspeed has three:
   `eagle` / `mtp` / `dflash` (`python/tokenspeed/runtime/execution/drafter/`).
   **[verified]**

**Net: on the accept/reject rule and on speculation *policy*, vLLM is ahead.
tokenspeed is ahead on exactly one axis — multi-depth MTP drafting — and that axis
happens to be the one the third-party evaluation measured.**

## 1.6 ABSORB — MTP

| # | What | Where from → where to | Cost | Notes |
|---|---|---|---|---|
| M1 | **Multi-depth Inkling MTP** (all depth layers, K>1) | `python/tokenspeed/runtime/models/inkling_nextn.py:146,156-164,197` + `drafter/mtp.py:612-646` → `unieinfra-vllm/vllm/models/inkling/nvidia/mtp.py` + a new multi-depth proposer | **L (2-4 wks)** | The single highest-value item. `Step3p5MTPProposer` (`third_party/vllm/vllm/v1/spec_decode/step3p5.py:24`) is the right base class: it already does per-layer draft-step selection and per-KV-group slot mappings. Needs the full-window re-run, which `EagleProposer`'s one-row-per-step loop does not do. |
| M2 | **Lookback KV repair** | `drafter/mtp.py:648-771`, `:218-288` | **M (1-2 wks)** | Only meaningful on top of M1, and only for backends implementing `configure_draft_lookback`/`enter_draft_lookback_window` (`mtp.py:380-385`, `:692-694`). Gate it behind an A/B env knob as tokenspeed does (`inkling_nextn.py:230`) and measure — it is not free. |
| M3 | **Per-depth EXTEND catch-up incl. mid-chunk** | `drafter/mtp.py:843-953`, rationale at `:1024-1033` | **M (1 wk)** | Required for M1 correctness on long prompts, not an optimization. Do not ship M1 without it. |
| M4 | **Never quantize the draft head by default** | `server_args.py:256,601-602`; `qwen3_5_nextn.py:166-169` → `unieinfra-vllm/vllm/config/speculative.py` | **S (1-2 days)** | Flip the default so the MTP head stays bf16 unless explicitly asked otherwise. Cheapest acceptance win available; measurable on any quantized target. |
| M5 | **Unconditional embed/LM-head sharing for MTP/EAGLE3, or a loud warning when sharing is refused** | `model_executor.py:390-394` → `llm_base_proposer.py:1443-1479` | **S (1-2 days)** | Minimum viable version: keep vLLM's `torch.equal` check but `logger.warning` on failure instead of silently duplicating. |
| M6 | **Freeze the acceptance thresholds at the lossless value** | `sampling/backends/base.py:40-41` | **XS** | Not a port — a policy assertion to write down and test, so nobody later buys acceptance with distribution drift. |

**Do NOT port:** the `Eagle` drafter itself (§1.2 — it is at parity with
`llm_base_proposer` and vLLM's version is better plumbed), or the rejection sampler
(§1.1 — parity, and vLLM has block verification we do not).

---

# Part 2 — Tool-calling correctness

## 2.1 The uncomfortable finding: tokenspeed has no tool-calling code

This is the claimed-advantage-with-no-visible-mechanism case, and it must be
recorded rather than explained away.

- `grep -rn "tool_call" python/tokenspeed/ --include=*.py` → **0 hits**.
  `tool_choice` → **0 hits**. **[verified]**
- `grammar_backend: str = "none"` — the default
  (`python/tokenspeed/runtime/utils/server_args.py:239`), CLI choices
  `["xgrammar", "none"]` (`:1516-1525`), and the launcher
  `python/tokenspeed/cli/serve_smg.py` never sets it (it sets
  `--reasoning-parser` / `--tool-call-parser` at `:312-340` and nothing else).
  A captured production server log confirms it at runtime:
  `benchmark/results/unieinfra/abl_both/server_startup_FAILED.log:17` prints
  `grammar_backend='none'`. **[verified]**
- With `grammar_backend="none"`, a request that *does* carry a constraint is
  **hard-aborted**, not degraded
  (`python/tokenspeed/runtime/grammar/grammar_manager.py:145-151`). **[verified]**
- `GrammarManager.process_req_with_grammar` admits unconstrained whenever
  `json_schema`, `regex`, `ebnf` and `structural_tag` are all `None`
  (`grammar_manager.py:137-143`). There is no "we have tools, synthesize a grammar"
  branch anywhere. **[verified]**
- The OpenAI-compatible HTTP surface is not in this repo: it is `smg`, a prebuilt
  Rust gateway pinned at `python/pyproject.toml:73-75`
  (`tokenspeed-smg==1.8.0.post20260720`). The two HTTP modules in the engine
  (`runtime/entrypoints/sglang_compat_http.py`, `vllm_compat_http.py`) are RL
  weight-sync control-plane routes only — no `/v1/chat/completions`
  (`sglang_compat_http.py:20-37`). **[verified]**
- The engine↔gateway contract is gRPC `SamplingParams`, whose entire constraint
  surface is `regex` / `json_schema` / `ebnf_grammar` / `structural_tag` — **no
  `tools`, no `tool_choice`, no `strict`**
  (`smg_grpc_proto/generated/tokenspeed_scheduler_pb2.pyi:62-65` in the venv).
  **[verified]**

**Conclusion.** Toolcall-15 100% was produced by the Rust gateway's parser, or by
the model, or by a launch configuration that turns xgrammar on — **not by anything
in `python/tokenspeed/`**. Since `grammar_backend` defaults to `none` and a
constrained request against that default *errors*, the only self-consistent reading
is that the gateway sends no constraint and relies on parsing.
**[unverifiable]** — settling this needs the `tokenspeed-smg` wheel, which is not
installed in this environment.

**So: tokenspeed does NOT constrain generation so that a malformed tool call is
unrepresentable. It parses. There is no retry loop either.** **[inferred, strongly]**

## 2.2 What `capturable_grammar.py` actually does

The name is accurate and the mechanism is real — it is just a *throughput*
mechanism, not the correctness mechanism the name might suggest.

**It runs the xgrammar matcher advance and bitmask fill as CUDA-graph host nodes**,
via `cudaLaunchHostFunc`:

- `python/tokenspeed/runtime/utils/hostfunc.py:78-80` — a JIT `load_inline` pybind11
  trampoline that acquires the GIL (driver callback threads do not hold it) and calls
  a stored `py::function`; `:101` — `cudaLaunchHostFunc(stream, trampoline, data)`.
- `hostfunc.py:173-193` — `launch_hostfunc` checks
  `torch.cuda.is_current_stream_capturing()`; when capturing it keeps the user data
  alive so the callback **replays on every graph replay**. **[verified]**

Per-step sequence (`capturable_grammar.py:333-363`, scheduled from
`runtime/execution/model_executor.py:954-960`, i.e. *before* the target forward):
record a fork event on the main stream, switch to a side stream, D2H the draft chain
into pinned `candidates_host`, then two host nodes — `fetch_batch()` and `build()` —
then an H2D of the bitmask. The main stream joins at `wait_bitmask()`
(`capturable_grammar.py:365-367`, called at `model_executor.py:989-990`) just before
sampling. All static buffers are preallocated in `__init__`
(`capturable_grammar.py:104-158`). **[verified]**

The clever part is the **deferred advance** (`capturable_grammar.py:28-39` docstring,
`:222-276`): step N's sampled tokens have not landed on the host at the tail of step
N, so `build()` in step **N+1** advances the matcher for step N *first*, then fills
step N+1's mask. Ordering is enforced by the fork event. Net effect: the matcher
advance for step N overlaps the forward of step N+1 and never touches the host
critical path. **[verified]**

Spec-decode integration is complete and correct: `_fill_current`
(`capturable_grammar.py:278-331`) emits one mask row per draft position and walks the
chain with `try_accept_token`, then **rolls the walk back** (`:327-331`); the real
advance uses only `accept_lengths` (`:256-268`), so rejected drafts never enter the
matcher. **[verified]**

Fallbacks: all-ones (`-1`) bitmask when no grammar is present (`:278-284`, `:294`),
so the captured sampler always takes the apply branch and there is no graph-level
branch (`bind_grammar_mask_buf` docstring, `:450-456`); an eager path
`EagerGrammarBuffers` with a synchronous per-step fill (`:383-438`, `:478-525`) for
non-CUDA or `--disable-capturable-grammar`; and a host-side rescue when no step N+1
is ever dispatched (`runtime/engine/generation_output_processor.py:567-577`).
**[verified]**

**How much of this does vLLM already have?** More than the name suggests:

- vLLM computes the bitmask on CPU in the engine-core process **concurrently with
  the GPU forward** — `execute_model(..., non_block=True)` then
  `get_grammar_bitmask(...)` then `future.result()`
  (`third_party/vllm/vllm/v1/engine/core.py:588-594`). **[verified]**
- It applies the mask **outside** the captured region, on post-graph logits, in a
  separate `sample_tokens()` entry point
  (`third_party/vllm/vllm/v1/worker/gpu_model_runner.py:4566-4570`;
  `vllm/v1/structured_output/utils.py:161`) — **no graph break, no eager fallback of
  the model forward**. **[verified]**
- It masks draft positions and rolls the grammar back after the speculative walk
  (`vllm/v1/structured_output/__init__.py:297-350`, rollback at `:349-350`), and
  pre-filters grammar-violating drafts
  (`vllm/v1/core/sched/scheduler.py:2095-2105`). **[verified]**
- It parallelizes bitmask fill across threads for batches ≥ 128
  (`vllm/v1/structured_output/__init__.py:61-68`, `:207-210`). **[verified]**

**So the genuine tokenspeed delta is narrower than "grammar inside the CUDA graph":**
it is (a) the *mask apply* being inside the captured region rather than an extra
eager launch after it, and (b) the matcher advance being a graph host node on a side
stream rather than Python work in the engine-core process that must complete before
the next `get_grammar_bitmask`. Both are real, both are small at low request counts,
and neither improves correctness. **[inferred]**

## 2.3 vLLM's real tool-calling gap — a default, not missing machinery

vLLM has everything needed and does not switch it on for the common case.

**Default `tools=[...]` + `tool_choice="auto"` builds NO grammar.** The gate:

```
third_party/vllm/vllm/tool_parsers/structural_tag_registry.py:106-110
    if not tools or tool_choice == "none":
        return None
    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
```

`_any_tool_strict` requires `tool.function.strict is True`
(`structural_tag_registry.py:85-93`), and `strict` defaults to `None`
(`vllm/entrypoints/openai/engine/protocol.py:293`). A stock OpenAI-SDK payload does
not set it. **[verified]** vLLM's own documentation states the consequence:

> `"auto"` | Only when `strict: true` is set on at least one tool … Without it, the
> model generates freely and tool calls are extracted from raw text.
> — `third_party/vllm/docs/features/tool_calling.md:118`
>
> Otherwise, vLLM extracts tool calls from raw text, so arguments may occasionally
> be malformed or violate the function's parameter schema.
> — `third_party/vllm/docs/features/tool_calling.md:155`

**[verified]**

Two further narrowings, both silent:

- Only **11 of 44** registered tool parsers declare a `structural_tag_model` at all
  (`deepseekv3/v31/v32/v4`, `hermes`, `kimi_k2`, `llama`, `glm47_moe`, `minimax_m2`,
  `qwen3_engine`); the rest return `None` with no log
  (`third_party/vllm/vllm/tool_parsers/abstract_tool_parser.py:174-175`).
  **[verified]**
- The fallthrough is silent too: `_apply_structural_tag` just
  `return request` (`third_party/vllm/vllm/parser/abstract_parser.py:547-548`), and
  `required`-mode parse failures are swallowed by
  `contextlib.suppress(ValidationError)` (`abstract_parser.py:464-468`), yielding an
  empty `tool_calls` list with no error surfaced. **[verified]**

Where vLLM *does* constrain, it does it well: `tool_choice="required"` and named
tool choice always build a grammar (`structural_tag_registry.py:255-267` for the
tag path, `vllm/tool_parsers/utils.py:350-377` and `:404-415` for the JSON-schema
path); structural tags are trigger-scoped so free text outside `<tool_call>` is
unconstrained (`structural_tag_registry.py:248-254`); the grammar is suspended during
`<think>` blocks and resumed at the exact boundary
(`vllm/v1/structured_output/__init__.py:361-379`, `:381-439`, `:462-486`); and there
is **no** silent "unsupported schema → unconstrained" fallback — `backend="auto"`
falls back to another *backend* (`vllm/sampling_params.py:1022-1061`) and an explicit
backend raises a 400 (`backend_xgrammar.py:314-317`). **[verified]**

## 2.4 Where vLLM is already better

- **44 tool parsers** (`third_party/vllm/vllm/tool_parsers/__init__.py:203-210`) and
  ~24 reasoning parsers (`vllm/reasoning/`) vs **zero** in tokenspeed. **[verified]**
- **4 structured-output backends** (xgrammar, guidance, outlines,
  lm-format-enforcer) with automatic cross-backend fallback vs tokenspeed's
  xgrammar-only (`python/tokenspeed/runtime/grammar/xgrammar_backend.py`).
  **[verified]**
- **A per-model tool-call structural-tag registry**
  (`third_party/vllm/vllm/tool_parsers/structural_tag_registry.py`). tokenspeed's
  only registry is for *reasoning channels*
  (`python/tokenspeed/runtime/grammar/reasoning_structural_tag.py:56-64`) — it maps
  `--reasoning-parser` names to xgrammar builtin templates and substitutes the
  response channel's `any_text` slot with the user's schema so a grammar armed at
  token 0 does not reject the model's `<think>` preamble
  (`reasoning_structural_tag.py:103-194`), failing closed on template drift
  (`:188-192`). Useful, but orthogonal to tool calls. **[verified]**

**Neither engine has jump-forward decoding or token healing**: SGLang's compressed-FSM
fast-forward was not carried into tokenspeed
(`grep -rni "jump.forward|token_healing|retokeniz" python/tokenspeed/` → 0 hits).
**[verified]**

## 2.5 ABSORB — tool calling

| # | What | Where from → where to | Cost | Notes |
|---|---|---|---|---|
| T1 | **Make structural tags the default for `tool_choice="auto"`** when the selected parser has a `structural_tag_model` — i.e. drop the `strict: true` requirement, behind a server flag that defaults ON | edit `unieinfra-vllm/vllm/tool_parsers/structural_tag_registry.py:106-110` | **S (2-3 days)** | Not a port from tokenspeed at all — tokenspeed has nothing here. This is a **default change on the vLLM base**, and it is the single highest-leverage tool-call correctness action available. Must be measured: constraining `auto` can suppress legitimate free-text answers if the tag is built wrong. |
| T2 | **Extend `structural_tag_model` coverage** from 11/44 parsers to the model families we serve | `unieinfra-vllm/vllm/tool_parsers/*_tool_parser.py:174-175` | **M (1 wk/family)** | Ordinary per-model work. Prioritize by the models on our support matrix. |
| T3 | **Log every silent degradation to unconstrained** | `structural_tag_registry.py:109-110`, `abstract_parser.py:547-548`, `abstract_tool_parser.py:174-177`, `abstract_parser.py:464-468` | **XS (1 day)** | Four `return None`s and one `suppress(ValidationError)` that currently hide "you asked for a tool call and got no constraint". Pure observability; no behavior change. |
| T4 | **In-graph mask apply** (capturable grammar) | `python/tokenspeed/runtime/grammar/capturable_grammar.py` + `runtime/utils/hostfunc.py` → `unieinfra-vllm/vllm/v1/worker/gpu_model_runner.py` sampling path | **L (3-5 wks)** | Real but **not correctness**. vLLM already overlaps mask *computation* with the forward (`core.py:588-594`) and applies outside the graph without a break (`gpu_model_runner.py:4566-4570`), so the remaining delta is one eager launch plus the CPU advance. **Profile before porting** — this needs a measured structured-output throughput gap to justify a `cudaLaunchHostFunc` dependency and a GIL-acquiring driver callback. |
| T5 | **Reasoning-aware structural-tag deferral registry** | `python/tokenspeed/runtime/grammar/reasoning_structural_tag.py:56-64,103-194` | **S (3-4 days)** | Mostly redundant: vLLM already suspends the grammar during reasoning (`vllm/v1/structured_output/__init__.py:361-379`). The transferable bit is the **fail-closed-on-template-drift** pattern (`:188-192`) and the harmony `tags_with_separator` fix (`:136-173`). Port the pattern, not the file. |
| T6 | **`--disable-any-whitespace`** (whitespace-restricted JSON) | `python/tokenspeed/runtime/utils/server_args.py:245,1551-1559`; `xgrammar_backend.py:151,158` | **XS (1 day)** | Guards the "model wedges into endless whitespace until length cutoff" failure mode. vLLM has `constrained_json_whitespace`; verify parity and default it on for tool calls. |

**Do NOT port:** tokenspeed's grammar *manager* / compile cache
(`base_grammar_backend.py:224-401`, `grammar_manager.py:103-370`) — it is well built
(shared futures, two-phase timeout, TTL'd failure markers, cross-rank admission
consensus) but vLLM's async-compile equivalent
(`vllm/v1/structured_output/__init__.py:166-175`, `request.py:50-59`) already covers
the same ground and is wired into a scheduler we now own.

---

# Part 3 — The decode weakness, briefly

Background only for the separate per-step decode-gap investigation (#42). Not a
study — hypotheses with pointers, for #42 to confirm or kill.

**Framing first, because it reorders everything.** The third-party tg128 numbers are
essentially **flat across 8k/16k/32k** on both engines (30.3/28.7/33.3 vs
41.5/41.3/45.3). A deficit that does not grow with context is a roughly **constant
per-decode-step cost**, not KV-scan cost. That demotes "no flash-decoding split-K"
and promotes launch/dispatch/metadata effects. It also matches the log: E36 solved
`out tok/s = C / (t_decode_step + prefill_share)` and concluded the C=128 gap is
"~25% of per-step decode cost — a kernel/runtime property, not a scheduling one"
(`docs/experiments-log.md:17`). **[measured]**

Ranked hypotheses, all **[verified]** in code unless marked:

1. **Decode declares `max_seqlen_k` = the full model context, not the batch's real
   max KV length.** `python/tokenspeed/runtime/layers/attention/backends/mha.py:979`
   passes `max_seqlen_k=self.max_context_len`, and
   `self.max_context_len = config.context_len` (`mha.py:157`). Over-declaring by
   4-16x is exactly a constant, context-insensitive tax on kernels that use it to
   size split counts and scratch. vLLM passes the true batch max
   (`third_party/vllm/vllm/v1/attention/backends/flash_attn.py:1048`, from
   `common_attn_metadata.max_seq_len`). **Cheapest experiment in this list.**
2. **No `num_splits` / scheduler metadata is ever computed for decode.**
   `mha_decode_with_kvcache` has no such parameter
   (`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/__init__.py:974-1090`).
   vLLM AOT-computes the schedule and **forces** a split count under CUDA graphs
   (`third_party/vllm/vllm/v1/attention/backends/flash_attn.py:524-540`, `:1061`;
   default 32 at `third_party/vllm/vllm/config/attention.py:42`). At bs=1 with one
   query row, split-K is what decides whether the GPU is filled.
3. **Kernel selection is shape-blind.** The decode traits carry `head_dim`,
   `page_size`, `sliding_window`, … but **not batch size or sequence length**
   (`tokenspeed-kernel/.../ops/attention/__init__.py:1029-1037`), and the selection
   cache key omits them too (`.../selection.py:325-345`). The same kernel is used at
   bs=1 and bs=160. The default `SelectionObjective.DEFAULT` also scores 0 for every
   kernel (`selection.py:386-391`), so latency-tagged kernels get no boost in exactly
   the single-stream regime.
4. **Page table is materialized at full context width on every decode step.**
   `block_size: int = 64` (`python/tokenspeed/runtime/utils/server_args.py:117`) and
   `max_num_pages = ceil_div(max_context_len, page_size)` (`mha.py:159`) size a
   `[max_bs, max_num_pages]` graph buffer (`mha.py:409-411`) that
   `gather_page_table_with_padding` re-fills at full width on **every** replay
   (`mha.py:622-631`) rather than the live `ceil(seq_len/64)` prefix. Page size 64
   (vLLM defaults to 16, `third_party/vllm/vllm/config/cache.py:47`) also quadruples
   the wasted tail read on the last partial page.
5. **CUDA graphs are NOT the problem — rule-out.** bs=1 is always in `capture_bs`
   (`runtime/execution/cuda_graph_wrapper.py:112-114`) and graphs are disabled only
   under `enforce_eager` (`:308`). But `init_forward_metadata_replay_cuda_graph` runs
   a page-table gather, seq-len clamps and (on sm_120) a full FlashInfer FA2 re-`plan`
   **outside** the graph every step (`mha.py:605-704`) — with nothing to amortize it
   against at bs=1.
6. **Speculative decoding is OFF by default — rule-out.**
   `speculative_algorithm: str | None = None`
   (`python/tokenspeed/runtime/utils/server_args.py:254`); nothing in
   `python/tokenspeed/cli/` sets it. The "extra drafter forwards" theory does not
   apply unless the eval harness passed a spec config.
7. **[inferred] Profile the GEMMs before the attention kernel.** bs=1 decode is
   M=1 GEMV-shaped; per (3) the selector never sees M, and the gemm registrations are
   arch/format-gated only. A kernel that is best at M=8192 — where tokenspeed wins
   `ctx_pp` — can easily be 20-30% off the best M=1 kernel. The flat-vs-context
   signature fits a fixed per-layer cost better than an attention-scan cost.

Suggested first three experiments for #42: (a) pass the batch's real `max_seq_len` at
`mha.py:979` and re-measure; (b) sweep the registered decode kernels at bs=1 via the
override hook (`selection.py:697-700`); (c) capture an Nsight timeline of one bs=1/8k
decode step and check the graph-external region (`mha.py:605-704`) plus the
linear-layer kernels against vLLM's.

---

# Part 4 — Prioritized ABSORB list

Ordered by (value ÷ cost), not by section.

| Rank | Item | Axis | Cost | Why here |
|---|---|---|---|---|
| 1 | **T3** — log every silent degradation to unconstrained | tool | XS (1 d) | Five call sites. Turns an invisible failure into a metric. Prerequisite for measuring T1. |
| 2 | **M4** — never quantize the draft head by default | MTP | S (1-2 d) | Cheapest acceptance win in the list; applies to every quantized deployment. |
| 3 | **M6** — assert the lossless acceptance threshold | MTP | XS | A test and a comment. Protects the correctness property we already have. |
| 4 | **T1** — structural tags default-on for `tool_choice="auto"` | tool | S (2-3 d) | The actual fix for tool-call malformation on the vLLM base. Must ship with T3 and an A/B. |
| 5 | **M5** — loud warning when embed/head sharing is refused | MTP | S (1-2 d) | Converts a silent acceptance tax into a log line. |
| 6 | **T6** — whitespace-restricted JSON for tool calls | tool | XS (1 d) | Closes a known wedge-until-max-tokens failure. |
| 7 | **M1 + M3** — multi-depth MTP (drafter + EXTEND catch-up) | MTP | L (3-5 wks) | **The single largest quality item in this document** and the only real explanation of the MTP acceptance result. Ship M1 and M3 together; M1 alone is incorrect on long prompts. Base it on `Step3p5MTPProposer`, not `EagleProposer`. |
| 8 | **T2** — extend structural-tag coverage per model family | tool | M (1 wk/family) | Linear, prioritized by the support matrix. |
| 9 | **M2** — lookback KV repair | MTP | M (1-2 wks) | Only after M1 lands and only if measured; it is an A/B knob in tokenspeed too. |
| 10 | **T5** — fail-closed reasoning-tag pattern | tool | S (3-4 d) | Mostly redundant with vLLM; port the pattern, not the module. |
| 11 | **T4** — in-graph grammar mask apply | tool/perf | L (3-5 wks) | Genuine but narrow. **Gate on a measured structured-output throughput gap** — vLLM already overlaps the expensive half. |

### Explicitly not on the list

- tokenspeed's `Eagle` drafter, rejection sampler, and grammar compile cache — all at
  parity or behind (§1.1, §1.2, §2.5).
- Anything claiming to port tokenspeed's tool-call correctness. **There is no such
  code** (§2.1).

### Open questions this study could not close

1. Does the `smg` gateway synthesize a tool-call structural tag from `tools`?
   **[unverifiable]** — needs the `tokenspeed-smg` wheel. If it does, T1 has a
   working reference implementation; if it does not, Toolcall-15 100% is a
   model/parser result and nothing in either engine explains it.
2. Was the third-party MTP acceptance comparison run at equal K? If vLLM was capped
   at K=1 by `speculative.py:1002-1007` while tokenspeed ran K=4, the headline is
   real but the correct statement is "vLLM cannot run this model's MTP head at
   depth", not "tokenspeed drafts better".
3. What is the actual acceptance of a *single-depth* head on both engines,
   back-to-back at equal K? §1.2 predicts parity. That prediction is the cheapest
   falsifiable test in this document and should be run before M1 is funded.
