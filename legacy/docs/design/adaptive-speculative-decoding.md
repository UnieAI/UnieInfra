<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# Design: Adaptive Speculative Decoding

Status: **Design — NOT implemented** · Date: 2026-07-25 · Owner: royshih (drafted by agent, doc-only)

## Method and honesty rules

This is a **read-only code study + design proposal**. No source was modified and no
GPU was used (three GPU jobs — including the MTP ablation this design depends on —
were running under other agents at the time of writing). Claims are labelled:

- **[code]** — read directly from source; cites `file:line`. Describes what exists
  *today*, not what this design proposes.
- **[measured]** — cites `docs/experiments-log.md`; an experiment actually run.
- **[design]** — a proposal in this document. Not implemented, not measured.
- **[estimated]** — arithmetic from code (buffer shapes/dtypes), not a profiler
  measurement. Flagged wherever I could not verify with `torch.cuda.memory_stats()`
  because no GPU was available this session.
- **[pending]** — blocked on the in-flight Qwen3.5-27B MTP ablation (GPU 1). Several
  of this design's concrete thresholds cannot be filled in until it lands.

Where I could not determine something by reading, I say so rather than guessing.

## 0. Problem (owner's framing)

Speculative decoding (MTP: draft + verify) pays for itself only when the GPU has
spare compute to absorb the draft+verify overhead — at low concurrency it wins
(TPOT drops), at high concurrency the GPU is already saturated and the draft work
steals cycles from real decode work, so it must **hurt**, not help. This is not
speculative: it's the direct consequence of speculative decoding trading extra
FLOPs for fewer decode *steps*, and FLOPs are the scarce resource at saturation.
So speculation must be **adaptive to load**, not a static launch flag.

The owner's two constraints, taken as first-class design inputs:

1. **The on/off (and depth) decision must not cost throughput on the hot path.**
   Proposed mechanism: a separate CPU thread computes the decision at coarse
   cadence and publishes it to a memory location; the hot path reads that location
   directly instead of computing anything expensive inline.
2. **Slow startup is acceptable; the running path must not be slow.** This licenses
   pre-capturing more CUDA graphs than we'd otherwise dare, and doing more
   validation/instrumentation at boot.

This document refines that sketch against what the codebase actually does, and is
explicit about where the sketch survives contact with the code unchanged, where it
needs correction, and where a real complication was found that the sketch didn't
anticipate.

## 1. Prior art: JetSpec, and where the adaptivity gap actually is

**[external, fetched 2026-07-25]** JetSpec (hao-ai-lab,
`github.com/hao-ai-lab/JetSpec`) is directly relevant prior art on the *drafter*
side of this problem, and it sharpens one part of this design.

- **Technique — causal parallel tree drafting.** A trained draft-head adapter reads
  fused hidden states from the frozen target model and emits **per-depth logits in
  a single parallel pass** (not one forward per depth, unlike autoregressive
  drafters; and unlike block-diffusion drafters, it keeps causal branch
  conditioning). The target then verifies the **entire token tree in one batched
  forward** under a **tree-causal attention mask**, using custom Triton
  tree-attention kernels. The accepted path follows the target's own logits, so
  decoding is **lossless by construction** — output distribution matches the
  target's, same correctness bar as chain speculative decoding.
- **Stack**: paged KV, Triton tree-attention kernels, CUDA graphs (batch-1 focus).
  Triton is our sm_120/portable default backend
  (`AGENTS.md`: "Prefer CuteDSL for NVIDIA... Use Triton for portable solutions
  across vendors"), so a JetSpec-style tree-attention kernel is a plausible future
  `unieinfra-kernel` tenant if we ever move beyond chain MTP.
- **Results (Qwen3-8B, B200, bf16, batch size 1)**: MATH-500 1150 tok/s (accept
  length 9.56, up to 9.64x e2e speedup), GSM8K 984 tok/s (accept length 7.94),
  HumanEval 867 tok/s (accept length 6.92). Tree shape is a **static** launch
  config (`tree_depth`, `tree_width`, `budget`, e.g. depth=20/width=7/budget=128).
- **The gap**: JetSpec has **no adaptive mechanism** — nothing adjusts tree
  shape or disables speculation based on batch size, GPU saturation, or measured
  acceptance rate, and every headline number is single-stream (batch=1).
  High-concurrency behavior is uncharacterized.

**What this means for this design.** JetSpec and this document solve orthogonal
problems: JetSpec makes single-stream speculation better (tree > chain,
one-pass drafting, lossless verify); this design makes speculation *load-aware* so
it stops hurting at concurrency. They compose — a future tree drafter under our
adaptive controller is strictly better than either alone. Concretely:

- **The controller must not be drafter-specific.** `BaseDrafter`
  (`python/tokenspeed/runtime/execution/drafter/base.py:39-83`) is already an
  abstract interface — `get_candidates`/`run`/`draft` — with three registered
  implementations (`Eagle` for `MTP`/`EAGLE3`, `DFlash` for `DFLASH`; see
  `DRAFTER_MAPPING`, `python/tokenspeed/runtime/execution/model_executor.py:93`,
  and the `InklingForConditionalGenerationNextN` special-case at
  `model_executor.py:98` routing to the multi-depth drafter in
  `execution/drafter/mtp.py`). The adaptive controller (§3-§6 below) should sit
  **above** this interface: it decides *whether* and *how deep* to speculate from
  load signals alone, and hands that decision to whichever `BaseDrafter` is
  configured via its existing `spec_num_tokens`/`spec_num_steps` constructor
  parameters (`drafter/base.py:42-43`). A JetSpec-style tree drafter would be a
  **new** `BaseDrafter` subclass (tree-shaped `get_candidates`/`run`) and would
  plug into the same controller unchanged — the controller's job (§3) never touches
  drafting math, only gates it.
- **Correctness invariant this design must preserve**: speculative decode's
  accept/reject verify step already guarantees the sampled output matches the
  no-speculation target distribution (that is what "verify" means; it is not
  something this design changes). The adaptive controller therefore **cannot**
  introduce a correctness regression by construction, *as long as* "spec off" is
  implemented as literally routing to today's already-correct non-spec decode path
  (§4) rather than a degraded/truncated spec path. This is a design constraint on
  the implementation, not a new guarantee to build — flagging it because dynamic
  on/off switching is exactly the kind of change that invites a shortcut (e.g.
  skipping verify "just this once") which would break it.
- **Tree drafting reinforces, not weakens, the "small discrete config set"
  recommendation in §6.** A tree has a fixed `(depth, width, budget)` shape, so —
  exactly like chain depth today (§4) — each tree configuration is its own
  CUDA-graph capture shape. If we ever adopt tree drafting, the number of
  practical points to capture becomes `{off} ∪ {chosen tree configs}`, same
  bounded-discrete-set logic as chain depth, not a new problem.

## 2. Control plane / data plane split

### 2.1 Signals that are (largely) free — already computed every step

**[code]** `EventLoop.event_loop()` calls `self._get_scheduler_stats()`
**unconditionally, every iteration**, before dispatch —
`python/tokenspeed/runtime/engine/event_loop.py:1662` (and the overlap-schedule
variant at `event_loop.py:1815`). The method:

```
python/tokenspeed/runtime/engine/event_loop.py:1550-1559
def _get_scheduler_stats(self):
    available = self.scheduler.available_kv_pages()
    active = self.scheduler.active_kv_pages()
    num_total_pages = self.max_total_num_tokens // self.server_args.block_size
    return {
        "num_active_pages": active,
        "num_cached_pages": num_total_pages - available,
        "num_queue_reqs": self.scheduler.waiting_size(),
    }
```

- **Running batch size** — `bs = len(forward_op.request_ids)`
  (`execution/model_executor.py:1266`, logged as `#running-req`,
  `model_executor.py:1329/1345`). A Python `len()` on a list already built by the
  scheduler's execution plan — genuinely free.
- **Queue depth** — `scheduler.waiting_size()` →
  `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp:213-221`:
  ```cpp
  std::size_t Scheduler::WaitingSize() const {
      std::size_t count = 0;
      for (const auto& [id, req] : requests_) {
          if (req->Is<fsm::Submitted>()) { count++; }
      }
      return count;
  }
  ```
  **Correction to the "a few integer ops" framing**: this is a **linear scan**
  over the live `requests_` map, not an O(1) counter read. It is bounded by
  `max_batch_size` (today up to `adaptive_batch_max=1024`,
  `python/tokenspeed/runtime/utils/server_args.py:105`), so in absolute terms it's
  still cheap (hundreds of pointer-chases, not a kernel launch) — but it is not
  "free" in the asymptotic sense the prompt's framing implied. What actually makes
  it safe to read every step is that it is **already paid unconditionally on the
  hot path today** (`event_loop.py:1662`), for logging/metrics purposes, regardless
  of whether this design exists. Reusing it costs nothing *additional*.
- **KV pressure / page ratio** — `scheduler.available_kv_pages()` →
  `scheduler.cpp:253-263` (radix path: `device_allocator_.AvailablePages()`,
  effectively an O(1) counter; flat path: `block_pool_.NumFreeBlocks()`, also
  O(1)) is genuinely cheap. But `scheduler.active_kv_pages()` →
  `scheduler.cpp:265-274` is **not**:
  ```cpp
  std::unordered_set<std::int32_t> active_pages;
  for (const auto& [_, req] : requests_) {
      if (req->Is<fsm::Prefilling>() || ... || req->Is<fsm::Decoding>()) {
          for (std::int32_t page : req->GetOccupiedPagesAllGroups()) {
              active_pages.insert(page);
  ```
  a per-step hash-set build over every running request's pages (dedups
  prefix-shared pages — comment at `scheduler.cpp:266-269`). Same conclusion as
  `WaitingSize`: cheap in absolute terms, bounded, and **already paid today**
  (`_get_scheduler_stats` is unconditional), but not literally free. `page_ratio`
  itself (`model_executor.py:1324-1326`) is one integer division on the result.
  Bindings: `waiting_size`/`available_kv_pages`/`active_kv_pages` at
  `tokenspeed-scheduler/bindings/python_module.cpp:464,468,469`; declared `const`
  in `tokenspeed-scheduler/csrc/scheduler/scheduler.h:73,76,77`.

**Bottom line for §2.1**: bs, queue depth, and page ratio are all suitable as the
Phase-1 "cheap" signal set — not because they cost nothing, but because the engine
pays for them every step regardless, and a comparison against a couple of
thresholds on top of already-computed integers is indeed negligible next to a
forward pass. The prompt's "a few integer ops, cheaper than a thread with
synchronization" framing is directionally right but overstates how free the
*scheduler-side* reads are; it is exactly right about the *comparison* itself.

### 2.2 Signals that are genuinely expensive — must be off the hot path

- **Acceptance-rate EWMA.** The codebase already computes an acceptance rate
  today, but only at coarse cadence and only for logging:
  `execute_forward_op_with_log` gates the whole block on
  `self.log_step % self.config.decode_log_interval == 0`
  (`execution/model_executor.py:1306-1308`; `decode_log_interval` defaults to 40
  steps, `server_args.py:144`/`1228`). Inside that gate:
  ```
  execution/model_executor.py:1313-1322
  avg_accept = self.num_generated_tokens / self.num_decode_steps ...
  accept_rate = (avg_accept - 1) / self.config.spec_num_steps ...
  ```
  This is the right *cadence* precedent (coarse, windowed) but the wrong
  *placement* precedent — it runs inline on the main decode thread, on whichever
  step happens to land on the interval, which is exactly the "computed from data
  inline" pattern the owner wants to avoid for the control decision. Phase 2 (§7)
  should move this computation (or a proper EWMA generalization of it) off-thread
  and publish the result instead of computing-then-branching inline.
- **SLO feedback (TTFT/TPOT headroom vs. target)** — no existing computation
  found in the hot path; genuinely new for Phase 2.
- **Autotuned thresholds** — genuinely new, Phase 3.

### 2.3 The owner's flag-publish pattern already exists in this codebase

**[code]** For a different subsystem (layerwise P/D KV transfer), the engine
already implements exactly the "write from one place, read a memory address
directly from another" idiom the owner described:
`python/tokenspeed/runtime/pd/utils.py:433-477`, class `StepCounter`:

```python
# pd/utils.py:465-471 — write side (device -> pinned host, async copy)
def record_cache(self):
    self.d_ready_cache_step = (self.d_ready_cache_step + 1) % self.COUNT_NUM_MAX
    self.h_ready_cache_step.copy_(self.d_ready_cache_step, non_blocking=True)

# pd/utils.py:473-474 — read side (direct address read, no lock, no torch op)
def query_ready_cache_step(self) -> int:
    return ctypes.c_int64.from_address(self.h_ready_cache_step.data_ptr()).value
```

This is a real precedent to model Phase 2's publisher on
(`ctypes.c_int64.from_address(tensor.data_ptr()).value`, no lock, no Python object
churn on the read side). The one adaptation: `StepCounter` exists because the
*writer* is a CUDA kernel and the *reader* is a CPU thread waiting on GPU progress,
so it uses a pinned device-mirrored tensor. Here both writer (a Python
`threading.Thread`) and reader (the event loop) are CPU-side; a plain
`ctypes`-backed shared integer (or a single-element pinned CPU tensor reused just
for its address-stability, mirroring the existing idiom for consistency) is
sufficient — no device tensor or `non_blocking` copy needed. Recommend copying the
*shape* of this pattern (one flag per decision — mode, depth — each a single
machine word, updated by `.store()`-then-visible, read via direct address) rather
than inventing a new synchronization primitive.

## 3. The real cost is CUDA-graph shape, not the branch

**[code]** Confirmed: `CudaGraphWrapper` (`execution/cuda_graph_wrapper.py`)
already treats "which captured batch size to replay" as a cheap dispatch — dict
lookup + `graph.replay()`:

- Capture ladder: `get_batch_sizes_to_capture` (`cuda_graph_wrapper.py:106-135`)
  builds `capture_bs`, default `[1,2,4] + [8,16,...,160]`
  (`cuda_graph_wrapper.py:112-114`), extended under `--enable-adaptive-batch`
  (default **on** since E25) with 32-wide buckets up to the KV-derived cap
  (`cuda_graph_wrapper.py:122-124`, mirroring `derive_adaptive_max_batch`,
  `engine/event_loop.py:113-140`).
- Capture loop: `capture()` (`cuda_graph_wrapper.py:317-351`) iterates
  `(variant, bs)` pairs, storing each captured graph and its output tensors in
  dicts keyed by `(variant, bs)`: `self.graphs`/`self.output_buffers`
  (`cuda_graph_wrapper.py:304-305`, populated `350-351`).
- Replay dispatch: `_can_use_graph`/`_padded_bs` (`cuda_graph_wrapper.py:1019-1047`)
  bisect the live batch size up to the nearest captured bucket, then
  `self.graphs[graph_key].replay()` (`cuda_graph_wrapper.py:1201-1203`) — **this
  part of the sketch is correct**: switching between two already-captured shapes
  within one ladder genuinely is a pointer-swap-cost dispatch, not a
  recompilation or a data-dependent branch inside a graph.

**Why spec and non-spec can't share a graph.** `max_tokens_per_req` —
`config.spec_num_tokens if config.spec_algo is not None else 1`
(`cuda_graph_wrapper.py:241-243`) — multiplies the token dimension of the
captured shape throughout: `input_num_tokens=bs * self.max_tokens_per_req`
(`cuda_graph_wrapper.py:408`), output slicing
`output_tokens[: bs * self.max_tokens_per_req]`
(`cuda_graph_wrapper.py:1212`), paged-cache block-table sizing
(`cuda_graph_wrapper.py:694,719,875,899`). A spec step at `bs=32` and a non-spec
step at `bs=32` are **different tensor shapes** end to end — exactly the prompt's
framing. Good so far.

**Where the sketch under-states the complication.** `max_tokens_per_req` is not
just baked into the *graph's* captured shape — it is baked into the **attention
backend's persistent state at construction**, one level below the graph:

```
cuda_graph_wrapper.py:250-257
init_backend_cuda_graph_state(
    attn_backend, self.max_bs, self.input_buffers.seq_lens_buf,
    ..., max_tokens_per_req=self.max_tokens_per_req, ...
)
```
which forwards to `backend.init_cuda_graph_state(max_bs, seq_lens_buf, **extras)`
(`layers/attention/backends/base.py:40-59`), and separately every
`AttentionBackend` instance stores `self.spec_num_tokens =
config.speculative_num_draft_tokens` **at construction**
(`backends/base.py:82`) as an immutable per-instance property. In other words: a
`CudaGraphWrapper` and the `AttentionBackend` object it wraps are **shape-locked
to one spec depth for the lifetime of the process** — today's engine builds
exactly one `attn_backend` (+ one `draft_attn_backend` when spec is on) at
`EventLoop.__init__` time (`create_attn_components` call,
`engine/event_loop.py:233-243`), sized once for whatever
`speculative_num_draft_tokens` the process was launched with.

**Implication for the dual-ladder sketch**: a true dual ladder (spec-shaped
graphs *and* non-spec-shaped graphs coexisting, selectable per step) is **not**
"two more entries in the `graphs` dict." It requires a **second full stack**:
its own `AttentionBackend` instance (and, for MTP, its own
`draft_attn_backend`), sized for the other `max_tokens_per_req`, wrapped in its
own `CudaGraphWrapper`. The alternative — generalizing `AttentionBackend` to
support a runtime-variable `max_tokens_per_req` inside one instance — is a much
larger, backend-by-backend change (every file under
`layers/attention/backends/`: `mha.py`, `flashmla.py`, `trtllm.py`,
`hybrid_linear_attn.py`, `deepseek_v4.py`, `inkling.py`, …) and is **not**
recommended for Phase 1. Recommendation: **build N independent stacks**
(§4/§5 quantify N and the resulting VRAM), each a normal
`(attn_backend[, draft_attn_backend], CudaGraphWrapper, drafter-or-None)` tuple
built exactly the way `EventLoop.__init__` builds one today, and make the
control-plane flag select **which stack's `CudaGraphWrapper.__call__` to invoke**
for the step — a Python attribute swap, decided once per step, strictly before
`_dispatch_forward` (`engine/event_loop.py:1686`). This keeps the owner's
constraint intact (the decision happens outside any captured graph, at Python
dispatch time) while being honest that it is closer to "N separate model
executors sharing the KV pool" than "one executor with a richer graph dict."

## 4. Owner's accepted tradeoff: pre-capture, but the spec ladder is short by construction

**Startup cost is accepted** (owner's constraint #2), so capturing every stack's
full ladder at boot is on the table. But captured graphs cost VRAM permanently —
persistent buffers taken from the same budget as the KV pool
(`gpu_memory_utilization`, `server_args.py:414-429`), so a wider capture set
lowers max concurrency, not just boot time.

**The codebase already half-encodes this insight, as a blunt static default**
(not yet a policy): when `speculative_algorithm` is set, the *default*
`max_cudagraph_capture_size` is **80**, vs **160** for non-spec, and default
`max_num_seqs` is likewise **80** vs **160**
(`server_args.py:417-434`). This design generalizes that existing instinct
("don't extend spec graphs as far") into a **measured, policy-driven** ladder
length (§8) instead of a hardcoded 80.

**Latent gap found**: `derive_adaptive_max_batch`
(`engine/event_loop.py:113-140`) and its call site
(`event_loop.py:260-289`) do **not** check `speculative_algorithm` at all — they
extend `max_num_seqs`/`max_cudagraph_capture_size` up to `adaptive_batch_max`
(default **1024**, `server_args.py:105`) purely from KV-token capacity. Under
today's defaults — `enable_adaptive_batch=True` since E25
(`server_args.py:98`, `docs/experiments-log.md:30`) — **running MTP with
adaptive-batch on today would try to extend the spec-shaped (4x-token-width, see
§5) ladder all the way to the KV-derived cap**, not stop at the old static-80
instinct. Nothing currently prevents this; it is a real, currently-latent cost
blow-up that predates this design and that any Phase-1 implementation must close
by making the derivation spec-aware (cap the spec stack's ladder independently of
the non-spec stack's ceiling).

**The key insight to build on** (owner's framing, developed): because the policy
*disables* speculation above some crossover batch `C*` anyway (§6/§8), the spec
stack's ladder only ever needs capturing up to `C*` (the prompt's example: ≤32),
never up to the full non-spec ceiling (which E25 already extends to ~480 and the
ceiling allows up to 1024). The extra VRAM/boot cost is **confined to a short
low-batch ladder**; the long high-batch ladder is exactly what E25 already builds
today, untouched. This saving is structural — it falls out of the policy itself,
not a separate optimization to implement.

**VRAM cost, quantified as far as code allows [estimated — no GPU this session]:**

Two different components of "captured graph memory":

1. **Pooled intermediate/activation memory**, shared via one process-wide
   `global_graph_memory_pool` (`cuda_graph_wrapper.py:138`, used at capture time
   `515`, updated `535`). PyTorch's CUDA-graph pool allocator reuses freed
   intermediate memory across captures that share a pool, so **within one
   stack**, adding more `bs` buckets to a ladder costs much less than
   `num_buckets × single_capture_footprint` — it's closer to the high-water mark
   of the single largest capture. **But** a second stack (§3) built as an
   independent `CudaGraphWrapper` would need its own pool unless explicitly wired
   to share `global_graph_memory_pool` — `cuda_graph_wrapper.py` does not do this
   across independently-constructed wrappers today, so this saving likely does
   **not** carry over to the dual-stack case as-is. **[inferred, unverified —
   flag for a Phase-1 spike.]**
2. **Genuinely per-`(variant, bs)` persistent tensors** that are *not* reclaimed
   for the life of the process, because the wrapper holds a live reference to
   every captured entry:
   - `output_buffers[(variant, bs)]` (`cuda_graph_wrapper.py:304-305,350-351`):
     `output_tokens`/`output_lengths`/`output_logprobs`, sized against
     `bs * max_tokens_per_req` (`cuda_graph_wrapper.py:1212-1219`).
   - Per-group paged-cache block tables sized `(bs, max_pages)` int32
     (`_capture_paged_cache_block_tables`, `cuda_graph_wrapper.py:610-631`), held
     by the attention backend's own capture-time state.
   - **[estimated]** Rough order of magnitude per captured point: block tables
     ≈ `bs × max_pages × 4 bytes × num_paged_cache_groups`; output
     tokens/lengths ≈ `bs × max_tokens_per_req × 8 bytes × 2-3 tensors`; if
     `--enable-output-logprobs` is on, output logprobs add a much larger
     `bs × max_tokens_per_req × top-k × 4 bytes` term (the CLI help for that flag
     already warns "extending the captured CUDA-graph footprint",
     `server_args.py` near `enable_output_logprobs`). A spec stack point costs
     roughly `max_tokens_per_req`× (default 4, see §5) more than a non-spec point
     at the same `bs` for these terms, **plus** an entirely separate
     `attn_backend`(+`draft_attn_backend`) persistent-state allocation that today
     is paid exactly once per process (§3) but becomes a genuine **second** fixed
     allocation under a dual/multi-stack design.
   - I could not produce a concrete MB number without running capture on a real
     GPU, which this task explicitly excludes (GPU 1/2/3 owned by other agents).
     **Recommendation**: Phase 1 should log `torch.cuda.memory_stats()` deltas
     around each stack's `capture()` call at boot — cheap instrumentation,
     consistent with "slow startup is acceptable" — and turn this estimate into a
     measured number before deciding how many stacks/discrete configs (§5) are
     affordable.

## 5. Not binary — but bounded to a small discrete set, not a continuous dial

**Mapping to real server args** — **[code]**:

- `speculative_num_steps: int = 3` (`server_args.py:250`, CLI
  `--speculative-num-steps` at `server_args.py:1591`) — the **chain depth** dial.
- `speculative_num_draft_tokens` defaults to `speculative_num_steps + 1`
  (`server_args.py:385-386`; also documented at
  `execution/model_executor.py:179`: `"spec_num_tokens == spec_num_steps + 1 for
  now (without Tree Attention)"`). This is the value that becomes
  `max_tokens_per_req` and therefore **directly sizes the CUDA-graph token
  dimension** (§3).
- `speculative_eagle_topk` is **hard-locked to 1**: `resolve_speculative_decoding`
  raises if it's anything else (`server_args.py:615-620`), and the CLI restricts
  `choices=[1]` (`server_args.py:1597-1601`) — "tree spec (topk>1) is not
  currently supported... only chain spec is wired end-to-end"
  (`server_args.py:617-619`). **So today there is no branching-factor dial, only
  chain length** — `speculative_num_steps` is the *entire* depth space. (This is
  also exactly the gap JetSpec-style tree drafting would fill — §1 — but it is
  out of scope for this design; noted so the depth dial described here is not
  confused with a future tree-width dial.)

**Can depth change per step, given the graph constraint?** No, not freely — §3
established that `max_tokens_per_req` is baked into the attention backend at
construction, so **each distinct depth needs its own stack** (own
`attn_backend`/`draft_attn_backend`/`CudaGraphWrapper`), exactly like the
spec-vs-non-spec split. Depth-continuous control (any integer 0..N per step) would
require as many stacks as depths, each with its own capture ladder (§4) — not
affordable, and JetSpec's static-tree-shape-per-launch precedent (§1) reinforces
that this is the *industry* pattern too, not just a limitation of our current
backend design.

**Recommendation**: a **small discrete set**, as the prompt suggests —
`{0 (off), 2, 4}` (i.e. `speculative_num_steps ∈ {0, 1, 3}` in this codebase's
units, since `num_draft_tokens = steps + 1`; confirm exact steps↔tokens target
against the pending ablation, §7). Three stacks total, each with its **own**
ladder truncated at its **own** crossover batch (§4) — and since deeper
speculation is expected to lose to saturation at a *lower* batch than shallow
speculation, the deepest stack should have the shortest ladder, compounding the
§4 VRAM saving rather than adding a third full-width ladder.

**Selection mechanism**: same as §3 — the control flag selects one of
`{stack_off, stack_shallow, stack_deep}` once per step, before dispatch. Not a
per-token or intra-round depth change; the round's shape is fixed once a stack is
chosen for that step.

## 6. Hysteresis / anti-flapping

**Proposed mechanics [design]**:

- **Dead-band**: turn a stack **on** at `bs <= C_on`, back **off** only at
  `bs >= C_off`, with `C_on < C_off` (e.g. `C_off = C_on + margin`), so a
  single-batch jitter around the crossover doesn't flip the decision every step.
- **Minimum dwell time**: once a transition fires, hold the new stack for at least
  `N` steps before allowing another transition, tracked as a plain step counter
  next to the flag (free to check, same tier as §2.1). `N` on the order of
  `decode_log_interval` (default 40, `server_args.py:144`) is a reasonable
  starting point — it's the cadence the engine already treats as "coarse enough
  to be stable" for its own accept-rate window (§2.2).

**Why this matters more here than in a generic threshold controller**: a
transition is not a cheap value flip — it re-points which `CudaGraphWrapper`
stack `_dispatch_forward` calls into (§3), which changes `max_batch_size`
admission behavior (§4's per-stack ladder), and cools/rewarms whatever the GPU's
L2 and kernel caches were warm with. Flapping near the crossover would therefore
not just waste cycles on the decision itself — it risks perturbing **admission**,
which is precisely the failure mode E25 fixed for the batch cap
(`docs/experiments-log.md:30`) and E28 hit for KV/mamba over-admission
(`docs/experiments-log.md:16`, §7 below). Hysteresis here is load-bearing for
admission stability, not just a throughput-smoothing nicety.

## 7. Multi-resource signals — the E28 cautionary tale, and why it applies here

**[measured]** E28 (`docs/experiments-log.md:16`, Qwen3.6-35B-A3B, bf16 MoE +
GDN, GPU2) found that `derive_adaptive_max_batch`
(`engine/event_loop.py:113-140`) derives its cap from **KV pages alone** and
**ignores the GDN/mamba state pool** — "~58 MB/sequence... only 114 chunks
available... adaptive-batch derives max_batch_size≈416 from KV pages alone,
ignoring the mamba pool ⇒ over-admission ⇒ retraction ('lost tree-owned Mamba
state') ⇒ C=64 hangs, C=128 device-side assert." The row explicitly says this
"belongs with the E19/E25 adaptive-batch work" — i.e. it is a known, unfixed gap
in the *existing* single-resource controller, not hypothetical.

**The generalizable rule**: an adaptive policy must be evaluated against **every
binding resource**, not the one signal that's easiest to read. Applied to this
design, the minimum resource set is:

1. **Compute saturation** — running batch size `bs` (§2.1, free).
2. **KV pressure** — and specifically, **spec decode increases KV consumption per
   step relative to non-spec decode at the same batch size**: verify writes up to
   `speculative_num_draft_tokens` KV entries per request per step, not 1. This is
   already visible in how the engine sizes the pool at boot:
   `decode_input_tokens = server_args.speculative_num_draft_tokens if
   server_args.speculative_algorithm is not None else 1`
   (`engine/event_loop.py:219-223`), passed to `create_attn_components`
   (`event_loop.py:241`) *and* into the C++ scheduler's config,
   `scheduler_cfg.decode_input_tokens` (`make_config(...,
   decode_input_tokens=decode_input_tokens, ...)`, `event_loop.py:460`), where the
   C++ side reserves exactly that many tokens per decode admission/schedule event
   (`tokenspeed-scheduler/csrc/fsm/forward_events.cpp`: e.g.
   `local_kv_allocator->Acquire(decode_input_tokens_)` at
   `forward_events.cpp:263,528`). **This is a further complication beyond what the
   prompt anticipated**: `decode_input_tokens` is a **single boot-time constant**
   baked into `scheduler_cfg` from whatever `speculative_num_draft_tokens` the
   process launched with. If depth becomes dynamic per step (§5), the C++
   scheduler's KV admission math is still calibrated for **one** fixed
   reservation size — switching stacks doesn't just swap CUDA graphs, it silently
   invalidates the scheduler's per-step KV reservation assumption unless either
   (a) the scheduler always reserves for the **deepest** configured stack (safe,
   but wastes KV budget whenever running shallower or off — a real throughput
   cost at the KV-constrained end of the range) or (b) `scheduler_cfg`/the C++
   side is made depth-aware (a materially bigger change, out of scope for Phase
   1). Recommend (a) for Phase 1, called out explicitly as a known conservatism
   to revisit.
3. **Any model-specific state pool** — `mamba_pool_total_chunks`
   (`engine/event_loop.py:428-433`) for GDN/hybrid models, per E28. Not every
   model has one; check for `has_mamba` (`event_loop.py:248-250`) and include it
   in the resource check when present.

So the "free" tier (§2.1) is **necessary but not sufficient** for this
controller by itself: page-ratio must be read **depth-aware** (a ratio safe for
`depth=0` may not be safe for `depth=4` at the same `bs`, per point 2 above), and
any present state pool must be checked too, exactly as the E28 postmortem
demands.

## 8. Where the thresholds come from — and what's blocked

Thresholds must be **measured, not hand-picked**, matching the project's
self-configuring principle. `docs/design/scheduling-improvements.md:16-18`
states the same discipline for the batch cap: "the optimal cap is load-dependent,
so it must be adaptive, not a hand-set constant" — and `derive_adaptive_max_batch`
(`engine/event_loop.py:113-140`) is exactly that: co-derived from measured KV
capacity (E19, `docs/experiments-log.md:36`), later flipped to the shipped
default once validated (E25, `docs/experiments-log.md:30`). This design proposes
the same trajectory for spec-decode thresholds: **[pending]** an MTP ablation is
running now (GPU 1, Qwen3.5-27B) measuring exactly the inputs this controller
needs:

| Measured quantity [pending] | Feeds |
|---|---|
| Crossover concurrency `C*` (batch where MTP throughput stops beating non-spec) | `C_off` (hysteresis upper threshold, §6); the spec stack's ladder top (§4) |
| Acceptance rate at `C=1` / `C=16` / `C=128` | (a) a model-level gate — near-zero acceptance at *every* `C` means don't build a spec stack for this model/draft pair at all, independent of batch; (b) the Phase-2 EWMA's initial value/prior instead of a cold start |
| TTFT/TPOT headroom vs. SLO target [Phase 2, needs SLO input] | Dynamically adjusts `C_on`/`C_off` — a tighter TPOT budget should push `C_off` down (turn spec off sooner, since verify latency threatens TPOT before raw throughput does) |

This document specifies the **mechanism and the parameterization slots**; it does
**not** fill in `C_on`/`C_off`/per-depth crossovers, because those numbers do not
exist yet. Phase 1 (§9) should ship the measured constants the same way E19→E25
shipped the KV-derived cap: land the mechanism gated/opt-in with the measured
value as the default once the ablation confirms it, not before.

## 9. Implementation sketch + phasing

### Phase 1 — static policy, cheap signals only, multi-stack ladder, hysteresis

Files:
- `server_args.py`: new flags, e.g. `--speculative-adaptive`,
  `--speculative-crossover-batch` (or per-depth crossover list),
  `--speculative-dwell-steps`; defaults filled from §8 once measured. Must also
  close the §4 latent gap: `derive_adaptive_max_batch`
  (`engine/event_loop.py:113-140`) needs a spec-aware variant so it stops
  extending a spec-shaped ladder to the same ceiling as the non-spec ladder.
- `engine/event_loop.py`: build N stacks instead of one at `EventLoop.__init__`
  (today's single `create_model_runner`/`create_attn_components`/
  `create_model_executor` call, `event_loop.py:211-315`, generalizes to a loop
  over configured depths); read the free signals (§2.1) once per step and select
  the active stack before `_dispatch_forward` (`event_loop.py:1686`).
- `execution/cuda_graph_wrapper.py`: `get_batch_sizes_to_capture` needs a
  per-stack ladder cap (§4); confirm whether `global_graph_memory_pool` sharing
  across independently-constructed wrappers is safe/possible (§4's flagged
  unknown) or whether each stack needs its own pool.
- `execution/model_executor.py`: forward dispatch reads the selected stack's
  `drafter` (or `None` for the off stack) instead of a single
  construction-time-fixed reference.
- `execution/drafter/mtp.py` / `eagle.py` / `dflash.py`: no change to drafting
  math; must simply be constructible per-stack (already true — `BaseDrafter`
  takes `spec_num_tokens`/`spec_num_steps` as plain constructor args,
  `drafter/base.py:42-43`).

Risk: **highest of the three phases.** Touches the assumption (§3) that one
process has exactly one `attn_backend` shape; correctness risk if the wrong
stack's metadata gets read; VRAM risk if §4's estimate is wrong. Mitigate with
the `torch.cuda.memory_stats()` boot-time instrumentation from §4, and fail loud
at boot if total captured VRAM would eat into the `gpu_memory_utilization`
budget (`server_args.py:414-429`) enough to shrink the KV-derived
`max_batch_size` below the static (non-adaptive) floor.

### Phase 2 — off-hot-path EWMA/SLO layer publishing to a flag

Files:
- New module, e.g. `engine/spec_decode_controller.py`: a `threading.Thread`
  generalizing today's inline, log-gated accept-rate computation
  (`execution/model_executor.py:1306-1322`) into a proper EWMA, plus SLO headroom
  (new). Publishes via the `pd/utils.py:433-477` `StepCounter` idiom (§2.3) —
  plain machine-word flags, no CUDA tensor needed since neither writer nor reader
  is a GPU kernel here.
- `event_loop.py`/`model_executor.py`: hot path reads the published flag
  (direct-address read, no lock) alongside the Phase-1 free signals before
  choosing a stack.

Risk: **medium.** New thread lifecycle to manage (start with the engine, stop
cleanly on shutdown — mirror `storage_executor.py`'s `threading.Event` stop
pattern, `cache/executor/storage_executor.py:189-196`); a staleness window
between publish cadence and hot-path read that must be bounded and documented,
not just assumed small.

### Phase 3 — autotune thresholds online

Files: extend the Phase-2 controller module with a bandit/hill-climb over
`C_on`/`C_off`/the active depth set, objective = live SLO attainment; needs the
learned thresholds to persist across restarts (self-config principle — nothing
auto should regress to a manual flag on every restart).

Risk: **highest correctness-of-outcome risk** — a mis-tuned online policy can
regress silently, no crash to catch it. Mitigate: run the Phase-4 validation
suite (below) continuously, not just at CI time; hard-floor Phase 3 to never tune
`C_off` **below** what §8's measurement established as safe — only refine within
a band around the measured value initially, the same conservative posture E19
took before E25 made adaptive-batch the default.

### Validation plan

- **High-concurrency regression gate**: the E13/E25-style sweep
  (`docs/design/scheduling-improvements.md`'s C=128/C=256 table), controller ON
  vs. static-off, at the concurrency the ablation (§8) identifies as past
  crossover — must be within noise. Directly tests that the controller correctly
  disables spec where it would hurt.
- **Low-concurrency floor**: same sweep at C=1/C=16 (or wherever §8 shows spec
  still wins), controller ON vs. static-on — must be within noise. Tests the
  controller isn't leaving throughput on the table by turning spec off too
  eagerly.
- **Anti-flapping test**: synthetic load oscillating around `C*` (e.g. a square
  wave between `C*-8` and `C*+8`); assert the stack-switch count stays bounded by
  the dwell-time window (§6) — a cheap regression test, runnable as soon as `C*`
  is known, independent of the full ablation.
- **Multi-resource / E28 regression test**: run the controller on a GDN/mamba
  model (the Qwen3.6-35B-A3B family E28 used) and assert no retraction/hang/
  assert under a batch that the KV-only signal alone would have admitted —
  encodes §7's rule as a concrete, CI-able check once GDN support stabilizes.
- **VRAM budget check**: assert total captured-graph VRAM (Phase 1's
  `torch.cuda.memory_stats()` instrumentation, §4) plus the KV pool fits within
  the existing `gpu_memory_utilization` budget, so the multi-stack design never
  silently shrinks max concurrency below what the user configured.
- **Correctness (lossless) check**: with the controller forcing rapid stack
  transitions (worst case for §1's correctness invariant), diff sampled output
  against a fixed-seed, spec-disabled baseline run under greedy sampling — must
  match token-for-token, proving the on/off switch never touches what gets
  sampled, only how many decode steps it costs to get there.

## 10. Open questions / blocked items

- Exact `C_on`/`C_off` and per-depth crossover numbers: **blocked** on the
  pending Qwen3.5-27B MTP ablation (GPU 1, §8).
- Concrete per-stack VRAM (MB): **not measured this session** — no GPU available
  under this task's constraints. Phase 1 should instrument and get real numbers
  (§4) before committing to a 3-stack design over a 2-stack (on/off only) one.
- Whether `AttentionBackend` could cheaply support a runtime-variable
  `max_tokens_per_req` without a second full instance per depth: **not resolved
  by reading** — flagged in §3/§5 as the single biggest engineering unknown here,
  worth a short spike before Phase 1 lands, since it changes N-stacks-of-buffers
  into potentially one-stack-with-a-parameter.
- Whether `global_graph_memory_pool` sharing (§4) extends safely across
  independently-constructed `CudaGraphWrapper` stacks, or whether each stack
  needs (and pays for) its own pool: **not resolved by reading**.
- Tree drafting (§1) is explicitly **out of scope** for this design — noted as
  future work that would plug into the same controller via a new `BaseDrafter`
  subclass, not a reason to block this design on adopting it first.
