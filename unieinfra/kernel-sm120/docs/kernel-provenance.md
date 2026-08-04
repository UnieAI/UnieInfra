# Kernel provenance

One entry per kernel. Each records what it was derived from, what it reuses
unchanged, what it changes, what it fuses, and what it deliberately does not do.
A kernel with no entry here is not finished.

The point is that "it runs and the numbers look right" is not a design. If the
next person cannot tell which decisions were reasoned and which were inherited,
they cannot improve it without redoing the work.

## Template

```
### <kernel name>
- **Operator**: family/mode, priority band
- **Derived from**: the specific implementation, with file path or upstream ref
- **Surveyed**: which engines' versions were compared, and the design table
- **Reused unchanged**: what came across as-is and why that was right
- **Changed**: what differs from the source, and the evidence for the change
- **Fused / unfused**: what is fused into this kernel, what deliberately is not
- **Declared traits**: what it claims to support, and what it refuses
- **Verified**: reference used, and the numbers
- **Known cost**: what is slow or unfinished about it
```

---

### `unieinfra_flashinfer_paged_mha_extend_sm120`

- **Operator**: `attention/mha_extend_with_kvcache`, `Priority.PLUGIN` (16)
- **Derived from**: FlashInfer 0.6.15's
  `BatchPrefillWithPagedKVCacheWrapper` — the general paged prefill wrapper, not
  the trtllm XQA path. Same family vLLM's FA2 backend is built on.
- **Surveyed**: **Not properly.** This kernel was chosen because it was the
  first FlashAttention-class path found that accepts the geometry this model
  forces (page_size 832, head_dim 256, GQA group 6), verified on-device before
  writing any code. It was **not** compared against SGLang's, TurboMind's or
  TRT-LLM's paged attention implementations, and no design alternatives were
  evaluated. That is a gap in the process, not a judgement that the alternatives
  are worse — see `.claude/skills/port-kernel/SKILL.md` §2 for the step that was
  skipped. Anyone revisiting this should start there.
- **Reused unchanged**: the wrapper's plan/run split, its NHD layout, and its
  scheduling. The kernel math is entirely FlashInfer's.
- **Changed**: two things, both about TokenSpeed's calling convention rather
  than the attention math.
  1. *Page index construction.* TokenSpeed hands over a dense
     `page_table [batch, max_pages_per_seq]` plus `cache_seqlens`; FlashInfer
     wants CSR-style `(kv_indptr, kv_indices, last_page_len)`. Built entirely on
     device — reading `cache_seqlens` on the host would serialise against the
     model's stream once per layer, 16 times per step.
  2. *Padded queries.* TokenSpeed pads `q` up to a CUDA-graph bucket, so
     `q.shape[0]` can exceed `qo_indptr[-1]`; the in-tree Triton kernel ignores
     the surplus through its grid bounds, while FlashInfer validates the two and
     refuses the call. This showed up as a startup crash
     (`q.shape[0] (16) does not match qo_indptr[-1] (1)`), not as a test
     failure. Slicing `q` would need `qo_indptr[-1]` as a Python int, and that
     device-to-host read is the thing being avoided. Instead the ragged index is
     extended by one absorber sequence covering the padding rows, given the null
     page so FlashInfer has something well-formed to attend over. With no
     padding the extra segment is empty and costs nothing.
- **Fused / unfused**: nothing is fused. KV-cache stores stay in TokenSpeed's
  `_save_kv_cache`, and the quantize-on-store path is untouched. Fusing the
  store would be a real optimization and is not attempted here.
- **Declared traits**: `is_causal` both; `sliding_window`, `support_sinks`,
  `support_logit_cap` all `{False}` — not implemented, so selection falls back
  to Triton for requests needing them. `return_lse` both.
- **Verified**: against the in-tree `triton_mha_extend_with_kvcache` through the
  ops dispatch, at 24 q heads / 4 kv heads / head_dim 256 / page_size 832:

  | shape | max abs | cosine |
  |---|---|---|
  | qlen 96, no padding | 6.104e-05 | 0.99999678 |
  | qlen 96, 32 pad rows | 6.104e-05 | 0.99999690 |
  | qlen 1, 15 pad rows | 6.104e-05 | 0.99999583 |
  | qlen 4, 12 pad rows | 6.104e-05 | 0.99999607 |

- **Known cost**: plans on every call. All 16 full-attention layers of a step
  pay for scheduling that one of them needed. Caching needs an invalidation key
  that survives in-place buffer reuse across steps, so a data-pointer key is
  unsafe. Unmeasured.

---

### `unieinfra_flashinfer_paged_mha_decode_sm120`

- **Operator**: `attention/mha_decode_with_kvcache`, `Priority.PLUGIN` (16)
- **Derived from**: the same FlashInfer paged **prefill** wrapper, used for
  decode.
- **Surveyed**: yes, for the CUDA-graph question specifically. Every vendored
  engine was read on one question — how do you run a paged-KV decode attention
  under a captured graph when the library wants a host-side scheduling pass?

  | engine | host "plan" object | where it runs | buffers pinned by | bucket granularity |
  |---|---|---|---|---|
  | vLLM FA2 (`v1/attention/backends/flash_attn.py`) | none | n/a | runner-owned `query_start_loc` / `seq_lens` / block table | any |
  | vLLM FA3 | `scheduler_metadata` tensor + fixed `max_num_splits` | `build()`, before capture and before replay | builder-owned persistent tensor, copy-in then tail-zero | one buffer, sliced |
  | vLLM FlashInfer (`v1/attention/backends/flashinfer.py`) | FlashInfer `plan_info` | `fast_plan_decode` inside `build()` | `paged_kv_*` CPU+GPU buffer pairs, sliced | **one wrapper per batch size** |
  | SGLang FlashInfer | FlashInfer `plan_info` | `init_forward_metadata_out_graph()`, called before `run_once()` at capture and before `replay()` | global `kv_indptr` / `kv_last_page_len` / `cuda_graph_kv_indices`, sliced | **one wrapper per batch size** |
  | LMDeploy | none — its own Triton paged kernel | n/a | `make_buffers_cudagraph` | power-of-2 then step-256 |
  | TensorRT-LLM | none — XQA schedules on GPU | n/a | `create_cuda_graph_metadata` | per (batch, draft-len) |

  Two engines need no plan at all because their kernels do tile assignment on
  the GPU. The three that use a host pass all solve it the same way, and that
  convergence is what this kernel copies.
- **Reused unchanged**: everything about the wrapper. The three-phase protocol
  is SGLang's, structurally unchanged: allocate persistent CSR buffers once per
  bucket, construct the wrapper over them with `use_cuda_graph=True`, plan
  outside the captured region at capture and again before every replay, capture
  only `run()`. Padding rows get `seq_len` clamped to 1 and `last_page_len` 1,
  also SGLang's convention (`get_cuda_graph_seq_len_fill_value` returns 1).
- **Changed**: five things.
  1. *The prefill wrapper, not the decode wrapper.* Every other engine uses
     `BatchDecodeWithPagedKVCacheWrapper`; that one refuses this model
     (`Unsupported group_size: 6`). `BatchPrefillWithPagedKVCacheWrapper` takes
     the same `use_cuda_graph=True` constructor and additionally requires a
     `qo_indptr_buf`, which for decode is a constant `arange` filled once.
  2. *`qo_indptr` from `max_seqlen_q`*, which TokenSpeed documents as uniform
     across the batch (1 for ordinary decode, the draft length under
     speculative decoding). `causal` is set only when `max_seqlen_q > 1` — a
     single query token has nothing to mask, while several packed tokens are a
     causal suffix of the cache, which is FlashInfer's bottom-right alignment
     and matches TokenSpeed's documented extend convention.
  3. *Sync-free CSR construction.* SGLang and the legacy in-tree
     `flashinfer/fa2.py` both compact the page ids with a mask and then use the
     result's length as a Python int, which is a device-to-host sync. Here the
     valid page ids are scattered to their CSR destinations and every
     out-of-range column is aimed at one trailing scratch slot nothing reads,
     so the whole index is built on device.
  4. *The hook is one function, not four.* The legacy tree exposes
     `fa2_decode_is_selected` / `make_fa2_decode_cuda_graph_wrapper` /
     `fa2_decode_cuda_graph_plan` / `fa2_decode_cuda_graph_run` and makes the
     MHA backend construct wrappers, own buffers, and branch between two call
     paths. This exposes one vendor-neutral op, `mha_decode_prepare`, which
     dispatches to a `prepare` attribute on the selected kernel; the kernel owns
     every FlashInfer object. Backends that select a kernel with no host pass
     get `False` back and change nothing. That keeps the runtime's only kernel
     boundary `tokenspeed-kernel`, which `AGENTS.md` requires.
  5. *Freshness is epoch-keyed, not pointer-keyed.* `prepare` bumps a counter;
     a slot planned in the current epoch against the same metadata tensors is
     reused. That is what makes all 16 full-attention layers of a step share one
     plan — the "plan-per-call" cost recorded against the previous version —
     while an un-hooked runtime (epoch stays 0) still re-plans every call and
     stays correct.
- **Why not the obvious decode kernels**: both were tried and both refuse this
  model. FlashInfer's trtllm XQA decode rejects page sizes outside
  {16, 32, 64, 128} and the GDN state row forces 832. FlashInfer's AOT
  `BatchDecodeWithPagedKVCacheWrapper` reports `Unsupported group_size: 6` at
  head_dim 256 — nothing to do with page size, and an earlier attribution of
  this failure to page size was wrong. The prefill wrapper has neither
  restriction.
- **Fused / unfused**: nothing fused.
- **Declared traits**: `sliding_window`, `support_sinks`, `support_logit_cap`
  all `{False}`; `return_lse` both.
- **Why this is safe under a graph, mechanically**: FlashInfer's `run()` bakes
  the host-side `plan_info` — grid dimensions and byte offsets into the int
  workspace — into the captured launch, so a later re-plan can only be seen
  through memory. `enable_cuda_graph` is what makes that sufficient. In
  `include/flashinfer/attention/scheduler.cuh`, `PrefillSplitQOKVIndptr` derives
  `cta_tile_q` from `total_num_rows - batch_size + 1` instead of the observed
  average, sets `padded_batch_size = max(max_batch_size_if_split,
  total_num_tiles_q)` where `max_batch_size_if_split` is `2 * SM_count /
  num_kv_heads`, and forces `split_kv` true. None of those depend on KV length,
  so every `aligned_alloc_offset` in `PrefillPlanImpl` returns the same value on
  every re-plan. Everything that does vary — `kv_chunk_size`,
  `request_indices`, `kv_tile_indices`, `merge_indptr`, `block_valid_mask` —
  lands in the device int workspace. `block_valid_mask[i] = i < new_batch_size`
  is the early exit that lets a fixed grid do less work than it was sized for,
  which is why `disable_split_kv=True` must **not** be passed here.
- **Verified**: against `triton_mha_decode_with_kvcache_cached` through the ops
  dispatch, at the live geometry (24 q heads / 4 kv heads / head_dim 256 /
  page_size 832), bf16:
  - eager, `max_seqlen_q` 1 and 4, batch 1/8/33: cosine 0.99999428-0.99999893
  - plan hoisted out, same shapes: cosine 0.99999428-0.99999851
  - **capture + replay**, batch 8 and 33: capture succeeds, and three replays
    per bucket with fresh page tables and sequence lengths give cosine
    0.99999452-0.99999851 against Triton run eagerly on the same inputs
  - **negative control** — replaying *without* the re-plan, across page-count
    changes (1→6, 6→2, 2→5 pages): cosine collapses to 0.246 / 0.427 / 0.465.
    That is the evidence the hoisted plan is load-bearing rather than
    decorative; a passing graph test alone would not have shown it.
  - padding rows (`seq_len` 0 and 1 mixed into a live batch): output stays
    finite, live rows cosine 0.99999440
  - capture with the hook missing raises a `RuntimeError` naming
    `prepare_paged_mha_decode`, instead of an opaque CUDA capture fault
- **Measured**: per-call wall clock against Triton at the same geometry, plan
  already hoisted, GPU 3 (shared): batch 1 4.8x, batch 8 5.4x, batch 32 1.13x,
  batch 128 1.10x, batch 32 with `max_seqlen_q=4` 1.53x. These include Python
  dispatch and were taken on a GPU shared with other work; they say the kernel
  is not slower, not how much a server gains. Serving-level throughput was
  **not** measured.
- **Known cost**:
  - `plan()` still copies three small int32 tensors to the host, so there is one
    device-to-host sync per step. vLLM and SGLang avoid it by handing FlashInfer
    a host mirror of the sequence lengths (`indptr_cpu=`,
    `global_override_indptr_cpu`); TokenSpeed's graph-path decode metadata has
    no such mirror, so it was left in.
  - Split-KV scratch is large at head_dim 256 — `num_qo_heads *
    padded_batch_size * cta_tile_q * head_dim * 4` bytes, ~148 MB for a small
    bucket and growing with batch. The workspace is grown on demand and the
    displaced buffer is deliberately retained, because graphs captured earlier
    baked its address as their scratch pointer.
  - Each bucket holds a private 8 MiB FlashInfer int workspace, which cannot be
    shared: a captured graph reads its schedule out of that buffer, and a
    target and draft graph are both planned before either replays.
  - **DFLASH block decode is not served.** Its `seq_lens` are written by the
    drafter *inside* the captured graph, so no schedule computed outside can
    describe the step that runs. The kernel reports the unprepared bucket rather
    than running on a schedule for the wrong lengths.

---

### `unieinfra_gdn_chunk_prefill_sm120`

- **Operator**: `attention/gdn_chunk_prefill`, `Priority.PLUGIN` (16). Displaces
  `triton_gdn_chunk_prefill` (`solution="triton"`, `PORTABLE` 4), which was the
  only admissible candidate on sm_120 — FlashInfer's GDN prefill gates on
  compute-capability **major 10** plus CUDA >= 13, so sm_120 never sees it.
  This matters more than most kernels: Qwen3.5-27B and Qwen3.6-35B-A3B are
  48 of 64 layers GDN, so this op is the dominant prefill cost.

- **Derived from**: the in-tree FLA pipeline itself
  (`tokenspeed_kernel/ops/attention/triton/linear/`), restructured. Not a port
  of anyone's kernel — the design it implements is TurboMind's, re-derived in
  Triton (see below).

- **Surveyed**: all seven vendored engines, per
  `.claude/skills/port-kernel/SKILL.md` §2. Design table:

  | | TokenSpeed (before) | vLLM | SGLang | TRT-LLM | lmdeploy TurboMind | modular MAX | tilert / JetSpec |
  |---|---|---|---|---|---|---|---|
  | prefill impl | Triton FLA | Triton FLA + CuteDSL sm100 | Triton FLA + CuteDSL sm100 | Triton FLA fork + FlashInfer sm100 adapter | **hand-written CUDA/CuTe, per-arch** | Mojo, token-sequential | none |
  | chunk size | 64 | 64 | 64 | 64 | 64 (sm_90), **32 (sm_120)**, 16 (pre-sm90) | 1 | — |
  | chunk_h tiles | BV=32 w4 s2, fixed | autotune BV{32,64} x w{2,4} x s{2,3,4} | 1 env-tunable config (in-place state write forbids autotune) | BK=64 unrolled, BV{32,64} autotuned | `BlockDv` {32,64} on sm_120, {32,64,128} on sm_90 | n/a | — |
  | chunk_o tiles | BK=128 BV=64 w4 s2, fixed | autotune 24 cfg, `BKV_LIST` gated on `check_shared_mem()` | hardcoded, autotune commented out | BK/BV {64,128} or {32,64} on low smem | n/a (fused) | n/a | — |
  | state layout | **`[N,H,K,V]` FLA-native** | `[N,H,V,K]` | `[N,H,V,K]` | `[N,H,V,K]` | `[N,H,K,V]` (TurboMind), `[N,H,V,K]` (PyTorch path) | `[N,H,K,V]` | — |
  | state dtype | h bf16, final fp32 | h bf16, final fp32 | h bf16, pool fp32 | h bf16, pool fp32 | templated fp32 or bf16 | pool bf16 | — |
  | intra-chunk inverse | 3 launches (kkt / solve_tril / w,u) | 3 launches | **2** (kkt+solve fused) | 3 launches | **fused into 1 `kkt_solve`** | n/a | — |
  | `h` checkpoint tensor | materialized in HBM | materialized | materialized | materialized | **never materialized** | n/a | — |
  | prefill launches | **8** | 6 core / 8 with conv+prep | 5 core / 9 total | 7 core / 9 with l2norm | **3** (5 with context-parallel) | 2 | — |
  | conv fused? | no | no (but split+l2norm+gating fused post-conv) | no | no | no | no | — |
  | varlen | `chunk_indices` computed in-op | CPU-precomputed in metadata builder, async H2D | `.tolist()` in-op → device sync | in-op | pointer-indirected TMA descriptors | n/a | — |
  | arch specials | none | sm100 tcgen05 + TMA + warp-spec | same, one gen behind | FlashInfer sm100/103 tcgen05/TMEM | sm_90 TMA + wgmma + 226 KB smem; **separate sm_120 kernel** | none | — |

  Two findings from the survey shaped this kernel:

  1. **Everyone materializes `h` except TurboMind.** lmdeploy's fused chunk
     kernel does `wy_fast` + `chunk_delta_h` + `chunk_o` in one persistent CTA
     so the `[B, NT, H, K, V]` checkpoint tensor never reaches HBM. At this
     model's geometry that tensor is 201 MB for a single 8192-token sequence,
     written by one kernel and read straight back by the next. That is the
     structural win, and it is available in Triton for the `chunk_h`+`chunk_o`
     half of it.
  2. **TokenSpeed is the only engine storing the state K-first.** Everyone
     else matches the runtime's `[N,H,V,K]` SSM pool, so nobody else pays a
     transpose. TokenSpeed's wrapper calls
     `initial_state.transpose(-2, -1).contiguous()` on every call of every one
     of the 48 GDN layers, because the FLA `.jit` kernel has no stride support
     for `h0`.

  lmdeploy's sm_120 kernel also drops chunk size from 64 to 32 relative to its
  own sm_90 kernel — an independent signal that the sm_90/sm_100 tile choices
  do not survive this device's shared memory.

- **Reused unchanged**: `l2norm_fwd`, `chunk_local_cumsum`,
  `chunk_scaled_dot_kkt_fwd`, `solve_tril`, `recompute_w_u_fwd` — imported from
  `tokenspeed_kernel`, not copied. Those five are 40% of the pipeline and the
  part that is genuinely hard to fuse: `solve_tril` is a 64x64 forward
  substitution, which is serial and would stall a chunk loop. SGLang fuses
  kkt+solve_tril into one launch and that is a real 1-launch win still on the
  table here; it was left alone because it is independent of this change.

- **Changed**: `chunk_gated_delta_rule_fwd_h` and `chunk_fwd_o` are replaced by
  one kernel, `_fused_chunk_ho_kernel`. One program owns
  `(sequence, v-head, V-tile)`, holds its `[K, BV]` slice of the recurrent
  state in registers across the whole sequence, and emits `o` per chunk.
  Consequences:
  - `h` is never allocated or written (201 MB at T=8192, 403 MB at T=16384).
  - `v_new` is never allocated (100 MB at T=8192); it existed only to carry
    corrected values from `chunk_h` to `chunk_o`.
  - `h0`/`ht` are addressed K-last directly through a block pointer with
    strides `(1, K)`, so the contract layout costs nothing and the
    `.contiguous()` copy disappears.
  - One fewer kernel launch.
  Evidence: 1.14x-1.54x end to end at N>=2, listed under **Measured**.

- **Fused / unfused**:
  - Fused: state passing (`chunk_h`) + output (`chunk_o`), including the
    `v_new` correction and the intra-chunk causal term.
  - Deliberately not fused: `wy_fast` (`w`/`u`) — TurboMind fuses it, but in
    this decomposition `w` is `[BT, K]` and would be recomputed once per
    V-tile, i.e. `V/BV` = 4 times.
  - Deliberately not fused: kkt + `solve_tril` (serial substitution, see above).
  - Deliberately not fused: the short conv. Every surveyed engine keeps it
    separate; vLLM fuses only the *post*-conv prep (split + l2norm + gating),
    which is a different kernel and a separate opportunity.
  - `gdn_decode_step` / `gdn_decode_mtp` are untouched: they are state-pool
    bound, not bandwidth bound, and the in-tree Triton kernel already addresses
    the pool K-last with no transpose.

- **Declared traits**: `head_dim {128}`, `head_v_dim {128}`,
  `head_v_eq_head_k {True}`, `num_v_gte_num_q {True}`,
  `qk_l2norm {False, True}`, and **`output_h {False}`**. The last one is the
  honest declaration that matters: `output_h=True` asks for exactly the
  per-chunk checkpoints this kernel exists in order not to write, and the
  runtime uses it on the speculative-decode h-tracking path
  (`hybrid_linear_attn.py`, `need_h_track`). Those requests select the in-tree
  Triton kernel. `capability` sets `max_arch_version=(12, 9)` so a future arch
  does not inherit this by version monotonicity.

- **Verified**: against `triton_gdn_chunk_prefill` through the ops dispatch
  (`solution="triton"` vs `solution="unieinfra_fused"`), at the production
  geometry Hk=16, Hv=48, K=V=128, bf16.

  **Max abs diff 0.0, mean abs diff 0.0, cosine 1.0000000000** on all 16 shapes
  tested — the output is *bit-identical*, not merely within bf16 tolerance.
  That is by construction: the fused loop performs the same operations in the
  same order and rounds `v_new` to the input dtype at the same point the split
  version does by storing it to a bf16 tensor.

  Shapes: T in {1, 63, 513, 2048, 3072, 4096, 8192, 16384}; N in {1, 2, 4, 5, 8};
  ragged `[777,1024,63,200]`, `[3000,17]`, `[65,1]`, `[6000,1,4095]`,
  `[1,1,1,1]`; sub-chunk and chunk-boundary lengths `[64,65,127,128,129]`;
  zero and non-zero initial state; explicit and defaulted `scale`;
  `qk_l2norm` on; bf16 and fp16. Both gate branches exercised.

  A negative control was run to prove the comparison can fail: perturbing the
  fused kernel's `scale` by 0.1% moves max abs diff from 0.0 to 4.9e-4.

  Not covered: `qk_l2norm=False`. The *reference* produces NaN there at this
  geometry (unnormalized q/k make the delta-rule inverse diverge), so there is
  nothing to compare against. Production always passes `qk_l2norm=True`.

- **Measured**: `gdn_chunk_prefill` end to end, both arms through the dispatch,
  interleaved and min-of-9-reps, GPU 3.

  | shape | seqlen | triton | plugin | speedup | branch |
  |---|---|---|---|---|---|
  | N=1 T=2048 | 2048 | 381.9 us | 387.0 us | 0.99x | split |
  | N=1 T=8192 | 8192 | 1252.2 us | 1252.5 us | 1.00x | split |
  | N=1 T=16384 | 16384 | 2584.0 us | 2584.0 us | 1.00x | split |
  | N=2 T=2048 | 1024 | 395.3 us | 309.4 us | **1.28x** | fused |
  | N=4 T=4096 | 1024 | 648.9 us | 536.0 us | **1.21x** | fused |
  | N=8 T=8192 | 1024 | 1327.8 us | 1050.2 us | **1.26x** | fused |
  | N=16 T=16384 | 1024 | 2745.5 us | 2136.8 us | **1.28x** | fused |
  | N=4 T=8192 | 2048 | 1314.1 us | 1110.1 us | **1.18x** | fused |
  | N=8 T=16384 | 2048 | 2674.6 us | 2190.3 us | **1.22x** | fused |
  | N=4 T=16384 | 4096 | 2684.7 us | 2354.0 us | **1.14x** | fused |
  | N=16 T=4096 | 256 | 766.1 us | 515.0 us | **1.49x** | fused |
  | N=16 T=8192 | 512 | 1390.7 us | 1055.1 us | **1.32x** | fused |
  | N=32 T=4096 | 128 | 891.1 us | 576.9 us | **1.54x** | fused |
  | N=32 T=32768 | 1024 | 5542.0 us | 4292.6 us | **1.29x** | fused |

  Per-stage cost of the pipeline this replaces, same geometry, one 8192-token
  sequence: l2norm x2 43.8, cumsum 25.8, kkt 58.0, solve_tril 107.4, wy 279.9,
  **chunk_h 383.5, chunk_o 332.4** us — the fused pair is 58% of the total.

  These are kernel-level wall clock including Python dispatch. Serving-level
  throughput was **not** measured. `nvidia-smi` utilisation is not quoted as
  evidence here: it reads 100% on idle GPUs 2 and 3 on this host (log E22).

- **Known cost**:
  - **The fusion loses at N=1 and is gated off there.** The fused kernel's only
    parallelism is `(V/BV) x N x Hv` = 192 blocks at N=1 against 188 SMs, and
    the chunk loop is strictly serial because each chunk consumes the previous
    chunk's state. The split kernels get an extra NT-way parallel dimension in
    `chunk_o`. Measured 0.90x-0.98x at N=1 before the gate. `FUSED_MIN_SEQS=2`
    delegates those calls to the in-tree kernel unchanged, which is why the
    N=1 rows above read 1.00x rather than a regression. A single long prefill
    is a common shape, so this is a real limitation, not a corner case.
  - **The fix for N=1 exists and is not implemented.** TurboMind splits one
    long sequence into segments, computes per-segment states with `FusedGdrH`,
    then stitches them with `CorrectInitialStates` (`ContextParallelPlan`).
    That is the principled way to give a single sequence more than 192 blocks.
    It needs a second pass and an output correction, and was out of scope.
  - `num_stages=1` — no software pipelining of the chunk loop. Measured ~2x
    faster than `num_stages=2` (N=1 T=2048: 268 us vs 884 us) because the
    serial state dependency gives the pipeliner nothing to overlap while the
    duplicated operand buffers cost shared memory. `num_stages>=3` does not
    compile on this device at all.
  - **Shared memory is the binding sm_120 constraint.** Opt-in limit here is
    101376 B; sm_100 offers ~227 KB. Every `num_stages=3` config of this kernel
    needs 105-140 KB and fails to launch. The same cliff hits the in-tree
    kernel: `chunk_h` at BV=64/stages=3 needs 107016 B and at BV=32/stages=4
    needs 123660 B, so a third of vLLM's 12-config `chunk_h` autotune space is
    silently unreachable on this device. `_pick_config()` therefore branches on
    the driver-reported `shared_memory_per_block_optin`, never on the arch name.
  - **Retuning the in-tree tiles is not the lever.** A sweep of `chunk_h`
    (BV x warps x stages, 10 configs) and `chunk_o` (BK x BV x warps x stages,
    8 configs) at this geometry found the shipped defaults within 1-3% of the
    best reachable config at every shape tested. TokenSpeed's hardcoded
    BV=32/w4/s2 and BK=128/BV=64/w4/s2 are already right for sm_120.
  - Still on the table, untouched: SGLang's kkt+solve_tril fusion (165 us of
    the 8192-token pipeline in two launches), vLLM's fused post-conv prep
    (l2norm is 44 us and is almost entirely launch overhead), and the
    `[B,T,H,BT]` `A` / `w` / `u` intermediates that TurboMind also avoids.
  - `register_gdn()` is a standalone entry point and is **not** wired into the
    package's `register()`; the caller must add it.

---

### `unieinfra_triton_unquant_moe_apply_sm120` and `unieinfra_triton_fp8_block_moe_apply_sm120`

One entry for both: they are the same kernel with `USE_FP8_W8A8` flipped, and
the design decisions are shared.

- **Operator**: `moe/apply`, `Priority.PLUGIN` (16), `solution="unieinfra_triton"`
- **Derived from**: the Triton `fused_moe` design that vLLM, SGLang and lmdeploy
  all converged on — `moe_align_block_size` → block-sorted grouped GEMM with
  index-only gather/scatter → separate activation → second GEMM with the router
  weight in the accumulator → top-k sum.
  Reference source read for the kernel body:
  `third_party/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:295`
  (`fused_moe_kernel`).
- **Surveyed**: all seven repos in the skill's table, before writing any code.

  | | expert sort | GEMM | act fused? | top-k weight | sm_120 status |
  |---|---|---|---|---|---|
  | **vLLM** | `moe_align_block_size` CUDA → `sorted_token_ids`/`expert_ids`/`num_tokens_post_pad`, block-padded | Triton `fused_moe_kernel`, index-only gather (`offs_token // top_k`) | no, separate `silu_and_mul` | in GEMM2 accumulator (`MUL_ROUTED_WEIGHT`) | Triton yes; **its own CUTLASS MoE excluded** (`cap < 90 or cap >= 110`); all `trtllm_*` excluded (`is_device_capability_family(100)`, `cap//10`); FlashInfer-CUTLASS explicitly allowed |
  | **SGLang** | same, plus `FUSE_SUM_ALL_REDUCE`, `c_sorted`, TMA descriptors | fork of the same kernel | no | same | Triton is the **default** for unquantized; persistent/TMA-gather explicitly disabled (`if is_sm120_supported(): is_persistent = False`) |
  | **lmdeploy** | 2-phase `tl.sort` + atomic, **no materialised permute**, `reindex_a`/`reindex_c` flags | Triton, 2-D grid `(tiles, E)`, over-provision + retire | no | separate `moe_reduce` kernel | arch-agnostic Triton, would run unmodified |
  | **TRT-LLM** | 3-kernel `cub::BlockScan` prefix sum → `expert_first_token_offset`, **explicit `expandInputRows` permute** | CUTLASS grouped GEMM (TMA-WS on 90/100/103/120) | FC1 gated act fused **only on the non-TMA-WS path**; FC2 `FINALIZE` fuses unpermute+scale+reduce | FC2 epilogue, or separate finalize | trtllm-gen: **no sm_120 cubins at all** (`CudaArch` = Sm90a/Sm100a/Sm100f/Sm103a; Python raises `if sm_version >= 120`). CUTLASS: sm_120 only instantiates NVFP4×NVFP4 and FP8×FP4; FP8×FP8 redirected to the Sm89 path, BF16 to Sm80 |
  | **Modular MAX** | `moe_create_indices` — one 32-thread block per expert scanning all tokens; single 64-bit atomic fuses slot-assign + prefix-sum; CSR offsets, **no block padding**; separate `gather` materialises the permute | dedicated `grouped_matmul` with CSR offsets | no (except two SM100-only fusions: `grouped_matmul_swiglu_nvfp4`, `mega_ffn`) | separate batched matmul, or folded into the EP combine kernel | **nothing**. `_is_sm12x_gpu` exists but is never used in the MoE path; sm_120 falls through to `naive_grouped_matmul` — one scalar dot product per thread, no smem, no tensor cores |
  | **tilert** | fused into one `ExpertSelectUpGateSiLU` op | closed-source prebuilt binary, `mma.m16n8k32` swizzles, weights pre-sharded across a hardcoded `num_sms = 128` | yes, fully | via `expert_probs_out` into the down op | B200-only binary, ABI-pinned; not portable |
  | **mlsys2026 contest** | — | — | — | — | **no implementations**, only task prompts; its `moe-fp8` track targets B200 and names `tcgen05`/TMEM explicitly |

  Two conclusions the table forced, neither of which was the starting assumption:
  1. **Nobody fuses the two expert GEMMs on portable hardware.** The only
     single-launch MoE fusions in any of the seven are vendor/closed kernels with
     no sm_120 build (FlashInfer `b12x`, trtllm-gen, Modular's `mega_ffn`,
     tilert). So "fuse it" is not a shortcut this port passed up; it is unclaimed
     territory that needs its own project.
  2. **Two independent trees route sm_120 away from their fast path by name**
     (SGLang disables persistent/TMA gather, Modular has no sm_120 branch at all
     and lands on a naive kernel). Those are scars, not oversights.
- **Reused unchanged**: the index-only gather/scatter contract — `sorted_token_ids`
  holding flat `token * top_k + slot` values, A gathered by `offs_token // top_k`,
  C scattered by `offs_token`, `num_valid_tokens` as the padding sentinel so a
  padded lane simply masks off, and `num_tokens_post_padded` read from device so
  over-provisioned M blocks retire. This is byte-identical in vLLM and SGLang and
  is the de-facto interface; matching it keeps the kernel swappable with theirs.
  Also reused: the `GROUP_SIZE_M` L2 swizzle, and applying the router weight in
  GEMM2's accumulator rather than in a separate reduction.
- **Changed**, and the evidence for each:
  1. *Tile tables taken from the vendored trees rather than invented.*
     BF16 from SGLang's
     `configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json`;
     FP8 from vLLM's `E=256,N=512,...,dtype=fp8_w8a8,block_shape=[128,128].json`.
     Both were tuned on **this GPU at this model's geometry** (E=256, N=512 is
     Qwen3.6-35B-A3B). Their shape is itself the evidence: `BLOCK_N <= 128` and
     `BLOCK_M <= 64` until M reaches the thousands, where the H100 tables in the
     same directories use 128×256×128.
  2. *Shared-memory clamp keyed on the device, not the arch.* `_smem_budget()`
     reads `shared_memory_per_block_optin` (101376 B measured here, against
     sm_100's 232448) and `_clamp_stages()` lowers `num_stages` until the
     pipelined A/B tiles fit. SGLang gates its MoE defaults on the same attribute
     (`< 128 KB`); TRT-LLM's `cutlass_heuristic.cpp:558` queries
     `cudaDevAttrMaxSharedMemoryPerBlockOptin` and, below 120 KB, deletes every
     sm_120 tile except 128×128×64, in a comment that names this GPU. vLLM has no
     such guard — the one place its defaults are wrong here. Modular's SM90/SM100
     paths hardcode `H100.`/`B200.shared_memory_per_multiprocessor` as compile-time
     constants, which is the same bug in a different language.
     **Honest cost**: the clamp can lower a stage count the tuner validated on
     this device (bf16 at 512 tokens: table says 3 stages for 32×128×128, the
     estimate says 40960 B/stage × 3 > budget, so it runs at 2). Deterministic and
     correct, possibly slower than letting Triton's own pipeliner decide. Untested
     for performance.
  3. *`BLOCK_K` pinned to the FP8 group size.* The kernel loads exactly one
     scale group per K iteration (`offs_ks = k * BLOCK_K // GROUP_K`), so
     `_fp8_config` forces `BLOCK_K = 128` regardless of the table's 256 entries.
     Conservative rather than clever; the alternative needs an inner scale loop.
  4. *Non-local EP routes go to a sentinel bucket, not an `expert_ids == -1`
     branch.* vLLM keeps the `-1` case and has the kernel write zeros. Here the
     global→local rebase happens on the host, out-of-range routes sort into a
     bucket that is never emitted, and the output cache is zero-filled instead
     (only when something was actually dropped, so the common `ep_size == 1` path
     pays nothing). Fewer kernel branches; same result, verified below.
  5. *Alignment in PyTorch, not a CUDA kernel.* `_align_blocks` reproduces
     `moe_align_block_size` with `argsort`/`bincount`/`searchsorted`, written to
     avoid a device-to-host sync: the padded total stays a device tensor and the
     grid is sized from the static worst case `num_routes + E_local * (BLOCK_M - 1)`.
     This is the largest known cost (see below).
- **Fused / unfused**: nothing fused beyond what the reference design fuses —
  the router weight into GEMM2's accumulator, and the dequant into GEMM's
  epilogue for FP8. SwiGLU and the top-k sum are separate launches, matching all
  three Triton trees. Deliberately **not** fused: the two expert GEMMs, and
  GEMM1+SwiGLU. See survey conclusion 1 for why that is a project and not an
  oversight.
- **Declared traits**: `weight_dtype` `{unquant}` / `{fp8}` with
  `fp8_scale_block_shape {(128, 128)}`; `activation {silu, swiglu}`;
  `routing_mode {precomputed_topk}`; `supports_ep {False, True}`;
  `ispp_alignment {1}` for unquant and **`{128}` for FP8**.
  Refused, and therefore left to the in-tree kernels: `kernel_routing`,
  `supports_bias`, `supports_deferred_finalize`, `supports_all_to_all_ep`,
  `internal_activation_dtype` other than `input`, and every other weight dtype.
  The FP8 `ispp_alignment` is a **correctness** requirement, not a tuning one:
  the loader lays w13's block scales out as `2 * ceil(ispp/128)` rows, which
  only equals `ceil(2*ispp/128)` when ispp is 128-aligned. Below that the two
  halves' scale grids do not tile the N axis and the kernel would silently read
  the wrong scale. gemma-4's ispp=704 is exactly that case, so it is refused —
  and falls through to `flashinfer_cutlass_fp8_moe_apply`, which raises. A
  refusal that lands on a broken fallback is still the right refusal.
- **Verified** through the ops dispatch, with `solution=` pinning both sides.
  Bar is cosine > 0.9999.

  | case | reference | max abs | mean abs | cosine |
  |---|---|---|---|---|
  | bf16, T=1 / 7 / 64 | in-tree `flashinfer_cutlass_unquant` | 7.8e-03 | 1.2e-03 | 0.9999871 / 0.9999868 / 0.9999869 |
  | bf16, T=512 | in-tree cutlass | 1.2e-02 | 1.2e-03 | 0.9999869 |
  | bf16, T=512 | fp32 oracle | 9.9e-03 | 1.1e-03 | 0.9999910 |
  | gemma-4 shape (E=128, D=2816, ispp=704), T=1/64/512 | in-tree cutlass | 1.6e-02 | 2.0e-03 | 0.9999875 / 0.9999872 / 0.9999871 |
  | fp8 block, T=1 / 64 / 512 | W8A8-matched oracle | 3.2e-02 | 3.4e-03 | 0.9999200 / 0.9999164 / 0.9999163 |
  | fp8 block, T=512 | plain bf16 oracle | 1.2e-01 | 1.7e-02 | 0.9979854 |
  | EP=2, rank0 / rank1 | local-only fp32 oracle | 7.5e-03 | 7.7e-04 | 0.9999908 / 0.9999911 |
  | EP=2, rank0 + rank1 | full fp32 oracle | 1.0e-02 | 1.2e-03 | 0.9999894 |

  Awkward cases, all against the fp32 oracle, all passing:

  | case | why it is awkward | cosine |
  |---|---|---|
  | fp16 at T=1/33/130 | declared in `signatures` and was untested until the contract audit | 0.9999999 |
  | T = 3 / 17 / 63 / 129 / 511 | not multiples of `BLOCK_M`, so every expert run has a partial last block | 0.9999909–0.9999911 |
  | all tokens → one expert | maximum load imbalance; 255 of 256 experts get an empty run and one gets `T * top_k` rows | 0.9999920–0.9999934 |
  | non-contiguous `x` (`stride(0) = 2 * D`) | the kernel indexes A by `stride_am`, so a sliced hidden-state buffer must work | 0.9999910 |
  | T = 0 | early return; returns `(0, 2048)` without launching | n/a |
  | selection unpinned | `solution=` omitted so the registry chooses, per §5 of the skill | 0.9999910 |

  The two FP8 rows are the point. Against a plain bf16 oracle the kernel looks
  like a 0.998 failure; against an oracle that applies the *same* fp8 rounding
  the kernel applies, it is 0.99992. The 4.4% L2 gap is the W8A8 recipe, not the
  kernel. There is no kernel reference available for FP8 on this arch — both
  in-tree candidates raise — which is precisely why the reference had to be built
  to match the recipe rather than the dtype.
- **Known cost**, in the order it is likely to matter:
  1. *`_align_blocks` is PyTorch.* An `argsort` + `bincount` + `searchsorted` +
     scatter per MoE layer per step, where vLLM and SGLang each have a single
     CUDA kernel (SGLang's has a `< 1024` routes fast path in one CTA). At 40
     layers this is 40 sorts a step. It is correct and sync-free, but it is the
     first thing to replace.
  2. *Two separate activation-quantization passes on the FP8 path*, also plain
     PyTorch, for the same reason: the in-tree `per_token_group_quant_fp8`
     returns a flattened scale on this platform, which is the bug that makes
     `flashinfer_trtllm_fp8_moe_apply` raise, so depending on it would import
     that ambiguity.
  3. *The bf16 registration displaces a working kernel on an unproven argument.*
     `flashinfer_cutlass_unquant_moe_apply` verifies at cosine 0.999996 — better
     than this kernel — and is only being outranked because it needs FlashInfer's
     JIT and startup autotune and then reports
     `No tuned config covers trtllm::fused_moe::gemm1 ... falling back to tactic=-1`
     at every token count tested. **No throughput A/B was run.** If one says
     otherwise, drop the unquant registration and keep the FP8 one, which has
     nothing to lose to.
  4. *`num_stages` may be lower than the tuner's answer* — see Changed (2).
  5. *No small-M / decode specialisation.* The tables bottom out at
     `BLOCK_M = 16..32` for M ≤ 8, which is what the tuner chose, but TRT-LLM's
     `min_latency_mode` (no permute, `swap_ab = true`, dense per-expert token
     matrix) and Modular's persistent `TileScheduler` are both architecture-neutral
     ideas that nothing here uses. Both are portable to sm_120; neither was tried.
- **Reachability (§4 trap 3, and worse than the trap describes)**: `solution=` is
  a hard filter, and for MoE the requestable set is a **closed enum** —
  `MoeBackend` in `runtime/layers/moe/utils.py` is
  `{auto, triton, gluon, flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl,
  deep_gemm_mega_moe, mega_moe}`. There is no plugin member, so
  `solution="unieinfra_triton"` can never come from `--moe-backend`, and these
  kernels are reachable **only** under `--moe-backend auto` (which passes
  `solution=None`). This is not fixable from out of tree; naming the solution
  `"triton"` to sneak in would shadow the in-tree Triton MoE kernels, which is
  worse. Benchmarking with any explicit `--moe-backend` will silently measure a
  different kernel.
  Related, and its own finding: on sm_120 `--moe-backend triton` for a bf16 MoE
  raises `NoKernelFoundError` outright — there is no in-tree NVIDIA Triton
  unquantized MoE apply at all, only AMD ones and the mxfp4/mxfp8 entries.
- **Not wired**: `__init__.py::register()` does not call `register_moe()` yet, so
  these kernels are inert in the live engine.

---

### `unieinfra_bmm_fp8_scaled_sm120`

- **Operator**: `gemm/bmm`, `Priority.PLUGIN` (16)
- **Derived from**: FlashInfer 0.6.15's `flashinfer.gemm.bmm_fp8`, which is a
  thin wrapper over cuBLASLt's strided-batched FP8 GEMM. Its internal entry
  point is named `fp8_gemm_sm100`, but the name is misleading: the path is
  cuBLASLt, not a tcgen05 kernel, and it runs correctly on sm_120.
- **Surveyed**: yes, all seven vendored engines. The relevant finding is that
  **vLLM and SGLang independently converged on this exact call** for FP8
  batched matmul — vLLM at `vllm/utils/flashinfer.py:642-662`, SGLang at
  `python/sglang/srt/layers/quantization/fp8_utils.py:176-203` (registered as
  `gemm.bmm_fp8`, commented "flashinfer cuBLAS backend"). Neither writes its
  own. Design table for `bmm`:

  | engine | dense bf16 bmm | fp8 bmm |
  |---|---|---|
  | vLLM | `torch.bmm` (`mla_attention.py:861,1148`) | FlashInfer `bmm_fp8` |
  | SGLang | `torch.bmm` (`forward_mla.py:550,988`) | FlashInfer `bmm_fp8` |
  | LMDeploy | `torch.bmm` (`deepseek_v2.py:377`) | none |
  | TRT-LLM | CuTe-DSL `cute_dsl_bf16_bmm_blackwell`, **SM100/103 only** — raises on sm_120 (`cute_dsl_custom_ops.py:7908-7911`) | own `fp8_block_scaling_bmm_out` |
  | MAX/Mojo | own `bmm.mojo`; SM100 branch excluded on sm_120 (`info.mojo:693-696`), falls to Ampere/SM80 tiles or cuBLASLt | `bmm_sm100_blockwise_scaled_fp8` |
  | TileRT | none (B200 only) | none |
  | flashinfer-contest | no bmm task; B200/tcgen05 only | n/a |

  So: **nobody has a hand-written bf16 bmm that runs on sm_120**, and the two
  engines closest to TokenSpeed use precisely the library call adopted here.
- **Reused unchanged**: all of it. The kernel math is cuBLASLt's.
- **Changed**: nothing in the math; three things in the calling convention.
  1. *Layout.* cuBLASLt FP8 requires TN — A row-major `[M, K]`, B column-major
     `[K, N]`. TokenSpeed's bmm contract already hands B as row-major
     `[batch, N, K]`, which *is* a column-major `[batch, K, N]`, so
     `B.transpose(1, 2)` is a free view. This is load-bearing and silent if
     got wrong: passing a genuinely contiguous `[batch, K, N]` tensor passes
     `bmm_fp8`'s shape check and returns garbage (measured cosine 0.0157,
     -0.0021, -0.0000 at three shapes). The kernel therefore *rejects*
     `B.stride(-1) != 1` rather than calling `.contiguous()`.
  2. *Small-shape delegation.* See "Known cost".
  3. *Scales.* `bmm_fp8` takes 0-dim or `(1,)` tensors; a Python float raises
     `TypeError`. Per-tensor only, hence the narrow signature.
- **Fused / unfused**: nothing fused. No bias epilogue (the kernel is not in
  `_KERNELS_WITH_FUSED_BIAS`, so dispatch applies bias as a post-GEMM add, same
  as the reference).
- **Declared traits**: `n_align_16`, `k_align_16` both `{True}` — cuBLASLt
  rejects other N/K with "the requested functionality is not supported"
  (measured at N=100 and at K=100). Registered **only** for the per-tensor
  `scaled-fp8` signature; the per-channel signature that
  `torch_bmm_fp8_scaled` also serves is deliberately left to the reference,
  because `bmm_fp8` takes scalar scales and cannot express per-row scaling.
  `out_dtype` is bf16/fp16 — fp32 raises, and is excluded by the signature.
- **Verified**: against `torch_bmm_fp8_scaled` (the kernel it displaces), FP32
  accumulation, per-tensor scale 0.05, on an RTX PRO 6000 Blackwell:

  | shape | max abs | mean abs | cosine |
  |---|---|---|---|
  | B32 M1 N512 K256 | 0.0020 | 0.0003 | 0.9999986 |
  | B32 M16 N512 K256 | 0.0039 | 0.0003 | 0.9999986 |
  | B8 M512 N512 K256 | 0.0039 | 0.0003 | 0.9999987 |
  | B40 M4096 N128 K128 | 0.0020 | 0.0002 | 0.9999986 |
  | B16 M1024 N5120 K512 | 0.0039 | 0.0004 | 0.9999987 |
  | B64 M512 N1024 K512 | 0.0039 | 0.0004 | 0.9999986 |

  Speed, min-of-9 blocks because GPU 3 has co-tenants:

  | shape | reference | plugin | speedup | plugin TFLOP/s |
  |---|---|---|---|---|
  | B32 M512 N512 K256 | 0.1699 ms | 0.0772 ms | 2.20x | 55.6 |
  | B40 M4096 N128 K128 | 0.3416 ms | 0.0817 ms | 4.18x | 65.7 |
  | B16 M1 N5120 K512 | 0.5302 ms | 0.0858 ms | 6.18x | 1.0 |
  | B16 M1024 N5120 K512 | 2.1741 ms | 0.1910 ms | **11.38x** | 449.8 |
  | B64 M512 N1024 K512 | 1.1810 ms | 0.0834 ms | **14.17x** | 412.1 |

- **Known cost**: `bmm_fp8` has a flat ~0.06-0.09 ms per-call floor on sm_120
  (cuBLASLt heuristic + FlashInfer workspace), nearly independent of shape,
  while the reference is dominated by materializing `B.float()` and so scales
  with `batch*N*K`. Below the crossover the reference is genuinely faster, and
  the *decode-shaped MLA absorb* (batch 32, M 1, N 512, K 256) sits below it.
  The trait system cannot express this — `gemm/bmm` builds traits from N and K
  only, never batch or M — so the kernel delegates back to
  `torch_bmm_fp8_scaled` when `batch*N*K < 8Mi and M < 256`. Both thresholds
  were fitted to a measured sweep over batch/M/N/K, not derived. Residual cost
  of the delegation is one registry lookup and one Python call: 0.92-0.97x on
  the delegated shapes (~2-3 us absolute).

---

### `unieinfra_mm_fp8_blockscale_sm120`

- **Operator**: `gemm/mm`, `Priority.PLUGIN` (16)
- **Derived from**: nothing new — it is a **dispatch decision** over the two
  in-tree kernels, `triton_mm_fp8_blockscale` and
  `flashinfer_mm_fp8_blockscale`, both called through the registry so no
  numerics are duplicated.
- **Why it has to exist out of tree**: priority is a per-kernel constant.
  `flashinfer_mm_fp8_blockscale` (SPECIALIZED+3 = 15) outranks
  `triton_mm_fp8_blockscale` (PERFORMANT+3 = 11) at *every* M, and the traits
  `gemm/mm` builds carry only N/K alignment — there is no `m` trait, so no
  in-tree registration can say "I win below M=256". Measured on sm_120, Triton
  is the faster of the two for M <= 256 on all three Qwen3.5-27B weight shapes.
- **Surveyed**: yes. The M-dependent split is the same conclusion vLLM and
  SGLang reached independently for sm_120 FP8 block-scale GEMM, both of which
  branch on small M and use a **swapAB tile N=32** CUTLASS variant there —
  vLLM `scaled_mm_blockwise_sm120_fp8_dispatch.cuh:164-172`, SGLang
  `fp8_blockwise_scaled_mm_sm120.cuh:428-455` (`m<=64 || m%4!=0` -> swapAB,
  "up to ~1.2x faster"; tile N=16 is impossible because the SM120 blockwise
  collective needs `EPI_TILE_N=32 | CTA_N`). This plugin gets the same effect
  by kernel *choice* rather than by tile choice, because TokenSpeed does not
  vendor an SM120 CUTLASS blockwise GEMM. Porting those CUTLASS kernels is the
  obvious next step and would likely beat both.
- **Reused unchanged**: both backends entirely, including
  `flashinfer_mm_fp8_blockscale`'s M-padding to a multiple of 4.
- **Changed**: the kernel performs its **own** online activation quantization
  when `A_scales is None`, using
  `quantize_fp8_with_scale(granularity="token_group", scale_encoding="float32",
  solution="triton")` — bit-identical to what the in-tree dispatcher already
  applies on sm_120 (`ops/gemm/__init__.py`, the
  `arch_version == ArchVersion(12, 0)` branch), and the layout both backends
  consume. This is necessary because `_online_quantize_mxfp8` dispatches on
  **kernel name** through a hard-coded if-chain and raises `ValueError` for any
  name it does not know — which is every out-of-tree kernel. See the companion
  in-tree change in the changelog.
- **Fused / unfused**: nothing fused. Activation quantization is a separate
  launch, as in tree.
- **Declared traits**: `n_align_128`, `k_align_128` `{True}` — inherited from
  the FlashInfer backend used above the split, so the kernel refuses shapes it
  could only serve on one side. Plus `self_quantizes_mxfp8` `{True}`, which is
  the opt-out described above. Registered only for the `(128,128)` float32
  block-scale signature; the `(1,32)` ue8m0 signatures are left alone.
- **Verified**: against an FP32 dequantize-and-matmul reference (both operands
  dequantized from the same quantized tensors the kernel receives), 18 shapes
  across three Qwen3.5-27B weight shapes. Max abs 0.0078-0.0156, mean abs
  0.0011, **cosine 0.9999985-0.9999987 at every shape** — indistinguishable
  from either backend alone, as it must be. Also verified end-to-end through
  `tokenspeed_kernel.mm(quant="mxfp8", A_scales=None)`: cosine 0.999664, no
  NaN, kernel selected.
- **Speed** (min-of-9, vs `flashinfer_mm_fp8_blockscale` alone):

  | N,K | M=1 | M=16 | M=64 | M=256 | M=512 | M=4096 |
  |---|---|---|---|---|---|---|
  | 5120, 5120 | 1.30x | **1.54x** | 1.36x | 1.29x | 1.00x | 1.00x |
  | 34816, 5120 | 0.93x | 1.39x | 1.13x | 0.96x | 0.99x | 0.97x |
  | 5120, 17408 | 1.20x | 1.25x | 1.26x | 1.21x | 1.00x | 0.96x |

  Peak 690 TFLOP/s at M=4096.
- **Known cost**: the split point is a single constant (M=256) fitted to three
  weight shapes; it is wrong for the widest one. At N=34816 (gate_up) Triton
  loses at M=1 (0.93x) and M=256 (0.96x) while winning 1.39x at M=16, so a
  scalar threshold cannot capture it — the real boundary depends on N as well
  as M. Not fixed, because fitting a second parameter to three shapes is
  overfitting; the correct fix is the SM120 CUTLASS blockwise kernel above.
  The M=4096 rows should read exactly 1.00x (they route to FlashInfer and do
  nothing else); the 0.96-0.97x there is residual measurement noise from GPU
  co-tenants, not a real cost.
- **Not wired**: `register_gemm()` is a standalone entry point and is **not**
  called from the package's `register()`; the caller must add it.

---

### `unieinfra_flashinfer_quantize_mxfp4_sm120`

- **Operator**: `quantization/mxfp4`, `Priority.PLUGIN` (16)
- **Derived from**: FlashInfer 0.6.15's `fp4_quantize` with
  `sf_vec_size=32, sf_use_ue8m0=True`, which dispatches through
  `get_fp4_quantization_module("120")` to a purpose-built **`sm120a`** JIT
  module (`flashinfer/quantization/fp4_quantization.py:182-190, 228-230`).
  The device code is TRT-LLM's `cvt_warp_fp16_to_fp4`
  (`csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh:431`).
- **Surveyed**: all five vendored engines plus `modular` and the FlashInfer
  contest tree. The MXFP4 activation-quant landscape on sm_120:

  | engine | MXFP4 act-quant on sm_120 | shared exponent | E2M1 encode | SF layout |
  |---|---|---|---|---|
  | **TRT-LLM / FlashInfer** | **yes** — `__CUDA_ARCH__ >= 1000` covers 1200 | `ceil(log2(amax/6))` via `__nv_cvt_float_to_e8m0(..., cudaRoundPosInf)`, **never clips** | PTX `cvt.rn.satfinite.e2m1x2.f32` | linear / 128x4 / 8x4 |
  | **vLLM** | **no** — `mxfp4_experts_quant.cu:380` returns `cap >= 100 && cap < 120`, and the source is absent from `FP4_SM120_SRCS` (`CMakeLists.txt:998-1014`) | OCP `floor(log2(round_up(amax))) - 2`, **can clip to 6.0** (`nvfp4_utils.cuh:244-260`) | same PTX cvt | swizzled 128x4 |
  | **SGLang** | **no** — `mxfp4.py:359` raises `"requires SM90 or SM100"`; only Marlin W4A16 remains | `ceil(log2(amax/6))` (reference path only, `mxfp4_tensor.py:65-73`) | Python LUT, round-**half-up**, not RTNE | via FlashInfer |
  | **lmdeploy / tilert** | none (FP4 is weight-only, activations are FP8) | — | — | — |

  So TRT-LLM/FlashInfer is not merely the convenient choice, it is the only
  engine in the survey whose MXFP4 activation quantizer is not arch-excluded on
  this card. vLLM — which runs on this GPU today — deliberately excludes it,
  which is why its exclusion is worth naming rather than ignoring.
- **Reused unchanged**: all of it. No wrapper math, no re-derivation of the
  scale. The public `fp4_quantize` entry point is called directly.
- **Changed**: nothing in the kernel. Three things at the call boundary:
  1. *The global scale is passed as a constant `1.0` and documented as inert.*
     `fp4_quantize` takes one unconditionally, but the UE8M0 branch computes
     the exponent purely from the block amax and never reads it
     (`quantization_utils.cuh:471-480`). Verified rather than assumed:
     quantizing the same input with global scales `1.0` and `137.0` gives
     byte-identical data **and** scale factors. This matters because the
     tokenspeed contract declares `has_global_scale=False`, and silently
     honouring a scale that the format has no place for would be worse than
     refusing one.
  2. *`scale_layout` is mapped onto FlashInfer's two booleans* —
     `linear -> is_sf_swizzled_layout=False`, `128x4 -> (True, False)`,
     `8x4 -> (True, True)` — so all three layouts the tokenspeed API names are
     reachable, including its own default of `128x4`.
  3. *Non-contiguous inputs are densified.* `fp4_quantize` treats
     `stride(-2) == 1` as column-major and transposes; a strided view would
     otherwise be quantized along the wrong axis with no error.
- **Fused / unfused**: nothing fused. TRT-LLM ships fused
  `fused_rmsnorm_fp4_quantize` (`thop/rmsNormFp4Quant.cpp:243`) and vLLM ships
  `silu_and_mul_nvfp4_quant`
  (`csrc/.../fp4/activation_nvfp4_quant_fusion_kernels.cu:38`); neither is used
  here because tokenspeed's `quantize_mxfp4` is a standalone op with no
  producer in its signature. Fusing it would need a new op in the tree, not a
  plugin.
- **Declared traits**: `scale_size {32}`, `scale_layout {linear, 128x4, 8x4}`,
  `has_global_scale {False}`, `scale_encoding {ue8m0}`. Refuses `scale_size 16`
  — the kernel asserts `sfVecSize == 32` whenever `sfUseUE8M0` is set.
- **Verified**: against an independent float32 model of the FlashInfer scheme
  (E2M1 grid with RTNE and tie-to-even, E8M0 exponent by `ceil(log2(amax/6))`),
  built from torch primitives — **not** against another kernel.
  - UE8M0 scale bytes: **100.0000%** identical to the reference.
  - E2M1 codes: **100.0000%** identical to the reference.
  - Reconstruction vs the float32 input, 512x1024 bf16: max abs error is
    **exactly 0.5x the format's own quantization step** (the widest E2M1 gap,
    `2.0 * 2^e`) at every magnitude from 1e-3 to 1024 — i.e. the RTNE
    half-step bound, met with equality and never exceeded.
  - Cosine vs float32 input: 0.9933 (randn*4), 0.9930 (randn*0.02), 0.9916
    (heavy-tailed). This is the 4-bit format's own loss, not the kernel's:
    the in-tree Triton kernel scores 0.9937 / 0.9937 / 0.9918 on the same data.
  - Layouts: `128x4` output is exactly `nvfp4_block_scale_interleave(linear)`,
    and the packed data is byte-identical across all three layouts. Checked at
    `M=300` so row padding to 128 is exercised.
- **Known cost / stated precision difference**: **not bit-identical to the
  Triton path, by design, and slightly worse on mean error.** The two use
  different shared-exponent recipes — `ceil(log2(amax/6))` here vs OCP
  `floor(log2(amax)) - 2` in-tree — which disagree on **11.8%** of blocks and
  change **3.2%** of dequantized elements, by up to one full E2M1 step. The
  disagreeing blocks are *exactly* the ones where the OCP recipe has to clip:
  both figures are 11.804% on the same input, which is the mechanism, not a
  coincidence. The
  trade is real and goes both ways: the OCP recipe packs the block max closer to
  6.0 and wins ~4% on mean absolute error, but must clip it in **11.8%** of
  blocks; this recipe never clips. Both stay inside the half-step bound.
  Anything comparing an activation quantized here against a weight scale
  produced by the other recipe will lose accuracy silently — vLLM and
  TRT-LLM have exactly this incompatibility between them.

### `unieinfra_flashinfer_quantize_nvfp4_sm120`

- **Operator**: `quantization/nvfp4`, `Priority.PLUGIN` (16)
- **Derived from**: the same FlashInfer `fp4_quantize`, with `sf_vec_size=16`
  and E4M3 scale factors.
- **Surveyed**: same table. NVFP4 is the one FP4 path with real sm_120
  support across engines — vLLM builds `nvfp4_quant_kernels.cu` for `12.0a;12.1a`
  under `-DENABLE_NVFP4_SM120=1` (`CMakeLists.txt:974-996`) and admits it at
  runtime with `sm >= 120 && sm < 130` (`nvfp4_quant_entry.cu:69`).
- **Reused unchanged**: everything, including the reciprocal convention.
- **Changed**: only that `scale` is reciprocated at the boundary, because
  tokenspeed's public API passes the *actual* scale while FlashInfer wants the
  inverse. This mirrors the in-tree `flashinfer_quantize_nvfp4` on sm_100
  exactly, so the two architectures cannot disagree about what `scale` means.
  A scalar is also promoted to a `[1]` float32 device tensor — the in-tree
  version passes `1.0 / scale` straight through, which raises on a Python
  float.
- **Fused / unfused**: nothing fused.
- **Declared traits**: `scale_layout {linear, swizzled}`, `has_scale {True}`.
- **Verified**: against a float32 model of the NVFP4 scheme (per-16 E4M3 scale
  factor `e4m3(amax/6/global)`, E2M1 data). Scale-factor bytes **100.0000%**
  identical; reconstruction error exactly **0.5x** the format's step; mean
  error 0.31 on 512x1024 bf16 randn*4.
- **Known cost**: this registration exists because `quantization/nvfp4` had
  **zero** kernels on sm_120, so there is nothing to compare it against for
  speed. Untimed.

### `unieinfra_triton_quantize_fp8_with_scale_sm120`

- **Operator**: `quantization/fp8_with_scale`, `Priority.PLUGIN` (16)
- **Derived from**: written here, but the recipe is the one every surveyed
  engine agrees on: `scale = amax / 448`, then `clamp(x / scale, -448, 448)`.
- **Surveyed**: the E4M3 bound and the amax floor, where the engines actually
  differ:

  | engine | E4M3 max | clamp | amax / scale floor |
  |---|---|---|---|
  | vLLM | 448 (`utils.cuh:27`), **224** for fnuz | explicit `fmaxf/fminf` **and** `__NV_SATFINITE` | scale floor `1/(448*512)` = 4.39e-6 |
  | SGLang | 448 (`utils.h:375`), 224 on ROCm | explicit + `__NV_SATFINITE` | `1e-10` on the group path; per-token path has **none**, branches on `scale == 0` |
  | TRT-LLM | 448 (`cudaFp8Utils.h:39`) | **saturating instruction only**, no clamp | `1e-10` (1x128), scale floor 4.39e-6 elsewhere |
  | lmdeploy | `finfo.max` | explicit `tl.clamp` | `1e-6` Triton / `1e-8` turbomind |
  | tilert | 448 hard-coded | explicit | `1e-4` |

  Five orders of magnitude of disagreement on the floor. This kernel takes
  `1e-10`, matching the in-tree token-group kernel it sits beside, so the two
  granularities on this card do not disagree with each other.
- **Reused unchanged**: `torch.finfo(float8_e4m3fn).max` rather than a literal,
  so the constant cannot drift from the dtype.
- **Changed** relative to the in-tree kernels: the reduction loops over the row
  in `BLOCK`-sized tiles instead of materialising `next_pow2(N)` lanes per
  program, so cost tracks `N` rather than the next power of two above it. The
  per-tensor scale is a max over the `[M]` row-amax vector, not a second full
  pass over `[M, N]`.
- **Fused / unfused**: nothing fused. Every surveyed engine fuses this into the
  preceding RMSNorm or SiLU (vLLM `rms_norm_dynamic_per_token_quant`, TRT-LLM
  `fused_add_rms_norm_quant`, SGLang `dsv4_norm_rope.cu`, lmdeploy
  `v4_swiglu_and_quant_fp8`), and that is the single largest improvement left
  on the table here. It is not attempted because tokenspeed's
  `quantize_fp8_with_scale` is a standalone op whose signature has no producer
  in it.
- **Declared traits**: `granularity {tensor, token}`, `scale_encoding {float32}`.
  Deliberately **not** claimed: `token_group` (the in-tree Triton kernel already
  covers it correctly here) and `ue8m0` (its encoded integer layout is
  backend-specific, and there is no consumer on this card to validate against —
  a kernel that wins selection and hands back a differently-encoded scale is
  worse than no kernel).
- **Verified**: against a float32 E4M3 model (RTNE by binade, SATFINITE at 448),
  512x1024 bf16 including a 50x heavy-tailed row and an all-zero row:

  | granularity | codes == ref | max abs err / (16*scale) | max rel err, normal range | underflow to 0 |
  |---|---|---|---|---|
  | tensor | 100.0000% | 0.9884 (bound 1.0) | 0.0581 (bound 2^-4 = 0.0625) | 0.0269% |
  | token | 100.0000% | 1.0000 (bound 1.0) | 0.0588 (bound 0.0625) | 0.0004% |

  Scales match the reference to 0.0e+00 relative error. Cosine vs the float32
  input is 0.99965 (tensor) / 0.99965 (token) on randn*4, 0.99964 / 0.99976 on
  heavy-tailed data — the per-token advantage on heavy tails is the expected
  one.
- **Known cost**: one program per row, so at `M=16384, N=128` it runs 16384
  programs of 128 elements — correct but launch-bound. Untimed against
  anything, because `granularity="tensor"` and `"token"` had **no admissible
  kernel at all** on this card; there is nothing to compare against.

### `unieinfra_triton_quantize_fp8_sm120`

- **Operator**: `quantization/fp8`, `Priority.PLUGIN` (16)
- **Derived from**: the in-tree `triton_quantize_fp8`
  (`ops/quantization/triton.py:105`), same math, different launch geometry.
- **Surveyed**: every engine's plain FP8 cast is a flat vectorized elementwise
  kernel — vLLM `vectorize_with_alignment<16>` (`common.cu:44`), SGLang
  `vec_t<T,16>` (`per_token_quant_fp8.cu:44`), TRT-LLM `float4` load / `uint2`
  store (`cudaFp8Utils.cu:74`). None of them tiles by row. The in-tree kernel
  is the outlier.
- **Reused unchanged**: the numerics, exactly — including the
  `x * (1.0 / scale)` order, so results are **bit-identical**.
- **Changed**: flat 1-D indexing over `numel` instead of
  `BLOCK_M x next_pow2(N)` per program. The in-tree heuristic ramps `BLOCK_M`
  with `M` (4 / 16 / 32) while `BLOCK_N` is `next_pow2(N)`, so at
  `M=4096, N=7168` one program holds `16 x 8192 = 131072` elements and spills
  catastrophically. `BLOCK=2048, num_warps=4` chosen from an interleaved sweep
  over `{1024, 2048, 4096, 8192} x {4, 8}`; 1024 was the only setting that
  loses to the in-tree kernel anywhere.
- **Fused / unfused**: nothing fused; PDL is forwarded unchanged.
- **Declared traits**: `has_scale {True, False}` — the same surface as the
  kernel it displaces, which is what lets it claim bit-identity.
- **Verified**: bit-identical to the in-tree kernel at `(512,1024)`,
  `(333,577)`, `(2048,7168)`, `(17,129)`, with and without a scale; and
  **100.0000%** identical to a float32 E4M3 reference at all four shapes.
- **Known cost**: interleaved A/B on a **shared** GPU, medians of 7, with the
  drift of the A-series reported because the machine is not stationary:

  | shape | in-tree | plugin | speedup | A drift |
  |---|---|---|---|---|
  | (8192, 1024) | 6.3 us | 4.3 us | 1.47x | 0.9% |
  | (1024, 2112) | 4.2 us | 2.2 us | 1.93x | 1.0% |
  | (4096, 7168) | 804.6 us | 12.8 us | **63x** | 8.8% |
  | (16384, 128) | 2.2 us | 2.1 us | 1.02x | 1076% |

  Only the first three rows are worth quoting; the last is pure noise and is
  shown to say so. The 63x is the register-spill cliff, not a tuning win — it
  reproduced at 38x in an earlier, differently-loaded run. **No serving-level
  throughput was measured.**

---

### `unieinfra_fa2_mha_prefill_sm120` / `unieinfra_fa2_mha_extend_sm120` / `fa2_mha_decode_with_kvcache`

- **Operator**: `attention/mha_prefill` and `attention/mha_extend_with_kvcache`
  at `Priority.PLUGIN + 1` (17); `attention/mha_decode_with_kvcache` written and
  verified but **not registered by default** — see *Declared traits*.
- **Derived from**: vLLM's flash-attention fork, `flash_attn_varlen_func` with
  `fa_version=2`, called through the `vllm_flash_attn` extension exactly as
  `vllm/v1/attention/backends/flash_attn.py` calls it. No kernel source was
  written or modified; this is a wrapper onto a prebuilt binary.
- **Provenance of the binary, which matters more than usual here.** FA2 is not
  installable from PyPI in any form that works against this image: the
  `vllm-flash-attn` distribution is abandoned at 2.6.2 (torch 2.4 / cu121) and
  `flash-attn` 2.8.3 ships sdist only, while `tokenspeed:sm120-dev` has no
  `nvcc`. The working artifact is the extension inside vLLM's container images.
  `vllm/vllm-openai:cu130-nightly` is torch 2.11.0+cu130 / Python 3.12 /
  cxx11-ABI, an exact match for this image, and its
  `_vllm_fa2_C.abi3.so` loads unmodified. The `v0.26.0-cu129` image's copy does
  **not** — it links `libcudart.so.12` and this image has only CUDA 13.
- **What actually executes is Ampere PTX, JIT-compiled.** `cuobjdump --list-elf`
  on every shipped `_vllm_fa2_C.abi3.so` inspected (cu129 and cu130) returns
  `sm_80` cubins and nothing else; `--list-ptx` returns the matching
  `compute_80` PTX. The fork's `CMakeLists.txt` clamps FA2 unconditionally:
  `cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" "${CUDA_ARCHS}")`.
  Building for `12.0` produces the same binary. So "FA2 on sm_120" means the
  driver JIT-compiling Ampere FlashAttention-2 — `mma.m16n8k16`, no TMA, no warp
  specialisation. There is no sm_120-tuned FA2 in existence; the only fork that
  emits `sm_120a` FA2 SASS is sgl-kernel's `infllm_ops`, and it covers only
  bf16 head_dim 64/128, which this model's head_dim 256 excludes.
- **Surveyed**: yes, on the question "what FlashAttention can run here at all".

  | candidate | state on sm_120 | evidence |
  |---|---|---|
  | FA4 CuteDSL (in-tree, `ops/attention/flash_attn`) | registered `min==max==(10,0)`; needs `tcgen05` TMEM | `ptxas` rejects `tcgen05.st` |
  | FA3 (`flash_attn_3` wheel in this image) | 62 `sm_90` cubins, **zero PTX** | `no kernel image is available for execution on the device` |
  | FA2, vLLM fork | **works**, via `compute_80` PTX JIT | this entry |
  | FA2, Dao-AILab upstream | source-only build; same `8.0+PTX` clamp would apply | `flash-attn` 2.8.3 sdist, no `nvcc` in image |
  | FA2, sgl-kernel `infllm_ops` | genuinely emits `sm_120a` SASS | bf16 hdim 64/128 split kernels only |
  | FlashInfer paged prefill | works; the incumbent | `unieinfra_flashinfer_paged_mha_*_sm120` above |

  Head dims compiled into the shipped FA2: `[32, 64, 96, 128, 192, 256]` —
  note 160 is **not** there, and 256 is, which is the one this model needs.
- **Reused unchanged**: everything. TokenSpeed's extend contract *is* FA2's
  varlen-paged contract: `cu_seqlens_q` is FA2's `cu_seqlens_q`, `page_table` is
  its `block_table`, `cache_seqlens` is its `seqused_k`. Nothing is gathered,
  compacted, transposed or copied on the way in — which is the whole reason this
  is faster than the incumbent, whose CSR page-index rebuild alone costs 3-5 ms
  per call.
- **Changed**: three small things, none of them attention math.
  1. *`cu_seqlens_kv` is dropped.* FA2 rejects being given both a `cu_seqlens_k`
     and a `seqused_k`; the per-request visible length is the one that is right
     for a paged cache.
  2. *Decode synthesises its `qo_indptr`.* The decode dispatch does not pass one,
     and for a uniform query length it is a constant of the bucket, so it is
     built once and cached on `(device, batch, seqlen_q)` — a value key, never a
     data pointer, and the tensor is constant, which makes it safe to bake into
     a captured graph.
  3. *LSE is transposed.* FA2 returns `[num_heads, total_q]`; the operator
     contract is `[total_q, num_heads]`.
- **Padded `q` is handled by doing nothing**, and that is the interesting part.
  FA2's grid is `(cdiv(max_seqlen_q, block_m), batch, num_heads)` and each block
  clips against `cu_seqlens_q`, so rows past `cu_seqlens_q[batch]` are never
  read and never written. Verified: with 0, 7 and 64 padding rows appended to a
  `batch=4, q_len=32, ctx=4096` call, the real rows are **bit-identical** to the
  unpadded call. No slice, no `out=` view, no device-to-host read of
  `cu_seqlens_q[-1]`. The incumbent FlashInfer extend kernel **raises** on the
  same input (`q.shape[0] (192) does not match qo_indptr[-1] (128)`) — see the
  note under its own entry above, which documents an absorber-sequence
  mitigation that is not present in the code as it currently stands.
- **Fused / unfused**: nothing fused. KV-cache stores stay in TokenSpeed's
  `_save_kv_cache`. FA2 could fuse neither anyway.
- **Declared traits**: `is_causal` both; `return_lse` both; `sliding_window`,
  `support_sinks`, `support_logit_cap` all `{False}`. The last three are
  understated on purpose — FA2 implements sliding windows and logit softcap
  natively and this wrapper could forward both in two lines, but neither is
  verified here, so they stay off and selection keeps routing those requests to
  Triton. Attention sinks are a genuine gap: FA2 raises `NotImplementedError`
  for `s_aux`.

  **Decode is deliberately unregistered** (`UNIEINFRA_FA2_REGISTER_DECODE=1`
  turns it on). Kernel for kernel FA2 loses decode to FlashInfer at every shape
  swept, and the reason is structural: FA2's varlen paged path enables split-KV
  only when `max_seqlen_q == 1` (`flash_api.cpp::mha_varlen_fwd` gates
  `set_params_splitkv` on `seqlenq_ngroups_swapped`, comment "Only apply split-k
  for decoding"; the else-branch hard-checks `num_splits <= 1, "num_splits > 1
  is not supported for varlen paged KV"`). A speculative draft block therefore
  gets no KV parallelism at all and wastes all but 4 rows of a 64-row query
  tile. It cannot be forced on.
- **Verified**: against `triton_mha_*` through the ops dispatch, at 24 q heads /
  4 kv heads / head_dim 256 / page_size 832, bf16 unless stated. 22 comparisons,
  no failures; cosine bar 0.9999, and `rel_l2` is quoted alongside because
  cosine is blind to a wrong `softmax_scale`.

  | case | cosine | rel_l2 |
  |---|---|---|
  | extend b4 q128 ctx1024 | 0.99999571 | 2.89e-03 |
  | extend b8 q32 ctx4096 | 0.99999547 | 3.02e-03 |
  | extend b1 q4096 ctx0 | 0.99999803 | 2.00e-03 |
  | extend, 64 padding rows | 0.99999541 | 3.02e-03 |
  | extend, 7 padding rows | 0.99999547 | 3.01e-03 |
  | extend ragged q+ctx, partial last page (831/832/833/1663) | 0.99999577 | 2.92e-03 |
  | extend non-causal | 0.99999619 | 2.79e-03 |
  | extend fp16 | 1.00000000 | 3.61e-04 |
  | extend `return_lse`, out / lse | 0.99999571 / 1.00000000 | 2.91e-03 / 4.79e-08 |
  | decode b32 ctx4096 q1 | 0.99999559 | 2.96e-03 |
  | decode b64 ctx2048 q1 | 0.99999583 | 2.87e-03 |
  | decode b16 ctx4096 q4 (speculative) | 0.99999535 | 3.02e-03 |
  | decode ragged ctx | 0.99999583 | 2.90e-03 |
  | decode fp16 | 0.99999994 | 3.52e-04 |
  | prefill b4 s1024 | 0.99999839 | 1.72e-03 |
  | prefill ragged (1/333/1024/2047) | 0.99999857 | 1.67e-03 |

  `rel_l2` sits at 2-3e-03 for every bf16 row because that is bf16 output
  rounding (2^-9 = 2.0e-03), not disagreement; the fp16 rows drop to 3.5e-04,
  which is the same story one format down. FA2 was also compared directly
  against the FlashInfer kernel on the same inputs and agrees to the same
  tolerance.

  Under CUDA-graph capture: capture and replay succeed with **no prepare hook
  and no plan**, cosine 1.00000000 against eager; after mutating
  `cache_seqlens` and `q` in place, replay still tracks eager at cosine
  1.00000000 — FA2 reads its schedule off the device, so there is nothing
  host-side to refresh.
- **Known cost**: three.
  1. **Cold-start JIT.** The first launch of each shape costs tens of seconds on
     a cold CUDA JIT cache — measured 31 s (extend) and 53 s (decode) in this
     image. The driver caches the result under `CUDA_CACHE_PATH` (default
     `~/.nv`), so it is paid once per (kernel, arch, driver); a container with
     no persistent `~/.nv` pays it on every cold start.
  2. **No sm_120 tuning exists.** What runs is Ampere PTX. Kernel for kernel,
     FlashInfer's paged prefill is equal or faster at every shape measured; FA2
     wins on the extend path only because the incumbent's per-call `plan()` and
     CSR rebuild cost more than the entire kernel. If someone hoists that plan,
     the extend justification for this kernel largely disappears.
  3. **Padding rows of the output are uninitialised.** They are whatever
     `torch.empty_like` returned, exactly as with the incumbent, because zeroing
     them would cost a launch and detecting them would cost a sync.
