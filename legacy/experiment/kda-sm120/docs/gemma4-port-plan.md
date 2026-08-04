# Gemma 4 on TokenSpeed — gap analysis of the EXISTING implementation

Status: gap analysis. Written 2026-08-02. Supersedes the "port plan" framing —
Gemma 4 is already implemented; this document says what that implementation
does, what it cannot do, and what it does silently wrong.

Subject: `/workspace/roy/UnieInfra/experiment/legacy/python/tokenspeed/runtime/models/gemma.py`
(608 lines, + 28 lines uncommitted).
Hardware: RTX PRO 6000 Blackwell (sm_120, 101376 B smem/SM), GPU 3.

---

## 0. Headline

The existing `gemma.py` is a careful, well-documented implementation whose
**model-side** handling of all four hard config fields is correct — and on
`attention_k_eq_v` it is **better than vLLM, SGLang and TensorRT-LLM**, all
three of which take a naive path it avoids.

It is registered and it will load. But it **cannot serve `gemma-4-31B-it`
correctly**, for a reason that lives entirely outside the model file: the KV
cache pool has no concept of per-layer head_dim, so the 10 full-attention
layers write 2048 elements/token into rows sized for 4096. **This does not
raise. It silently corrupts.** Measured, §4.

Second finding: the head_dim-512 backend question the briefing called "the
first thing to find out" has a clean answer (FlashInfer, verified numerically
on sm_120) — but it is **not the binding constraint**. The binding constraint
is at head_dim **256**, which breaks the Triton prefill kernel on sm_120 and
therefore affects Gemma 4's 50 *sliding* layers. §5.

---

## 1. Corrections to the briefing

Two rounds of correction. The first four were wrong in the original premise;
the fifth is a correction to the redirect.

| # | Stated | Actually | Evidence |
|---|---|---|---|
| 1 | "TokenSpeed has no gemma model implementation at all" | A 608-line Gemma 4 implementation exists and is registered | `experiment/legacy/.../models/gemma.py` |
| 2 | "it appears NOT to be registered — grepping `gemma` in registry.py returns nothing" | **It IS registered.** `registry.py` contains no model names *at all* — it auto-discovers by walking `tokenspeed.runtime.models` with `pkgutil` and reading each module's `EntryClass`. `gemma.py:608` exports `EntryClass = [Gemma4ForConditionalGeneration, Gemma4ForCausalLM]`. Grepping for model names in that file will always return nothing, for every model. | `registry.py:96-106`; `gemma.py:608` |
| 3 | `attention_k_eq_v` — "K and V are the SAME tensor" (model-wide) | True only for `full_attention` layers. Sliding layers have a real `v_proj`. In the 26B checkpoint, exactly 25 of 30 layers have `v_proj` — the 5 missing ones are the full-attention positions. | safetensors index, both checkpoints |
| 4 | "it is an omni model … + audio_config" | `audio_config: null` in **both** checkpoints. Vision only. (The 12B "unified" variant has audio; these two do not.) | `config.json` |
| 5 | `global_head_dim: 512` is the full-layer difference | Full layers **also** change KV head count: `num_global_key_value_heads` = 4 (31B) / 2 (26B) vs `num_key_value_heads` = 16 / 8. Full layers are therefore **cheaper** per token (2048 elems) than sliding layers (4096), not more expensive. | `config.json`, tensor shapes |

### Measured geometry (31B)

```
layer 0 (sliding_attention)        layer 5 (full_attention)
  q_proj  (8192, 5376) = 32x256      q_proj  (16384, 5376) = 32x512
  k_proj  (4096, 5376) = 16x256      k_proj  ( 2048, 5376) =  4x512
  v_proj  (4096, 5376) = 16x256      v_proj  ABSENT
  q/k_norm (256,)                    q/k_norm (512,)
  layer_scalar = 0.0894              layer_scalar = 0.8789
```

KV bytes/token/layer (bf16, K and V both stored): sliding `16384`, full `8192`.
The 50 sliding layers are capped at the 1024-token window; the 10 full layers
are unbounded. So at long context the full layers — the k_eq_v ones — are
essentially the entire cache cost (`10 x 8192 = 80 KiB/token`).

---

## 2. Gap analysis against the four hard fields

### 2.1 `attention_k_eq_v` — model side BEST-IN-CLASS, cache side absent

`gemma.py:271-283, 345-349`:

```python
if self.k_eq_v:
    self.v_proj = None
...
if self.v_proj is None:
    v = k_raw          # raw k_proj output, BEFORE k-norm and RoPE
```

**This is better than all three reference implementations.** vLLM
(`weight.clone()`), SGLang (a second `weight_loader(param, w, "v")` call) and
TensorRT-LLM (`weights.setdefault(v_key, value)`) each *duplicate the K weight
into a V slot* and then run the projection twice. `gemma.py` builds no `v_proj`
at all and reuses the K activation. That saves, per full layer, a
`2048 x 5376` bf16 weight (~22 MB, ~220 MB across the 31B's 10 full layers) and
one GEMM per token.

The ordering is also right, and it is subtle. `v_norm` consumes the **raw**
`k_proj` output, before `k_norm` mutates it. `gemma.py` takes `v = k_raw` at
line 346 and only norms `k` afterwards at 355. TRT-LLM comments this same
ordering as a trap.

**Gap: the KV cache still stores K and V separately.** `v` is passed to
`self.attn(q, k, v, ...)` like any other model, so the pool holds both. No
capacity is saved. Measured: the pool reports `K size: 1.59 GB, V size: 1.59 GB`
for a 4096-token probe — two independent allocations.

**Is the halving actually available?** Yes, and no framework has taken it.
K and V are not equal — they diverge after the norms:

```
V = n                       where n = k_proj(x) / rms(k_proj(x))
K = RoPE_pos(n * w_k)       w_k = k_norm weight
```

(`v_norm` has no learned weight, `k_norm` does; both share the same `rms`.)
So aliasing the two is wrong — but reconstructing K from V needs **no
reduction**, only a broadcast multiply by a `[512]` vector and a rotation at the
token's own position. Both are elementwise and both already sit in every
attention kernel's K-tile load path. That is the genuine 2x on the dominant
cache term (`80 -> 40 KiB/token`), and it is a kernel project, not a config
change. Verify `RoPE(V * w_k) == K_reference` on real layer-5 weights before
anyone touches a kernel.

Caveat worth stating: this trades bandwidth for ALU on the *read* path. It
should win on bandwidth-bound decode and may not on prefill. Measure across
concurrency with period-blocked means; do not report a single number.

### 2.2 `global_head_dim` 512 vs `head_dim` 256 — model side correct, cache side BROKEN

Model side is right (`gemma.py:225-245`): per-layer `head_dim`, per-layer
`num_kv_heads`, per-layer RoPE, and `group_id=layer_type` passed to
`PagedAttention` so the two layer types land in separate paged-cache groups.

**Cache side has no per-layer head_dim at all.** `MHATokenToKVPool` takes a
single scalar `head_dim` (`kv_cache/mha.py:95`) and allocates every layer as
`(size, head_num, head_dim)` (`:267`); `_layer_row_view` reshapes to
`(-1, heads_l, self.head_dim)` (`:557`) — the head *count* can vary per layer,
the head *dim* cannot.

There is a heterogeneous-KV mechanism (`layer_kv_head_counts` /
`kv_alloc_head_count` / `group_page_sizes`, added for Inkling, issue #647) that
would fit Gemma 4 almost exactly. **Nothing populates it for Gemma 4.** A grep
for `gemma` across the entire legacy runtime returns only `GemmaRMSNorm` used
by *other* models (`qwen3_5_nextn`, `minimax_m3`) — there is zero Gemma 4
plumbing outside `models/gemma.py`. No config module, no attention-registry
hook, no hetero-KV publisher.

Consequence, measured in §4: `layer_kv_head_counts` is `None`, every layer gets
a `16 x 256` row, and the full layers write half a row.

**What would fix it.** The Inkling mechanism generalizes if the two geometries
are normalized to a common head dim of 256 — `4 x 512` is the same element
count as `8 x 256`:

| layer type | real | elems/token | normalized heads @256 | page tokens |
|---|---|---|---|---|
| sliding | 16 x 256 | 4096 | 16 | `page_size` |
| full | 4 x 512 | 2048 | **8** | `2 x page_size` |

So `layer_kv_head_counts = (16,16,16,16,16,8) * 10`, `kv_alloc_head_count = 16`,
`group_page_sizes = {"full_attention": 2*page_size}` — and all the existing byte
accounting, page-geometry solving and slab guards keep working untouched. The
full layers get 2x tokens per slot, which is what you want since they are the
unbounded ones.

The one genuinely new thing needed is a per-layer head_dim in the *view*:
`_layer_row_view` must be able to return `(-1, 4, 512)` instead of
`(-1, 8, 256)`. Same bytes, contiguous, pure reshape — the helper just cannot
express it today. ~6 additive lines, backward-compatible (no-op when the new
tuple is `None`).

Second-order blocker: `MHAAttnBackend.__init__` binds one `head_dim` for the
whole backend (`backends/mha.py:140,156`, `self.plan = partial(mha_plan,
head_dim=self.head_dim, ...)`). Two head dims need either one backend instance
per cache group or a per-call head_dim. vLLM and TRT-LLM both took the
"one attention wrapper per distinct plan geometry" route; TRT-LLM additionally
had to *defer* `plan()` because multiple wrappers share one workspace buffer.

### 2.3 `final_logit_softcapping: 30.0` — HANDLED, confirmed reachable

`gemma.py:549-553` passes the **text** sub-config to `BaseCausalLM`, and
`layers/logits_processor.py:234` reads
`getattr(self.config, "final_logit_softcapping", None)`, applying it at `:647`
via `fused_softcap_generic` (stable tanh form, no overflow at cap=30). Passing
the outer `Gemma4Config` instead would have silently disabled it — the code
gets this right and comments why.

Correct omission: Gemma 4 has **no** attention-logit softcapping (unlike
Gemma 2). `gemma.py:249` sets `self.scaling = 1.0` with the right reason
("no `query_pre_attn_scalar`; q/k are unit-RMS after qk-norm"). All three
references agree.

### 2.4 `sliding_window: 1024`, 5:1 alternation — HANDLED

`gemma.py:300-309` passes `sliding_window_size` and `group_id=layer_type` per
layer. `paged_cache_spec.group_specs_from_layer_types` already collapses layers
into one group per distinct `(retention, window)` and Gemma 4's config labels
are literally the strings it expects. Zero new code, as the briefing predicted.

### 2.5 Things the briefing did not ask about, which the file gets right

- **Plain RMSNorm, not Gemma's `(1 + w)`.** Gemma 4 dropped the Gemma 1/2/3
  convention. `gemma.py:35-37` calls this out explicitly and uses the standard
  `RMSNorm`. Verified against the checkpoint: `q_norm.weight ≈ 1.02`,
  `input_layernorm.weight` mean ≈ 4.9 — direct multipliers, not offsets. Using
  the repo's existing `GemmaRMSNorm` here would have been silently wrong; the
  file correctly does not.
- **`layer_scalar`.** A trained per-layer output scalar, loaded as a parameter
  and applied at `:447`. Absent from every other model in the package; omitting
  it degrades output without erroring.
- **Proportional RoPE.** `build_rope_inv_freq` (`:89-131`) matches HF exactly,
  including the two details all three references flag as the silent-accuracy
  trap: the frequency exponent divides by the **full** `head_dim` (512), not the
  rotary dim (128); and the zero-padded tail keeps the rotation full-width so
  `rotate_half` pairs `dim[i]` with `dim[i+256]`, not `dim[i+64]`.
- **Embedding normalizer under CUDA graph capture.** The 28-line uncommitted
  diff fixes a real crash: `embeds.new_tensor(hidden**0.5)` built a CPU scalar
  and copied it to device every forward, which is illegal during capture
  (`Cannot copy between CPU and CUDA tensors during CUDA graph capture`) and
  "took gemma-4-31B down at startup before it served a request". Now cached per
  dtype as a Python float. This is evidence the model was run far enough to hit
  capture — but capture precedes any correctness check, so it is not evidence
  the KV path works.

---

## 3. Where the existing implementation is weaker than vLLM 0.26

| Axis | vLLM 0.26 | existing `gemma.py` | verdict |
|---|---|---|---|
| k_eq_v weights/compute | duplicates `k_proj` into a `v_proj` slot, runs the GEMM twice | no `v_proj` built, reuses the K activation | **`gemma.py` better** |
| k_eq_v cache capacity | stores both | stores both | tie (both leave 2x on the table) |
| Per-layer head dim, model side | per-layer `Attention` objects | per-layer, equivalent | tie |
| Per-layer head dim, **cache side** | vLLM's KV cache is per-layer-spec by construction; works | single global head_dim; **silently corrupts** | **vLLM better — this is the blocker** |
| MoE (`26B-A4B`) | full: `Gemma4Router` + `FusedMoE`, fp32-softmax→top-8→renormalize→`per_expert_scale`, dense MLP in parallel, fused Triton routing kernel | **not implemented** (documented TODO at `:380-386`) | **vLLM better** |
| Vision tower | full (`gemma4_mm.py`, 1718 lines) | not implemented, weights skipped | **vLLM better** |
| Bidirectional vision mask | `mm_prefix_clamp_sliding_window` on sliding layers | n/a | **vLLM better** |
| KV sharing (`num_kv_shared_layers`) | implemented | not implemented, not guarded | vLLM better (moot: 0 in both target checkpoints) |
| Unknown-weight handling | `AutoWeightsLoader` reports | `if name not in params_dict: continue` (`:595-596`) — **silently drops** | **vLLM better** |
| Fused qk-norm | — | uses generic `RMSNorm` per head, plus a `.reshape(-1, head_dim)` round trip per forward | vLLM slightly better |

The `load_weights` silent-skip is worth its own line: pointed at the 26B MoE
checkpoint, this implementation would load the dense subset, silently discard
every expert/router tensor, and serve fluent garbage. It should raise.

Also note vLLM gates `use_k_eq_v` on `is_full_attention` and *then* picks
`num_global_key_value_heads`, i.e. it treats the head-count change as a
consequence of k_eq_v. `gemma.py:236-240` does the same. Both are
observationally correct for these checkpoints, but the head count is really a
property of the `k_proj` shape and would be independent if a checkpoint ever
shipped full-attention layers with a real `v_proj`.

---

## 4. The blocker, measured

Built the real `MHATokenToKVPool` at gemma-4-31B text parameters
(60 layers, 5:1 alternation, `head_num=16`, `head_dim=256`, `page_size=64`) and
exercised the store path for one layer of each type.

```
pool built. k_buffer[0] shape: (4160, 16, 256)
k_buffer[5] (full layer) shape: (4160, 16, 256)
layer_kv_head_counts: None
row view layer 0: (4160, 16, 256)
row view layer 5: (4160, 16, 256)
```

`layer_kv_head_counts` is `None` — the heterogeneous mechanism is not engaged.
Layer 5 (full attention) gets the same `16 x 256` = 4096-elem row as a sliding
layer, but produces K of `[T, 4, 512]` = 2048 elems/token.

Store path:

```
layer 0 (sliding, 16x256): STORE OK
layer 5 (full,    4x512):  STORE OK   <-- no error raised
```

Read-back of what actually landed:

```
row elems/token : 4096   written elems/token: 2048
first 2048 match source : True
tail 2048 nonzero count  : 0 of 16384
```

**The K tensor lands in the leading half of each row and the trailing half
stays zero.** Nothing raises. Half the allocated full-layer cache is dead, and
any read that views the row at the layer's real `4 x 512` geometry straddles two
tokens' worth of storage.

This is the worst failure mode available: not a crash, not a shape error — a
silent wrong answer. It also explains why the model could get as far as CUDA
graph capture (the §2.5 normalizer crash) without anyone noticing the KV path
was wrong.

---

## 5. Which attention backend runs head_dim 512 on sm_120

Tested on GPU 3, not assumed. `tokenspeed:sm120-dev`, torch 2.11.0+cu130,
flashinfer 0.6.15.post1, triton 3.6.0, sm_120, 101376 B smem/SM.

### 5.1 FlashInfer direct — works, and is numerically correct

| backend | hd=256 | hd=512 |
|---|---|---|
| `torch.nn.functional.scaled_dot_product_attention` | PASS | PASS |
| `flashinfer.BatchPrefillWithPagedKVCacheWrapper` | PASS | **PASS** |
| `flashinfer.BatchDecodeWithPagedKVCacheWrapper` | PASS | **PASS** |
| `flashinfer.BatchPrefillWithRaggedKVCacheWrapper` | PASS | **PASS** |
| `flash_attn` | namespace stub, exports nothing (`__file__ is None`) | — |

Verified against an fp32 reference, not just "returned a tensor":

```
ragged  HD=512 HKV=4  win=None : rel err 0.00444   OK
ragged  HD=256 HKV=16 win=1024 : rel err 0.00426   OK   (sliding path)
ragged  HD=256 HKV=16 win=None : rel err 0.00382   OK
decode  HD=512 paged            : rel err 0.00000   OK
```

This contradicts TensorRT-LLM's own test suite, which claims a "FlashInfer
head_dim=512 limitation on B200" (`test_gemma4_e2e_dummy.py:17-19`) and patches
configs down to 128/256 to avoid it. That note is stale or B200-specific.
Measured beats documented.

### 5.2 `tokenspeed_kernel` — two separate problems, and the worse one is at 256

| entry point | solution | hd=128 | hd=256 | hd=512 |
|---|---|---|---|---|
| `mha_decode_with_kvcache` | default | — | PASS | **FAIL** (hard gate) |
| `mha_decode_with_kvcache` | `triton` | — | PASS | **PASS** |
| `mha_decode_with_kvcache` | `flashinfer` | — | PASS | **FAIL** (hard gate) |
| `mha_prefill` | `triton` | PASS | **FAIL** | **FAIL** |
| `mha_prefill` | `flashinfer`/`fa3`/`fa4` | — | not registered | not registered |

**(a) A closed head_dim allowlist.**

```
ValueError: Invalid head_dim: 512, must be divisible by 16 and in range [16, 256]
```

A closed range check, not a capability probe — the same class of bug as the
sm_120 capability tables found previously. FlashInfer itself handles 512 fine
(§5.1); the wrapper refuses to ask. The `triton` solution bypasses it and works.
Note this gate is **not** present in the legacy tree's own kernel package —
it comes from the newer `tokenspeed_kernel` installed in the image.

**(b) Triton prefill over-subscribes shared memory — at hd=256 too.**

```
hd=128: PASS
hd=256: OutOfResources: shared memory required 114688, limit 101376
hd=512: OutOfResources: shared memory required 102400, limit 101376
```

**This reorders the risk register. The blocker is not 512 — it is 256.**
Gemma 4's 50 *sliding* layers are hd=256, so prefill is blocked on sm_120
regardless of anything to do with the full layers, and this is a
model-independent sm_120 gap that any hd=256 model hits. hd=512 misses the
limit by only 1024 bytes; a modest `BLOCK_M`/`BLOCK_N` reduction clears both.
That fix lives in `ops/attention/triton/mha_prefill.py`, owned by another
agent — **hand-off item**. (Its current uncommitted diff is an unrelated
early-exit optimization for mixed batches, not a shared-memory fix.)

### 5.3 What already exists in the legacy tree

`tokenspeed-kernel/.../ops/attention/flashinfer/fa2.py` (532 lines, uncommitted)
registers FlashInfer **FA2** `mha_extend_with_kvcache` and
`mha_decode_with_kvcache` specifically for sm_120, with CUDA-graph plan/run
wrappers, explicitly because "the trtllm kernels are gated out on sm_120". So a
FlashInfer path into the MHA backend already exists on this hardware — it is
extend+decode only, not ragged prefill. Whether it carries the §5.2(a) head_dim
gate needs checking before it is counted on for the full layers.

### 5.4 Recommendation

Bring up on **FlashInfer prefill + Triton decode**; file (a) and (b) against
their owners. Neither blocks correctness validation, which should be done at
the tensor level anyway (§7).

---

## 6. What is missing, in priority order

1. **Per-layer head_dim in the KV pool** (§2.2, §4). *This is the only thing
   standing between the existing code and a correct 31B run.* Two changes:
   a Gemma 4 hetero-KV publisher (mirroring `_apply_inkling_hetero_kv`) and a
   per-layer head dim in `_layer_row_view`.
2. **Per-group attention backend instances** (§2.2) — one `head_dim` per
   backend today.
3. **A guard that fails loudly** on the geometry the pool cannot express, so
   this class of bug errors instead of corrupting.
4. **`load_weights` must raise on unknown tensors**, not `continue` — currently
   a MoE checkpoint loads its dense subset and serves garbage.
5. **The MoE variant** (`26B-A4B`): router (weightless RMSNorm → `x hidden^-0.5`
   → learned per-dim scale → *replicated* projection), fp32 softmax over all 128
   → top-8 → renormalize → `x per_expert_scale`, GEGLU experts at
   `moe_intermediate_size=704`, dense MLP running **in parallel** and summed via
   two extra post-norms. Router reads the raw residual; experts read
   `pre_feedforward_layernorm_2` of it.
6. **The k_eq_v cache halving** (§2.1) — 2x on the dominant cache term.
7. **Vision tower** and the bidirectional vision mask (sliding layers only).
8. **TP > 1**: `num_global_key_value_heads` is 4 (31B) / **2** (26B); at TP=8 the
   26B's full layers have fewer KV heads than ranks. TRT-LLM sidesteps the whole
   area by asserting `pp_size == 1` and `cp_size == 1`.
9. **Guards for unimplemented config fields** — `num_kv_shared_layers`,
   `hidden_size_per_layer_input`, `use_double_wide_mlp` are all silently ignored
   today. All are 0/false in these two checkpoints, so this is latent.

---

## 7. Verification strategy

Oracle: HF `transformers` (eager/sdpa) on the same checkpoint, or vLLM 0.26.
Reproducing the reference's bad output is a PASS; matching expectation is not
the test.

Cheapest first — the first three need no full model load, which matters because
GPU 3 is shared and the 31B will not fit alongside other tenants:

1. **RoPE tensor equality** — `build_rope_inv_freq("full_attention", cfg)`
   against HF `_compute_proportional_rope_parameters` at
   `head_dim=512, prf=0.25, theta=1e6`.
2. **Single-layer forward** — layer 0 (sliding, has `v_proj`) and layer 5 (full,
   k_eq_v) against HF, same hidden states in. Catches the norm convention,
   `layer_scalar`, and the k_eq_v ordering.
3. **`K = RoPE(V * w_k)` identity** on real layer-5 weights — gates the §2.1
   halving work before anyone touches a kernel.
4. **KV round-trip at both geometries** — the §4 probe, as a regression test.
5. **Text-only generation parity** through the chat template, greedy. The real
   gate, and only meaningful after item 1 of §6 lands.

---

## 8. Status of code written this session

Per the redirect, **no new implementation was added to the upstream tree**;
`/workspace/roy/tokenspeed/python/tokenspeed/runtime/{models,configs}/` contain
no gemma4 files. Work started before the redirect is retained, clearly labelled,
as **proposals only** in `gemma4-proposals/`:

- `PROPOSAL_gemma4_config.py.txt` — the genuinely additive part: derivations for
  per-layer geometry and, in particular, `normalized_kv_head_counts` /
  `group_page_tokens`, which express the 256/512 split in the terms the existing
  Inkling hetero-KV mechanism already understands (§2.2). This is the piece the
  existing implementation is missing and is worth lifting.
- `PROPOSAL_gemma4_model.py.txt` — a from-scratch model file. **Largely
  redundant**: the existing `gemma.py` is equal or better on every axis except
  MoE. Useful only as a diff source for (a) the MoE block, (b) `load_weights`
  raising on unknown tensors, (c) explicit config guards. Do not adopt wholesale.
