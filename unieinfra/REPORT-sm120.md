# sm_120 on UnieInfra — status report

Qwen3.5-27B bf16, one RTX PRO 6000 Blackwell (sm_120), C=32 / ILEN=4096 /
OLEN=256. All arms 600 s fixed duration with early convergence disabled, scored
on the last 200 s, run back to back in one session on the same card.

## Headline

| | tok/s | vs vLLM |
|---|---|---|
| Start of this work | **did not start at all** | — |
| Upstream tokenspeed, once it started | 208.0 | −16.2% |
| **+ fp8 KV cache** | **245.8** | **−0.9%** |
| vLLM 0.26 | 248.1 | — |

Essentially parity, from a position of not running.

The fp8-KV arm oscillates between two levels rather than sitting at its mean:

```
199 212 204 212 265 273 273 273 273 273 272 247 211 211 212 212 273 273 273 273
```

Bimodal, 211 against 273, 25.1% window spread. **The high mode already beats
vLLM by 10%.** Closing the remaining gap is a matter of holding it there, not of
raising a mean — and the tail mean is phase-dependent, which is why 245.8 and a
later 246.4 should not be read as different numbers.

## What made it start

Upstream does not run Qwen3.5 on this card. Three bugs, each found by launching
and reading the traceback, all of the same shape — **the hardware is capable and
a declaration excludes it**:

| what | why | commit |
|---|---|---|
| XQA decode chosen for a page size it cannot compile | traits declared sliding window, sinks, logit cap, LSE — but not `page_size`, which callers already pass | `48a80ebb` |
| TRT-LLM-gen FMHA chosen for an arch it was not built for | `min_arch_version` alone is an open-ended comparison every newer part satisfies | `48a80ebb` |
| Triton prefill tile does not fit in shared memory | the tile ladder branched on arch family; sm_120 is `is_hopper_plus` by version and carries sm_89's ~100 KB | `73b27362` |

The third had a second layer worth recording: the obvious fix — ask the device
for its budget — did nothing, because `PlatformInfo.max_shared_memory_per_sm`
read `getattr(props, "max_shared_memory_per_block", 0)` and **torch has no
attribute by that name**, so the field was 0 on every CUDA device ever. Nothing
else read it, which is why it was never noticed (`228886f0`).

## What made it fast

**fp8 KV cache, +19.0%** (206.5 → 245.8, verified by reversing arm order).

Budgeted as a kernel port; it was two declarations. Both Triton kernels already
load a quantized cache and convert up to the query dtype. What made fp8 KV
unreachable was that `format_signatures` puts one dtype on every role, so
declaring `{fp16, bf16}` across q/k_cache/v_cache meant selection would only ever
offer a cache matching the query — and the useful combination, a high-precision
query over an fp8 cache, had no signature to match (`996b05f4`, `b8a392f8`).

## What was tried and rejected

Recording these because a measured negative is worth as much as a win and costs
the same to produce twice.

**`--mamba-ssm-dtype bfloat16`** — halves the state row, which pulls the page
size back from 1600 to 832 and doubles the page count from 149 to 294. Both
configurations fit 49 concurrent requests, one via coarser pages and one via more
of them. Measured 246.4 against 245.8: the same number. It changes the precision
of the recurrent state for nothing.

**Shrinking the attention page P** (the original capacity plan) — would have
grown state memory 13x. `flat_state_slabs` allocates one full conv+ssm state row
**per page id**, so P=64 means ~1937 rows × 3.2 MB = 6.2 GB against 478 MB at
P=832. The inflation bounds state memory; the 14.7% KV rounding waste is the
deliberate price. The file says so eleven lines from the code that would have
been changed.

**w4a16** — the TurboMind port works (2 → 222 registered kernels, 28/28 numerics,
2.3–3.3x on decode GEMMs, `legacy/patches/`), but there is no path into this
engine to reuse: `CompressedTensorsWNA16` references ten names it never imports,
`marlin.so` exports only a repacker for a GEMM that is not in the tree, and
`BASE_QUANTIZATION_METHODS` has neither gptq nor awq. Quantization is also never
automatic — TurboMind repacks the layout at load time, but the int4 weights must
exist first, and uncalibrated round-to-nearest measured **+7.2% perplexity**
(17.48 → 18.73 on 17k tokens of natural prose), which is the floor of the range,
not a deployment number.

## What is unfinished, and why

**Mixed prefill+decode batching.** The remaining ~10% and the only identified
cause of the oscillation. Capacity is ruled out — 49 requests fit against C=32 —
leaving the strict prefill/decode alternation, which is exactly what mixed
batching addresses.

The full-attention half is done (`64e632d8`). The recurrent half is not, and the
attempt is the most useful thing in this report:

Five serve-crash-read cycles, each fixing the previous failure and finding the
next — query-length arrays, then tensor alignment on the kernel's output buffer,
then an illegal address in the state gather. All three were symptoms of one
placement error: **the split was written in the attention backend, which does not
own the tensors.**

Surveying the other engines answered it in one file read each:

| engine | where the split lives | order | how the halves are addressed |
|---|---|---|---|
| vLLM | mixer (model layer) | decode first | `torch.split` on the token dim |
| TRT-LLM | mixer | **prefill first** | `torch.split`, output preallocated and split too |
| SGLang | mixer | **prefill first** | `torch.split` |
| TurboMind | layer plan | decode first | **offsets, no slicing at all** |

SGLang's `hybrid_linear_attn_backend.py` — which ours shares a lineage with —
has the same gap ours does and puts its mixed handling one level down. That is
the confirmation that the level was wrong. TurboMind never materialises a slice:
two launch plans carrying offsets into shared buffers, which removes both the
concat and the alignment trap that cost two of the five cycles.

The machinery is landed and disabled (`146c4175`); `supports_mixed_batch` stays
False and a startup guard refuses the flag rather than letting a step fail
mid-serve (`c81fa7fa`). The design to build is: split in the layer, prefill-first
to match TokenSpeed's own convention, and pass offsets rather than slices.

## Repository

The work also required fixing how it is kept. UnieInfra had been forked from
tokenspeed by **copying the tree**, so it shared no git ancestor with upstream:
rebase was impossible and divergence was invisible until someone diffed two
working trees by hand. Re-forked with a shared ancestor (`1dd493da`); `git
rebase upstream/main` now works and is verified clean against a synthetic
upstream commit touching a file we also changed.

Measured against the real fork point, UnieAI's actual divergence was 3381 lines
in `tokenspeed-kernel`, 3003 in the runtime, and **zero** in
`tokenspeed-scheduler` — where a naive diff had shown 165 and 32 files. Most of
the apparent divergence was upstream drift that no tool could separate from the
real changes.

Two results in the old repository turned out to be **unreproducible**: the code
behind E45's `r = P/N` capacity win and behind its mixed-batch measurement exists
in no commit, no diff, and not in the preserved working tree. Both were measured
against images built by copying files out of a working tree.

## Standing count

`git log --oneline upstream/main..main` — every UnieAI commit.
`git diff --stat upstream/main..main -- python tokenspeed-kernel` — every
upstream line touched. Both were impossible before the re-fork and are one
command now.
