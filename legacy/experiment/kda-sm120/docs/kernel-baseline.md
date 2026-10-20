# Kernel baseline — Qwen3.5-27B geometry on sm_120

Baseline sweep of the operators the target model actually hits, measured with
[`eval/kernel_bench.py`](../../unieai-kda/eval/kernel_bench.py). This is the
reference point every kernel port on this platform should be compared against,
and the measurement contract those ports should reuse rather than reinvent.

Raw results: [`kernel-baseline.json`](kernel-baseline.json).

---

## 1. Conditions

| | |
|---|---|
| device | NVIDIA RTX PRO 6000 Blackwell Server Edition, `sm_120`, 188 SMs |
| shared memory / block (opt-in) | **101376 B** (`shared_memory_per_block_optin`) |
| L2 | 128 MiB |
| memory | 94.97 GiB, 90.8 GiB free at start |
| image | `tokenspeed:sm120-dev` — torch 2.11.0+cu130, CUDA 13.0, flashinfer 0.6.15.post1, triton 3.6.0 |
| `tokenspeed_kernel` | `/usr/local/lib/python3.12/dist-packages/tokenspeed_kernel` (the image's copy, **not** the `/workspace/roy/tokenspeed` tree) |
| model geometry | Qwen3.5-27B: 24 q heads, 4 kv heads, head_dim 256, page_size 832, hidden 5120, intermediate 17408, bf16 |
| run | 2026-08-02 16:45:06–16:45:35 UTC, GPU 3, 3 repeats per case |

### Card calibration — the run that produced this table had the card to itself

GPU 3 is shared with other agents, so absolute numbers are only meaningful with
a calibration anchor attached. The harness runs a bf16 8192³ GEMM before and
after the sweep and divides by the free-card figure from
[log.md E22](log.md) (427.1 TFLOP/s, measured when the card was confirmed idle):

| | before | after |
|---|---|---|
| anchor peak | 437.1 TFLOP/s | 437.6 TFLOP/s |
| `card_share` vs free-card reference | **1.02** | **1.02** |
| power before our work | 88.0 W | 142.1 W |

`card_share` 1.02 and 88 W of idle draw mean we had the whole card, and the
before/after ratio of 1.001 means no neighbour arrived mid-sweep. **Earlier
attempts at this same sweep ran at `card_share` 0.47–0.62 and are discarded** —
see §5, this was the single largest measurement hazard found.

**Roofline anchors** used to interpret the numbers below:

* bf16 dense GEMM: **437 TFLOP/s** (measured, this run)
* device-to-device clone bandwidth: **1451 GB/s** (log.md E22)

---

## 2. Headline: three of the six operators have no working fast path

`Admitted` counts kernels that pass capability + format-signature + trait
filtering — i.e. what selection would consider. `Works` counts those that
actually return without raising at this geometry. **The gap between the two
columns is the finding.**

| operator | registered | admitted | works | best working kernel | band |
|---|---|---|---|---|---|
| `attention/mha_prefill` | 1 | 1 | **0** | — | — |
| `attention/mha_extend_with_kvcache` | 2 | 2 | **0** | — | — |
| `attention/mha_decode_with_kvcache` | 2 | 2 | 1 | `triton_mha_decode_with_kvcache_cached` | PORTABLE (4) |
| `gemm/mm` (bf16) | 10 | 1 | 1 | `torch_mm` | PORTABLE (7), `solution=reference` |
| `gemm/bmm` (bf16) | 3 | 1 | 1 | `torch_bmm` | PORTABLE (7), `solution=reference` |
| `gemm/mm` (mxfp8 128×128) | 10 | 4 | 4 (1 wrong) | `flashinfer_mm_fp8_blockscale` | SPECIALIZED (15) |
| `quantization/fp8` | 1 | 1 | 1 | `triton_quantize_fp8` | PORTABLE (4) |

Two whole operators on the prefill/extend path have **zero** working kernels at
this model's geometry, and both bf16 GEMM entry points run on a kernel whose
registered `solution` is literally `"reference"`. That is the shape of the gap
this effort is meant to close.

---

## 3. Why each kernel fails, separated by cause

Every failure below was reproduced across shapes to separate the real gate from
the incidental one. This matters: the obvious suspect (page_size 832, which the
GDN state row forces) turns out to be responsible for exactly one of them.

### 3.1 `head_dim 256` exceeds sm_120's shared memory — the Triton MHA blocker

```
OutOfResources: out of resource: shared memory,
Required: 114688, Hardware limit: 101376
```

Sweeping head_dim with everything else fixed:

| head_dim | `triton_mha_prefill` | `triton_mha_extend_with_kvcache` |
|---|---|---|
| 64 | ok | ok |
| 128 | ok | ok |
| 192 | ok | — |
| **256** | **OutOfResources** | **OutOfResources** |

The requirement is 112 KiB against a 99 KiB opt-in limit. It is **not** related
to page_size: extend fails identically at page_size 128 and 832. This is the
sm_100→sm_120 shared-memory cliff the port-kernel skill §3 warns about (227 KB
→ 100 KB), and it is the same root cause already recorded for GDN in the last
log.md entry, where a third of vLLM's `chunk_h` autotune space was unreachable
for exactly this reason. The fix has a precedent in-repo: **branch tile size on
the driver-reported `shared_memory_per_block_optin`, never on the arch name.**

Note `triton_mha_decode_with_kvcache_cached` is unaffected at head_dim 256 — the
decode path uses different tiling and stays under the limit.

### 3.2 flashinfer trtllm-gen extend is unsupported on sm_120, at every shape

```
RuntimeError: Error in function 'TllmGenFmhaRunner' at fmhaRunner.cuh:37:
Unsupported architecture
```

Tested at head_dim 128 and 256, page_size 128 and 832 — fails in all four. This
is an architecture gate, not a shape gate, and no change to the model's paging
will unlock it. It is admitted by selection at SPECIALIZED priority 12 because
its `CapabilityRequirement` is a version-monotonic lower bound that sm_120
satisfies while the kernel was only ever built for sm_100.

### 3.3 flashinfer decode is blocked **only** by page_size 832 — and is otherwise fine

```
ValueError: Invalid page_size: 832, only 16, 32, 64, 128 are supported
```

| head_dim | page_size | `flashinfer_trtllm_mha_decode` | `triton_..._decode` |
|---|---|---|---|
| 128 | 128 | **ok** | ok |
| 128 | 832 | rejected | ok |
| 256 | **128** | **ok** | ok |
| 256 | 832 | rejected | ok |

This is the most actionable result in the sweep. The SPECIALIZED-band flashinfer
decode kernel **works on sm_120 at head_dim 256**; the only thing standing
between the model and it is the page size, which is 832 because the GDN
recurrent-state row forces the attention block size. Anything that decouples the
attention page size from the state row — a separate attention pool, or a
page_size that divides 128 — makes a priority-12 kernel reachable where a
priority-4 fallback runs today.

### 3.4 `deep_gemm_mm_fp8_blockscale` silently returns all-NaN — and is the fastest candidate

The most dangerous result here. At the qkv shape it is the fastest of four
kernels (726 TFLOP/s), raises nothing, and every one of its 33,554,432 output
elements is NaN.

Isolated by holding one operand's scales at 1.0:

| A_scales | B_scales | deep_gemm | flashinfer | triton |
|---|---|---|---|---|
| ones | ones | **exact** | exact | exact |
| random | ones | **all NaN** | rel_l2 5.6e-5 | rel_l2 5.7e-5 |
| ones | random | **all NaN** | rel_l2 5.9e-5 | rel_l2 5.9e-5 |
| random | random | **all NaN** | rel_l2 5.7e-5 | rel_l2 5.7e-5 |

Reproduced at 4096×8192×5120 and 1024×2048×1024. The inputs are correct by the
in-tree contract — the same tensors are consumed correctly by three other
kernels, and the layout matches what `torch_mm_fp8_blockscale` asserts
(`A_scales [M, K/128]`, `B_scales [N/128, K/128]`, fp32).

Three things make this worse than an ordinary bug:

1. **It is invisible to an all-ones smoke test.** Scales of 1.0 are exactly what
   a quick correctness check writes, and they are the one case that works.
2. **It never raises.** A harness that only records "works / raises" scores it
   as a pass.
3. **It wins on speed.** `SelectionStrategy.AUTOTUNE` benchmarks candidates and
   picks the fastest; it would select this kernel. Its declared priority (14,
   SPECIALIZED+2) is already second only to flashinfer's 15.

Same root cause class as §3.2: `min_arch_version=ArchVersion(9, 0)` with no
upper bound, so a kernel qualified on sm_90/sm_100 is inherited by sm_120.

---

## 4. Measured baseline

All numbers: median of 3 independent repeats, each the median of 15 CUDA-event
samples. `TFLOP/s` and `GB/s` are both computed; the **primary** unit for each
row is the one in bold, chosen by which resource the shape is actually against.

### 4.1 Attention

| case | kernel | median | GB/s | TFLOP/s | rel_l2 | cosine |
|---|---|---|---|---|---|---|
| decode b32 ctx4096 | `triton_..._decode_..._cached` | 0.3756 ms | **1431.6** | 4.79 | 2.26e-3 | 0.999997 |
| decode b64 ctx2048 | `triton_..._decode_..._cached` | 0.3755 ms | **1433.8** | 4.79 | 2.25e-3 | 0.999997 |
| prefill b4 s1024 | — | \- | \- | \- | \- | \- |
| prefill b1 s4096 | — | \- | \- | \- | \- | \- |
| extend b4 q128 ctx1024 | — | \- | \- | \- | \- | \- |
| extend b8 q32 ctx4096 | — | \- | \- | \- | \- | \- |

**The decode kernel is at the memory roofline.** 1432 GB/s against a measured
1451 GB/s clone bandwidth is 98.7% of achievable. GB/s counts the KV bytes the
kernel must read at least once (`2 × Σ cache_seqlens × num_kv_heads × head_dim ×
2 B`). There is essentially nothing left on this shape for a port to win —
a faster decode has to come from reading *fewer* bytes (fp8 KV, which
[fp8-kv-works-on-sm120-via-triton] already showed is viable here), not from a
better kernel. The two shapes carry identical total KV bytes and land within
0.2% of each other, which is a useful internal consistency check on the byte
accounting.

bf16 vs an fp32 reference at rel_l2 2.3e-3 / cosine 0.999997 is normal bf16
rounding and comfortably inside the skill's cosine > 0.9999 bar.

### 4.2 GEMM — bf16 (`torch_mm`, the only admitted kernel)

| layer | M | N | K | median | **TFLOP/s** | % of 437 anchor | rel_l2 |
|---|---|---|---|---|---|---|---|
| qkv_proj | 4096 | 8192 | 5120 | 0.8814 ms | **389.8** | 89% | 1.7e-4 |
| o_proj | 4096 | 5120 | 6144 | 0.6420 ms | **401.4** | 92% | 2.1e-3 |
| gate_up_proj | 4096 | 34816 | 5120 | 3.6957 ms | **395.1** | 90% | 1.7e-4 |
| down_proj | 4096 | 5120 | 17408 | 1.7860 ms | **408.8** | 94% | 2.1e-3 |
| qkv_proj | 32 | 8192 | 5120 | 0.0311 ms | 86.4 ⚠ | — | 2.6e-3 |
| o_proj | 32 | 5120 | 6144 | 0.0232 ms | 86.9 ⚠ | — | 2.6e-3 |
| gate_up_proj | 32 | 34816 | 5120 | 0.2565 ms | 44.5 | — | 1.7e-4 |
| down_proj | 32 | 5120 | 17408 | 0.1343 ms | 42.5 | — | 2.6e-3 |

Prefill-shaped GEMM reaches 89–94% of the card's dense bf16 peak. **cuBLAS is
already close to the roofline and is not where a port should start.**

⚠ The two M=32 rows marked with a warning have working sets (81 MiB, 61 MiB)
that fit in the 128 MiB L2, and report 2716–2780 GB/s — above the card's
1451 GB/s DRAM bandwidth. Those numbers are **L2-fed and optimistic**: a real
forward pass streams each weight from DRAM once per layer, while a benchmark
loop re-reads it from L2. Treat decode-shaped GEMM figures here as upper bounds.
The M=32 rows whose weights exceed L2 (gate_up 340 MiB, down 170 MiB) do not
have this problem and read 42–45 TFLOP/s, i.e. weight-streaming bound as
expected.

### 4.3 GEMM — mxfp8 128×128 block-scaled (the only multi-candidate operator)

| kernel | band | median | **TFLOP/s** | rel_l2 | verdict |
|---|---|---|---|---|---|
| `deep_gemm_mm_fp8_blockscale` | SPECIALIZED (14) | 0.4732 ms | ~~726.2~~ | **NaN** | **WRONG — see §3.4** |
| `flashinfer_mm_fp8_blockscale` | SPECIALIZED (15) | 0.4760 ms | **721.9** | 5.8e-5 | best working |
| `triton_mm_fp8_blockscale` | PERFORMANT (11) | 0.6129 ms | **560.6** | 5.9e-5 | correct, 1.29× slower |
| `torch_mm_fp8_blockscale` | PORTABLE (6) | 7.1837 ms | **47.8** | 0 (self) | the designated reference |

fp8 gives **1.85×** over the best bf16 GEMM at the same shape (721.9 vs 389.8
TFLOP/s), which is the largest single lever visible in this sweep. Selection
already picks the right kernel here — flashinfer at priority 15 — and it happens
to also be the fastest *correct* one, so the NaN kernel does not currently win.
That is luck, not design: one point of priority, or an autotune-based policy,
flips it.

`triton_mm_fp8_blockscale` also emits `Config file not found ... Using default
W8A8 Block FP8 kernel config. Performance might be sub-optimal!` for this
device — there is no tuned config for
`N=8192,K=5120,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition`. Its
1.29× deficit is plausibly just an untuned table, which is a cheap thing to
check before porting anything.

### 4.4 BMM and quantization

| case | kernel | median | primary | note |
|---|---|---|---|---|
| bmm b24 ×1024×256×256 | `torch_bmm` | 0.0229 ms | **140.6 TFLOP/s** | 27 MiB working set, L2-resident ⚠ |
| bmm b4 ×4096×5120×5120 | `torch_bmm` | 2.0832 ms | **412.3 TFLOP/s** | not reproducible, ratio 2.19 — see §5 |
| quantize_fp8 4096×5120 | `triton_quantize_fp8` | 0.2173 ms | **289.5 GB/s** | 60 MiB, L2-resident ⚠ |
| quantize_fp8 scaled 4096×17408 | `triton_quantize_fp8` | 2.8093 ms | **76.1 GB/s** | 204 MiB, exceeds L2 |

**`quantize_fp8` is the clearest optimization target in the sweep.** At a working
set that exceeds L2 — the honest case — it reaches 76 GB/s against 1451 GB/s of
device bandwidth: **5.2% of roofline** on an operation that is nothing but a
strided read, a divide and a narrower write. Even the L2-resident case only
reaches 289 GB/s. Both are bit-exact against the reference (rel_l2 0.0), so this
is purely a performance gap, and it is on a PORTABLE-band kernel with no
competitor registered.

---

## 5. Measurement failure modes: guarded, and not

The point of this baseline is as much the contract as the numbers. Each guard
below is in `kernel_bench.py`, and most were added because they caught something
during this sweep.

### Guarded — and it fired here

| failure | guard | what it caught |
|---|---|---|
| **co-tenant on a shared card** | bf16 8192³ calibration anchor before *and* after the sweep, divided by the free-card reference | The decisive one. Three earlier sweeps ran at `card_share` 0.47–0.62 and produced GEMM numbers ~2× too low, all internally consistent and all stationary. The anchor read 203–265 TFLOP/s against E22's 427 and the host showed a 359 W neighbour on an otherwise-90 W card. Without this, this document would have quoted 186 TFLOP/s for qkv_proj and called cuBLAS slow. |
| **the calibration anchor itself measuring a cold card** | the anchor re-measures until its peak stops improving by >2%, and records the ramp (`ramped`, `ramp_tflop_s_per_round`) | An anchor that does not ramp reports an idle-but-cold card as a busy one. Measured: 20 warmup iterations (~30 ms) off an 88 W idle state read 203 TFLOP/s where the same card gave 437 seconds later. A guard that mis-fires this way is worse than none — it would have discarded good runs. |
| **NVML process list is blind in a container** | never used as evidence; power draw + the anchor used instead | `nvmlDeviceGetComputeRunningProcesses` returned **1** (ourselves) throughout, while another tenant held half the GPU — PID-namespace isolation. This is a *second* broken instrument on this host alongside E22's utilisation counter, and it fails the same way: it reads as "nothing else is running". |
| **`nvidia-smi` utilisation** | never read at all | E22: reports sm=100% on idle GPUs 2 and 3. No claim here rests on it. |
| **correctness judged against expectation** | every candidate compared to one *explicitly named* reference, printed in the table and stored in the JSON | Caught §3.4. Also note the references here are independent in-harness fp32 implementations, because for 4 of the 6 operators there is only ONE registered kernel and candidate-vs-candidate would be vacuous. |
| **cosine similarity hides a scale error** | `rel_l2` reported alongside and documented as the one to read first | A wrong `softmax_scale` or dropped `alpha` scores cosine 1.000000. |
| **a benchmark loop that lives in L2** | working set compared to L2 size; any GB/s above measured device bandwidth flagged | Fired on 5 cases. `mm_qkv_M32` reports 2716 GB/s on a 1451 GB/s card — real, and not what a forward pass sees. |
| **the timing loop length changes the answer** | inner-loop length snapped to a coarse ladder off a median-of-5 probe; flagged if it differs across repeats | A harness bug I introduced. Sizing the loop from a single noisy probe flipped `quantize_fp8` between inner=1 (5.09 ms) and inner=2 (2.79 ms) — an **1.8× swing** across repeats of an identical case, both stationary, clock pinned at 2422 MHz. |
| **stationary but not reproducible** | `--repeats N`: whole case re-run with fresh tensors, medians compared, `max/min > 1.10` refuses a single number | `gate_stationary` asks whether a series is still *moving*; a constant offset from a neighbour leaves it perfectly flat. `mha_decode_b32` measured 1433 GB/s and 799 GB/s in two processes, both with p10–p90 spread under 3%. |
| **still-ramping clocks** | warmup, then `kda.gates.gate_stationary` on the sample series; one automatic retry at 3× warmup; a still-moving series is reported as `null` with `rejected` naming the reason | `mm_o_M4096` was rejected at drift +4.3% in one run. |
| **a kernel that clobbers its inputs** | byte checksum of every input before and after each candidate; case rebuilt if mutated | Nothing fired, but a mutation would silently corrupt every later candidate's numerics and be attributed to the wrong kernel. |
| **"admissible" mistaken for "working"** | enumeration reported in stages (registered → capability → signature → traits), and every admitted candidate actually executed | This is the entire §2 table. 8 of 26 candidate runs raised. |
| **wrong unit** | each case declares its natural unit with a stated reason; both computed | A memory-bound op in TFLOP/s looks terrible and tells you nothing. |
| **plugin that failed to import** | import status recorded and printed loudly | Otherwise a plugin's absence reads as "the plugin kernel lost on merit". |

Two rows remained `NOT REPRODUCIBLE` in the final run and are **not** quoted as
single numbers above: `mm_qkv_M32` (inner-loop length differed, 89/55/89) and
`bmm_b4_m4096_n5120_k5120` (ratio 2.19). Both are recorded with their ranges in
the JSON.

### Not guarded — known gaps

1. **Triton autotune state is per-process and order-dependent.** Candidates run
   in one process in a fixed order; a kernel measured second may inherit a
   warmer allocator or a cached autotune choice. Warmup is per-candidate, but
   the caching allocator is not reset between them.
2. **L2 residency is flagged, not eliminated.** Flushing L2 between iterations
   would measure every call cold, which a real forward pass also is not. Both
   regimes are defensible; the harness pins one and says which. Any absolute
   figure on a case whose working set is near 128 MiB should be read as a band,
   not a point.
3. **No tolerance judgement for quantized paths.** The harness reports rel_l2,
   max abs diff and cosine for fp8/fp4; it does not know what is acceptable.
   That belongs to whoever owns the kernel.
4. **Kernel-level only.** Nothing here predicts serving throughput. Launch
   overhead, CUDA-graph capture and per-forward-pass planning (the
   `plan()`-per-layer trap in the skill §6) are invisible at this level and have
   dominated real results before.
5. **`max_seqlen_q > 1` decode is untested.** Speculative draft blocks take a
   different path through the decode kernel and are not covered by this sweep.
6. **Thermal drift across a long sweep** is recorded per candidate (clock, temp,
   power) but not corrected. This run stayed at 2422 MHz throughout.
7. **The image's `tokenspeed_kernel`, not the working tree.** If a port is
   developed in `/workspace/roy/tokenspeed` it is not what was measured here;
   use `--plugin` to load it into the same image.

---

## 6. What this says to a kernel port

In rough order of expected value:

1. **`quantization/fp8` at 5.2% of memory roofline** — a pure streaming op with
   one PORTABLE-band implementation and no competitor. Largest headroom, lowest
   risk, bit-exact reference already available.
2. **Unblock flashinfer decode by decoupling page_size from 832** (§3.3) — no
   kernel writing at all, and it promotes a priority-4 fallback to a priority-12
   specialized kernel. Note the decode kernel is already at 98.7% of the memory
   roofline, so the win here is in what the SPECIALIZED path enables (fp8 KV,
   CUDA-graph behaviour), not in bandwidth.
3. **`mha_prefill` / `mha_extend` at head_dim 256** (§3.1) — currently *nothing
   runs*. The fix is known and precedented in this repo: branch tile sizes on
   the reported shared-memory limit rather than the arch name. This is the only
   item on the list that is a correctness/availability gap rather than a
   performance one.
4. **File a defect for `deep_gemm_mm_fp8_blockscale` on sm_120** (§3.4) and
   consider a `max_arch_version` on its capability so it stops being inherited.
   No porting work, but it is a live footgun for any autotune-based selection.
5. **bf16 GEMM is not the place to start** — cuBLAS is at 89–94% of peak. The
   available win there is fp8 (1.85× measured), not a better bf16 kernel.

---

## 7. Development log — 開發日誌

### 目的 (Purpose)

Several agents are porting kernels into an out-of-tree plugin for sm_120 in
parallel. Left alone, each would invent its own timing loop and its own notion
of "correct", and the numbers would not be comparable — and, given this
project's history, some would be wrong in ways that survive review. The goal was
one shared instrument that makes a kernel claim on this platform *checkable*:
same shapes, same reference discipline, same refusal rules, one JSON schema.

### 方法 (Method) — what was gained by referencing and fusing existing work

The harness was not written from scratch; it is a fusion of three things that
already existed in this repo, plus the parts that only appear at kernel level.

* **Reused unchanged from `kda/gates.py`:** `gate_stationary`, imported rather
  than copied. The failure it blocks — quoting the mean of a series that is
  still moving — is identical at kernel and serving level, and a second copy
  would drift from the original. `kernel_bench.py` adds `sys.path` for it and
  degrades to a drift-only check if it is unavailable, reporting that it did so.
* **Reused in structure from `eval/steady_bench.py`:** the house pattern of
  *compute the diagnostic, then gate on it, then refuse* — the
  counter_agreement retraction happened because a check was computed, written to
  every result file, and never read. Hence `rejected` is a field that blanks the
  reported value, not a warning line.
* **Adapted from log.md E22:** the bf16 8192³ GEMM as a card-calibration probe.
  E22 used it once, by hand, to decide whether GPUs 2 and 3 were free. Making it
  an automatic before/after anchor turned a one-off diagnostic into a standing
  guard, and it is what caught the co-tenant contamination described below.
* **Adapted from the `port-kernel` skill:** §0's staged enumeration
  (`admissible ≠ working`) became the harness's four-stage candidate table and
  the requirement that every admitted candidate is actually executed; §3's
  "read the real shared-memory limit, never infer from the arch name" is what
  made the head_dim-256 result interpretable rather than just an error string;
  §5's "correctness against a reference, never against expectation" is why
  the reference is a mandatory, named field with no default.
* **New, because they only exist at kernel level:** CUDA-event timing with a
  deterministic inner loop; working-set-vs-L2 accounting; per-candidate input
  checksums; the cross-repeat reproducibility gate; and unit selection
  (TFLOP/s vs GB/s) as a declared property of the shape.

### 為什麼 (Why these choices)

Three of the guards exist specifically because the alternative was tried during
this work and produced a confident wrong answer:

* **Why a calibration anchor rather than the process list.** The obvious way to
  detect a neighbour is to ask NVML what else is running. Inside a container
  that returns `1` — ourselves — because of PID-namespace isolation, while a
  co-tenant holds half the card. A missing instrument that reads as "all clear"
  is the exact shape of the counter_agreement retraction, so the neighbour is
  detected by *arithmetic* (how much of the card are we actually getting?)
  rather than by asking.
* **Why a reproducibility gate on top of a stationarity gate.** They are
  different properties, and only one had a gate. A steady neighbour shifts a
  whole sample series by a constant and leaves it flat, so it passes
  `gate_stationary` cleanly. Observed directly: 1433 GB/s and 799 GB/s for the
  same case in two processes, both with under 3% spread.
* **Why the inner-loop length is pinned rather than adaptive.** Because it is a
  parameter that changes the answer, not an implementation detail. For any
  kernel whose working set straddles L2, `inner=1` measures cold calls and
  `inner=2` measures a half-warm one; adaptive sizing let probe noise pick
  between them and moved a result by 1.8×.

### 效果 (Effect)

* **19 cases, 26 candidate runs, 3 repeats each, ~30 s of GPU time per sweep.**
  Cheap enough to re-run on every port, which was the design constraint.
* **Caught a contamination that would have inverted a headline.** Three earlier
  sweeps ran at 47–62% of the card and reported prefill GEMM at 186–248
  TFLOP/s — internally consistent, stationary, and about half the truth. The
  clean run reads 390–409 TFLOP/s, i.e. 89–94% of peak. The published
  conclusion flips from "cuBLAS is leaving half the card on the table" to
  "bf16 GEMM is near roofline; go after fp8 and quantization instead."
* **Found a fast, silent, wrong kernel.** `deep_gemm_mm_fp8_blockscale` returns
  all-NaN for any non-unit scale, never raises, is the fastest candidate, and is
  correct under exactly the all-ones scales a smoke test would use.
* **Separated three attention failures that all looked like "page_size 832".**
  Only one of them is; one is head_dim 256 against a 99 KiB shared-memory limit,
  and one is a blanket architecture rejection. That turns a single vague blocker
  into three items with different owners and very different costs.
* **Established the roofline context** (437 TFLOP/s bf16, 1451 GB/s) that makes
  every later number in this effort interpretable, and showed the decode kernel
  is already at 98.7% of it — steering effort away from a dead end.
* **Two rows still refuse to report a number.** That is the harness working as
  intended, and both are recorded with ranges rather than quietly averaged.
* **The instrument cross-validates.** A later run with a neighbour back on the
  card read `card_share` 0.477 (before and after, stable to 1.00) and the decode
  case fell from 1432 to 803 GB/s — the measurement and the anchor moved
  together, which is the check that the anchor is measuring what it claims.
