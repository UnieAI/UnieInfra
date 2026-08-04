<!-- English canonical. Chinese: kernel-autotuning.zh.md (keep in sync). -->
# Kernel autotuning: measure the candidates, keep the fastest

## Why

Heuristic kernel ranking — `oracle > objective > priority > origin` — is a
*belief* about hardware written down at registration time. Priority bands in
particular encode "the narrowly-gated datacenter kernel is the best one". That
is exactly the assumption that fails on the hardware we run on: sm_120 is
arch-*newer* than sm_90 yet has no tcgen05, so an open-ended `>=` arch gate
admits kernels that then crawl or die (E30 catalogued these in our own tree;
E37 hit the identical shape in upstream vLLM's `ll_bf16` CuteDSL router GEMM,
gated on `has_device_capability(90)`).

**A measurement cannot be wrong about the hardware it ran on.** That is the
whole argument.

Two prior results say this is the right investment now rather than later:

* **E34** — letting the declarative trait gate choose across *all* dense
  `Linear`s measured **+4.2%** on batch-1 Qwen3-8B, against **+1.6%** for a
  hand-picked `qkv_proj,o_proj` list. The difference was `down_proj`, a layer
  the human list simply forgot. Automated selection already beat human curation
  on a *static* gate; autotuning is the dynamic version of the same argument.
* **E33 / E38** — first-party kernels can win (71.5% → 87.5% achieved DRAM
  bandwidth; fp8 **+7.4%** and nvfp4 **+17.8%** at C=1 against vLLM's own
  quantized paths). Generating more candidates only pays off if we can correctly
  choose between them, which makes autotuning the prerequisite for the KDA loop.

## What was broken

`_autotune_select` never timed anything. It called `_rank_by_objective`, logged
*"falling back to heuristic"*, and — worse — hardcoded
`SelectionObjective.DEFAULT`, discarding whatever objective the caller asked
for. So `AUTOTUNE` was strictly **worse** than `HEURISTIC`. `AutotuneParams` was
read by nothing.

## Design

```
select_kernel(family, mode, signature, traits=…, shape=…)
        │
        ├─ capability / signature / trait / pin filtering        (unchanged)
        │
        ├─ HEURISTIC  → _rank_by_objective                       (unchanged)
        │
        └─ AUTOTUNE   → _autotune_select
                          ├─ disk cache hit?            → done, no measurement
                          ├─ frozen, or no CUDA?        → heuristic winner
                          ├─ no shape, or <2 candidates?→ heuristic winner
                          ├─ inputs = input_generator(signature, shape)
                          ├─ time each candidate, each in a guard
                          └─ fastest wins, unless inside `margin` of the
                             heuristic winner (then the heuristic keeps it)
```

### 1. It measures

`unieinfra_kernel/selection/autotune.py` owns the timing. Warmup, then
`reps` repetitions of `bench_iters` CUDA-event-timed calls; the answer is the
**median across repetitions**, so a single scheduling hiccup cannot decide a
kernel.

Two fidelity details, both carried over from
`benchmark/scripts/bench_decode_gemv.py` rather than reinvented:

* **L2 rotation.** GB202's L2 is 128 MiB, larger than every weight matrix in an
  8B decode step. Timing a loop that re-reads one weight measures L2 bandwidth,
  not DRAM — it reported >200% of peak before this was fixed in the benchmark
  script. Inputs are allocated in several independent sets and cycled, so each
  call starts cold, exactly as a real decode step does when 16 GB of weights
  stream past between two visits to one layer.
* **A margin.** The measured winner only overturns the heuristic winner when it
  is faster by more than `AutotuneParams.margin` (default 2%). Without it,
  run-to-run noise picks the winner and the disk cache then freezes that coin
  flip for the life of the machine. `margin=0.0` gives pure fastest-wins.

### 2. It honours the caller's objective

`_autotune_select` now takes `objective` and passes it to `_rank_by_objective`.
The heuristic ranking it produces is the fallback answer *and* the tiebreaker,
so a caller asking for `LATENCY` gets a latency-shaped tie broken their way.

### 3. It never tunes during serving

`unieinfra_kernel/selection/tuning.py` (ported from the retiring tree's
`ops/tuning.py`, semantics and names unchanged) holds one flag.

Tuning during serving is a hazard twice over: it blocks the launch thread for
the whole profiling run, and under tensor parallelism ranks that begin tuning at
different times miss their next collective and **deadlock**.

So tuning happens in exactly one place — startup:

```
GPUModelRunner.load_model()
   … weights resident …
   unieinfra_autotune_startup(self.model)      ← measure, then freeze
   … torch.compile, CUDA-graph capture …       ← captures the tuned decision
   … serving …                                 ← frozen; cache hits or heuristic
```

Under TP every rank reaches this point over the same shapes in the same order
with no collective in between, so the ranks stay in lockstep. After the freeze,
a shape first seen during serving resolves heuristically — possibly a worse
kernel, never a stall and never a hang.

The shapes come from **walking the loaded model**, not from a table: `N` and `K`
are fixed when a `Linear` is built, so this covers exactly the projections that
will run — including any a hand-maintained list would have missed. That is E34's
lesson applied to the tuner itself. Duplicate `(N, K)` collapse, so 36 identical
decoder layers cost one measurement, not 36.

### 4. It caches on disk, keyed on the device and not just the arch

`~/.cache/unieinfra-kernel/autotune/<device>-<arch>.json`
(`UNIEINFRA_KERNEL_AUTOTUNE_CACHE` overrides the directory). Each entry is keyed
by

```
(device_name, arch, family, mode, format_signature, traits, shape-bucket)
```

**`device_name`, not just `arch`, on purpose**: sm_120 spans the RTX PRO 6000
(188 SMs, server clocks) and the RTX 5090 (170 SMs). A kernel whose whole trick
is maximising grid size against the SM count — which is precisely what E33's
decode GEMV does — does not have to win on both.

The shape bucket rounds `M` **up to a power of two** and keeps every other
dimension exact. `M` decides whether a GEMV beats a tensor-core GEMM and E33
measured the crossover between `M = 2` and `M = 8`, so 1/2/4/8 must each keep
their own answer; `N` and `K` are architectural constants with only a handful of
values per model, so coarsening them buys nothing and could merge a shape that
passes `n_le_16384` with one that fails it.

A missing, unreadable or version-stale cache file is ignored, never fatal.

### 5. It fails safe

Every candidate is timed inside a guard. One that raises — a bad arch gate, a
Triton compile failure, an unsupported layout — is **dropped from the ranking**
and logged, not propagated. E30 noted this alone would have caught every
hard-failing mis-gate it found: *a kernel that wins ranking and then dies at
runtime is exactly what measurement should eliminate*, before it ever sees a
request. If every candidate fails, the heuristic winner stands.

### 6. Ablations stay possible: pins

`kernel_override` forces one function and bypasses the pipeline. That is the
wrong tool for an ablation, which wants "select as usual, but only among X".
`SelectionPin` filters the *candidate set* and lets normal filtering and ranking
run on what is left:

```python
with pin_kernel(origin="vendor"):        # incumbent arm
    run()
with pin_kernel(origin="first_party"):   # our kernels only
    run()
with pin_kernel("gemm", "mm", name="unieinfra_decode_gemv_triton"):
    run()
```

Also available as `UNIEINFRA_KERNEL_PIN=origin=first_party` (or a bare kernel
name), and per-operator as `UNIEINFRA_KERNEL_PIN_GEMM_MM=…`. The pin is part of
the memo key, so switching arms in one process cannot serve a stale decision.

## Reused, not reinvented

| Piece | Came from | Where it is now |
|---|---|---|
| Warmup + CUDA-event timing | `benchmark/runner.py::_time_kernel` (retiring tree, now `legacy/`) | `selection/autotune.py::time_kernel` |
| Input generator registry | `numerics/inputs.py` (`legacy/`) | `selection/inputs.py` — class collapsed to a plain callable; nothing used the class's state |
| Benchmark shapes | `numerics/inputs.py::get_benchmark_shapes` | `selection/inputs.py`, populated from `KernelContract.workloads` — the contract already declared "representative shapes for autotune / benchmarking" |
| Freeze lifecycle | `ops/tuning.py` (`legacy/`) | `selection/tuning.py`, names unchanged |
| L2-defeating rotation | `benchmark/scripts/bench_decode_gemv.py` | `selection/autotune.py::_rotation_count`, same 640 MiB constant |

The generator is a *registry*, keyed by `(family, mode)` and populated by the
operator's contract module, so the selection core never learns what a GEMM's
arguments look like. `gemm.mm` registers one generator that reads the **format
signature** (not the checkpoint) to decide whether to build dense, scaled-fp8 or
NVFP4 operands — because the signature is what selection matched on, and a
weight-only fp8 request and a symmetric W8A8 request are two different requests.

## Operating it

| Variable | Effect |
|---|---|
| `UNIEINFRA_AUTOTUNE=1` | Engine startup measures instead of guessing. Off by default. |
| `UNIEINFRA_KERNEL_AUTOTUNE_CACHE=<dir>` | Where measurements persist. |
| `UNIEINFRA_KERNEL_PIN=<spec>` | Ablation pin: `name=…`, `solution=…`, `origin=…`, or a bare kernel name. |
| `UNIEINFRA_KERNEL_PIN_<FAMILY>_<MODE>=<spec>` | Same, one operator only. |

Programmatically:

```python
from unieinfra_kernel.selection import enable_autotuning, warmup_selection

enable_autotuning([("gemm", "mm")])
seconds = warmup_selection(ops, freeze=True)   # tune, then close the door
```

`benchmark/scripts/autotune_validate.py` runs the selection-correctness checks
against E33/E34's known answers, reports the cold and warm tuning cost, and
prints every case where measurement disagreed with the heuristic.

## The M=256 case: selection can only fix what it can choose between

E44 attributed ~8% of a 45.7 ms C=128/C=256 decode step to one shape: at
`M = 256` cuBLASLt moves the QKV projection's 50.3 MB in **147.9 µs instead of
43.0 µs**. That is the canonical "the machine has a faster way and the
dispatcher does not find it" failure, so it is the natural acceptance test for a
measuring selector. Measured on an idle RTX PRO 6000 with L2-defeating rotation
(`benchmark/scripts/probe_m256_qkv.py`, `autotune_m256_demo.py`):

**1. Today the registry has exactly one candidate at that shape.**

| M | candidates for `gemm.mm` at `N=6144, K=4096`, bf16 |
|---|---|
| 1 | `unieinfra_decode_gemv_triton`, `torch_linear` |
| 8 | `unieinfra_decode_gemv_triton`, `torch_linear` |
| **256** | **`torch_linear` only** |

The decode GEMV is excluded by its `m_le_8` gate; the fp8/nvfp4 kernels are
excluded by signature. `_autotune_select` returns without measuring when there
are fewer than two runnable candidates, so **autotuning cannot capture E44's
M=256 loss as the tree stands — not because the selector is weak, but because
there is nothing to select between.** The fix is a *candidate*, not a selector.

**2. Given a second real candidate, measurement does find the win.** Registering
`F.linear` against a K-major weight (same arithmetic, different cuBLAS heuristic
input, relayout done once via `KernelSpec.weight_preprocessor`) at a *lower*
priority band, so the heuristic can never pick it:

| M | heuristic | autotune | µs (incumbent → chosen) |
|---|---|---|---|
| 1 | `unieinfra_decode_gemv_triton` | `unieinfra_decode_gemv_triton` | 35.0 (held; runner-up 37.3) |
| 64 | `torch_linear` | `torch_linear` | 39.3 (held; runner-up 45.3) |
| 128 | `torch_linear` | `torch_linear` | 46.4 (held; runner-up 49.3) |
| **256** | `torch_linear` | **`demo_torch_linear_kmajor`** | **60.0 → 58.2 µs (+3.0%)** |
| 512 | `torch_linear` | `torch_linear` | 96.5 (runner-up 95.8, +0.8%, inside margin) |

The same instrument that flips M=256 *declines* to flip M=64/128 (where the
alternative is 13-6% slower) and M=512 (where it is 0.8% faster, inside the
margin). A hand-written rule reading "K-major is better at large M" would have
been wrong at four of the five points.

**3. E44's 147.9 µs does not reproduce in isolation, and that matters.** On an
idle GPU every PyTorch-reachable path does M=256 QKV in **58-61 µs (~930 GB/s)**,
not 147.9 µs (~340 GB/s). What *does* reproduce is the direction: achieved
bandwidth collapses monotonically with M —

| M | 1 | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| GB/s (best path) | 1420 | 1360 | 1233 | 1391 | 1168 | **940** | 653 |

so M=256 really is ~32% off the rate M=64 sustains, and M=512 is 53% off. But the
isolated deficit is ~2.5× smaller than E44's. Either E44's number is a
serving-context effect (concurrent kernels, cuBLASLt workspace/heuristic state,
or chunked prefill making the real M not exactly 256), or the two measurements
differ methodologically. **That discrepancy should be resolved before anyone
sizes a kernel project against the 147.9 µs figure.**

**What would actually capture it.** Alternate BLAS entry points are worth ≤4% on
this box, not the 3.4× E44 implies. Closing a 32% large-M bandwidth gap needs a
first-party large-M GEMM — a KDA target — and autotuning is what will then
choose it correctly, per shape, without anyone writing the crossover rule down.

## Known limits

* Only `gemm.mm` has an input generator today, so only `gemm.mm` is tunable.
  Adding an operator is one function registered from its contract module.
* **Autotuning is worthless where there is one candidate**, which today is every
  `M > 8` dense GEMM (see the M=256 section). It raises the value of adding
  candidates; it does not substitute for them.
* Symmetric W8A8 / W4A4 signatures are declined by the generator rather than
  synthesized. No first-party kernel claims them, so there is nothing to tune
  between; inventing a quantized activation would tune a path that never runs.
* On the `tokenspeed` registry backend the boundary drops `shape=`/`strategy=`
  and selection stays heuristic. That base is retiring; the standalone core is
  the target.
* Tuning measures the kernel in isolation, not inside the captured graph. A
  kernel whose advantage depends on fusion with its neighbours will not show it
  here.
