<!-- English canonical. Chinese mirror TODO: scheduling-improvements.zh.md -->
# Design: High-Concurrency Scheduling Improvements

Status: **Draft (R&D roadmap, evidence-backed)** · Date: 2026-07-25

## Problem (measured)
TokenSpeed/UnieInfra continuous batching saturates at high concurrency because
the **max running batch is tied to the static CUDA-graph capture cap (~160)**.
Measured on 1× RTX PRO 6000, Qwen3-8B (experiment E13):

| config | C=128 | C=256 |
|---|---|---|
| baseline (graph cap 160) | 3024 tok/s, TTFT 1.8s | 2581 tok/s, **TTFT 12.8s** |
| tuned (`max_num_seqs=256`, `max_cudagraph_capture_size=256`) | 2457 tok/s, TTFT 3.2s | **4713 tok/s, TTFT 0.85s** |

Raising the caps fixes C=256 (+83% throughput, 15× lower TTFT) but **regresses
C=128** → the optimal cap is load-dependent, so it **must be adaptive**, not a
hand-set constant.

## How the mature engines avoid this (studied in-tree; third_party/)
- **vLLM**: `max_num_seqs` (default 128) is **decoupled** from the graph cap
  (`max_cudagraph_capture_size = min(max_num_seqs*2, 512)`, `vllm/config/compilation.py:703`);
  live decode batches are **padded up** to the nearest captured bucket
  (`v1/cudagraph_dispatcher.py:72-91`); over-cap batches fall back to *compiled*
  (PIECEWISE torch.compile) kernels, not naive eager — no throughput cliff.
- **SGLang**: capture list is extended to `num_max_requests` (the KV-pool request
  capacity) so the **max running batch is always captured**
  (`model_executor/runner/base_cuda_graph_runner.py:86-95`); `max_running_requests`
  is re-derived from the KV pool (`kv_pool_runtime.py:89-101`); batches padded to
  bucket. Batch cap and graph cap are **co-derived from KV capacity**.
- **LMDeploy / TurboMind**: a **persistent C++ batch** with incremental slot
  refill (`engine.cc:755,779-782`), **no CUDA graphs at all** (fused kernels sized
  to `max_batch_size` → no eager cliff), and **async double-buffered
  schedule/forward overlap** (`engine.cc:829-846`) hiding per-step scheduler cost.

## Roadmap for the UnieInfra (TokenSpeed) C++ scheduler — prioritized
1. **Decouple `max_num_seqs` from the graph cap; capture up to the KV-admissible
   batch.** Derive `max_num_seqs` from KV block capacity (admission is already
   KV-gated) and capture decode graphs bucketed across that full range. Removes
   the 160 wall. (vLLM `compilation.py:703`; SGLang `base_cuda_graph_runner.py:86-95`,
   `kv_pool_runtime.py:89-101`.)
2. **Pad the decode batch up to the nearest captured bucket** instead of treating
   the graph max as a hard batch ceiling. (vLLM `cudagraph_dispatcher.py:72-91`;
   SGLang `_pad_to_bucket` `base_cuda_graph_runner.py:138-152`.)
3. **Make the over-cap path cheap** — a fused/compiled decode for batches above
   the top bucket (or, TurboMind-style, a graph-free fused decode) so it loses
   only replay overhead, not kernel-fusion speed.
4. **Adaptive admission with reserved decode headroom (anti-thrash).** Before
   admitting a waiting request, subtract a reserved decode-token estimate for each
   running seq from the KV budget; lower that reservation ratio for one step after
   a retraction, then decay. Prevents the admit→retract oscillation that spikes
   TTFT at C=256. (SGLang `schedule_policy.py:556-563,587`; `new_token_ratio`
   adaptation `scheduler.py:3238,3266`.) TokenSpeed's victim policy (evict longest
   decode) already matches SGLang/vLLM — the gap is *reactive* vs *reserved*.
5. **Async double-buffer the `submit→next_execution_plan→forward→advance` loop**
   so scheduling of step N+1 overlaps the GPU forward of step N. (TurboMind
   `engine.cc:829-846`; vLLM `v1/core/sched/async_scheduler.py`.)

Items 1–3 remove the cap/cliff; 4 stabilizes RPS@SLO under load; 5 recovers
per-step overhead that compounds at high concurrency. Together these are the
concrete path to **self-tuning, SLO-aware scheduling** (the answer to "can the
cap be adaptive?" — yes: co-derive caps from KV, pad to bucket, reserve decode
headroom, and adapt the reservation from observed retractions).
