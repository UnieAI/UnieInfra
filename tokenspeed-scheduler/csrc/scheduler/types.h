// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <vector>

#include "cache/core/cache_config.h"
#include "utils.h"

namespace tokenspeed {

struct SchedulerConfig {
    std::int32_t block_size{};
    struct {
        // Page 0 is the null placeholder, so usable = total - 1.
        std::int32_t total_pages{};
    } host_allocator;

    struct {
        std::int32_t total_pages{};
    } device_allocator;

    std::vector<PagedCacheGroupConfig> paged_cache_groups{};

    bool HasHostCache() const { return !disable_l2_cache && host_allocator.total_pages > 1; }

    // Decode uses Host cache only for best-effort Retraction and recovery. It
    // does not continuously stream ordinary Device cache entries to Host.
    bool StreamsDeviceCacheToHost() const { return HasHostCache() && role != Role::kD; }

    std::int32_t max_scheduled_tokens{};
    std::int32_t max_batch_size{};
    std::int32_t decode_input_tokens{1};
    // Number of scheduler iterations that may be dispatched before the
    // accepted decode length is committed. The current event loop supports
    // only the non-overlapped (0) and one-step-overlapped (1) contracts.
    std::int32_t overlap_schedule_depth{0};
    bool disable_l2_cache{false};
    bool enable_l3_storage{false};
    bool enable_kv_cache_events{false};
    bool enable_mixed_prefill_decode{false};
    // Charge the whole remaining prompt when a request is first admitted,
    // instead of only this chunk plus one decode token. Ported from vLLM's
    // `scheduler_reserve_full_isl` (config/scheduler.py, default True).
    //
    // Without it nothing converts the pool size into an admission limit, so
    // `max_num_seqs` becomes the only gate and has to be guessed: measured on
    // Qwen3.5-27B at 128 concurrent, 452 requests entered a pool that carries
    // far fewer, and the resulting arena was full for the whole run with a
    // median of ONE request able to decode -- every other resident request was
    // waiting for a page. Throughput at a full batch matched vLLM (1036 vs
    // 1036 tok/s); the run averaged 215 because the batch was almost never
    // full. See unieinfra/xref/25-admission.md.
    // Cap on requests left with an INCOMPLETE prefill at the end of a step.
    // A request whose prefill finishes this step does not count, so this
    // bounds fragmentation of the token budget, not concurrency.
    //
    // Measured on gemma-4-31B / RAG: with no cap the scheduler admitted 11 new
    // sequences into one 8,192-token step -- ~745 tokens each against ~1,149
    // token prompts, so none of them finished and 21 more sat queued behind
    // them. vLLM on the same shape at c=64 held TTFT p99 at 3,435 ms and
    // 640 tok/s; we managed 6,925 ms p50 and 481 tok/s.
    //
    // This is vLLM's max_num_partial_prefills, which their V1 scheduler
    // plumbs but never reads -- their actual throttle is memory-based
    // (`scheduler_reserve_full_isl`), and ours does not bind because the pool
    // is large enough that capacity is never the limit on this shape.
    //
    // 0 means unlimited, which is the behaviour before this field existed.
    // A cap of 1 is NOT the safe default: the comment in buildForwardOperations
    // records that holding a single incomplete prefill as head of line cost
    // TTFT p50 18.1 s against vLLM's 3.4 s on the 35B agent shape, where a
    // 43k-token prompt occupies the line for six consecutive steps.
    std::int32_t max_partial_prefills{0};
    bool reserve_full_prefill_at_admission{true};
    // Preempt when a single request is blocked for capacity, instead of only
    // when the whole step produced no operation at all. Ported from vLLM,
    // which preempts inside its running loop for the one request that cannot
    // get blocks (v1/core/sched/scheduler.py).
    //
    // The existing trigger is `operations.empty() && admission_failed`, and a
    // second guard refuses to retract while ANY request owes a forward result.
    // Together they mean one request with room left in its tail page keeps the
    // step non-empty and preemption never fires -- measured as a median of ONE
    // decoding request with 128 outstanding and the arena pinned at 100%.
    //
    // Reserving the prompt at admission (above) is only half of vLLM's design:
    // it charges the prompt, not the output, so its preemption path stays live
    // and is a normal mechanism rather than a last resort.
    bool preempt_on_blocked_request{false};
    // Admission reserves `prompt + max_new_tokens * new_token_ratio`, ported
    // from SGLang (srt/managers/schedule_policy.py, new_token_ratio_tracker).
    // vLLM's reservation charges the prompt only, which is enough when a
    // request's footprint is dominated by its prompt and not enough when it is
    // not: on 1024-in/1024-out traffic the output doubles what admission
    // believed it was charging, so 128 requests entered a pool sized for far
    // fewer and every one of them stalled.
    //
    // The ratio is a feedback loop, not a constant -- it is the one admission
    // among the five surveyed that learns its own conservativeness. It decays
    // toward `min` on every step that does not retract, and is re-estimated
    // upward from measured output whenever one does.
    bool reserve_generation_at_admission{true};
    double new_token_ratio_init{1.0};   // generation-full by default; see server_args.admission_reserve
    double new_token_ratio_min_factor{1.0};  // 1.0 = no decay (TRT-LLM); 0.14 = SGLang
    std::int32_t new_token_ratio_decay_steps{600};

    Role role{Role::kFused};
    bool enable_pd_cache{false};

    bool disable_prefix_cache{false};
    // Minimum prompt tail that must be recomputed after a prefix-cache hit.
    // Zero preserves the default logits contract, which already recomputes at
    // least the final prompt token. The effective hit is page-aligned down.
    std::int32_t prefix_replay_tokens{0};
};

}  // namespace tokenspeed
