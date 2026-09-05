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

    // Second device arena, for the recurrent-state groups only.
    //
    // The LCM parent is sized by the widest geometry family, so a group whose
    // page is narrower strands the difference on every parent it binds. On
    // Qwen3.8-27B a GDN state page is 48.94 MiB against a 96 MiB parent and
    // packs 1, so the three recurrent groups take 206 of 346 parents -- 60 of
    // every 68 of them retained snapshots -- and leave full attention 188,928
    // tokens of prefix cache against vLLM's 447,146. Splitting them into an
    // arena whose parent IS the state page takes both utilizations to 1.00.
    //
    // 0 keeps every group on `device_allocator`, which is the old behaviour.
    struct {
        std::int32_t total_pages{};
    } state_allocator;

    bool HasStateArena() const { return state_allocator.total_pages > 1; }

    std::vector<PagedCacheGroupConfig> paged_cache_groups{};

    bool HasHostCache() const { return !disable_l2_cache && host_allocator.total_pages > 1; }

    // Decode uses Host cache only for best-effort Retraction and recovery. It
    // does not continuously stream ordinary Device cache entries to Host.
    bool StreamsDeviceCacheToHost() const { return HasHostCache() && role != Role::kD; }

    // How the per-step token budget is divided among the requests that are
    // mid-prefill. Ported from TRT-LLM's `ContextChunkingPolicy`
    // (cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:126 for the
    // kEQUAL_PROGRESS specialisation); it is the only one of the five surveyed
    // engines that makes this a choice rather than a side effect.
    //
    //   kFirstComeFirstServed  what we have always done: `schedulePrefill`
    //     takes min(remaining, unscheduled), so the first continuation in
    //     priority order consumes the whole step and the next request is
    //     offered whatever is left, usually nothing.
    //
    //   kEqualProgress  round-robin: every mid-prefill request is offered one
    //     `context_chunk_unit` per pass, repeatedly, until the budget is spent
    //     or nobody can take more. Same total budget, same makespan; what
    //     changes is WHICH requests finish early.
    //
    // Which is better is a function of how unequal the prefill work is, and
    // that is what this exists to measure rather than assume. With equal work
    // FCFS strictly dominates on mean completion time -- 16 identical prompts
    // at one step each finish at steps 1..16 under FCFS and ALL at step 16
    // under equal progress. With unequal work the long prompts under FCFS wait
    // behind everything and miss the deadline: measured on
    // agent-26k/Qwen3.6-35B-A3B c=16, where prompt lengths span 23x, our TTFT
    // correlates +0.575 with a request's own length while vLLM's is flat
    // (-0.026), and the four requests that miss the 10 s bound are literally
    // the four longest prompts in the corpus.
    //
    // NOT the same knob as `--chunked-prefill-size`. That sets
    // `max_scheduled_tokens`, the whole step; shrinking it makes the step
    // smaller, not shared, and measured 8192 -> 1024 on that cell cost 74% of
    // throughput while the length correlation barely moved (+0.456 -> +0.409).
    enum class ContextChunkPolicy { kFirstComeFirstServed, kEqualProgress };
    ContextChunkPolicy context_chunk_policy{ContextChunkPolicy::kFirstComeFirstServed};
    // Granularity of one round-robin pass. 0 means "one cache block", which
    // keeps every offer page-aligned -- an unaligned chunk is refused outright
    // by AlignPrefillChunk on hybrid models.
    std::int32_t context_chunk_unit{0};

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
    // Cap on requests admitted in one step whose prefix matched NOTHING in the
    // cache. Zero means unlimited, which is the behaviour before this field.
    //
    // Measured on gemma-4-31B / RAG at c=64: the first step of a measured run
    // admits ten requests and the whole step reports 1,088 cached tokens --
    // about one request's worth -- because all ten probe the cache in the same
    // step, before any of them has written to it. They then prefill the same
    // ~1,000-token shared prefix ten times over. The cache fills in as the run
    // proceeds (1,088 -> 6,784 -> 17,664 cached across successive steps), so
    // the redundancy is confined to the cold start, but that start is what TTFT
    // measures: p50 4,022 ms on a cold cache against 1,837 ms on a warm one.
    //
    // Admitting one cold request per step lets it publish its prefix before the
    // rest probe. It costs a step of latency when prompts share nothing, which
    // is why it is off by default and set from measurement.
    std::int32_t max_cold_prefills{0};
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
