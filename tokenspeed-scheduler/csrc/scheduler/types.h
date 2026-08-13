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

#include <optional>
#include <unordered_map>
#include <variant>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>
#include <memory>

#include "fsm/forward_events.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/types.h"
#include "scheduler/operations/inc.h"

namespace tokenspeed {

class TreeNode;

enum class DisaggregationMode {
    kNone,
    kPrefill,
    kDecode,
};
// `PagedCacheGroupFamily` is defined in
// resource/allocator/paged_cache_group.h (transitively included above).

template <ResourceType>
class NodeRef;
using HostNodeRef = NodeRef<ResourceType::Host>;
using DeviceNodeRef = NodeRef<ResourceType::Device>;

struct SchedulerStats {
    std::int64_t total_batches = 0;
    std::int64_t mixed_batches = 0;
    std::int64_t retract_count = 0;
    std::int64_t abort_count = 0;
    std::int64_t schedule_latency_count = 0;
    std::int64_t schedule_latency_sum_us = 0;
    std::int64_t schedule_latency_max_us = 0;
    std::int64_t prefix_cache_hit_tokens = 0;
    std::int64_t prefix_cache_req_tokens = 0;

    std::int64_t pending_queue_size = 0;
    std::int64_t plan_queue_size = 0;
    std::int64_t event_queue_size = 0;
    std::int64_t active_requests = 0;
};

// Opt-in spec for the paged-cache prefix-cache adjunct. Unset means paged-cache
// groups are transport-only (no snapshot chain, no prefix-cache reuse).
struct PrefixCacheAdjunctSpec {
    std::vector<std::string> required_groups{};
};

struct SchedulerConfig {
    std::int32_t block_size{};
    struct {
        // Flat builds: page 0 is the null placeholder (device convention), so usable = total - 1.
        std::int32_t total_pages{};
    } host_allocator;

    struct {
        std::int32_t total_pages{};
    } device_allocator;

    std::vector<PagedCacheGroupConfig> paged_cache_groups{};

    // GCD of every group's effective block_size (per-group override, else the global
    // block_size): the base page granularity all group block sizes are multiples of.
    std::int32_t BaseBlockSize() const {
        std::int32_t base = 0;
        for (const auto& g : paged_cache_groups) {
            std::int32_t bs = g.block_size > 0 ? g.block_size : block_size;
            base = base == 0 ? bs : std::gcd(base, bs);
        }
        return base == 0 ? block_size : base;
    }

    // Streaming-sink (flat L2) enablement: an L2 host tier exists (> 1: page 0 is the null
    // placeholder) and this role writes to it. Orthogonal to disable_prefix_cache by design:
    // that flag gates MATCHING only, the sink gates STORING.
    bool FlatStreamingSinkEnabled() const {
        return !disable_l2_cache && host_allocator.total_pages > 1 && role == Role::kFused;
    }

    // Unset means paged-cache groups are transport-only.
    std::optional<PrefixCacheAdjunctSpec> prefix_cache_adjunct{};

    std::int32_t max_scheduled_tokens{};
    std::int32_t max_batch_size{};
    std::int32_t decode_input_tokens{1};
    // Number of scheduler iterations that may be dispatched before the
    // accepted decode length is committed. The current event loop supports
    // only the non-overlapped (0) and one-step-overlapped (1) contracts.
    std::int32_t overlap_schedule_depth{0};
    bool disable_l2_cache{false};
    bool enable_l3_storage{false};
    std::int32_t prefetch_threshold{4};  // num pages
    bool enable_kv_cache_events{false};
    bool enable_mixed_prefill_decode{false};
    // Fraction of max_batch_size above which a step is allowed to mix.
    //
    // Mixing is not free on the recurrent path: a decode row becomes an extend
    // row of length 1, so it runs the chunked-scan prefill kernel instead of
    // the fused decode one (9 launches per layer against 2, see
    // unieinfra/xref/40-decode.md). Measured on Qwen3.5-27B over HTTP, always
    // mixing costs 96 -> 60 tok/s at 64 and 128 concurrent, and always
    // splitting fails outright at 256 -- 346 of 512 requests dropped -- because
    // prefill never gets a turn. Neither setting wins everywhere, so the choice
    // belongs per step rather than per server.
    //
    // Keyed on how full the running set is, because that is what decides
    // whether a waiting prefill can get a step of its own: with room to spare a
    // pure prefill step costs one step, and mixing would tax every decode row
    // in the batch for it. Near capacity there is no such step to be had.
    // Queue depth, as a fraction of max_batch_size, above which a step mixes.
    //
    // Measured on Qwen3.5-27B: 0.02-0.03 at 16 concurrent, 0.09-0.13 at 64,
    // 0.16-0.27 at 128, 0.44-0.60 at 256, 0.90-1.13 at 512 -- so 0.35 sits in a
    // clean gap between the two regimes. Two earlier signals did not:
    // running/max_batch_size never fires because a starving engine has a small
    // running set by construction (10-44 running against 210-246 waiting at 256
    // concurrent), and waiting/running fires indiscriminately because 64 and 256
    // concurrent produce overlapping ranges (3.3-15.0 against 5.6-17.3).
    //
    // Re-measured after the recurrent split was fixed, because the first
    // comparison was made while every mixed step silently did double work on
    // the wrong rows. The conclusion held and the margin grew: on the real RAG
    // dataset, always mixing (0.0) measures 223 tok/s at 64 concurrent and 266
    // at 256, against 390 and 516 at this threshold. So the length-1 decode row
    // on the chunked scan was not the only cost of mixing every step -- the
    // extra launch, the rejoin, and the more ragged batch shape remain.
    double mixed_prefill_decode_min_load{0.35};
    // How many prompts may be partially prefilled in one step, vLLM's
    // `max_num_partial_prefills` (config/scheduler.py:70, default 1). This is
    // the knob that lets mixing stay on unconditionally: prefill's share of a
    // step is bounded by a count, so decode is never starved and no per-step
    // mix/no-mix decision is needed. Zero means unbounded, which is what this
    // scheduler did before.
    std::int32_t max_num_partial_prefills{0};
    // Cap on how far ONE prefill may advance in a single step, vLLM's
    // `long_prefill_token_threshold` (v1/core/sched/scheduler.py:478).
    //
    // Different from max_num_partial_prefills, which caps how many prefills a
    // step may carry: this caps each one's stride, so several requests advance
    // together within the same total token budget instead of one consuming it.
    // Measured at 64 concurrent, this engine puts 5955 tokens of an 8192 budget
    // into 9.27 requests -- one or two whole prompts a step -- and its TTFT is
    // 4297 ms against vLLM's 2630 while its throughput is within 15%. Nothing
    // is queued (18 waiting) and the batch is 73% full, so the gap is when each
    // request's first token arrives, not how much work gets done.
    //
    // Zero means unbounded, which is what this scheduler did before.
    std::int32_t long_prefill_token_threshold{0};

    std::int32_t num_pages_reserved_for_retracted_or_running{};
    Role role{Role::kFused};
    // Explicit opt-in for the Flat KV Prefill/Decode lifecycle. This keeps
    // legacy non-flat PD and Flat standalone scheduling unchanged.
    // Charge a request's whole prompt against the pool when it is admitted,
    // rather than only the chunk being scheduled. Without it the sole bound on
    // admission is the request-slot count, so the pool oversubscribes and every
    // request stalls together instead of a few being preempted. vLLM's
    // equivalent (`scheduler_reserve_full_isl`) is also on by default; see
    // unieinfra/xref/25-admission.md.
    bool reserve_full_prompt_on_admit{true};

    bool enable_flatkv_pd{false};

    bool disable_prefix_cache{false};
    bool enable_mamba{false};
    std::int32_t mamba_cache_chunk_size{64};
    std::int32_t mamba_pool_total_chunks{0};
    bool enable_mamba_l2{false};
    std::int32_t mamba_l2_host_slots{0};
    // Slots in the *checkpoint* pool, a second index space for tree-owned
    // snapshots. Zero keeps the old arrangement, where a retained snapshot is
    // taken from the same pool a running sequence draws its working and
    // checkpoint slots from -- which is what made retaining several per chunk
    // fail at 64 concurrent requests. Non-zero means the radix allocates here
    // instead and a cached state can never starve a live one.
    std::int32_t mamba_ckpt_pool_slots{0};
};

}  // namespace tokenspeed
