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
#include <optional>
#include <string>

namespace tokenspeed {

enum class PagedCacheGroupFamily { History, State };

enum class PagedCacheTransferPolicy {
    Unspecified,
    FullSuffix,
    LatestSnapshot,
};

struct PagedCacheGroupConfig {
    enum class Retention {
        FullHistory,
        SlidingWindow,
    };

    std::string group_id;
    std::int32_t rows_per_page{};
    std::int32_t entry_stride_tokens{};
    std::int32_t total_pages{};
    // Number of this group's CacheBlocks packed into one physical LCM block.
    std::int32_t cache_blocks_per_lcm_block{1};
    // Blocks between recurrent-state checkpoints; 1 backs every page. Only a
    // state group reads it. See KvCacheSpec::checkpoint_stride_blocks.
    std::int32_t checkpoint_stride_blocks{1};
    Retention retention{Retention::FullHistory};
    std::optional<std::int32_t> sliding_window_tokens{};
    PagedCacheGroupFamily family{PagedCacheGroupFamily::History};
    PagedCacheTransferPolicy transfer_policy{PagedCacheTransferPolicy::Unspecified};

    // NOTE: must stay LAST -- the nanobind constructor aggregate-initialises
    // this struct positionally, so a field inserted earlier shifts every
    // argument after it. Set it through def_rw, not the constructor.
    // The largest token span one scheduling step can prefill. A sliding-window
    // group never HOLDS more than its window plus the step being written --
    // `ReclaimExpired` nulls every page that has fully slid out between steps --
    // so this bounds what admission must reserve for it. Without it the base
    // `AdmissionBlocksNeededFor` charges the whole prompt span, which on
    // gemma-4 (window 1024, P 64, chunk 8192) over-charges each of the five
    // sliding groups 172 pages where 145 is the true peak.
    std::int32_t max_scheduled_tokens{0};

    std::int32_t CacheBlockTokens() const { return rows_per_page * entry_stride_tokens; }
    void Validate() const;
};

}  // namespace tokenspeed
