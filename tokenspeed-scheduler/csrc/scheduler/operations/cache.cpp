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

#include "scheduler/operations/cache.h"

#include <stdexcept>

#include "scheduler/types.h"

namespace tokenspeed {

std::int32_t AlignPrefillChunk(std::int32_t first_pos, std::int32_t unscheduled, std::int32_t token_budget,
                               std::int32_t page_size, std::int32_t promotion_boundary_tokens) {
    _assert(first_pos >= 0 && unscheduled >= 0 && token_budget >= 0, "prefill positions must be non-negative");
    _assert(page_size > 0, "page_size must be > 0");
    std::int32_t chunk_size = std::min(unscheduled, token_budget);
    if (promotion_boundary_tokens > first_pos) {
        chunk_size = std::min(chunk_size, promotion_boundary_tokens - first_pos);
    }
    if (chunk_size == unscheduled) {
        // The last round used to carry the whole remainder, so prefill ended at
        // `prefill_size` -- page-aligned only by luck. A mamba/state group
        // publishes nothing when `num_computed_tokens % cache_block_tokens != 0`
        // (cacheCompletedBlocksForGroup), and SweepThenConverge takes a min
        // across groups, so one unaligned endpoint drops the whole converged
        // prefix back to the last aligned publish -- in practice the last
        // COMPLETE prefill chunk.
        //
        // Measured on Qwen3.8-27B with the engine's own per-chunk log: a
        // prefix's FIRST reuse matched exactly
        // floor(prefill / chunked_prefill_size) * chunked_prefill_size and
        // recomputed the rest -- 6,400 / 5,632 / 4,352 / 3,584 tokens at
        // prompts of 6.5k / 14.0k / 29.0k / 61.0k, i.e. 745-940 ms. In an agent
        // workload every reuse is a first reuse: turn N+1 resumes turn N's tail
        // exactly once, so this was paid on every turn.
        //
        // End the last full round on the grid and leave the ragged tail -- less
        // than one page -- to the next round, so the recurrent state
        // materializes where it can be published. Worst-case reuse loss goes
        // from one chunk to one page. A prompt that fits inside a single page
        // keeps its single round: `chunk_size > ragged` is false there.
        const std::int32_t ragged = (first_pos + chunk_size) % page_size;
        if (ragged != 0 && chunk_size > ragged) {
            return chunk_size - ragged;
        }
        return chunk_size;
    }

    const std::int32_t page_offset = first_pos % page_size;
    if (page_offset != 0) {
        const std::int32_t tokens_to_boundary = page_size - page_offset;
        if (chunk_size < tokens_to_boundary) {
            return 0;
        }
        // Re-align to the page grid, then keep going. Returning only
        // `tokens_to_boundary` capped every misaligned chunk at less than one
        // page, so a request that started off-grid -- which on a high-reuse
        // shape is most of them, because a prefix hit rarely ends on a page
        // boundary -- crawled forward under 128 tokens per prefill step.
        //
        // Measured on Qwen3.5-27B, RAG at C=128: single-sequence prefill steps
        // carried 96-127 new tokens, i.e. exactly `page_size - page_offset`,
        // and 19 of 36 prefill steps carried one sequence. Prompts of 606-1015
        // tokens were taking several steps each to admit.
        //
        // Alignment is preserved: the chunk still ends on a page boundary,
        // because everything past the first partial page is a whole number of
        // pages.
        const std::int32_t past_boundary = chunk_size - tokens_to_boundary;
        return tokens_to_boundary + (past_boundary - past_boundary % page_size);
    }
    return chunk_size - chunk_size % page_size;
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    _assert(config.block_size > 0, "scheduler block_size must be > 0");
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        _assert(group.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        const std::int32_t cache_block_tokens = group.CacheBlockTokens();
        _assert(cache_block_tokens > 0, "cache group must have positive tokens per cache block");
        _assert(config.block_size % cache_block_tokens == 0,
                "cache group block tokens must divide the scheduler cache block tokens");
        // family=State marks trailing-window prefix reuse and covers both SWA and
        // linear-attention groups; only a State group WITHOUT SlidingWindow
        // retention is a mamba-style state group.
        const bool final_state_manager = group.family == PagedCacheGroupFamily::State &&
                                         group.retention != PagedCacheGroupConfig::Retention::SlidingWindow;
        if (config.enable_pd_cache) {
            const PagedCacheTransferPolicy expected =
                final_state_manager ? PagedCacheTransferPolicy::LatestSnapshot : PagedCacheTransferPolicy::FullSuffix;
            if (group.transfer_policy == PagedCacheTransferPolicy::Unspecified) {
                throw std::invalid_argument("PD cache group '" + group.group_id +
                                            "' requires an explicit transfer_policy");
            }
            if (group.transfer_policy != expected) {
                throw std::invalid_argument("PD cache group '" + group.group_id +
                                            "' transfer_policy does not match its scheduler "
                                            "destination layout");
            }
        }
        if (final_state_manager) {
            specs.push_back(KvCacheSpec{
                .kind = AttnKind::kMambaState,
                .sliding_window = 0,
                .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
                .cache_block_tokens = cache_block_tokens,
                .checkpoint_stride_blocks = group.checkpoint_stride_blocks,
            });
            continue;
        }
        const bool is_swa = group.retention == PagedCacheGroupConfig::Retention::SlidingWindow;
        if (is_swa && (!group.sliding_window_tokens || *group.sliding_window_tokens <= 0)) {
            throw std::invalid_argument("Cache group '" + group.group_id + "' requires positive sliding_window_tokens");
        }
        specs.push_back(KvCacheSpec{
            .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
            .sliding_window = is_swa ? *group.sliding_window_tokens : 0,
            .cache_blocks_per_lcm_block = group.cache_blocks_per_lcm_block,
            .cache_block_tokens = cache_block_tokens,
            // Only a sliding group reads it; a full group's admission charge is
            // the whole span by definition.
            .max_scheduled_tokens = group.max_scheduled_tokens,
        });
    }
    return specs;
}

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables) {
    if (tables.empty()) {
        return;  // request never got tables, or a failure path already released them
    }
    coordinator.Free(tables);
}

std::map<std::string, std::vector<std::int32_t>> BuildBlockTables(const KvCacheCoordinator& coordinator,
                                                                  const std::vector<BlockTable>& tables,
                                                                  std::span<const std::string> group_ids) {
    _assert(tables.size() == group_ids.size(), "BuildBlockTables: tables/group_ids size mismatch");
    _assert(tables.size() == static_cast<std::size_t>(coordinator.NumGroups()),
            "BuildBlockTables: tables/coordinator size mismatch");
    std::map<std::string, std::vector<std::int32_t>> out;
    for (std::size_t i = 0; i < tables.size(); ++i) {
        out.emplace(group_ids[i], coordinator.GroupManager(static_cast<std::int32_t>(i)).BlockTablePageIds(tables[i]));
    }
    return out;
}

}  // namespace tokenspeed
