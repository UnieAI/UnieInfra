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

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/manager/kv_cache_manager.h"
#include "utils.h"

namespace tokenspeed {

class SwaManager : public KvCacheManager {
public:
    SwaManager(std::int32_t cache_block_tokens, std::int32_t sliding_window)
        : SwaManager(cache_block_tokens, /*cache_blocks_per_lcm_block=*/1, sliding_window, /*group_id=*/0) {}

    SwaManager(std::int32_t cache_block_tokens, std::int32_t cache_blocks_per_lcm_block, std::int32_t sliding_window,
               std::uint32_t group_id = 0)
        : KvCacheManager(cache_block_tokens, cache_blocks_per_lcm_block, group_id), sliding_window_{sliding_window} {
        _assert(sliding_window > 0, "sliding_window must be > 0");
    }

    // Non-closed: shortening a match can cut its trailing run below the window, so match bound-first.
    bool MatchIsPrefixClosed() const override { return false; }
    std::int32_t BoundaryLookbackBlocks() const override { return pagesNeededToResume(); }

    // Positions between checkpoints share the page of the checkpoint that
    // closes their stride: the recurrent state is rewritten every step, so a
    // token in the middle of a stride updates the slot that will become that
    // stride's snapshot. The base mapping sends an unbacked position to page 0
    // -- the null placeholder -- which makes every such write from every
    // request land on the same page. That is not a missing id the kernel can
    // notice; it is a silent data race, and it aborted the engine as soon as
    // admission let a second request in.
    bool BacksEveryPage() const override { return checkpoint_stride_blocks_ <= 1; }

    std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) const override {
        std::vector<std::int32_t> ids = KvCacheManager::BlockTablePageIds(table);
        if (checkpoint_stride_blocks_ <= 1) {
            return ids;
        }
        // Fill backwards: a hole takes the next backed page, which is the one
        // closing its stride. A trailing hole cannot occur -- Acquire always
        // backs the live tail -- so nothing is left pointing at page 0 except
        // slots that slid out of the window, which are never read.
        std::int32_t next_backed = 0;
        for (std::size_t i = ids.size(); i > 0; --i) {
            const std::size_t idx = i - 1;
            if (table.Blocks()[idx]) {
                next_backed = ids[idx];
            } else if (next_backed != 0) {
                ids[idx] = next_backed;
            }
        }
        return ids;
    }

    std::int64_t BackedPagesFor(std::int64_t pages) const override {
        if (checkpoint_stride_blocks_ <= 1 || pages <= 0) {
            return pages;
        }
        // Same shape as the admission estimate: one page per stride, plus the
        // live tail the request is always writing into.
        return (pages + checkpoint_stride_blocks_ - 1) / checkpoint_stride_blocks_ + 1;
    }

    // Admission-only view of the demand. A sliding-window group never holds
    // more than its window at once: ReclaimExpired nulls each block as it
    // slides out, so a long prompt walks the window rather than accumulating.
    // Charging admission for the whole extent over-reserves by
    // (extent / window), while the growth path still allocates block by block
    // as the request advances -- which is why this is separate from
    // BlocksNeededFor rather than an override of it.
    //
    // On Qwen3.6-35B-A3B this is the difference between running and not. Its
    // three GDN state groups are MambaStateManagers with a 2-token window and
    // each 20.5 MB block claims a whole 40 MB LCM parent; a 6,656-token chunk
    // was charged 52 parents per group against 55 free, so admission refused
    // every new request and concurrency sat at one while vLLM held twelve.
    // vLLM charges 2 blocks per request here (MambaSpec, "align" mode);
    // SGLang and TRT-LLM charge one. None of them charge the sequence.
    // See unieinfra/xref/46 6.1.
    // Blocks per checkpoint. 1 keeps a snapshot in every page, which is what
    // this manager has always done and what makes a long prompt reserve one
    // LCM parent per page. Larger strides keep the table the same length and
    // null the pages between checkpoints, so a prefix match resumes from the
    // nearest earlier snapshot instead of an exact one.
    void SetCheckpointStrideBlocks(std::int32_t blocks) {
        _assert(blocks > 0, "checkpoint stride must be positive");
        checkpoint_stride_blocks_ = blocks;
    }
    std::int32_t CheckpointStrideBlocks() const { return checkpoint_stride_blocks_; }

    // Allocates a real block only where a checkpoint lands, plus the live tail
    // the request is writing into now. Everything else becomes a null hole --
    // the same shape the materialized-suffix path already produces, and the
    // same shape ReclaimExpired leaves behind once a page slides out.
    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens,
                 std::int32_t reserve_tokens = 0) override {
        if (checkpoint_stride_blocks_ <= 1) {
            return KvCacheManager::Acquire(pool, table, num_tokens, reserve_tokens);
        }
        _assert(num_tokens >= 0 && reserve_tokens >= 0, "token demand and reserve must be non-negative");
        const std::int32_t num_pages = KvCacheManager::BlocksNeededFor(table, num_tokens + reserve_tokens);
        if (num_pages == 0) {
            return KvCacheManager::Acquire(pool, table, num_tokens, reserve_tokens);
        }
        const std::int32_t first_index = static_cast<std::int32_t>(table.Blocks().size());
        std::int32_t real = 0;
        for (std::int32_t i = 0; i < num_pages; ++i) {
            if (IsBackedIndex(first_index + i) || i == num_pages - 1) {
                ++real;
            }
        }
        std::vector<CacheBlockRef> refs = pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, real);
        if (static_cast<std::int32_t>(refs.size()) < real) {
            return false;
        }
        AppendSparse(table, num_tokens, num_pages, first_index, std::move(refs));
        return true;
    }

    // What a sparse group actually takes. Acquire backs only checkpoints and
    // the live tail, so charging admission for every page refuses requests
    // over capacity the group will never occupy -- and until both sides agree
    // the sparse allocation is invisible: admission is the gate, so a stride
    // change with this left at the dense count measures exactly nothing.
    // Right->left scan for a run backing a resumable boundary; slots left of it stay holes.
    GroupPrefixProbe Probe(const BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                           std::int32_t max_blocks) const override {
        const std::int32_t end_blocks =
            static_cast<std::int32_t>(std::min(keys.size(), static_cast<std::size_t>(std::max(max_blocks, 0))));
        GroupPrefixProbe probe;
        if (begin_blocks >= end_blocks) {
            return probe;
        }
        // W == 1: no lookback, so every boundary is resumable with no cached page at all.
        if (pagesNeededToResume() == 0) {
            probe.hits.resize(static_cast<std::size_t>(end_blocks - begin_blocks));
            return probe;
        }
        const auto [boundary, hits_begin] = findResumableBoundary(
            [&](std::int32_t i) { return ContainsCachedBlock(pool, keys[static_cast<std::size_t>(i)]); }, begin_blocks,
            end_blocks);
        if (boundary == begin_blocks) {
            return probe;
        }
        probe.hits.resize(static_cast<std::size_t>(boundary - begin_blocks));
        for (std::int32_t i = hits_begin; i < boundary; ++i) {
            probe.hits[static_cast<std::size_t>(i - begin_blocks)] = 1;
        }
        return probe;
    }

    // Punches null holes so the table never shrinks and slot alignment stays stable.
    void ReclaimExpired(BlockPool& /*pool*/, BlockTable& table, std::int32_t num_computed_tokens) override {
        std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            CacheBlockRef old = table.EvictToNull(i);
            if (!old) {
                break;  // already null -> earlier slots are null too
            }
        }
    }

    // Only blocks uniquely owned by this table reach the free list, so shared ones don't count.
    std::int32_t BlocksReclaimableAt(const BlockTable& table, std::int32_t num_computed_tokens,
                                     bool count_uncached) const override {
        std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        std::int32_t freed = 0;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(i)];
            if (!block) {
                break;  // already null -> earlier slots are null too
            }
            const bool cached = ContainsCachedBlock(block);
            const bool only_table_and_cache_owners = cached && block.use_count() == 2;
            if (only_table_and_cache_owners || (count_uncached && !cached && block.unique())) {
                ++freed;
            }
        }
        return freed;
    }

    std::vector<CacheBlockLocation> ReclaimableBlockLocationsAt(
        const BlockTable& table, std::int32_t num_computed_tokens,
        std::span<const CacheBlockLocation> released_locations) const override {
        const std::int32_t skipped_blocks = fullySlidOutBlocks(table, num_computed_tokens);
        std::vector<CacheBlockLocation> locations;
        for (std::int32_t i = skipped_blocks - 1; i >= 0; --i) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(i)];
            if (!block) {
                break;
            }
            const bool cached = ContainsCachedBlock(block);
            const std::uint32_t released_owners =
                static_cast<std::uint32_t>(std::ranges::count(released_locations, block->Location()));
            if ((cached && block.use_count() == 2 + released_owners) || (!cached && block.unique())) {
                locations.push_back(block->Location());
            }
        }
        return locations;
    }

protected:
    bool IsBackedIndex(std::int32_t logical_index) const override {
        return (logical_index + 1) % checkpoint_stride_blocks_ == 0;
    }

private:
    std::int32_t checkpoint_stride_blocks_{1};

    // Cached pages a boundary needs behind it: they cover the window's last (window - 1) tokens.
    std::int32_t pagesNeededToResume() const {
        return (sliding_window_ - 1 + cache_block_tokens_ - 1) / cache_block_tokens_;
    }

    struct ResumableBoundary {
        std::int32_t boundary;    // == begin_blocks when no boundary qualifies
        std::int32_t hits_begin;  // probe hits cover [hits_begin, boundary)
    };

    // Core scan shared by device and host lookup: the highest boundary backed by enough
    // consecutive probe hits -- pagesNeededToResume(), or fewer bottoming out at begin_blocks.
    template <typename Probe>
    ResumableBoundary findResumableBoundary(const Probe& probe, std::int32_t begin_blocks,
                                            std::int32_t end_blocks) const {
        const std::int32_t pages_needed = pagesNeededToResume();
        for (std::int32_t boundary = end_blocks; boundary > begin_blocks;) {
            std::int32_t hits_begin = boundary;
            while (hits_begin > begin_blocks && probe(hits_begin - 1)) {
                --hits_begin;
                if (boundary - hits_begin >= pages_needed) {
                    return {boundary, hits_begin};  // enough pages behind the boundary
                }
            }
            if (hits_begin == begin_blocks && hits_begin < boundary) {
                return {boundary, hits_begin};  // fewer, but nothing below begin_blocks is needed
            }
            // The miss at hits_begin-1 cuts every boundary in (hits_begin-1, boundary] short -- retry below it.
            boundary = hits_begin - 1;
        }
        return {begin_blocks, begin_blocks};
    }

    // Pages [0, result) fully slid out: the next query reads keys [num_computed - window + 1, num_computed].
    std::int32_t fullySlidOutBlocks(const BlockTable& table, std::int32_t num_computed_tokens) const {
        std::int32_t skipped = num_computed_tokens - sliding_window_ + 1;
        if (skipped <= 0) {
            return 0;  // all tokens still inside the window
        }
        std::int32_t skipped_blocks = skipped / cache_block_tokens_;  // only fully-slid-out pages
        // Safety cap: FSM-consistent input never engages it.
        return std::min(skipped_blocks, table.NumBlocks());
    }

    std::int32_t sliding_window_;
};

}  // namespace tokenspeed
