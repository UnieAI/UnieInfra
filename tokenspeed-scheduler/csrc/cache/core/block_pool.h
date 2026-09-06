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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// Physical LCM placement only. It deliberately has no cache key, LRU node,
// CacheBlock pointer, or ownership count.
class BlockPool {
public:
    explicit BlockPool(std::int32_t num_lcm_blocks) : lcm_blocks_(checkedLcmBlockCount(num_lcm_blocks)) {
        for (std::int32_t id = 1; id <= num_lcm_blocks; ++id) {
            free_parent_ids_.push_back(id);
        }
        usable_parents_ = num_lcm_blocks;
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    ~BlockPool() noexcept { FatalCheck(NumOccupiedSlots() == 0, "BlockPool destroyed with live block references"); }

    // Number of physical LCM blocks. Kernel page 0 is reserved separately.
    std::int32_t NumLcmBlocks() const noexcept { return static_cast<std::int32_t>(lcm_blocks_.size()); }
    std::int32_t NumEmptyLcmBlocks() const noexcept {
        if (usable_from_ <= 1 && usable_parents_ >= NumLcmBlocks()) {
            return static_cast<std::int32_t>(free_parent_ids_.size());
        }
        std::int32_t n = 0;
        for (std::int32_t id : free_parent_ids_) {
            n += usableParent(id) ? 1 : 0;
        }
        return n;
    }

    // Parents above this id stop being handed out; the pool shrinks to it as
    // they drain. The whole pool by default, which is the old behaviour.
    //
    // This is the primitive a movable arena boundary needs. The right split
    // between the attention arena and the recurrent-state one is a function of
    // the workload -- measured on Qwen3.8-27B, the agent shape at c=16 wants
    // 112 state pages per group and collapses to 0.55x at 180, while RAG at
    // c=128 wants 180 and sits at 0.69x with 112 -- so the boundary has to be
    // able to move after the engine has seen traffic. It can only move over
    // parents that are free, and a shrinking arena has to stop handing out the
    // ones it is giving up before it can wait for them to come back.
    void SetUsableParents(std::int32_t n) {
        FatalCheck(n >= 1 && n <= NumLcmBlocks(), "usable parent count is outside the pool");
        usable_from_ = 1;
        usable_parents_ = n;
    }
    // The mirror: keep the HIGHEST n parents and give up the low ones.
    //
    // Which end a pool keeps is not cosmetic. The two arenas share one buffer,
    // and both reach the kernels as strided tensor views -- torch has no
    // negative stride, so neither region can be laid out counting down from
    // its far end. With both laid out ascending, the only way the free bytes
    // land where the *other* arena can reach them is for one pool to allocate
    // from the top. The recurrent-state pool is the one that does: its free
    // rows then sit immediately above the attention parents, which is exactly
    // the gap a moved boundary hands over.
    void SetUsableTopParents(std::int32_t n) {
        FatalCheck(n >= 1 && n <= NumLcmBlocks(), "usable parent count is outside the pool");
        usable_from_ = NumLcmBlocks() - n + 1;
        usable_parents_ = NumLcmBlocks();
    }
    std::int32_t UsableParents() const noexcept { return usable_parents_ - usable_from_ + 1; }
    std::int32_t UsableFrom() const noexcept { return usable_from_; }
    std::int32_t UsableTo() const noexcept { return usable_parents_; }
    // True once nothing outside the usable range is still bound: the boundary
    // may move. Both ends matter -- a pool that gave up its low parents has to
    // wait for those, not for the ones above its cap.
    bool DrainedOutsideUsable() const {
        for (std::int32_t id = 1; id <= NumLcmBlocks(); ++id) {
            if (!usableParent(id) && lcmBlock(id).bound_group) {
                return false;
            }
        }
        return true;
    }
    bool DrainedAboveUsable() const { return DrainedOutsideUsable(); }

    CacheBlockRef AcquireBlock(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block) {
        std::vector<CacheBlockRef> blocks = AcquireBlocks(group_id, cache_blocks_per_lcm_block, 1);
        if (blocks.empty()) {
            return {};
        }
        return std::move(blocks.front());
    }

    std::vector<CacheBlockRef> AcquireBlocks(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block,
                                             std::int32_t num) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (num <= 0) {
            return {};
        }

        std::vector<CacheBlockLocation> locations;
        if (cache_blocks_per_lcm_block == 1) {
            if (NumEmptyLcmBlocks() < num) {
                return {};
            }
            locations.reserve(static_cast<std::size_t>(num));
            for (std::int32_t id : free_parent_ids_) {
                if (!usableParent(id)) {
                    continue;  // given up to the other arena; do not hand it out
                }
                locations.push_back(CacheBlockLocation{.lcm_block_id = id, .slot_index = 0});
                if (static_cast<std::int32_t>(locations.size()) == num) {
                    break;
                }
            }
        } else {
            locations = planLocations(group_id, cache_blocks_per_lcm_block, static_cast<std::size_t>(num));
        }
        if (locations.size() != static_cast<std::size_t>(num)) {
            return {};
        }

        std::vector<CacheBlockRef> out;
        out.reserve(locations.size());
        for (CacheBlockLocation location : locations) {
            out.push_back(createBlockRef(group_id, cache_blocks_per_lcm_block, location));
        }
        return out;
    }

    std::optional<std::uint32_t> BoundGroup(std::int32_t lcm_block_id) const {
        return lcmBlock(lcm_block_id).bound_group;
    }
    std::int32_t OccupiedCount(std::int32_t lcm_block_id) const {
        return static_cast<std::int32_t>(lcmBlock(lcm_block_id).occupied_count);
    }
    bool IsOccupied(CacheBlockLocation location) const {
        const LcmBlock& lcm_block = lcmBlock(location.lcm_block_id);
        return location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < lcm_block.occupancy.size() &&
               lcm_block.occupancy[static_cast<std::size_t>(location.slot_index)];
    }
    std::int32_t NumOccupiedSlots() const noexcept {
        std::int32_t count = 0;
        for (const LcmBlock& block : lcm_blocks_) {
            count += static_cast<std::int32_t>(block.occupied_count);
        }
        return count;
    }
    std::vector<CacheBlockLocation> OccupiedLocations(std::int32_t lcm_block_id) const {
        const LcmBlock& lcm_block = lcmBlock(lcm_block_id);
        std::vector<CacheBlockLocation> locations;
        locations.reserve(lcm_block.occupied_count);
        for (std::size_t slot = 0; slot < lcm_block.occupancy.size(); ++slot) {
            if (lcm_block.occupancy[slot]) {
                locations.push_back(
                    CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = static_cast<std::int32_t>(slot)});
            }
        }
        return locations;
    }

    void Release(CacheBlockLocation location) noexcept {
        FatalCheck(location.lcm_block_id > 0 && static_cast<std::size_t>(location.lcm_block_id) <= lcm_blocks_.size(),
                   "CacheBlock location has invalid LCM block id");
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        FatalCheck(location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < parent.occupancy.size(),
                   "CacheBlock location has invalid slot");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(parent.occupancy[slot] && parent.occupied_count > 0, "CacheBlock location is not occupied");
        parent.occupancy[slot] = false;
        --parent.occupied_count;
        if (parent.occupied_count == 0) {
            parent.bound_group.reset();
            parent.occupancy.clear();
            FatalCheck(free_parent_ids_.size() < lcm_blocks_.size(),
                       "free LCM block queue cannot exceed the pool size");
            free_parent_ids_.push_back(location.lcm_block_id);
        }
    }

private:
    bool usableParent(std::int32_t id) const noexcept {
        return id >= usable_from_ && id <= usable_parents_;
    }

    std::int32_t usable_from_{1};
    std::int32_t usable_parents_{0};

    struct LcmBlock {
        std::optional<std::uint32_t> bound_group;
        std::vector<bool> occupancy;
        std::uint32_t occupied_count{0};
    };

    static std::size_t checkedLcmBlockCount(std::int32_t num_lcm_blocks) {
        _assert(num_lcm_blocks >= 0, "num_lcm_blocks must be >= 0");
        return static_cast<std::size_t>(num_lcm_blocks);
    }

    const LcmBlock& lcmBlock(std::int32_t lcm_block_id) const {
        _assert(lcm_block_id > 0 && static_cast<std::size_t>(lcm_block_id) <= lcm_blocks_.size(),
                "LCM block id out of range");
        return lcm_blocks_[static_cast<std::size_t>(lcm_block_id - 1)];
    }

    CacheBlockRef createBlockRef(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) {
        auto* control = new internal_cache_block_ref::CacheBlockControl(*this, location);
        // Allocate the control before mutating the pool, then commit the
        // location before publishing its RAII owner: CacheBlock destruction
        // releases this location and therefore requires it to be occupied.
        occupy(group_id, slots_per_parent, location);
        return CacheBlockRef{*control};
    }

    void occupy(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) noexcept {
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        if (parent.occupied_count == 0) {
            // Usually the front, and that stays O(1). But a pool with a usable
            // range skips the ids it has given up, so the parent it picked can
            // sit anywhere in the deque -- release appends, so a free list of
            // [7, 3] under a cap of 5 hands out 3, which is not the front.
            // Asserting "front" here made that a crash rather than a placement.
            auto it = free_parent_ids_.begin();
            if (it == free_parent_ids_.end() || *it != location.lcm_block_id) {
                it = std::find(free_parent_ids_.begin(), free_parent_ids_.end(), location.lcm_block_id);
            }
            FatalCheck(it != free_parent_ids_.end(), "empty LCM placement must consume a free parent");
            FatalCheck(parent.occupancy.empty(), "empty LCM parent must not retain child slots");
            parent.occupancy.assign(static_cast<std::size_t>(slots_per_parent), false);
            free_parent_ids_.erase(it);
            parent.bound_group = group_id;
        }
        FatalCheck(
            parent.bound_group == group_id && parent.occupancy.size() == static_cast<std::size_t>(slots_per_parent),
            "LCM parent binding changed while occupied");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(slot < parent.occupancy.size(), "LCM child slot is out of range");
        FatalCheck(!parent.occupancy[slot], "LCM child slot already occupied");
        parent.occupancy[slot] = true;
        ++parent.occupied_count;
    }

    std::vector<CacheBlockLocation> planLocations(std::uint32_t group_id, std::int32_t slots_per_parent,
                                                  std::size_t count) const {
        std::vector<std::int32_t> partially_filled_parent_ids;
        partially_filled_parent_ids.reserve(lcm_blocks_.size());
        for (std::size_t i = 0; i < lcm_blocks_.size(); ++i) {
            const LcmBlock& parent = lcm_blocks_[i];
            if (parent.bound_group != group_id) {
                continue;
            }
            _assert(parent.occupancy.size() == static_cast<std::size_t>(slots_per_parent),
                    "group packing changed while LCM block is occupied");
            if (static_cast<std::size_t>(parent.occupied_count) < parent.occupancy.size()) {
                partially_filled_parent_ids.push_back(static_cast<std::int32_t>(i + 1));
            }
        }

        // Allocate max occupied lcmBlock in this group at first.
        std::ranges::sort(partially_filled_parent_ids, [this](std::int32_t lhs_id, std::int32_t rhs_id) {
            const std::uint32_t lhs_occupied = lcmBlock(lhs_id).occupied_count;
            const std::uint32_t rhs_occupied = lcmBlock(rhs_id).occupied_count;
            if (lhs_occupied != rhs_occupied) {
                return lhs_occupied > rhs_occupied;
            }
            return lhs_id < rhs_id;
        });

        std::vector<CacheBlockLocation> locations;
        locations.reserve(count);
        for (std::int32_t lcm_block_id : partially_filled_parent_ids) {
            const LcmBlock& parent = lcmBlock(lcm_block_id);
            for (std::size_t slot = 0; slot < parent.occupancy.size() && locations.size() < count; ++slot) {
                if (!parent.occupancy[slot]) {
                    locations.push_back(CacheBlockLocation{
                        .lcm_block_id = lcm_block_id,
                        .slot_index = static_cast<std::int32_t>(slot),
                    });
                }
            }
            if (locations.size() == count) {
                return locations;
            }
        }

        // Allocate free lcmBlock if partially-filled block is not enough.
        for (std::int32_t lcm_block_id : free_parent_ids_) {
            if (!usableParent(lcm_block_id)) {
                continue;  // given up to the other arena; a packed group may not take it either
            }
            for (std::int32_t slot = 0; slot < slots_per_parent && locations.size() < count; ++slot) {
                locations.push_back(CacheBlockLocation{
                    .lcm_block_id = lcm_block_id,
                    .slot_index = slot,
                });
            }
            if (locations.size() == count) {
                return locations;
            }
        }
        return {};
    }

    std::vector<LcmBlock> lcm_blocks_;
    // Free parents are interchangeable: release appends and allocation consumes
    // the front. Bound parents are selected separately by planLocations().
    std::deque<std::int32_t> free_parent_ids_;
};

}  // namespace tokenspeed
