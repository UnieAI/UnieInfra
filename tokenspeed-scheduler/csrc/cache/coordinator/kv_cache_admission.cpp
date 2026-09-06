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

#include "cache/coordinator/kv_cache_coordinator.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <optional>

#include <spdlog/spdlog.h>
#include <span>
#include <tuple>
#include <unordered_set>

#include "utils.h"

namespace tokenspeed {
namespace {

struct AdmissionPlan {
    KvCacheCoordinator::PrefixProbe prefix;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
};

class AdmissionPlanner {
public:
    // `pools` are the distinct device arenas, `pool_of_group` maps each group to
    // one of them. With a single arena both collapse to the old behaviour.
    //
    // Two arenas is not a generalization for its own sake: the LCM parent is
    // sized by the widest geometry family, so a group whose page is narrower
    // strands the difference on every parent it binds. On Qwen3.8-27B a GDN
    // state page is 48.94 MiB against a 96 MiB parent, and the recurrent groups
    // hold 206 of 346 parents -- 60 of every 68 of them retained snapshots.
    // Giving those groups an arena whose parent IS their page takes both
    // utilizations to 1.00 and full attention from 188,928 to ~369,024 tokens
    // of prefix cache.
    //
    // Admission has to fit each arena separately: a free parent in one is not
    // available to a group in the other.
    AdmissionPlanner(const std::vector<CacheGroup>& groups, std::vector<const BlockPool*> pools,
                     std::vector<std::size_t> pool_of_group, std::span<const GroupDemand> demands,
                     const KvCacheCoordinator::PrefixProbe& prefix,
                     std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases,
                     std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims)
        : groups_{groups},
          pools_{std::move(pools)},
          pool_of_group_{std::move(pool_of_group)},
          demands_{demands},
          prefix_{prefix},
          pending_store_releases_{pending_store_releases},
          victims_{victims},
          local_free_slots_(groups.size()),
          blocks_needed_(groups.size()) {
        _assert(!pools_.empty(), "admission needs at least one device pool");
        _assert(pool_of_group_.size() == groups.size(), "pool map must cover every group");
        empty_parent_count_.assign(pools_.size(), 0);
        remaining_occupied_.resize(pools_.size());
        for (std::size_t p = 0; p < pools_.size(); ++p) {
            remaining_occupied_[p].assign(static_cast<std::size_t>(pools_[p]->NumLcmBlocks()) + 1, 0);
        }
    }

    // Where refusals are tallied; index 0 is the shared arena.
    void RecordShortagesInto(std::int64_t* counters) { short_arena_ = counters; }

    bool Plan() {
        victims_.clear();
        initializeCapacity();

        // Existing local holes plus empty parents are the zero-eviction
        // capacity. Do not discard cache merely to make placement denser.
        if (fits()) {
            return true;
        }

        collectCandidates();
        while (!fits()) {
            if (victim_candidates_.empty()) {
                // fits() sums a per-group ceiling against only the fully empty
                // parents, so a request needing one block in each of many
                // groups can be refused while bound parents still hold free
                // slots. Record both sides once per refusal to tell that apart
                // from a genuine shortage.
                static const bool trace =
                    std::fopen("/workspace/roy/.plan_trace.on", "r") != nullptr;
                if (trace) {
                    static std::FILE* f = std::fopen("/workspace/roy/.fits_trace.txt", "a");
                    if (f != nullptr) {
                        std::int64_t need = 0, free_slots = 0, blocks = 0;
                        for (std::size_t i = 0; i < groups_.size(); ++i) {
                            const std::int64_t remaining =
                                std::max<std::int64_t>(blocks_needed_[i] - local_free_slots_[i], 0);
                            const std::int64_t slots = groups_[i].Manager().CacheBlocksPerLcmBlock();
                            need += (remaining + slots - 1) / slots;
                            free_slots += local_free_slots_[i];
                            blocks += blocks_needed_[i];
                        }
                        std::fprintf(f,
                                     "refuse groups=%zu blocks_needed=%lld free_slots=%lld "
                                     "parents_needed=%lld empty_parents=%d |",
                                     groups_.size(), static_cast<long long>(blocks),
                                     static_cast<long long>(free_slots),
                                     static_cast<long long>(need), empty_parent_count_.empty() ? 0 : empty_parent_count_[0]);
                        for (std::size_t i = 0; i < groups_.size(); ++i) {
                            std::fprintf(f, " g%zu(need=%lld free=%lld slots=%d)", i,
                                         static_cast<long long>(blocks_needed_[i]),
                                         static_cast<long long>(local_free_slots_[i]),
                                         groups_[i].Manager().CacheBlocksPerLcmBlock());
                        }
                        std::fprintf(f, "\n");
                        std::fflush(f);
                    }
                }
                return false;
            }
            std::ranges::pop_heap(victim_candidates_, evictedAfter);
            const VictimCandidate candidate = victim_candidates_.back();
            victim_candidates_.pop_back();
            removeOccupant(candidate.group_id, candidate.location);
            victims_.emplace_back(candidate.group_id, candidate.location);
        }

        // Once removing an eviction prefix fits, keeping the entire unpopped
        // tail also fits. Tentatively restore the prefix newest-first using
        // only the planner's shadow occupancy. If a restore makes admission
        // infeasible, undo it and keep that block as a required victim.
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> required_victims;
        required_victims.reserve(victims_.size());
        for (std::size_t i = victims_.size(); i > 0; --i) {
            const auto& [group_id, location] = victims_[i - 1];
            restoreOccupant(group_id, location);
            if (!fits()) {
                removeOccupant(group_id, location);
                required_victims.emplace_back(group_id, location);
            }
        }
        std::ranges::reverse(required_victims);
        victims_ = std::move(required_victims);
        return true;
    }

private:
    // Current prefix hits are protected before candidates reach this policy.
    // A request-only block with no CacheEntry is reclaimed first. Cached
    // entries then compare request access epoch, followed within one epoch by
    // the tier order below. Position keeps the deeper unproven non-closed
    // boundary, while a closed prefix is reclaimed from its suffix.
    enum class EvictionTier {
        kUncached,  // physically allocated, but owned only by the request table
        kProbationaryBoundary,
        kEstablishedBoundary,
        kClosedPrefix,
    };

    struct VictimCandidate {
        std::uint32_t group_id;
        CacheBlockLocation location;
        std::uint64_t last_access_epoch;
        EvictionTier eviction_tier;
        std::int64_t position_rank;
    };

    // Depth-before-recency, opt-in, default off.
    //
    // The default below is LRU, and LRU has a worst case this workload sits in.
    // The agent corpora interleave N sessions round-robin, so a session's next
    // turn arrives after N-1 other requests. Measured on gemma-4-31B swe_smith
    // (32 sessions, 10,084 tokens each): surviving one reuse distance needs
    // 312,912 tokens against a 172,928-token pool. LRU keeps the most recent
    // ~16 sequences WHOLE and drops the rest WHOLE, and the reuse distance is
    // 32, so every reuse misses by construction. The engine's own probe agrees:
    // 87% of matches converge at exactly 1024 tokens -- the cross-session
    // system prompt and nothing else. 1024/10,084 = 10.2% against a measured
    // hit rate of 9-11%. Reordering the same corpus so a session's turns are
    // adjacent, with no engine change at all, takes it to 74.0%.
    //
    // Ordering depth first inverts the failure: under pressure the deep suffix
    // of every sequence is reclaimed before the shallow prefix of any of them,
    // so all N sessions keep a prefix instead of 16 keeping everything. The
    // equilibrium depth is capacity/N and needs no tuning parameter.
    //
    // It is off by default because it is a POLICY change, not a fix. Recency
    // wins when there is temporal locality; depth wins when access is cyclic
    // and the working set exceeds capacity. Six tests in
    // test_kv_cache_coordinator.cpp assert the recency contract by name
    // (`FullEvictionPrefersOlderRequestBeforePosition`,
    // `StateEvictionUsesAccessEpochBeforePosition`, ...) and they keep passing:
    // they do not set the variable. Flipping the default is a decision that
    // needs both workload shapes measured, which this switch is what makes
    // possible.
    static bool depthFirstEviction() {
        static const bool on = std::getenv("TOKENSPEED_EVICT_DEPTH_FIRST") != nullptr;
        return on;
    }

    static auto evictionKey(const VictimCandidate& candidate) {
        // Recency first, tier second. Tried the reverse on 2026-08-21 to keep
        // resumable SWA boundaries alive: gemma-4-31B agent-26k c=1, median
        // TTFT 3547.73 -> 5335.39 ms (+50.4%), P99 9085.58 -> 9456.02 against
        // a 39 ms spread over three runs of the same config. Reverted.
        //
        // The order decides the FIRST victim as much as the last, and
        // kUncached means "allocated, owned only by the request table" -- a
        // page an in-flight request is writing into. Promoting tier made every
        // one of those the first thing evicted, and the recompute cost more
        // than the retained boundaries saved. Keeping older boundaries alive
        // needs a mechanism that does not also reorder the live set: refresh
        // the epoch of a boundary when a match lands on it, or exclude
        // kUncached from candidacy, not a global re-sort.
        //
        // `last_access_epoch == 0` is an uncached block -- a page an in-flight
        // request is writing into. Under the depth-first policy those are
        // partitioned out FIRST and keep their existing order among themselves,
        // so the live set is not reordered. That is the distinction the
        // 2026-08-21 attempt missed: it re-sorted everything and paid +50.4%
        // median TTFT for it.
        const std::int64_t live = candidate.last_access_epoch == 0 ? 0 : 1;
        const std::int64_t depth =
            (depthFirstEviction() && live == 1) ? candidate.position_rank : std::int64_t{0};
        return std::tuple{live,
                          depth,
                          candidate.last_access_epoch,
                          candidate.eviction_tier,
                          candidate.position_rank,
                          candidate.group_id,
                          candidate.location.lcm_block_id,
                          candidate.location.slot_index};
    }
    static bool evictedAfter(const VictimCandidate& lhs, const VictimCandidate& rhs) {
        return evictionKey(rhs) < evictionKey(lhs);
    }

    void initializeCapacity() {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            const KvCacheManager& manager = groups_[i].Manager();
            // The admission view, which a sparse group narrows to the blocks
            // it will actually back. Acquire allocates exactly this many, so
            // the two stay consistent and the Admit-time FatalCheck holds.
            // Admit() runs ClaimHitBlocks then AppendHostExtension before it
            // acquires, so this group's own pages begin that far along the
            // table. Both tiers contribute positions, not just backed pages:
            // a hole still occupies a slot and still shifts the checkpoints.
            const std::span<const std::uint8_t> device_hits =
                prefix_.device.num_common_tokens > 0 && !prefix_.device.per_group.empty()
                    ? std::span<const std::uint8_t>(prefix_.device.per_group[i].hits)
                    : std::span<const std::uint8_t>{};
            const std::span<const std::uint8_t> host_hits =
                prefix_.host.per_group.empty() ? std::span<const std::uint8_t>{}
                                               : std::span<const std::uint8_t>(prefix_.host.per_group[i].hits);
            const std::int32_t host_base =
                static_cast<std::int32_t>(demand.table->Blocks().size() + device_hits.size());
            const std::int32_t host_blocks = manager.BackedHostExtensionPages(host_base, host_hits);
            const std::int32_t device_blocks =
                demand.materialized_suffix_start < 0
                    ? manager.AdmissionBlocksNeededFor(*demand.table, demand.num_tokens + demand.reserve_tokens,
                                                       host_base + static_cast<std::int32_t>(host_hits.size()))
                    // The sparse view, same as the branch above. A
                    // checkpointing group allocates one page per stride plus
                    // the live tail, so charging the dense span refuses
                    // requests over capacity it will never occupy: 78 pages on
                    // a 10k prompt where Acquire takes 2. Invisible while the
                    // recurrent groups share the 346-parent arena, which
                    // absorbs it; fatal once they have their own, where the
                    // first prefix-hit request cannot be admitted and
                    // #running-req sits at 1. ddefef3a fixed the other branch
                    // and left this one.
                    : manager.CapAdmissionPages(static_cast<std::int32_t>(
                          manager.BackedPagesFor(manager.BlocksNeededFor(*demand.table, demand))));
            blocks_needed_[i] = static_cast<std::int64_t>(device_blocks) + host_blocks;
        }

        for (std::size_t p = 0; p < pools_.size(); ++p) {
            const BlockPool& pool = *pools_[p];
            for (std::int32_t parent_id = 1; parent_id <= pool.NumLcmBlocks(); ++parent_id) {
                const std::optional<std::uint32_t> group_id = pool.BoundGroup(parent_id);
                if (!group_id) {
                    ++empty_parent_count_[p];
                    continue;
                }

                _assert(*group_id < groups_.size(), "LCM parent has invalid group binding");
                _assert(poolIndex(*group_id) == p, "a parent is bound to a group from another arena");
                const std::int32_t occupied = pool.OccupiedCount(parent_id);
                const std::int32_t slots = groups_[*group_id].Manager().CacheBlocksPerLcmBlock();
                _assert(0 < occupied && occupied <= slots, "bound LCM parent has invalid occupancy");
                remaining_occupied_[p][static_cast<std::size_t>(parent_id)] = occupied;
                local_free_slots_[*group_id] += slots - occupied;
            }
        }
    }

    void collectCandidates() {
        // Keyed by (arena, location). A parent id is only unique within its own
        // arena, so with more than one arena a bare location aliases: the same
        // (parent, slot) in two state arenas collides, and the second group's
        // page is dropped as a duplicate -- leaving that group with no eviction
        // candidates and admission refusing at #running-req 1.
        using ArenaLocation = std::pair<std::size_t, CacheBlockLocation>;
        struct ArenaLocationHash {
            std::size_t operator()(const ArenaLocation& value) const {
                return CacheBlockLocationHash{}(value.second) * 1000003U + value.first;
            }
        };
        std::unordered_set<ArenaLocation, ArenaLocationHash> protected_locations;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::vector<CacheBlockLocation> hits = groups_[i].Manager().MatchedBlockLocations(
                poolFor(static_cast<std::uint32_t>(i)), prefix_.group_keys[i], /*begin_blocks=*/0,
                prefix_.device.per_group[i]);
            for (CacheBlockLocation hit : hits) {
                protected_locations.insert({poolIndex(static_cast<std::uint32_t>(i)), hit});
            }
        }

        std::unordered_set<ArenaLocation, ArenaLocationHash> candidates;
        const auto add_candidate = [&](std::uint32_t group_id, CacheBlockLocation location) {
            const ArenaLocation keyed{poolIndex(group_id), location};
            if (protected_locations.contains(keyed) || !candidates.insert(keyed).second) {
                return;
            }
            const KvCacheManager& manager = groups_[group_id].Manager();
            const std::optional<KvCacheManager::CachedBlockMetadata> metadata =
                manager.CachedBlockMetadataFor(poolFor(group_id), location);
            const std::uint64_t last_access_epoch = metadata ? metadata->last_access_epoch : 0;
            const std::int32_t logical_block_index = metadata ? metadata->logical_block_index : -1;
            const CacheBoundaryKind boundary_kind = metadata ? metadata->boundary_kind : CacheBoundaryKind::kChunk;
            const bool is_prefix_closed = manager.MatchIsPrefixClosed();
            const bool is_probationary_boundary = !is_prefix_closed && boundary_kind == CacheBoundaryKind::kChunk &&
                                                  !(metadata && metadata->was_acquired) && logical_block_index >= 0;
            const EvictionTier eviction_tier = [&] {
                if (last_access_epoch == 0) {
                    return EvictionTier::kUncached;
                }
                if (is_probationary_boundary) {
                    return EvictionTier::kProbationaryBoundary;
                }
                return is_prefix_closed ? EvictionTier::kClosedPrefix : EvictionTier::kEstablishedBoundary;
            }();
            std::int64_t position_rank = 0;
            if (is_probationary_boundary) {
                // Retain the longer unproven frontier.
                position_rank = logical_block_index;
            } else if (is_prefix_closed && logical_block_index >= 0) {
                // Reclaim a closed prefix from its suffix.
                position_rank = -static_cast<std::int64_t>(logical_block_index);
            }
            victim_candidates_.push_back(VictimCandidate{
                .group_id = group_id,
                .location = location,
                // Access epochs start at one. Zero puts an uncached block
                // ahead of every reusable cache entry.
                .last_access_epoch = last_access_epoch,
                .eviction_tier = eviction_tier,
                .position_rank = position_rank,
            });
        };

        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::uint32_t group_id = static_cast<std::uint32_t>(i);
            const KvCacheManager& manager = groups_[i].Manager();
            std::vector<CacheBlockLocation> group_pending_store_releases;
            for (const auto& [released_group_id, location] : pending_store_releases_) {
                if (released_group_id == group_id) {
                    group_pending_store_releases.push_back(location);
                }
            }
            for (CacheBlockLocation location :
                 manager.EvictableBlockLocationsAfterReleasing(poolFor(group_id), group_pending_store_releases)) {
                add_candidate(group_id, location);
            }
            if (demands_[i].num_computed_tokens >= 0) {
                for (CacheBlockLocation location : manager.ReclaimableBlockLocationsAt(
                         *demands_[i].table, demands_[i].num_computed_tokens, group_pending_store_releases)) {
                    add_candidate(group_id, location);
                }
            }
        }
        std::ranges::make_heap(victim_candidates_, evictedAfter);
    }

    void removeOccupant(std::uint32_t group_id, CacheBlockLocation location) {
        _assert(poolFor(group_id).BoundGroup(location.lcm_block_id) == group_id,
                "released admission location belongs to another group");
        const std::size_t p = poolIndex(group_id);
        std::int32_t& occupied = remaining_occupied_[p][static_cast<std::size_t>(location.lcm_block_id)];
        _assert(occupied > 0, "admission released the same location twice");
        const std::int32_t slots = groups_[group_id].Manager().CacheBlocksPerLcmBlock();
        if (occupied == 1) {
            local_free_slots_[group_id] -= slots - 1;
            occupied = 0;
            ++empty_parent_count_[p];
        } else {
            --occupied;
            ++local_free_slots_[group_id];
        }
    }

    void restoreOccupant(std::uint32_t group_id, CacheBlockLocation location) {
        const std::size_t p = poolIndex(group_id);
        std::int32_t& occupied = remaining_occupied_[p][static_cast<std::size_t>(location.lcm_block_id)];
        const std::int32_t slots = groups_[group_id].Manager().CacheBlocksPerLcmBlock();
        if (occupied == 0) {
            _assert(empty_parent_count_[p] > 0, "restoring an admission victim underflowed empty parents");
            --empty_parent_count_[p];
            occupied = 1;
            local_free_slots_[group_id] += slots - 1;
        } else {
            _assert(occupied < slots, "restoring an admission victim overflowed its parent");
            ++occupied;
            --local_free_slots_[group_id];
        }
    }

    bool fits() const {
        // Per arena: a free parent in one is not available to a group in the
        // other, so a single sum would admit a request neither arena can hold.
        std::vector<std::int64_t> parents_needed(pools_.size(), 0);
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t remaining = std::max<std::int64_t>(blocks_needed_[i] - local_free_slots_[i], 0);
            const std::int64_t slots = groups_[i].Manager().CacheBlocksPerLcmBlock();
            parents_needed[poolIndex(static_cast<std::uint32_t>(i))] += (remaining + slots - 1) / slots;
        }
        for (std::size_t p = 0; p < pools_.size(); ++p) {
            if (parents_needed[p] > empty_parent_count_[p]) {
                // Which arena is short, so the split can be derived instead of
                // swept. The right split is a function of the workload, not a
                // constant: on this model a state page is 48.94 MiB and an
                // attention page 8 MiB, so the state arena has to cover
                // concurrency x live pages per request while the attention one
                // covers the working set. Measured, no single value serves
                // both -- agent c=16 wants 112 state pages per group and gets
                // 0.55x at 180, RAG c=128 wants 180 and gets 0.69x at 112.
                if (short_arena_ != nullptr) {
                    ++short_arena_[p];
                }
                return false;
            }
        }
        return true;
    }

    const std::vector<CacheGroup>& groups_;
    std::vector<const BlockPool*> pools_;
    // Optional per-arena refusal counter, owned by the coordinator.
    std::int64_t* short_arena_{nullptr};
    std::vector<std::size_t> pool_of_group_;
    const BlockPool& poolFor(std::uint32_t group_id) const { return *pools_[pool_of_group_[group_id]]; }
    std::size_t poolIndex(std::uint32_t group_id) const { return pool_of_group_[group_id]; }
    std::span<const GroupDemand> demands_;
    const KvCacheCoordinator::PrefixProbe& prefix_;
    // These locations are still pinned by in-flight Store tickets. The planner only discounts those ticket refs to
    // decide whether waiting for Store ACK makes admission feasible; it never mutates the real pool.
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases_;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims_;
    std::vector<std::vector<std::int32_t>> remaining_occupied_;
    std::vector<std::int64_t> local_free_slots_;
    std::vector<std::int64_t> blocks_needed_;
    std::vector<std::int64_t> empty_parent_count_;
    std::vector<VictimCandidate> victim_candidates_;
};

std::optional<AdmissionPlan> planAdmission(const std::vector<CacheGroup>& groups,
                                           std::vector<const BlockPool*> pools,
                                           std::vector<std::size_t> pool_of_group,
                                           KvCacheCoordinator::PrefixProbe&& prefix,
                                           std::span<const GroupDemand> demands, std::int64_t* shortages) {
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
    const std::vector<const BlockPool*> planner_pools = pools;
    AdmissionPlanner planner{groups, std::move(pools), std::move(pool_of_group), demands, prefix, {}, victims};
    planner.RecordShortagesInto(shortages);
    if (!planner.Plan()) {
        // `TOKENSPEED_LOG_ADMISSION=1`: name the group that could not be
        // satisfied. In-flight concurrency stalls well below the client's
        // level while KV still has thousands of blocks free, and nothing in
        // the log says which pool actually refuses -- the request simply is
        // not scheduled.
        static const bool trace = std::getenv("TOKENSPEED_LOG_ADMISSION") != nullptr;
        if (trace) {
            std::string per_group;
            for (std::size_t i = 0; i < groups.size(); ++i) {
                per_group += (i ? " " : "");
                per_group += "g" + std::to_string(i) + "=" + std::to_string(demands[i].num_tokens) + "tok";
            }
            std::string empties;
            for (std::size_t p = 0; p < planner_pools.size(); ++p) {
                empties += (p ? "/" : "") + std::to_string(planner_pools[p]->NumEmptyLcmBlocks());
            }
            spdlog::info("[ADMIT-REJECT] empty_parents={} demands: {}", empties, per_group);
        }
        return std::nullopt;
    }
    return AdmissionPlan{.prefix = std::move(prefix), .victims = std::move(victims)};
}

}  // namespace

bool KvCacheCoordinator::CanAdmitAfterReleasing(
    const PrefixProbe& prefix, std::span<const GroupDemand> demands,
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases) const {
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
    std::vector<const BlockPool*> pools;
    std::vector<std::size_t> pool_of_group(groups_.size());
    devicePoolViews(pools, pool_of_group);
    AdmissionPlanner planner{groups_,  std::move(pools),      std::move(pool_of_group), demands,
                             prefix,   pending_store_releases, victims};
    return planner.Plan();
}

std::int32_t KvCacheCoordinator::PromotionBoundaryTokens(const PrefixProbe& prefix) const {
    const std::int32_t matched_tokens = std::max(prefix.device.num_common_tokens, prefix.host.num_common_tokens);
    const std::int32_t prefix_closed_tokens =
        std::max(prefix.device.prefix_closed_tokens, prefix.host.prefix_closed_tokens);
    return prefix_closed_tokens > matched_tokens ? prefix_closed_tokens : 0;
}

std::optional<KvCacheCoordinator::AdmissionResult> KvCacheCoordinator::Admit(
    PrefixProbe&& prefix, std::span<const GroupDemand> demands, std::optional<std::uint64_t> request_access_epoch) {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    for (const GroupDemand& demand : demands) {
        _assert(demand.table != nullptr, "group demand requires a block table");
        _assert(demand.new_page_hash_begin >= 0 &&
                    static_cast<std::size_t>(demand.new_page_hash_begin) <= demand.page_hashes.size(),
                "new page hash begin is outside the hash history");
        const bool has_new_page_hashes =
            static_cast<std::size_t>(demand.new_page_hash_begin) < demand.page_hashes.size();
        _assert(demand.completed_boundary_kind.has_value() == has_new_page_hashes,
                "completed boundary kind must match newly completed page hashes");
    }

    std::vector<const BlockPool*> plan_pools;
    std::vector<std::size_t> plan_pool_of_group(groups_.size());
    devicePoolViews(plan_pools, plan_pool_of_group);
    arena_shortages_.resize(plan_pools.size(), 0);
    std::optional<AdmissionPlan> candidate =
        planAdmission(groups_, std::move(plan_pools), std::move(plan_pool_of_group), std::move(prefix), demands,
                      arena_shortages_.data());
    if (!candidate) {
        return std::nullopt;
    }
    AdmissionPlan plan = std::move(*candidate);

    if (request_access_epoch.has_value()) {
        _assert(*request_access_epoch > 0 && *request_access_epoch <= next_access_epoch_,
                "request access epoch was not issued by this coordinator");
    }
    const std::uint64_t access_epoch = request_access_epoch.has_value() ? *request_access_epoch : ++next_access_epoch_;
    const std::int32_t promotion_boundary_tokens = PromotionBoundaryTokens(plan.prefix);
    AcquiredPrefix acquired_prefix = acquirePrefix(std::move(plan.prefix), access_epoch);
    AdmissionResult result{
        .device_prefix_tokens = acquired_prefix.device.num_common_tokens,
        .host_prefix_tokens = acquired_prefix.host.num_common_tokens,
        .promotion_boundary_tokens = promotion_boundary_tokens,
        .access_epoch = access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
    };
    if (acquired_prefix.device.num_common_tokens > 0) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            groups_[i].Manager().ClaimHitBlocks(*demands[i].table, std::move(acquired_prefix.device.per_group[i]));
        }
    }
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> prospective_victims;
    prospective_victims.reserve(plan.victims.size());
    // A reclaimable table block may still be pinned by both the request and
    // cache here. Evict what is already free, then slide the request tables and
    // retry the blocks whose request reference has just been released.
    for (const auto& victim : plan.victims) {
        const auto& [group_id, location] = victim;
        if (!evictCachedBlock(group_id, location)) {
            prospective_victims.push_back(victim);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = demands[i];
        if (demand.completed_boundary_kind) {
            cacheDeviceCompletedBlocksForGroup(i, demand, access_epoch);
        }
        if (demand.num_computed_tokens >= 0) {
            groups_[i].Manager().ReclaimExpired(PoolForGroup(i), *demand.table, demand.num_computed_tokens);
        }
    }
    for (const auto& [group_id, location] : prospective_victims) {
        if (!evictCachedBlock(group_id, location)) {
            FatalCheck(!PoolForGroup(group_id).IsOccupied(location), "admission victim changed before acquisition");
        }
    }

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = demands[i];
        if (!acquired_prefix.host.per_group.empty() && !acquired_prefix.host.per_group[i].blocks.empty()) {
            groups_[i].Manager().AppendHostExtension(
                PoolForGroup(i), *demand.table, std::move(acquired_prefix.host.per_group[i].blocks),
                result.load_pairs);
        }
        const std::int32_t first_new_block = demand.table->NumBlocks();
        const bool acquired = groups_[i].Manager().Acquire(PoolForGroup(i), *demand.table, demand);
        FatalCheck(acquired, "admission plan no longer fits the block pool");
        const std::span<const CacheBlockRef> blocks = demand.table->Blocks();
        for (std::int32_t block = first_new_block; block < demand.table->NumBlocks(); ++block) {
            if (!blocks[static_cast<std::size_t>(block)]) {
                continue;
            }
            result.new_page_ids[i].push_back(
                groups_[i].Manager().ResolveKernelPageId(blocks[static_cast<std::size_t>(block)]->Location()));
        }
    }
    return result;
}

}  // namespace tokenspeed
