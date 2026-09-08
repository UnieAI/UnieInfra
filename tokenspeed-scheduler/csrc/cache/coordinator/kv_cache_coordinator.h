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
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "cache/manager/cache_group.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

struct KvCacheCoordinatorTestAccess;

enum class CacheTier { kDevice, kHost };

// num_common_tokens is in tokens at the one shared CacheBlock granularity P.
// per_group[i] is group i's PrefixMatch at exactly that length.
struct CoordinatorMatch {
    std::int32_t num_common_tokens{0};
    std::vector<PrefixMatch> per_group;
};

// Multi-group fan-out over the per-attention managers, one shared BlockPool. Holds no per-request
// state; the request access clock is global, while each request carries its issued epoch.
class KvCacheCoordinator {
public:
    enum class CacheMutation { kStored, kRemoved };
    using CacheMutationSink = std::function<void(const CacheKey&, CacheMutation)>;

    // The Host pool is available to explicit tier operations. Streaming controls
    // whether ordinary Device prefix publication also feeds the Host tier.
    KvCacheCoordinator(std::vector<CacheGroup> groups, std::int32_t cache_block_tokens, BlockPool& pool,
                       BlockPool* host_pool = nullptr, bool stream_device_cache_to_host = true,
                       std::vector<BlockPool*> state_pools = {});

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    std::int32_t CacheBlockTokens() const noexcept { return cache_block_tokens_; }
    // Extra resume points inside a step's newly completed range, in cache
    // blocks. 0 publishes only the run ending at the boundary. See
    // SchedulerConfig::boundary_stride_blocks for why this is a knob.
    void SetBoundaryStrideBlocks(std::int32_t blocks) noexcept { boundary_stride_blocks_ = blocks; }
    bool HasMambaStateGroup() const;

    const BlockPool& DevicePool() const { return pool_; }

    // The device pool a group draws from.
    //
    // One arena today, so every group gets the same one and this is an
    // identity. It exists because the arena's parent is sized by the widest
    // geometry family and a group whose page is narrower than that wastes the
    // difference on every parent it binds: on Qwen3.8-27B a GDN state page is
    // 48.94 MiB against a 96 MiB parent and packs 1, so the three recurrent
    // groups take 206 of 346 parents -- 60 of every 68 of them retained
    // snapshots -- and leave full attention 188,928 tokens of prefix cache
    // against vLLM's 447,146. No packing fixes it: the parent is the sum of the
    // attention family's planes, which are twice the recurrent family's (k and
    // v per layer against one state), so the ratio is 16/32 for every packing
    // and every parent size. A second pool whose parent IS the state page is
    // the smallest thing that does.
    BlockPool& PoolForGroup(std::size_t group_index) {
        const std::optional<std::size_t> slot = statePoolSlot(group_index);
        return slot ? *state_pools_[*slot] : pool_;
    }
    const BlockPool& PoolForGroup(std::size_t group_index) const {
        const std::optional<std::size_t> slot = statePoolSlot(group_index);
        return slot ? *state_pools_[*slot] : pool_;
    }
    // Distinct device pools this coordinator draws from, in a stable order.
    // Admission has to fit each one separately: a parent is only ever a page of
    // its own arena, so a shortage in one says nothing about the other.
    // Refusals attributed to each arena, index 0 being the shared one.
    const std::vector<std::int64_t>& ArenaShortages() const { return arena_shortages_; }
    std::size_t DevicePoolIndexForGroup(std::size_t group_index) const {
        const std::optional<std::size_t> slot = statePoolSlot(group_index);
        return slot ? *slot + 1U : 0U;
    }
    // The arenas and the group->arena map admission plans against.
    void devicePoolViews(std::vector<const BlockPool*>& pools, std::vector<std::size_t>& pool_of_group) const {
        pools.clear();
        pools.push_back(&pool_);
        for (const BlockPool* pool : state_pools_) {
            pools.push_back(pool);
        }
        pool_of_group.resize(groups_.size());
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            pool_of_group[i] = DevicePoolIndexForGroup(i);
        }
    }
    KvCacheManager& GroupManager(std::int32_t i) { return groups_[static_cast<std::size_t>(i)].Manager(); }
    const KvCacheManager& GroupManager(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Manager(); }
    AttnKind GroupKind(std::int32_t i) const { return groups_[static_cast<std::size_t>(i)].Spec().kind; }

    struct PrefixProbe {
        struct Tier {
            std::int32_t num_common_tokens{0};
            // Common coverage after all prefix-closed groups, before a
            // window/state group shortens the resumable boundary.
            std::int32_t prefix_closed_tokens{0};
            std::vector<GroupPrefixProbe> per_group;
        };

        std::vector<std::vector<CacheKey>> group_keys;
        Tier device;
        Tier host;
    };
    struct AdmissionResult {
        std::int32_t device_prefix_tokens{0};
        std::int32_t host_prefix_tokens{0};
        // Longer prefix-closed coverage worth materializing for non-closed groups.
        std::int32_t promotion_boundary_tokens{0};
        std::uint64_t access_epoch{0};
        std::vector<BlockTransfer> load_pairs;
        // Fresh device child pages appended by ordinary Acquire, aligned by
        // group_id. Cache hits and host-loaded destinations are excluded.
        std::vector<std::vector<std::int32_t>> new_page_ids;
    };

    // ProbePrefix is read-only. Cache state must not change before its
    // result is passed to Admit. Admit leaves the probe intact when capacity is
    // unavailable so the caller may perform a hypothetical-release check.
    // A missing epoch starts a new request; a supplied epoch continues that
    // request. Once commit starts, an internal plan/pool mismatch is fatal
    // because partial commit is not rolled back.
    PrefixProbe ProbePrefix(std::span<const std::string> content_hashes) const;
    // Decode-side PD reuses local history pages, while final-state groups are
    // restored from the remote endpoint snapshot. Their aligned null holes do
    // not count as cache hits.
    PrefixProbe ProbeDecodeDevicePrefix(std::span<const std::string> content_hashes) const;
    std::int32_t PromotionBoundaryTokens(const PrefixProbe& prefix) const;
    std::optional<AdmissionResult> Admit(PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                         std::optional<std::uint64_t> request_access_epoch = std::nullopt);
    bool CanAdmitAfterReleasing(
        const PrefixProbe& prefix, std::span<const GroupDemand> demands,
        std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases) const;
    // Number of physical parents that become reclaimable after dropping the
    // exact request-owned refs in tables. Used only to rank Retraction victims.
    std::int32_t NumNewlyReleasableLcmBlocks(std::span<const BlockTable> tables) const;

    std::int32_t NumAvailableLcmBlocks() const;

    // Registers an exact range, used for transferred prefix blocks and tests.
    // Runtime publication during Admit follows each manager's boundary contract.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::uint64_t access_epoch, std::int32_t first_slot = 0,
                         CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk);
    void CacheCompletedBlocks(std::span<BlockTable> tables, std::span<const std::string> page_hashes,
                              std::uint64_t access_epoch, std::int32_t first_new_page, std::int32_t num_computed_tokens,
                              CacheBoundaryKind boundary_kind);
    void ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens);
    void ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens);
    void Free(std::span<BlockTable> tables);
    // Clears only the Device prefix index. Returns false without mutation when
    // any cached block still has an owner outside its Manager.
    bool ClearDeviceCache();
    // Clears both Device and Host prefix indexes. Returns false without
    // mutation when either tier still has a pinned cached block.
    bool ClearCache();

    struct StoreCandidate {
        CacheKey key;
    };
    // Retry ordinary D2H Store for already-published Device cache entries.
    // Missing keys and an absent Host tier are silently skipped.
    void QueueCachedBlocksForStore(std::span<const std::string> page_hashes);
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }
    CacheBlockRef AcquireDeviceCachedBlock(const CacheKey& key) const;
    CacheBlockRef AcquireHostBlock(std::uint32_t group_id);
    // Collection/pinning follows host-tier presence, so the slide credit flips count_uncached on this.
    bool StreamsDeviceCacheToHost() const { return stream_device_cache_to_host_; }
    bool ContainsHostCachedBlock(const CacheKey& key) const;
    bool IsHostCachedBlock(CacheBlockLocation location) const;
    std::int32_t NumHostCachedBlocks() const;
    std::int32_t NumPinnedHostCachedBlocks() const;
    void CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key);

    // Reports real device-cache entry insertions and removals. The scheduler
    // folds the per-group mutations into one externally visible prefix event.
    void SetCacheMutationSink(CacheMutationSink sink) { cache_mutation_sink_ = std::move(sink); }

private:
    friend struct KvCacheCoordinatorTestAccess;

    struct AcquiredPrefix {
        CoordinatorMatch device;
        CoordinatorMatch host;
    };

    std::vector<CacheKey> keysForGroup(std::span<const std::string> content_hashes, std::uint32_t group_id) const;
    std::vector<std::vector<CacheKey>> buildGroupKeys(std::span<const std::string> content_hashes) const;
    // Device tier resolves per group; the Host tier is one pool for all.
    template <CacheTier Tier>
    BlockPool& tierPool(std::size_t group_index);
    template <CacheTier Tier>
    const BlockPool& tierPool(std::size_t group_index) const;
    template <CacheTier Tier>
    BlockPool& tierPool();
    template <CacheTier Tier>
    const BlockPool& tierPool() const;
    template <CacheTier Tier>
    PrefixProbe::Tier probeTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                        std::span<const std::size_t> match_order, std::int32_t num_cache_blocks,
                                        std::int32_t floor_tokens) const;
    template <CacheTier Tier>
    CoordinatorMatch acquireTierWithKeys(std::span<const std::vector<CacheKey>> group_keys, std::int32_t floor_tokens,
                                         PrefixProbe::Tier&& probe, std::uint64_t access_epoch);
    AcquiredPrefix acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch);
    template <CacheTier Tier>
    void cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table, std::span<const CacheKey> keys,
                                 std::int32_t first_cache_block, std::uint64_t access_epoch,
                                 CacheBoundaryKind boundary_kind);
    template <CacheTier Tier>
    void cacheCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand, std::uint64_t access_epoch);
    void cacheDeviceCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand,
                                            std::uint64_t access_epoch);
    bool evictCachedBlock(std::uint32_t group_id, CacheBlockLocation location);
    std::vector<CacheGroup> groups_;
    // Closed groups first, so non-closed groups match against a settled bound.
    std::vector<std::size_t> match_order_;
    BlockPool& pool_;
    // One arena per recurrent-state group, in group order. Empty keeps every
    // group on the one arena, which is byte-for-byte the old behaviour.
    std::vector<BlockPool*> state_pools_;
    // How often each arena was the one that could not be satisfied. The right
    // split between them is a function of the workload -- measured, agent c=16
    // wants 112 state pages per group and collapses to 0.55x at 180, while RAG
    // c=128 wants 180 and sits at 0.69x with 112 -- so it cannot be a constant,
    // and this is the signal a rebalance would act on.
    mutable std::vector<std::int64_t> arena_shortages_;
    // Which of them this group draws from, if any.
    std::optional<std::size_t> statePoolSlot(std::size_t group_index) const {
        if (state_pools_.empty() || groups_[group_index].Spec().kind != AttnKind::kMambaState) {
            return std::nullopt;
        }
        std::size_t slot = 0;
        for (std::size_t i = 0; i < group_index; ++i) {
            if (groups_[i].Spec().kind == AttnKind::kMambaState) {
                ++slot;
            }
        }
        _assert(slot < state_pools_.size(), "more recurrent groups than state arenas");
        return slot;
    }
    BlockPool* host_pool_{nullptr};
    bool stream_device_cache_to_host_{false};
    std::int32_t cache_block_tokens_{0};
    std::int32_t boundary_stride_blocks_{0};
    std::uint64_t next_access_epoch_{0};
    std::vector<StoreCandidate> pending_stores_;
    CacheMutationSink cache_mutation_sink_;
};

// One CacheGroup per spec (group_id = index), sharing one scheduler prefix
// domain P while each manager may use a smaller cache-page token count.
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, std::int32_t cache_block_tokens, BlockPool& pool,
                                   BlockPool* host_pool = nullptr, bool stream_device_cache_to_host = true,
                                   std::vector<BlockPool*> state_pools = {});

}  // namespace tokenspeed
