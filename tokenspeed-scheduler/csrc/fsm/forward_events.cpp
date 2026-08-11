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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/token_container.h"
#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "fsm/pd_states.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/radix_tree/node_range.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"

#if TOKENSPEED_FLAT_KVCACHE
#include "cache/forward_cache_ops.h"
#include "scheduler/page_hasher.h"
#endif

namespace {

std::vector<tokenspeed::TransferPair> BuildWriteBackPairs(const std::vector<tokenspeed::TreeNode*>& write_diff) {
    std::vector<tokenspeed::TransferPair> pages_to_transfer;
    for (tokenspeed::TreeNode* n : write_diff) {
        const auto& dev_pages = n->Device().Pages();
        const auto& host_pages = n->Host().Pages();
        for (std::size_t i = 0; i < dev_pages.size(); ++i) {
            pages_to_transfer.push_back(
                tokenspeed::TransferPair{tokenspeed::CacheKind::kKV, dev_pages[i], host_pages[i]});
        }
    }
    return pages_to_transfer;
}

std::vector<tokenspeed::TreeNode*> MambaNodesForTransferPairs(const std::vector<tokenspeed::TreeNode*>& candidates,
                                                              const std::vector<tokenspeed::TransferPair>& transfers) {
    std::unordered_set<std::int32_t> src_slots;
    for (const auto& transfer : transfers) {
        if (transfer.kind == tokenspeed::CacheKind::kMamba) {
            src_slots.insert(transfer.src);
        }
    }
    std::vector<tokenspeed::TreeNode*> nodes;
    nodes.reserve(src_slots.size());
    for (tokenspeed::TreeNode* node : candidates) {
        if (node != nullptr && node->HasMamba() && src_slots.find(node->MambaSlotIndex()) != src_slots.end()) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

// Below this fraction of the device pool free, a finished request gives its
// pages up as soon as they are on host. Above it, it keeps them.
//
// Demoting eagerly costs a full PCIe round trip on the next partial hit: the
// prefix has to be dragged back a layer at a time. Measured on Qwen3-8B with a
// 3890-token shared prefix, that is 11.46 ms of loadback per request -- 15.7 MB
// per layer at 49 GB/s, i.e. the whole prefix, every time -- while ~68 GB of
// device KV cache sat unused. Keeping them costs nothing that is not already
// paid: the host copy exists, so `EnsureCapacity` can evict these nodes for
// free the moment anyone actually needs the pages.
//
// One eighth rather than zero because eviction is not instantaneous: leaving a
// margin means a burst of arrivals finds pages ready instead of waiting on an
// eviction pass.
constexpr double kEagerDemoteFreeFraction = 0.125;

bool DeviceUnderPressure(tokenspeed::KVPrefixCache* kv_prefix_cache) {
    const std::int32_t total = kv_prefix_cache->TotalPagesOfType<tokenspeed::ResourceType::Device>();
    if (total <= 0) return true;
    const std::int32_t available = kv_prefix_cache->AvailablePagesOfType<tokenspeed::ResourceType::Device>();
    return static_cast<double>(available) < kEagerDemoteFreeFraction * static_cast<double>(total);
}

void DemoteWrittenBackDevice(tokenspeed::KVPrefixCache* kv_prefix_cache,
                             tokenspeed::HybridPrefixCache* hybrid_prefix_cache, tokenspeed::TreeNode* device_node,
                             bool only_under_pressure = false) {
    if (kv_prefix_cache == nullptr || device_node == nullptr) return;
    if (only_under_pressure && !DeviceUnderPressure(kv_prefix_cache)) return;
    kv_prefix_cache->ReleaseDeviceResourcesPresentOnHost(device_node, [hybrid_prefix_cache](tokenspeed::TreeNode* n) {
        if (hybrid_prefix_cache != nullptr) {
            hybrid_prefix_cache->OnKVDeviceDemote(n);
        }
    });
}

// Depth at which this chunk's recurrent snapshot must be published, or -1.
//
// One page BELOW the last page this chunk inserts, not at it. A later match can
// never reach the inserted terminal's own depth: an exact repeat must leave at
// least one token to compute, and a diverging prefix (shared document, new
// question) splits the node somewhere inside that last page. Measured both ways
// on Qwen3.5-27B -- snapshot at 4096 against a match at 4032, and snapshot at
// 3008 against a match at 2944 -- and in both cases `augmentMatch` found no
// ancestor snapshot and discarded a valid multi-thousand-token KV match, so
// reuse read `#cached-token: 0`.
//
// One page of reuse is given up in the best case, which is the price of the
// snapshot being reachable at all.
std::int32_t MambaCheckpointPublishDepth(tokenspeed::HybridPrefixCache* hybrid_cache, std::int32_t chunk_begin,
                                         std::int32_t chunk_size, std::int32_t page_size) {
    if (hybrid_cache == nullptr || chunk_size <= 0 || page_size <= 0) return -1;
    const std::int32_t final_len = chunk_begin + chunk_size;
    const std::int32_t publish_depth = (final_len / page_size) * page_size - page_size;
    if (publish_depth <= chunk_begin) return -1;

    // The GPU harvests this state from the chunked scan's intermediate
    // checkpoints, which exist only on its own chunk grain.
    const std::int32_t track_len = publish_depth - chunk_begin;
    if (hybrid_cache->AlignMambaCacheSeqlen(track_len) != track_len) return -1;
    return publish_depth;
}

// Stride between retained snapshots, in tokens.
//
// Reuse on a hybrid is min(KV depth, deepest usable snapshot), so a divergence
// loses at most one stride of reuse. 512 keeps that loss under ~19% of a
// 2690-token document while retaining 5 snapshots rather than the 42 the scan
// actually produces. See unieinfra/xref/35-hybrid-prefix-reuse.md for the
// stride/cost table.
// Default only. The stride is a server arg (`--mamba-track-interval`, 256),
// which existed and was never read here -- the engine carried two strides with
// different values, one configurable and one compiled in.
//
// It matters more than it looks: on the RAG dataset (16 documents of ~4900
// tokens asked 64 questions each) this engine reuses 16.2% of its tokens where
// vLLM reuses 38.5%, so it recomputes more than twice the prefill, which is
// most of the mid-concurrency gap. With a 512 stride and a cap of 6 the deepest
// retained snapshot sits at 3072 of 4900 tokens, and a match cannot reach past
// it.
constexpr std::int32_t kMambaSnapshotStrideTokensDefault = 512;

// Ceiling on retained snapshots per chunk. The pool is shared with every
// in-flight sequence's working state, so an unbounded count would let one long
// document starve the batch. Six covers ~3k tokens at the stride above, which
// is the document size this was measured on.
// Chosen when the pool held 80 slots and every retained snapshot competed with
// a running sequence's working slot. It now holds 221 (bf16 state, mamba memory
// ratio 2.0), so the ceiling that made 6 the right number has moved.
constexpr std::int32_t kMambaSnapshotMaxPerRequestDefault = 6;

// Read once from the environment so the stride/cap pair can be swept without
// threading config through the FSM. Both were fixed constants chosen when the
// state pool held 80 slots; it now holds 221, and reuse is the measured gap at
// mid concurrency (16.2% of tokens against vLLM's 38.5%). If a setting wins,
// it becomes a real server arg -- `--mamba-track-interval` already exists for
// the stride and is currently ignored here.
inline std::int32_t EnvInt(const char* name, std::int32_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    const int parsed = std::atoi(v);
    return parsed > 0 ? parsed : fallback;
}

inline std::int32_t SnapshotStrideTokens() {
    static const std::int32_t v = EnvInt("TOKENSPEED_MAMBA_SNAPSHOT_STRIDE", kMambaSnapshotStrideTokensDefault);
    return v;
}

inline std::int32_t SnapshotMaxPerRequest() {
    static const std::int32_t v = EnvInt("TOKENSPEED_MAMBA_SNAPSHOT_MAX", kMambaSnapshotMaxPerRequestDefault);
    return v;
}

// Every depth this chunk should retain a snapshot at, ascending.
//
// The last entry is what the old single-snapshot rule published -- one page
// below the chunk end, which serves an exact repeat. The earlier entries are
// what makes a *diverging* request find an ancestor at all: they sit on the
// stride grid, below any plausible divergence point, so a match that stops at
// the end of a shared document still has a snapshot above it.
//
// Every returned depth satisfies the same alignment invariant as the single
// one: the GPU only leaves intermediate states on its own chunk grain, and a
// depth off that grain is silently dropped rather than rejected. Getting this
// wrong publishes nothing at all -- measured, when a 256-token setback was
// tried, exact repeats stopped reusing too.
std::vector<std::int32_t> MambaCheckpointPublishDepths(tokenspeed::HybridPrefixCache* hybrid_cache,
                                                       std::int32_t chunk_begin, std::int32_t chunk_size,
                                                       std::int32_t page_size, std::int32_t max_count,
                                                       std::int32_t stride_tokens) {
    std::vector<std::int32_t> depths;
    if (hybrid_cache == nullptr || chunk_size <= 0 || page_size <= 0 || max_count <= 0) return depths;

    const std::int32_t deepest = MambaCheckpointPublishDepth(hybrid_cache, chunk_begin, chunk_size, page_size);
    if (deepest < 0) return depths;

    const std::int32_t final_len = chunk_begin + chunk_size;
    const std::int32_t stride = stride_tokens > 0 ? stride_tokens : kMambaSnapshotStrideTokensDefault;
    for (std::int32_t depth = stride; depth < deepest; depth += stride) {
        if (depth <= chunk_begin) continue;  // already covered by an earlier chunk
        if (depth >= final_len) break;
        const std::int32_t track_len = depth - chunk_begin;
        if (hybrid_cache->AlignMambaCacheSeqlen(track_len) != track_len) continue;
        depths.push_back(depth);
    }
    depths.push_back(deepest);

    // Keep the deepest ones when the pool is short: a shallow snapshot is worth
    // less than a deep one to every future match.
    if (static_cast<std::int32_t>(depths.size()) > max_count) {
        depths.erase(depths.begin(), depths.end() - max_count);
    }
    return depths;
}

// A mandatory Mamba slot, reclaiming a cached snapshot if the pool is empty.
//
// Retained snapshots are a cache and `EnsureMambaCapacityByEvict` frees the
// coldest of them, but nothing in this file called it: the pool only ever grew,
// and a request that found it empty threw. That was survivable while each chunk
// retained a single snapshot, and became a hard failure at concurrency once a
// chunk retained up to `kMambaSnapshotMaxPerRequest` of them -- measured as
// "SchedulePrefillEvent: failed to allocate Mamba checkpoint slot" on the
// unique-prompt shape at 64 concurrent requests, where nothing is ever reused
// and every finished request leaves six snapshots behind.
//
// Eviction skips any node whose device resource is still referenced, so the
// snapshots this step is about to publish -- held by the in-flight request --
// cannot be freed out from under the forward that writes them.
//
// Returning false still throws at the call site: that means the pool cannot
// satisfy even one working set, which is a sizing error rather than pressure.
bool AllocateMambaCheckpointOrEvict(tokenspeed::LocalMambaAllocator* local,
                                    tokenspeed::HybridPrefixCache* hybrid_cache) {
    if (local->AllocateCheckpoint()) return true;
    if (hybrid_cache == nullptr) return false;
    if (!hybrid_cache->EnsureMambaCapacityByEvict(1)) return false;
    return local->AllocateCheckpoint();
}

bool AllocateMambaWorkingOrEvict(tokenspeed::LocalMambaAllocator* local,
                                 tokenspeed::HybridPrefixCache* hybrid_cache) {
    if (local->AllocateWorking()) return true;
    if (hybrid_cache == nullptr) return false;
    if (!hybrid_cache->EnsureMambaCapacityByEvict(1)) return false;
    return local->AllocateWorking();
}

}  // namespace

namespace tokenspeed::fsm {

#if TOKENSPEED_FLAT_KVCACHE
namespace {

// The throw makes it noreturn so callers with non-default-constructible returns compile.
[[noreturn]] void FlatRetractUnsupported() {
    _assert(false, "flat path: retract/writeback/loadback unsupported in C slice");
    throw std::logic_error("flat path: retract/writeback/loadback unsupported in C slice");
}

// Base-page hashes one decode step hands to registration, anchored at begin_page.
struct HashesToRegister {
    std::int32_t begin_page{0};
    std::vector<std::string> hashes;
};

// Seed from a full-prefix hash list; the tail keeps the suffix from the last fold-grid boundary.
HashChain MakeHashChain(const std::vector<std::string>& hashes, std::int32_t fold_align_pages) {
    const std::int32_t n = static_cast<std::int32_t>(hashes.size());
    const std::int32_t tail_begin = n / fold_align_pages * fold_align_pages;
    return HashChain{.num_hashed_pages = n,
                     .last_hash = n > 0 ? hashes.back() : std::string{},
                     .tail_begin_page = tail_begin,
                     .tail = {hashes.begin() + tail_begin, hashes.end()}};
}

// Hash the newly filled pages onto the chain; the returned batch reaches back to the fold grid
// (fold_align_pages = lcm/base) so coarse groups fold the blocks completed this step. The
// re-covered overlap is idempotent (m == 1 degenerates to exactly the fresh pages).
HashesToRegister AdvanceHashChain(HashChain& chain, const std::vector<std::span<const std::int32_t>>& paged,
                                  std::int32_t filled_pages, std::int32_t fold_align_pages) {
    _assert(filled_pages > chain.num_hashed_pages, "caller must pre-check hash-chain progress");
    _assert(filled_pages <= static_cast<std::int32_t>(paged.size()),
            "flat decode hashing: filled pages exceed the container's full pages");
    const std::vector<std::span<const std::int32_t>> fresh(paged.begin() + chain.num_hashed_pages,
                                                           paged.begin() + filled_pages);
    std::vector<std::string> new_hashes = ComputePagedHashes(fresh, chain.last_hash);

    HashesToRegister batch{.begin_page = chain.tail_begin_page, .hashes = std::move(chain.tail)};
    batch.hashes.insert(batch.hashes.end(), new_hashes.begin(), new_hashes.end());

    const std::int32_t tail_begin = filled_pages / fold_align_pages * fold_align_pages;
    chain.num_hashed_pages = filled_pages;
    chain.last_hash = batch.hashes.back();
    chain.tail_begin_page = tail_begin;
    chain.tail.assign(batch.hashes.begin() + (tail_begin - batch.begin_page), batch.hashes.end());
    return batch;
}

}  // namespace
#endif

void InsertHybridCache(HybridPrefixCache* hybrid_cache,
                       const std::vector<std::span<const std::int32_t>>& full_paged_tokens,
                       std::unique_ptr<DeviceNodeRef>& device_node_ref, LocalKVAllocator* local_kv_allocator,
                       LocalMambaAllocator* local_mamba_allocator, std::int32_t chunk_begin, std::int32_t chunk_size,
                       std::int32_t page_size, const std::vector<std::int32_t>* prefix_pages_override) {
    if (hybrid_cache == nullptr) return;

    std::vector<std::int32_t> computed_prefix_pages;
    const std::vector<std::int32_t>* prefix_pages = prefix_pages_override;
    if (prefix_pages == nullptr) {
        computed_prefix_pages = DevicePagesFromRoot(device_node_ref->Node());
        prefix_pages = &computed_prefix_pages;
    }
    std::int32_t new_page_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages->size());
    if (new_page_count <= 0) {
        if (local_mamba_allocator != nullptr && local_mamba_allocator->HasCheckpoint()) {
            local_mamba_allocator->DetachCheckpoint();
        }
        return;
    }

    // Allocator may hold fewer pages than new_page_count by insert time; clamp page-aligned so
    // prefix + inserted == full (a count-only clamp would relocate the overflow into Insert's TakeLast).
    const std::vector<std::span<const std::int32_t>>* tokens_for_insert = &full_paged_tokens;
    std::vector<std::span<const std::int32_t>> clamped_tokens;
    const std::int32_t avail = static_cast<std::int32_t>(local_kv_allocator->PageCount());
    if (new_page_count > avail) {
        const std::int32_t keep = static_cast<std::int32_t>(prefix_pages->size()) + avail;
        clamped_tokens.assign(full_paged_tokens.begin(), full_paged_tokens.begin() + keep);
        tokens_for_insert = &clamped_tokens;
        new_page_count = avail;
    }

    OwnedPages pages_to_insert = local_kv_allocator->TakeFirst(new_page_count);
    auto insert_result = hybrid_cache->GetKVPrefixCache().Insert<ResourceType::Device>(
        *tokens_for_insert, *prefix_pages, std::move(pages_to_insert));

    if (local_mamba_allocator != nullptr && local_mamba_allocator->HasCheckpoint()) {
        // How many depths this chunk is worth retaining. Asked for here rather
        // than at the four allocation sites because only here are chunk_begin
        // and chunk_size known -- a short prompt wants one snapshot and a long
        // document wants several, and reserving the maximum per request would
        // starve a pool shared with every other in-flight sequence.
        local_mamba_allocator->ClearPublishedCheckpoints();
        const auto wanted =
            MambaCheckpointPublishDepths(hybrid_cache, chunk_begin, chunk_size, page_size,
                                         SnapshotMaxPerRequest(), SnapshotStrideTokens());
        const std::int32_t extra = static_cast<std::int32_t>(wanted.size()) - 1;
        if (extra > 0) {
            // Best-effort; a refusal here costs reuse depth, not correctness.
            local_mamba_allocator->AllocateCheckpoints(extra);
        }

        // One slot per depth, ascending, as many as the pool gave us. The
        // depths and the slots are paired by position and must stay that way:
        // a slot filed under a depth it does not describe would resume a later
        // hit from the wrong recurrent state, silently.
        auto slots = local_mamba_allocator->DetachCheckpoints();
        const auto publish_depths = MambaCheckpointPublishDepths(
            hybrid_cache, chunk_begin, chunk_size, page_size, static_cast<std::int32_t>(slots.size()),
            SnapshotStrideTokens());
        std::size_t next_slot = 0;
        for (std::int32_t publish_depth : publish_depths) {
            if (next_slot >= slots.size()) break;
            // SplitAt resolves, or carves out, the ancestor at that exact depth
            // -- Insert leaves one node spanning the whole chunk, so the node a
            // match will land on usually does not exist yet. Same primitive the
            // paged-cache snapshot path already uses.
            TreeNode* attach_node =
                publish_depth > 0
                    ? hybrid_cache->GetKVPrefixCache().GetRadixTree().SplitAt(insert_result.last_node, publish_depth)
                    : nullptr;
            if (attach_node != nullptr && !attach_node->HasMamba()) {
                const std::int32_t slot_index = slots[next_slot]->Index();
                hybrid_cache->InsertMamba(attach_node, std::move(slots[next_slot]));
                // The operation built after this event is what tells the GPU
                // where to write; ownership has just moved to the tree, so the
                // index has to be carried out by hand.
                local_mamba_allocator->RecordPublishedCheckpoint(slot_index);
            }
            ++next_slot;
        }
        // Anything left over (no node to attach to, or a node that already had
        // a state) is released here by going out of scope.
    }
    device_node_ref = std::make_unique<DeviceNodeRef>(insert_result.last_node);
}

// Submitted -> PrefillDone / Prefilling
std::variant<PrefillDone, Prefilling> SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "SchedulePrefillFirstChunkEvent: flat path requires a coordinator");
    TokenContainer* token_container = state.GetTokenContainer();

    // Slot first: Allocate() throws on exhaustion; order kept for determinism.
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());

    std::vector<BlockTable> tables(coordinator_->NumGroups());
    const std::int32_t device_hit_tokens = flat_hit_.num_common_tokens;
    const std::int32_t hit_tokens = std::max(device_hit_tokens, flat_host_.num_common_tokens);
    coordinator_->ClaimCommonPrefix(tables, std::move(flat_hit_));
    // Extension appends between claim and fresh acquire so slots stay
    // [device hit | host ext | new pages]; ext pages are FULL, composing with the gate.
    flat_load_pairs_ = LoadHostExtension(*coordinator_, tables, std::move(flat_host_));
    // Host boundary is absolute (floor included); default-constructed (no host pool) it is 0.
    // Loaded pages become device-cached now (SWA holes skipped by the IsNull guard); the sink
    // re-collects them and the drain dedupes against the host index. The extension ends at the
    // page-aligned hit boundary, so the state group's final page is exactly the loaded snapshot.
    coordinator_->CacheFullBlocks(tables, flat_ext_hashes_,
                                  /*first_slot=*/device_hit_tokens / state.GetPageSize(),
                                  /*end_tokens=*/hit_tokens);
    bool acquired = false;
    if (role_ == Role::kD && enable_flatkv_pd_) {
        _assert(hit_tokens + tokens_this_round_ == token_container->PrefillSize(),
                "FlatKV PD decode destination must acquire the complete prompt atomically");
        acquired = coordinator_->AcquireDecodeDestination(tables, token_container->PrefillSize(), tokens_this_round_,
                                                          decode_input_tokens_);
    } else {
        // Role kD cannot defer its decode reserve until RemotePrefillDone: the
        // remote transfer may complete after other requests consume the pool.
        const std::int32_t tokens_to_acquire = tokens_this_round_ + (role_ == Role::kD ? decode_input_tokens_ : 0);
        acquired = coordinator_->Acquire(tables, tokens_to_acquire);
    }
    if (!acquired) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    TokenContainer::Window window{.begin = hit_tokens, .size = tokens_this_round_};
    bool is_last_chunk = (window.begin + window.size) == token_container->PrefillSize();
    if (is_last_chunk && role_ != Role::kD) {
        PrefillDone done{token_container, state.GetPageSize(),  nullptr, nullptr, nullptr, std::move(req_pool_index),
                         window,          decode_input_tokens_, nullptr};
        done.SetBlockTables(std::move(tables));
        return done;
    }
    Prefilling prefilling{token_container, state.GetPageSize(),       nullptr, nullptr,
                          nullptr,         std::move(req_pool_index), window,  nullptr};
    prefilling.SetBlockTables(std::move(tables));
    return prefilling;
#else
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    if (!disable_l2_cache_ && (match_result_.host.DepthInPage() > match_result_.device.DepthInPage())) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(
            match_result_.NodesWithout<ResourceType::Device>());
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }

    auto local_kv_allocator = std::make_unique<LocalKVAllocator>(device_allocator_, tokens_this_round_);
    // Reserve token slots for draft multi-step decode.
    local_kv_allocator->Acquire(decode_input_tokens_);

    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());

    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator;
    if (mamba_allocator_ != nullptr) {
        // Snapshots come from the checkpoint pool when one is configured, so a
        // cached state cannot take the slot a running sequence needs.
        local_mamba_allocator = std::make_unique<LocalMambaAllocator>(
            mamba_allocator_,
            hybrid_prefix_cache_ != nullptr ? hybrid_prefix_cache_->CheckpointAllocator() : nullptr);
        if (!AllocateMambaWorkingOrEvict(local_mamba_allocator.get(), hybrid_prefix_cache_)) {
            throw std::logic_error("SchedulePrefillFirstChunkEvent: failed to allocate Mamba working slot");
        }
        // Best-effort, not fatal. This slot is where the GPU writes a snapshot
        // for the radix cache to keep; the request does not read it to make
        // progress. Throwing here turned a full cache into a dead engine --
        // "ScheduleDecodeEvent: failed to allocate Mamba checkpoint slot",
        // 1024 of 1024 requests lost at 512 concurrent. Without a slot the op
        // carries -1, no snapshot is published, and the only cost is reuse
        // depth for a later request.
        AllocateMambaCheckpointOrEvict(local_mamba_allocator.get(), hybrid_prefix_cache_);
    }

    TokenContainer* token_container = state.GetTokenContainer();

    std::int32_t max_matched_pages =
        disable_l2_cache_ ? match_result_.device.DepthInPage()
                          : std::max(match_result_.device.DepthInPage(), match_result_.host.DepthInPage());
    std::int32_t window_begin = max_matched_pages * state.GetPageSize();
    TokenContainer::Window window{.begin = window_begin, .size = tokens_this_round_};

    bool is_last_chunk = (window.begin + window.size) == token_container->PrefillSize();
    if (is_last_chunk && role_ != Role::kD) {
        std::int32_t reserve_num_tokens_in_next_schedule_event = decode_input_tokens_;
        return PrefillDone{token_container,
                           state.GetPageSize(),
                           std::move(host_node_ref),
                           std::move(device_node_ref),
                           std::move(local_kv_allocator),
                           std::move(req_pool_index),
                           window,
                           reserve_num_tokens_in_next_schedule_event,
                           std::move(local_mamba_allocator)};
    } else {
        return Prefilling{token_container,
                          state.GetPageSize(),
                          std::move(host_node_ref),
                          std::move(device_node_ref),
                          std::move(local_kv_allocator),
                          std::move(req_pool_index),
                          window,
                          std::move(local_mamba_allocator)};
    }
#endif
}

// Prefilling -> Prefilling / PrefillDone
std::variant<PrefillDone, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "SchedulePrefillEvent: flat path requires a coordinator");
    const std::vector<std::string> hashes = FlatWindowPageHashes(state.GetFullPagedTokens(false), state.GetPageSize(),
                                                                 state.window.begin, state.window.size);
    // Prior chunks 0..k-1 (state.window is the PREVIOUS chunk); the gate credited this same slide value.
    const std::int32_t num_computed_tokens = state.window.begin + state.window.size;

    auto tables = std::move(state).TakeBlockTables();
    if (!PrefillChunk(*coordinator_, tables, hashes, tokens_this_round_, num_computed_tokens)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    TokenContainer::Window window{.begin = state.window.begin + state.window.size, .size = tokens_this_round_};
    bool is_last_chunk = (window.begin + window.size) == state.GetTokenContainer()->PrefillSize();
    if (is_last_chunk) {
        PrefillDone done{state.GetTokenContainer(),
                         state.GetPageSize(),
                         nullptr,
                         nullptr,
                         nullptr,
                         std::move(state).TakeReqPoolIndex(),
                         window,
                         reserve_num_tokens_in_next_schedule_event_,
                         nullptr};
        done.SetBlockTables(std::move(tables));
        return done;
    }
    Prefilling prefilling{state.GetTokenContainer(),
                          state.GetPageSize(),
                          nullptr,
                          nullptr,
                          nullptr,
                          std::move(state).TakeReqPoolIndex(),
                          window,
                          nullptr};
    prefilling.SetBlockTables(std::move(tables));
    return prefilling;
#else
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    // Only insert pages from the beginning up to the end of the last processed chunk.
    auto paged_tokens = state.GetFullPagedTokens(false);
    std::int32_t end_of_window_pages = (state.window.begin + state.window.size) / state.GetPageSize();
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    InsertHybridCache(hybrid_prefix_cache_, paged_tokens, device_node_ref, local_kv_allocator.get(),
                      local_mamba_allocator.get(), state.window.begin, state.window.size, state.GetPageSize());
    local_kv_allocator->Acquire(tokens_this_round_);

    if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr) {
        // Best-effort, not fatal. This slot is where the GPU writes a snapshot
        // for the radix cache to keep; the request does not read it to make
        // progress. Throwing here turned a full cache into a dead engine --
        // "ScheduleDecodeEvent: failed to allocate Mamba checkpoint slot",
        // 1024 of 1024 requests lost at 512 concurrent. Without a slot the op
        // carries -1, no snapshot is published, and the only cost is reuse
        // depth for a later request.
        AllocateMambaCheckpointOrEvict(local_mamba_allocator.get(), hybrid_prefix_cache_);
    }

    TokenContainer::Window window{.begin = state.window.begin + state.window.size, .size = tokens_this_round_};

    bool is_last_chunk = (window.begin + window.size) == state.GetTokenContainer()->PrefillSize();
    if (is_last_chunk) {
        return PrefillDone{state.GetTokenContainer(),
                           state.GetPageSize(),
                           std::move(host_node_ref),
                           std::move(device_node_ref),
                           std::move(local_kv_allocator),
                           std::move(state).TakeReqPoolIndex(),
                           window,
                           reserve_num_tokens_in_next_schedule_event_,
                           std::move(local_mamba_allocator)};
    } else {
        return Prefilling{state.GetTokenContainer(),
                          state.GetPageSize(),
                          std::move(host_node_ref),
                          std::move(device_node_ref),
                          std::move(local_kv_allocator),
                          std::move(state).TakeReqPoolIndex(),
                          window,
                          std::move(local_mamba_allocator)};
    }
#endif
}

// PrefillDone -> Decoding: insert prefill pages into tree, then transition to decode.
Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "ScheduleDecodeEvent: flat path requires a coordinator");
    const std::vector<std::string> hashes = FlatWindowPageHashes(state.GetFullPagedTokens(false), state.GetPageSize(),
                                                                 state.window.begin, state.window.size);
    const std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    // Full prefill length (window end == PrefillSize()); the PrefillDone gate credited the same value.
    const std::int32_t num_computed_tokens = state.window.begin + state.window.size;

    auto tables = std::move(state).TakeBlockTables();
    if (!FinalizePrefillAndReserveDecode(*coordinator_, tables, hashes, reserve, num_computed_tokens)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    Decoding decoding{state.GetTokenContainer(),           state.GetPageSize(),  nullptr, nullptr, nullptr,
                      std::move(state).TakeReqPoolIndex(), decode_input_tokens_, nullptr};
    decoding.SetBlockTables(std::move(tables));
    decoding.SetHashChain(MakeHashChain(hashes, coordinator_->LcmBlockSize() / coordinator_->BaseBlockSize()));
    return decoding;
#else
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    // Only insert pages from the beginning up to the end of the last processed chunk.
    auto paged_tokens = state.GetFullPagedTokens(false);
    std::int32_t end_of_window_pages = (state.window.begin + state.window.size) / state.GetPageSize();
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    InsertHybridCache(hybrid_prefix_cache_, paged_tokens, device_node_ref, local_kv_allocator.get(),
                      local_mamba_allocator.get(), state.window.begin, state.window.size, state.GetPageSize());
    if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr) {
        // Best-effort, not fatal. This slot is where the GPU writes a snapshot
        // for the radix cache to keep; the request does not read it to make
        // progress. Throwing here turned a full cache into a dead engine --
        // "ScheduleDecodeEvent: failed to allocate Mamba checkpoint slot",
        // 1024 of 1024 requests lost at 512 concurrent. Without a slot the op
        // carries -1, no snapshot is published, and the only cost is reuse
        // depth for a later request.
        AllocateMambaCheckpointOrEvict(local_mamba_allocator.get(), hybrid_prefix_cache_);
    }

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    local_kv_allocator->Acquire(reserve);

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_,          std::move(local_mamba_allocator)};
#endif
}

// Decoding -> Decoding: allocate pages for next decode step.
Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "ScheduleDecodeEvent: flat path requires a coordinator");
    const std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    // Size() includes this round's pending decode tail; sliding at Size() would free a page its
    // query still reads. scheduleDecode's gate credited the slide with this same value.
    const std::int32_t num_computed_tokens = state.GetTokenContainer()->Size() - decode_input_tokens_;

    HashChain chain = std::move(state).TakeHashChain();
    const std::int32_t filled_pages = num_computed_tokens / state.GetPageSize();
    // A page fills only once every page_size steps; skip the span walk on the other steps.
    HashesToRegister to_register{.begin_page = chain.num_hashed_pages};
    if (filled_pages > chain.num_hashed_pages) {
        to_register = AdvanceHashChain(chain, state.GetFullPagedTokens(false), filled_pages,
                                       coordinator_->LcmBlockSize() / coordinator_->BaseBlockSize());
    }

    auto tables = std::move(state).TakeBlockTables();
    if (!DecodeStep(*coordinator_, tables, to_register.hashes, to_register.begin_page, reserve, num_computed_tokens)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    Decoding decoding{state.GetTokenContainer(),           state.GetPageSize(),  nullptr, nullptr, nullptr,
                      std::move(state).TakeReqPoolIndex(), decode_input_tokens_, nullptr};
    decoding.SetBlockTables(std::move(tables));
    decoding.SetHashChain(std::move(chain));
    return decoding;
#else
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    local_kv_allocator->Acquire(reserve);

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_,          std::move(local_mamba_allocator)};
#endif
}

// Retracted -> Decoding: recover via LoadBack (host -> device).
Decoding ScheduleDecodeFromRetractedEvent::operator()(Retracted&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    FlatRetractUnsupported();
#else
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    if (match_result_.host.DepthInPage() > match_result_.device.DepthInPage()) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(
                match_result_.NodesWithout<ResourceType::Device>())) {
            throw std::logic_error(
                "ScheduleDecodeFromRetractedEvent: failed to allocate device pages for host cache recovery");
        }
        // Device pages were just attached along the host-matched chain: pinning the HOST last node is not a typo.
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_kv_allocator = std::move(state).TakeKVAllocator();
    auto old_mamba_allocator = std::move(state).TakeMambaAllocator();
    old_mamba_allocator.reset();
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator;
    if (mamba_allocator_ != nullptr) {
        // Snapshots come from the checkpoint pool when one is configured, so a
        // cached state cannot take the slot a running sequence needs.
        local_mamba_allocator = std::make_unique<LocalMambaAllocator>(mamba_allocator_, mamba_ckpt_allocator_);
        if (!local_mamba_allocator->AllocateWorking()) {
            throw std::logic_error("ScheduleDecodeFromRetractedEvent: failed to allocate Mamba recovery working slot");
        }
        if (!local_mamba_allocator->AllocateCheckpoint()) {
            throw std::logic_error(
                "ScheduleDecodeFromRetractedEvent: failed to allocate Mamba recovery checkpoint slot");
        }
    }
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    local_kv_allocator->Acquire(decode_input_tokens_);
    return Decoding{token_container,
                    page_size,
                    std::move(host_node_ref),
                    std::move(device_node_ref),
                    std::move(local_kv_allocator),
                    std::move(req_pool_index),
                    decode_input_tokens_,
                    std::move(local_mamba_allocator)};
#endif
}

// Decoding/PrefillDone -> Draining/Finished, driven by the Python side's Advance.
template <typename ForwardStateT>
std::variant<Draining, Finished> FinishEvent::apply(ForwardStateT&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "FinishEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
    return Finished{};
#else
    auto full_paged_tokens = state.GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(state.GetDeviceNode());
    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    // Overlap / disagg handoff can grow the token container past the pages this
    // request actually Acquired (the terminal token's Acquire is skipped). Clamp
    // to owned pages, as scheduleRetract does.
    const std::int32_t owned_pages = local_allocator->PageCount();
    if (alloc_count > owned_pages) {
        full_paged_tokens.resize(static_cast<std::size_t>(owned_pages) + prefix_pages.size());
        alloc_count = owned_pages;
    }
    if (alloc_count > 0) {
        // Same page-aligned clamp as InsertHybridCache, keeping prefix + inserted == full.
        const std::int32_t avail = static_cast<std::int32_t>(local_allocator->PageCount());
        if (alloc_count > avail) {
            full_paged_tokens.resize(prefix_pages.size() + avail);
            alloc_count =
                static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
        }
        OwnedPages alloc_pages = local_allocator->TakeFirst(alloc_count);

        kv_prefix_cache_->Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages),
                                                       page_hashes_);

        if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr &&
            (local_mamba_allocator->HasCheckpoint() || local_mamba_allocator->HasWorking())) {
            MatchResult post_match = kv_prefix_cache_->Match(full_paged_tokens);
            TreeNode* terminal = post_match.device.last_node;
            if (terminal != nullptr && !terminal->HasMamba()) {
                if (local_mamba_allocator->HasCheckpoint()) {
                    hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachCheckpoint());
                } else {
                    hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachWorking());
                }
            }
        }
    }
    // local_mamba_allocator dropped here — destructor frees remaining slots

    MatchResult match = kv_prefix_cache_->Match(full_paged_tokens);
    if (!disable_l2_cache_ && (match.device.DepthInPage() > match.host.DepthInPage())) {
        std::vector<TreeNode*> write_diff = match.NodesWithout<ResourceType::Host>();
        std::int32_t host_pages_num = 0;
        for (TreeNode* node : write_diff) {
            host_pages_num += node->Device().NumPages();
        }
        std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match.host.last_node);
        if (!kv_prefix_cache_->EnsureCapacityByEvict<ResourceType::Host>(host_pages_num)) {
            return Finished{};
        }
        kv_prefix_cache_->AllocateResourceOfType<ResourceType::Host>(write_diff);
        std::unique_ptr<DeviceNodeRef> device_node_ref = std::make_unique<DeviceNodeRef>(match.device.last_node);
        std::unique_ptr<HostNodeRef> host_node_ref = std::make_unique<HostNodeRef>(match.device.last_node);

        auto pages_to_transfer = BuildWriteBackPairs(write_diff);
        std::vector<TreeNode*> mamba_writeback_nodes;
        if (hybrid_prefix_cache_ != nullptr) {
            auto mamba_pairs = hybrid_prefix_cache_->PrepareMambaHostWriteBack(write_diff);
            mamba_writeback_nodes = MambaNodesForTransferPairs(write_diff, mamba_pairs);
            pages_to_transfer.insert(pages_to_transfer.end(), std::make_move_iterator(mamba_pairs.begin()),
                                     std::make_move_iterator(mamba_pairs.end()));
        }
        return Draining{std::move(pages_to_transfer), std::move(device_node_ref), std::move(host_node_ref),
                        std::move(mamba_writeback_nodes)};
    }
    return Finished{};
#endif
}

std::variant<Draining, Finished> FinishEvent::operator()(Decoding&& state) {
    return apply(std::move(state));
}

std::variant<Draining, Finished> FinishEvent::operator()(PrefillDone&& state) {
    return apply(std::move(state));
}

// EOS mid-writeback: downcast so WriteBackDoneEvent takes the existing WritingBack -> Finished path.
WritingBack FinishEvent::operator()(Retracting&& state) {
    return static_cast<WritingBack&&>(state);
}

// Draining -> WritingBack: the two RAII node-ref locks alone keep the pages pinned until WriteBackDone.
WritingBack CommitDrainingEvent::operator()(Draining&& state) {
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();
    auto mamba_writeback_nodes = std::move(state).TakeMambaWriteBackNodes();
    return WritingBack{std::move(device_node_ref), std::move(host_node_ref), std::move(mamba_writeback_nodes)};
}

// WritingBack -> Finished: written-back cache keeps its device pages unless the
// device pool is under pressure, so a later hit is served from device rather
// than loaded back over PCIe. Under pressure it demotes as before.
Finished WriteBackDoneEvent::operator()(WritingBack&& state) {
    TreeNode* device_node = state.DeviceNode();
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(state.MambaWriteBackNodes());
    }
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node, /*only_under_pressure=*/true);
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->DemoteIdleMambaDeviceCopiesPresentOnHost();
    }
    return Finished{};
}

Retracted WriteBackDoneEvent::operator()(Retracting&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    FlatRetractUnsupported();
#else
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    TreeNode* device_node = state.DeviceNode();
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(state.MambaWriteBackNodes());
    }
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node);
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->DemoteIdleMambaDeviceCopiesPresentOnHost();
    }
    auto host_ref = std::move(static_cast<WritingBack&&>(state)).TakeHostNodeRef();
    std::unique_ptr<LocalKVAllocator> local_device_allocator = std::move(state).TakeKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeMambaAllocator();
    auto mamba_node_lock = std::move(state).TakeMambaNodeLock();
    // DeviceNodeRef inside WritingBack base is released here (unique_ptr dtor).
    return Retracted{token_container,          page_size,
                     std::move(host_ref),      std::move(local_device_allocator),
                     std::move(local_mamba_allocator), std::move(mamba_node_lock)};
#endif
}

Finished AbortEvent::operator()(Bootstrapping&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Submitted&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Prefetching&& state) {
    return Aborting{std::move(state).TakeHostPages()};
}

Finished AbortEvent::operator()(Draining&&) {
    return Finished{};
}

Finished AbortEvent::operator()(PrefetchDone&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Aborting&& state) {
    return std::move(state);
}

Finished AbortEvent::operator()(Prefilling&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "AbortEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
#else
    (void)state;
#endif
    return Finished{};
}

Finished AbortEvent::operator()(PrefillDone&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "AbortEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
#else
    (void)state;
#endif
    return Finished{};
}

Finished AbortEvent::operator()(Decoding&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "AbortEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
#else
    (void)state;
#endif
    return Finished{};
}

Finished AbortEvent::operator()(Retracting&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};
}

#if TOKENSPEED_FLAT_KVCACHE
template <typename ForwardStateT>
Submitted FlatRetractEvent::applyRetract(ForwardStateT&& state) {
    _assert(coordinator_ != nullptr, "FlatRetractEvent: flat path requires a coordinator");
    TokenContainer* token_container = state.GetTokenContainer();
    const std::int32_t page_size = state.GetPageSize();
    // Generated tokens rebase into the prefill window so the requeued prefill recomputes them.
    token_container->RebasePrefill();
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
    return Submitted{token_container, page_size};
}

Submitted FlatRetractEvent::operator()(Decoding&& state) {
    return applyRetract(std::move(state));
}

Submitted FlatRetractEvent::operator()(PrefillDone&& state) {
    return applyRetract(std::move(state));
}
#endif

template <typename ForwardStateT>
Retracting ScheduleRetractEvent::applyRetract(ForwardStateT&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    FlatRetractUnsupported();
#else
    std::unique_ptr<DeviceNodeRef> device_node_ref = nullptr;
    std::unique_ptr<HostNodeRef> host_node_ref = nullptr;
    std::vector<Retracting::PagePair> pages_to_transfer;
    std::vector<TreeNode*> mamba_writeback_nodes;

    if (match_result_.device.DepthInPage() > match_result_.host.DepthInPage()) {
        std::vector<TreeNode*> write_diff = match_result_.NodesWithout<ResourceType::Host>();
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Host>(write_diff)) {
            throw std::logic_error("ScheduleRetractEvent: failed to allocate host pages for device cache writeback");
        }
        pages_to_transfer = BuildWriteBackPairs(write_diff);
        if (hybrid_prefix_cache_ != nullptr) {
            auto mamba_pairs = hybrid_prefix_cache_->PrepareMambaHostWriteBack(write_diff);
            mamba_writeback_nodes = MambaNodesForTransferPairs(write_diff, mamba_pairs);
            pages_to_transfer.insert(pages_to_transfer.end(), std::make_move_iterator(mamba_pairs.begin()),
                                     std::make_move_iterator(mamba_pairs.end()));
        }
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    } else {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    }

    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();

    // Save Mamba state into the prefix cache before retract so loadback can recover it.
    std::unique_ptr<MambaNodeLock> mamba_node_lock;
    if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr &&
        (local_mamba_allocator->HasCheckpoint() || local_mamba_allocator->HasWorking())) {
        TreeNode* terminal = match_result_.device.last_node;
        if (terminal != nullptr && !terminal->HasMamba()) {
            if (local_mamba_allocator->HasCheckpoint()) {
                hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachCheckpoint());
            } else {
                hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachWorking());
            }
        }
        // The state is tree-owned from here, so the request keeps no local slot
        // -- but it is not a cache entry either: this is the only copy of the
        // recurrent state this request can resume from, and losing it aborts
        // the request rather than costing a recompute. Lock it for the duration
        // of the retraction; the guard releases on resume and on abort alike.
        //
        // Leaving it evictable is what produced "Retracted request ... lost
        // tree-owned Mamba state, aborting request" once several snapshots per
        // chunk made the slot pool tight enough to evict under load.
        if (terminal != nullptr && terminal->HasMamba()) {
            mamba_node_lock = std::make_unique<MambaNodeLock>(terminal);
        }
        local_mamba_allocator.reset();
    }

    return Retracting{token_container,
                      page_size,
                      std::move(host_node_ref),
                      std::move(device_node_ref),
                      std::move(local_allocator),
                      std::move(pages_to_transfer),
                      std::move(mamba_writeback_nodes),
                      std::move(local_mamba_allocator),
                      std::move(mamba_node_lock)};
#endif
}

Retracting ScheduleRetractEvent::operator()(Decoding&& state) {
    return applyRetract(std::move(state));
}

Retracting ScheduleRetractEvent::operator()(PrefillDone&& state) {
    return applyRetract(std::move(state));
}

}  // namespace tokenspeed::fsm
