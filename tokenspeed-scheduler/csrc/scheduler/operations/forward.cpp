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

#include "scheduler/scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/tier/transfer.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "utils.h"
#include <unordered_map>

namespace tokenspeed {

namespace {

template <typename Operation>
void fillBlockTables(Operation& operation, Request& request, const KvCacheCoordinator& coordinator,
                     std::span<const std::string> group_ids) {
    operation.block_tables = BuildBlockTables(coordinator, request.BlockTablesRef(), group_ids);
}

std::vector<GroupDemand> makeGroupDemands(std::vector<BlockTable>& tables, GroupDemand prototype) {
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (BlockTable& table : tables) {
        prototype.table = &table;
        demands.push_back(prototype);
    }
    return demands;
}

void appendCompletedPageHashes(std::vector<std::string>& page_hashes,
                               const std::vector<std::span<const std::int32_t>>& paged_tokens,
                               std::int32_t filled_pages) {
    const std::int32_t first_new_page = static_cast<std::int32_t>(page_hashes.size());
    _assert(filled_pages > first_new_page, "caller must pre-check page-hash progress");
    const std::string previous_hash = page_hashes.empty() ? std::string{} : page_hashes.back();
    std::vector<std::string> new_hashes = AdvancePagedHashes(paged_tokens, first_new_page, previous_hash, filled_pages);
    page_hashes.insert(page_hashes.end(), std::make_move_iterator(new_hashes.begin()),
                       std::make_move_iterator(new_hashes.end()));
}

bool canConsumeReservedTokensInPlace(const KvCacheCoordinator& coordinator, std::span<const BlockTable> tables,
                                     std::int32_t num_tokens, std::int32_t num_computed_tokens) {
    for (std::int32_t i = 0; i < coordinator.NumGroups(); ++i) {
        const KvCacheManager& manager = coordinator.GroupManager(i);
        const BlockTable& table = tables[static_cast<std::size_t>(i)];
        if (manager.BlocksNeededFor(table, num_tokens) != 0 ||
            !manager.ReclaimableBlockLocationsAt(table, num_computed_tokens, {}).empty()) {
            return false;
        }
    }
    return true;
}

CacheBoundaryKind consumeCompletedBoundaryKind(fsm::CacheProgress& cache_progress, std::int32_t num_computed_tokens,
                                               std::int32_t prefill_size) {
    if (cache_progress.promotion_boundary_tokens > 0 &&
        num_computed_tokens >= cache_progress.promotion_boundary_tokens) {
        const bool reached_exactly = num_computed_tokens == cache_progress.promotion_boundary_tokens;
        cache_progress.promotion_boundary_tokens = 0;
        if (reached_exactly) {
            return CacheBoundaryKind::kPromoted;
        }
    }
    return num_computed_tokens == prefill_size ? CacheBoundaryKind::kEndpoint : CacheBoundaryKind::kChunk;
}

struct CompletedCachePages {
    std::int32_t first_new_page{0};
    std::optional<CacheBoundaryKind> boundary_kind;
};

CompletedCachePages updateCompletedPageHashes(Request& request, fsm::CacheProgress& cache_progress,
                                              std::int32_t num_computed_tokens, std::int32_t cache_block_tokens) {
    CompletedCachePages completed{
        .first_new_page = static_cast<std::int32_t>(cache_progress.page_hashes.size()),
    };
    const std::int32_t filled_pages = num_computed_tokens / cache_block_tokens;
    if (filled_pages > static_cast<std::int32_t>(cache_progress.page_hashes.size())) {
        appendCompletedPageHashes(cache_progress.page_hashes, request.FullPagedTokens(false), filled_pages);
    }
    if (completed.first_new_page < static_cast<std::int32_t>(cache_progress.page_hashes.size())) {
        completed.boundary_kind =
            consumeCompletedBoundaryKind(cache_progress, num_computed_tokens, request.PrefillSize());
    }
    return completed;
}

template <typename Event>
    requires(std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent> || std::same_as<Event, fsm::SchedulePrefillEvent>)
PrefillOperation applyPrefillEvent(Request& request, Event& event, const KvCacheCoordinator& coordinator,
                                   std::span<const std::string> group_ids) {
    const fsm::PrefillSource source = [&] {
        if constexpr (std::same_as<Event, fsm::SchedulePrefillFirstChunkEvent>) {
            return event.Source();
        }
        return request.PrefillSource();
    }();
    request.Apply(event);
    const PrefillInfo info = request.CurrentPrefillInfo();

    PrefillOperation operation;
    operation.request_id = request.Id();
    operation.request_pool_index = request.RequestPoolIndex();
    operation.input_length = info.extend_len;
    operation.prefill_length = request.PrefillSize();
    operation.input_ids.assign(info.input_ids.begin(), info.input_ids.end());
    operation.shifted_input_ids = info.shifted_input_ids;
    operation.extend_prefix_len = info.already_scheduled_len;
    operation.local_prefill = source == fsm::PrefillSource::kLocal;
    fillBlockTables(operation, request, coordinator, group_ids);
    return operation;
}

DecodeOperation applyDecodeEvent(Request& request, fsm::ScheduleDecodeEvent event, std::int32_t decode_input_tokens,
                                 const KvCacheCoordinator& coordinator, std::span<const std::string> group_ids) {
    request.Apply(std::move(event));

    DecodeOperation operation{{
        .request_id = request.Id(),
        .request_pool_index = request.RequestPoolIndex(),
        .input_length = decode_input_tokens,
        .prefill_length = request.PrefillSize(),
    }};
    fillBlockTables(operation, request, coordinator, group_ids);
    return operation;
}

}  // namespace

Scheduler::AdmissionMatch Scheduler::matchPrefixAtAdmission(Request* request) {
    const auto probe = [this, request](std::span<const std::string> hashes) {
        if (config_.role == Role::kD && !request->Is<fsm::Retracted>()) {
            return coordinator_.ProbeDecodeDevicePrefix(hashes);
        }
        return coordinator_.ProbePrefix(hashes);
    };
    const std::int32_t cache_block_tokens = coordinator_.CacheBlockTokens();
    // The final prompt token is always recomputed to produce logits. Some
    // consumers additionally require a larger prompt tail (for example, to
    // rebuild request-persistent state that is not stored in the KV cache).
    // Limit the probe itself so excluded hit pages are never claimed: admission
    // will allocate private writable pages for the replayed suffix.
    const std::int32_t replay_tokens = std::max(config_.prefix_replay_tokens, 1);
    const std::int32_t max_cacheable_tokens = std::max(request->PrefillSize() - replay_tokens, 0);
    const std::int32_t probe_pages = max_cacheable_tokens / cache_block_tokens;
    const std::int32_t candidate_pages = std::max((request->PrefillSize() - 1) / cache_block_tokens, 0);
    std::vector<std::span<const std::int32_t>> paged_tokens = request->FullPagedTokens(false);
    paged_tokens.resize(std::min(paged_tokens.size(), static_cast<std::size_t>(candidate_pages)));
    std::vector<std::string> hashes = ComputePagedHashes(paged_tokens, "");
    const auto probe_hashes =
        std::span<const std::string>(hashes).first(std::min(hashes.size(), static_cast<std::size_t>(probe_pages)));

    AdmissionMatch match;
    match.candidate_page_hashes = hashes;
    // Retraction recovery may reuse its own L2 snapshot even when ordinary
    // request-to-request prefix reuse is disabled.
    if (config_.disable_prefix_cache && !request->Is<fsm::Retracted>()) {
        match.probe = probe({});
        return match;
    }
    match.probe = probe(probe_hashes);
    const std::int32_t hit_pages =
        std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens) / cache_block_tokens;
    match.page_hashes.assign(hashes.begin(), hashes.begin() + hit_pages);

    const std::int32_t extension_pages =
        std::max(match.probe.host.num_common_tokens - match.probe.device.num_common_tokens, 0) / cache_block_tokens;
    const auto extension_begin = hashes.begin() + match.probe.device.num_common_tokens / cache_block_tokens;
    match.extension_hashes.assign(extension_begin, extension_begin + extension_pages);
    return match;
}

std::optional<KvCacheCoordinator::AdmissionResult> Scheduler::admit(PlanBuildContext& context,
                                                                    KvCacheCoordinator::PrefixProbe&& prefix,
                                                                    std::span<const GroupDemand> demands,
                                                                    std::optional<std::uint64_t> request_access_epoch) {
    std::optional<KvCacheCoordinator::AdmissionResult> result =
        coordinator_.Admit(std::move(prefix), demands, request_access_epoch);
    if (!result) {
        context.admission_failed = true;
        if (tier_transfers_.HasStoresInFlight()) {
            const auto pending_store_releases = tier_transfers_.DeviceLocationsReleasedOnStoreAck();
            context.waits_for_store_ack = context.waits_for_store_ack ||
                                          coordinator_.CanAdmitAfterReleasing(prefix, demands, pending_store_releases);
        }
        return std::nullopt;
    }

    _assert(result->new_page_ids.size() == cache_group_ids_.size(),
            "admission fresh-page groups must match scheduler config");
    for (std::size_t i = 0; i < result->new_page_ids.size(); ++i) {
        auto& page_ids = result->new_page_ids[i];
        auto& pending = context.plan.pages_to_zero[cache_group_ids_[i]];
        pending.insert(pending.end(), page_ids.begin(), page_ids.end());
    }
    return result;
}

std::optional<KvCacheCoordinator::AdmissionResult> Scheduler::admit(PlanBuildContext& context,
                                                                    std::span<const GroupDemand> demands,
                                                                    std::uint64_t request_access_epoch) {
    return admit(context, coordinator_.ProbePrefix({}), demands, request_access_epoch);
}

bool Scheduler::admitWithKvEventTracking(PlanBuildContext& context, Request& request,
                                         const fsm::CacheProgress& cache_progress, std::int32_t new_page_hash_begin,
                                         std::span<const GroupDemand> demands) {
    std::vector<CacheKey> event_keys = registerKvEventPages(request, cache_progress.page_hashes, new_page_hash_begin);
    const bool admitted = admit(context, demands, cache_progress.access_epoch).has_value();
    discardUncachedKvEventPages(event_keys);
    return admitted;
}

std::optional<fsm::SchedulePrefillFirstChunkEvent> Scheduler::schedulePrefillFirstChunk(
    PlanBuildContext& context, Request* request, std::int32_t remaining, std::int32_t decode_input_tokens,
    std::int32_t* out_hit_tokens) {
    if (req_pool_allocator_.AvailableSlots() == 0) {
        return std::nullopt;
    }

    AdmissionMatch match = matchPrefixAtAdmission(request);
    const std::int32_t hit_tokens = std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
    // Every admission, refused or not, with BOTH boundaries. `common` is what
    // the coordinator will actually serve -- a std::min across cache groups --
    // and `closed` is the coverage the prefix-closed groups alone supported
    // before a window or state group shortened it. Refusals already print
    // `common`; the pair is what distinguishes an empty cache from a veto.
    //
    // On a six-group model (gemma: 1 full-attention + 5 sliding) a hit pattern
    // of all-or-nothing at full size is the signature of a veto: an empty
    // cache gives graded partial matches. `closed>0, common=0` says the
    // sliding groups discarded the full-attention group's match;
    // `closed=0, common=0` says there was nothing to discard. Different
    // findings, and the pair separates them in one line.
    // Requested by roy-ad for the gemma sliding-window reuse investigation.
    {
        static const bool match_trace =
            std::fopen("/workspace/roy/.plan_trace.on", "r") != nullptr;
        if (match_trace) {
            static std::FILE* match_file = std::fopen("/workspace/roy/.admit_trace.txt", "a");
            if (match_file != nullptr) {
                std::fprintf(match_file, "match req=%s common=%d closed=%d unscheduled=%d\n",
                             request->Id().c_str(), hit_tokens,
                             std::max(match.probe.device.prefix_closed_tokens,
                                      match.probe.host.prefix_closed_tokens),
                             request->PrefillSize() - hit_tokens);
                std::fflush(match_file);
            }
        }
    }
    if (out_hit_tokens != nullptr) {
        *out_hit_tokens = hit_tokens;
    }
    const std::int32_t promotion_boundary_tokens = coordinator_.PromotionBoundaryTokens(match.probe);
    _assert(promotion_boundary_tokens == 0 ||
                (promotion_boundary_tokens % coordinator_.CacheBlockTokens() == 0 &&
                 promotion_boundary_tokens > hit_tokens && promotion_boundary_tokens < request->PrefillSize()),
            "promotion boundary must be page-aligned and inside the unmatched prompt");

    const std::int32_t unscheduled = request->PrefillSize() - hit_tokens;

    // Charge the whole remaining prompt before letting this request in, the
    // way vLLM's `scheduler_reserve_full_isl` does. Admitting on one chunk
    // makes `max_num_seqs` the only limit on residency, and a pool that cannot
    // carry them then stalls every resident request instead of a few.
    //
    // Skipped while nothing else is resident, so a prompt larger than the pool
    // still makes progress chunk by chunk rather than being refused forever.
    if (config_.reserve_full_prefill_at_admission &&
        req_pool_allocator_.AvailableSlots() < req_pool_allocator_.Size()) {
        // prompt + max_new_tokens * ratio (SGLang). With the ratio at its
        // floor this is close to vLLM's prompt-only reservation; at its init
        // it is close to TRT-LLM's charge-to-completion.
        std::int32_t generation_reserve = decode_input_tokens;
        if (config_.reserve_generation_at_admission && request->MaxNewTokens() > 0) {
            generation_reserve = std::max<std::int32_t>(
                decode_input_tokens,
                static_cast<std::int32_t>(request->MaxNewTokens() * new_token_ratio_));
        }
        std::vector<BlockTable> full_tables(static_cast<std::size_t>(coordinator_.NumGroups()));
        const std::vector<GroupDemand> full_demands = makeGroupDemands(
            full_tables, GroupDemand{.num_tokens = unscheduled, .reserve_tokens = generation_reserve});
        if (!coordinator_.CanAdmitAfterReleasing(match.probe, full_demands, {})) {
            // Record what was asked against what was free. The step counters
            // establish that every new request is turned away here; they
            // cannot say whether the shortfall is the prompt, the generation
            // reserve, or an arena more fragmented than its token total
            // suggests -- and those three have different fixes.
            static const bool admit_trace =
                std::fopen("/workspace/roy/.plan_trace.on", "r") != nullptr;
            if (admit_trace) {
                static std::FILE* admit_file = std::fopen("/workspace/roy/.admit_trace.txt", "a");
                if (admit_file != nullptr) {
                    std::fprintf(admit_file,
                                 "refused req=%s unscheduled=%d gen_reserve=%d matched=%d "
                                 "free_lcm=%d\n",
                                 request->Id().c_str(), unscheduled, generation_reserve,
                                 match.probe.device.num_common_tokens,
                                 coordinator_.NumAvailableLcmBlocks());
                    std::fflush(admit_file);
                }
            }
            context.admission_failed = true;
            context.capacity_blocker = request->Id();
            return std::nullopt;
        }
    }

    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (coordinator_.HasMambaStateGroup() || promotion_boundary_tokens > 0) {
        tokens_this_round = AlignPrefillChunk(hit_tokens, unscheduled, remaining, coordinator_.CacheBlockTokens(),
                                              promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? decode_input_tokens : 0;
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator_.NumGroups()));
    std::vector<GroupDemand> demands =
        makeGroupDemands(tables, GroupDemand{.num_tokens = tokens_this_round, .reserve_tokens = decode_reserve});

    const fsm::PrefillSource source = config_.role == Role::kD && request->Is<fsm::Submitted>()
                                          ? fsm::PrefillSource::kRemote
                                          : fsm::PrefillSource::kLocal;
    if (config_.enable_pd_cache && source == fsm::PrefillSource::kRemote) {
        for (std::size_t i = 0; i < demands.size(); ++i) {
            if (config_.paged_cache_groups[i].transfer_policy == PagedCacheTransferPolicy::LatestSnapshot) {
                demands[i].num_tokens = request->PrefillSize();
                demands[i].materialized_suffix_start =
                    (request->PrefillSize() - 1) /
                    coordinator_.GroupManager(static_cast<std::int32_t>(i)).CacheBlockTokens();
            }
        }
    }

    std::vector<CacheKey> event_keys = registerKvEventPages(*request, match.candidate_page_hashes, 0);
    std::optional<KvCacheCoordinator::AdmissionResult> admission = admit(context, std::move(match.probe), demands);
    if (!admission) {
        context.capacity_blocker = request->Id();
        discardUncachedKvEventPages(event_keys);
        return std::nullopt;
    }
    _assert(admission->promotion_boundary_tokens == promotion_boundary_tokens,
            "promotion boundary changed between probe and admission");

    if (!match.extension_hashes.empty()) {
        coordinator_.CacheFullBlocks(tables, match.extension_hashes, admission->access_epoch,
                                     admission->device_prefix_tokens / coordinator_.CacheBlockTokens());
    }
    discardUncachedKvEventPages(event_keys);
    return fsm::SchedulePrefillFirstChunkEvent{
        tokens_this_round,
        decode_reserve,
        &req_pool_allocator_,
        source,
        &coordinator_,
        std::move(tables),
        hit_tokens,
        fsm::CacheProgress{
            .page_hashes = std::move(match.page_hashes),
            .access_epoch = admission->access_epoch,
            .promotion_boundary_tokens = admission->promotion_boundary_tokens,
        },
        std::move(admission->load_pairs),
    };
}

std::optional<fsm::SchedulePrefillEvent> Scheduler::schedulePrefill(
    PlanBuildContext& context, Request* request, std::int32_t remaining,
    std::int32_t reserve_num_tokens_in_next_schedule_event) {
    const std::int32_t unscheduled = request->UnscheduledPrefillSize();
    const std::int32_t first_pos = request->PrefillSize() - unscheduled;
    fsm::CacheProgress cache_progress = request->CacheProgress();
    std::int32_t tokens_this_round = std::min(remaining, unscheduled);
    if (coordinator_.HasMambaStateGroup() || cache_progress.promotion_boundary_tokens > 0) {
        tokens_this_round = AlignPrefillChunk(first_pos, unscheduled, remaining, coordinator_.CacheBlockTokens(),
                                              cache_progress.promotion_boundary_tokens);
        if (tokens_this_round == 0) {
            return std::nullopt;
        }
    }

    const bool completes_prefill = tokens_this_round == unscheduled;
    const std::int32_t decode_reserve = completes_prefill ? reserve_num_tokens_in_next_schedule_event : 0;
    const PrefillInfo previous = request->CurrentPrefillInfo();
    const std::int32_t num_computed_tokens = previous.already_scheduled_len + previous.extend_len;
    const CompletedCachePages completed =
        updateCompletedPageHashes(*request, cache_progress, num_computed_tokens, coordinator_.CacheBlockTokens());

    std::vector<BlockTable>& tables = request->BlockTablesRef();
    std::vector<GroupDemand> demands = makeGroupDemands(tables, GroupDemand{
                                                                    .num_tokens = tokens_this_round,
                                                                    .page_hashes = cache_progress.page_hashes,
                                                                    .new_page_hash_begin = completed.first_new_page,
                                                                    .completed_boundary_kind = completed.boundary_kind,
                                                                    .num_computed_tokens = num_computed_tokens,
                                                                    .reserve_tokens = decode_reserve,
                                                                });
    if (!admitWithKvEventTracking(context, *request, cache_progress, completed.first_new_page, demands)) {
        context.capacity_blocker = request->Id();
        return std::nullopt;
    }

    return fsm::SchedulePrefillEvent{
        tokens_this_round,
        decode_reserve,
        std::move(cache_progress),
    };
}

std::optional<fsm::ScheduleDecodeEvent> Scheduler::scheduleDecode(PlanBuildContext& context, Request* request) {
    std::vector<BlockTable>& tables = request->BlockTablesRef();
    const std::int32_t reserve_tokens = request->ReserveNumTokensInNextScheduleEvent();
    fsm::CacheProgress cache_progress = request->CacheProgress();
    std::int32_t num_computed_tokens = 0;
    if (request->Is<fsm::PrefillDone>()) {
        const PrefillInfo previous = request->CurrentPrefillInfo();
        num_computed_tokens = previous.already_scheduled_len + previous.extend_len;
    } else {
        num_computed_tokens = request->TokenSize() - config_.decode_input_tokens;
    }

    const CompletedCachePages completed =
        updateCompletedPageHashes(*request, cache_progress, num_computed_tokens, coordinator_.CacheBlockTokens());

    if (completed.first_new_page == static_cast<std::int32_t>(cache_progress.page_hashes.size()) &&
        canConsumeReservedTokensInPlace(coordinator_, tables, reserve_tokens, num_computed_tokens)) {
        coordinator_.ConsumeReservedTokens(tables, reserve_tokens);
    } else {
        std::vector<GroupDemand> demands =
            makeGroupDemands(tables, GroupDemand{
                                         .num_tokens = reserve_tokens,
                                         .page_hashes = cache_progress.page_hashes,
                                         .new_page_hash_begin = completed.first_new_page,
                                         .completed_boundary_kind = completed.boundary_kind,
                                         .num_computed_tokens = num_computed_tokens,
                                     });
        if (!admitWithKvEventTracking(context, *request, cache_progress, completed.first_new_page, demands)) {
            context.capacity_blocker = request->Id();
            return std::nullopt;
        }
    }

    return fsm::ScheduleDecodeEvent{config_.decode_input_tokens, std::move(cache_progress)};
}

PrefillOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillFirstChunkEvent event,
                                                        std::vector<LoadBackOperation>& load_back_operations) {
    PrefillOperation operation = applyPrefillEvent(*request, event, coordinator_, cache_group_ids_);
    std::vector<BlockTransfer> load_pairs = event.TakeLoadPairs();
    if (load_pairs.empty()) {
        return operation;
    }

    load_back_operations.push_back(tier_transfers_.StartPrefixLoad(std::move(load_pairs)));
    return operation;
}

PrefillOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::SchedulePrefillEvent event) {
    return applyPrefillEvent(*request, event, coordinator_, cache_group_ids_);
}

DecodeOperation Scheduler::applyEventAndBuildOperation(Request* request, fsm::ScheduleDecodeEvent event) {
    const bool needs_bootstrap_token = request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    const std::int32_t bootstrap_token = needs_bootstrap_token ? request->LastToken() : -1;
    DecodeOperation operation =
        applyDecodeEvent(*request, std::move(event), config_.decode_input_tokens, coordinator_, cache_group_ids_);
    if (needs_bootstrap_token) {
        operation.decode_input_id = bootstrap_token;
    }
    return operation;
}

std::optional<WriteBackOperation> Scheduler::beginRetraction(Request& request) {
    fsm::CacheProgress cache_progress = request.CacheProgress();
    const std::int32_t num_computed_tokens = request.TokenSize() - config_.decode_input_tokens;
    const CompletedCachePages completed =
        updateCompletedPageHashes(request, cache_progress, num_computed_tokens, coordinator_.CacheBlockTokens());
    if (completed.boundary_kind) {
        coordinator_.CacheCompletedBlocks(request.BlockTablesRef(), cache_progress.page_hashes,
                                          cache_progress.access_epoch, completed.first_new_page, num_computed_tokens,
                                          *completed.boundary_kind);
    }
    coordinator_.QueueCachedBlocksForStore(cache_progress.page_hashes);
    std::optional<WriteBackOperation> write_back = tier_transfers_.StartPendingStores();
    request.Apply(fsm::RetractionEvent{&coordinator_});
    recovery_queue_.push_back(request.Id());
    return write_back;
}

void Scheduler::retractForCapacity(PlanBuildContext& context, const std::vector<Request*>& candidates,
                                   std::vector<WriteBackOperation>& write_back_operations,
                                   const std::unordered_set<std::string>& scheduled_this_step) {
    // The invariant is that a RETRACT VICTIM must not owe a forward result;
    // requiring the whole map to be empty is a conservative stand-in for it,
    // and it is what makes preemption unreachable once any op is scheduled.
    // With per-request triggering the victim filter below enforces the real
    // invariant instead. The other three conditions are unrelated to it and
    // stay as they are.
    const bool blanket_forward_guard = !config_.preempt_on_blocked_request && !pending_forward_results_.empty();
    if ((config_.role != Role::kFused && config_.role != Role::kD) || !context.admission_failed ||
        blanket_forward_guard || !pd_transfer_pins_.empty() || context.waits_for_store_ack ||
        tier_transfers_.HasLoadBacksInFlight()) {
        return;
    }
    // A victim must owe NO outstanding forward result before it is torn down:
    // filtering on `scheduled_this_step` instead crashed the scheduler with
    // "FSM transition invalid: event=ExtendResultEvent; state=Retracted",
    // because at overlap depth 1 a request can owe a result issued a step
    // earlier and that result arrives after the retraction.
    //
    // Requiring the whole map to be empty before even looking used to stand in
    // for that. It is far stronger than the invariant needs, and once the step
    // stopped ending itself on a blocked prefill it was never satisfied: some
    // request has been scheduled in essentially every step, so retraction
    // stopped running at all and capacity was never reclaimed. The two-phase
    // path below is the real implementation -- mark the victim now, tear it
    // down when its result lands -- so the per-request filter is enough.
    // Both halves are needed. `pending_forward_results_` alone misses a
    // request whose operation was pushed in THIS step but whose counter has
    // not been raised yet, and retracting one of those tore it down before its
    // result arrived: "FSM transition invalid: event=ExtendResultEvent;
    // state=Submitted". `scheduled_this_step` alone misses a request that owes
    // a result issued a step earlier, which is the other half of the same
    // crash. Either condition sends the request down the deferred path
    // instead, which is where both cases are safe.
    const auto owes_forward_result = [this, &scheduled_this_step](const Request* request) {
        return pending_forward_results_.contains(request->Id()) || scheduled_this_step.contains(request->Id());
    };
    // Already marked victims are not candidates again, and are not counted as
    // resident capacity that could be freed twice.
    const auto already_marked = [this](const Request* request) {
        return pending_retraction_.contains(request->Id());
    };

    if (config_.role == Role::kD) {
        Request* victim = nullptr;
        std::optional<std::tuple<std::int32_t, std::int32_t, std::string>> victim_rank;
        for (Request* request : candidates) {
            if (!request->Is<fsm::Decoding>() || owes_forward_result(request)) {
                continue;
            }
            auto rank = std::tuple{-coordinator_.NumNewlyReleasableLcmBlocks(request->BlockTablesRef()),
                                   request->TokenSize(), request->Id()};
            if (!victim_rank || rank < *victim_rank) {
                victim = request;
                victim_rank = std::move(rank);
            }
        }
        if (victim == nullptr) {
            // Under per-request triggering every resident request may already
            // owe a forward result this step. That is not a wedge -- the next
            // step retries once those results land -- so it must not abort.
            // It used to abort anyway, because the FatalCheck here was
            // predicated on a config flag rather than on the situation being
            // unrecoverable.
            spdlog::debug("[Scheduler] retract: no eligible Decode victim; admitting nothing this step");
            return;
        }
        recovery_barrier_ = context.capacity_blocker;
        if (!recovery_barrier_ || *recovery_barrier_ == victim->Id()) {
            const auto waiting = std::ranges::find_if(candidates, [victim](const Request* request) {
                return request != victim && (request->Is<fsm::Submitted>() || request->Is<fsm::Prefilling>());
            });
            recovery_barrier_ =
                waiting == candidates.end() ? std::nullopt : std::optional<std::string>{(*waiting)->Id()};
        }
        if (auto operation = beginRetraction(*victim)) {
            write_back_operations.push_back(std::move(*operation));
        }
        spdlog::info("[Scheduler] retract: released request {} ({} tokens) with best-effort L2 store", victim->Id(),
                     victim->TokenSize());
        return;
    }

    Request* request_to_retract = nullptr;
    Request* deferred_victim = nullptr;
    for (Request* request : candidates) {
        if (!(request->Is<fsm::Decoding>() || request->Is<fsm::PrefillDone>()) || already_marked(request)) {
            continue;
        }
        if (owes_forward_result(request)) {
            // Cannot be torn down now, but can be marked: it stops receiving
            // work immediately and is retracted once its result lands.
            if (deferred_victim == nullptr || request->TokenSize() > deferred_victim->TokenSize()) {
                deferred_victim = request;
            }
            continue;
        }
        if (request_to_retract == nullptr || request->TokenSize() > request_to_retract->TokenSize()) {
            request_to_retract = request;
        }
    }
    // A retract means the reservation was too optimistic; SGLang re-estimates
    // upward from what the victims actually generated rather than resetting to
    // a constant.
    if (config_.reserve_generation_at_admission) {
        new_token_ratio_ = std::min(1.0, new_token_ratio_ + (config_.new_token_ratio_init - new_token_ratio_) * 0.5);
    }
    if (request_to_retract == nullptr && deferred_victim != nullptr) {
        pending_retraction_.insert(deferred_victim->Id());
        spdlog::info("[Scheduler] retract: marked request {} ({} tokens); it owes a forward result",
                     deferred_victim->Id(), deferred_victim->TokenSize());
        return;
    }
    if (request_to_retract == nullptr) {
        // Every resident request is either scheduled this step or owes a
        // forward result, so there is nothing to free right now. That is a
        // step which admits nobody, not an engine that cannot continue: the
        // next step sees a different set. Crashing here took down the 35B
        // agent-127k cell, where a single prompt legitimately needs more arena
        // than one step can free.
        spdlog::debug("[Scheduler] retract: no eligible victim; admitting nothing this step");
        return;
        return;
    }
    if (config_.HasHostCache() && request_to_retract->Is<fsm::Decoding>()) {
        if (auto operation = beginRetraction(*request_to_retract)) {
            write_back_operations.push_back(std::move(*operation));
        }
        spdlog::info("[Scheduler] retract: released request {} ({} tokens) with best-effort L2 store",
                     request_to_retract->Id(), request_to_retract->TokenSize());
        return;
    }
    request_to_retract->Apply(fsm::RetractEvent{&coordinator_});
    spdlog::info("[Scheduler] retract: released request {} ({} tokens) for LCM capacity", request_to_retract->Id(),
                 request_to_retract->TokenSize());
}

std::pair<std::vector<ForwardOperation>, std::vector<LoadBackOperation>> Scheduler::buildForwardOperations(
    ExecutionPlan& plan, std::vector<Request*> candidates, std::vector<WriteBackOperation>& write_back_operations) {
    PlanBuildContext context{plan};

    // SGLang decays the ratio on every step that did not retract, so a run
    // that is coping drifts toward admitting more.
    if (config_.reserve_generation_at_admission) {
        const double floor = config_.new_token_ratio_init * config_.new_token_ratio_min_factor;
        const double step = config_.new_token_ratio_decay_steps > 0
                                ? (config_.new_token_ratio_init - floor) / config_.new_token_ratio_decay_steps
                                : 0.0;
        new_token_ratio_ = std::max(floor, new_token_ratio_ - step);
    }

    // Phase two of a deferred retract: the debt has cleared, and this is a plan
    // build, so a write-back operation has somewhere to go.
    for (auto it = pending_retraction_.begin(); it != pending_retraction_.end();) {
        Request* victim = findRequest(*it);
        if (victim == nullptr || victim->Is<fsm::Finished>() || victim->Is<fsm::Retracted>()) {
            it = pending_retraction_.erase(it);
            continue;
        }
        if (pending_forward_results_.contains(*it)) {
            ++it;  // still owes a result; try again next step
            continue;
        }
        if (config_.HasHostCache() && victim->Is<fsm::Decoding>()) {
            if (auto operation = beginRetraction(*victim)) {
                write_back_operations.push_back(std::move(*operation));
            }
        } else {
            victim->Apply(fsm::RetractEvent{&coordinator_});
        }
        spdlog::info("[Scheduler] retract: released marked request {} ({} tokens)", victim->Id(),
                     victim->TokenSize());
        it = pending_retraction_.erase(it);
    }
    while (!recovery_queue_.empty()) {
        Request* request = findRequest(recovery_queue_.front());
        if (request != nullptr && !request->Is<fsm::Finished>()) {
            break;
        }
        recovery_queue_.pop_front();
    }
    if (recovery_barrier_) {
        Request* request = findRequest(*recovery_barrier_);
        if (request == nullptr || request->Is<fsm::Finished>() || request->Is<fsm::Retracted>()) {
            recovery_barrier_.reset();
        }
    }
    const auto priority = [this](const Request* request) {
        const bool recovery_front = !recovery_queue_.empty() && request->Id() == recovery_queue_.front();
        const bool local_decode_prefill =
            request->Is<fsm::Prefilling>() && request->PrefillSource() == fsm::PrefillSource::kLocal;
        if (config_.role == Role::kD && (local_decode_prefill || request->Is<fsm::PrefillDone>() ||
                                         (recovery_front && request->Is<fsm::Decoding>()))) {
            // Keep the oldest retracted request at the head of line through
            // local recovery and Decode completion. Starting another recovery
            // earlier can make the two requests repeatedly evict each other.
            return 0;
        }
        if (recovery_barrier_ && request->Id() == *recovery_barrier_) {
            return 1;
        }
        if (request->Is<fsm::Retracted>() && !recovery_queue_.empty() && request->Id() == recovery_queue_.front()) {
            return 2;
        }
        if (request->Is<fsm::Prefilling>()) {
            return 3;
        }
        if (request->Is<fsm::Submitted>()) {
            return 4;
        }
        if (request->Is<fsm::Decoding>() || request->Is<fsm::PrefillDone>()) {
            return config_.enable_mixed_prefill_decode ? 3 : 5;
        }
        return 10;
    };
    std::ranges::sort(candidates, [&](const Request* lhs, const Request* rhs) {
        const int lhs_priority = priority(lhs);
        const int rhs_priority = priority(rhs);
        return lhs_priority != rhs_priority ? lhs_priority < rhs_priority : lhs->Id() < rhs->Id();
    });

    const bool has_local_prefill = std::ranges::any_of(candidates, [this](const Request* request) {
        return (request->Is<fsm::Prefilling>() && request->PrefillSource() == fsm::PrefillSource::kLocal) ||
               (config_.role != Role::kD && request->Is<fsm::Submitted>()) ||
               (request->Is<fsm::Retracted>() && !recovery_queue_.empty() && request->Id() == recovery_queue_.front());
    });
    const std::int32_t state_prefill_reserve =
        config_.enable_mixed_prefill_decode && coordinator_.HasMambaStateGroup() && has_local_prefill
            ? coordinator_.CacheBlockTokens()
            : 0;

    std::vector<ForwardOperation> operations;
    std::vector<LoadBackOperation> load_back_operations;
    std::int32_t token_budget = config_.max_scheduled_tokens;
    // Capacity ran out, or an incomplete prefill is holding the line. Either
    // way nothing NEW may be admitted -- but the requests already resident can
    // still decode, and ending the whole step instead is what left 8192 tokens
    // of budget carrying 3.5 decodes.
    bool admissions_closed = false;
    bool pushed_prefill = false;
    bool pushed_decode = false;
    // Requests that received an operation in THIS step. The retract victim
    // must not be one of them. `pending_forward_results_` is the wrong set to
    // filter on: it holds every request still awaiting a result from ANY
    // earlier step, and at overlap depth 1 that is nearly every resident
    // request, so a filter built on it finds no eligible victim in steady
    // state and preemption silently does nothing (51 retracts over a run that
    // needed thousands).
    std::unordered_set<std::string> scheduled_this_step;
    // Why a step admitted what it did. Two very different defects look
    // identical from the outside -- a step that ran out of token budget and a
    // step that had budget left but never offered it to the next request --
    // and the running-request histogram cannot tell them apart. Off unless
    // TOKENSPEED_LOG_PLAN_TRACE is set; one line per plan build when it is.
    // Triggered by the existence of a file, not an environment variable, and
    // written straight to disk rather than through spdlog. Both indirections
    // were forced: the engine runs the scheduler in a spawned process that
    // neither forwards this variable nor surfaces its spdlog sink in the
    // harness log, so an env-gated spdlog trace stays silent and reads exactly
    // like code that never ran.
    static const char* plan_trace_path = "/workspace/roy/.plan_trace.txt";
    static const bool plan_trace = std::fopen("/workspace/roy/.plan_trace.on", "r") != nullptr;
    const std::int32_t trace_budget_start = token_budget;
    const std::size_t trace_candidates = candidates.size();
    const char* trace_stop = "candidates exhausted";
    std::int32_t trace_prefill_ops = 0, trace_decode_ops = 0, trace_prefill_tokens = 0;
    std::int32_t trace_skip_retract = 0, trace_skip_reserve = 0, trace_no_event = 0;
    std::int32_t trace_first_chunk_tokens = -1;
    // Which state each candidate arrived in, and which call turned it away.
    // "candidates exhausted with nothing scheduled" is the common outcome and
    // it names no culprit; these do.
    std::int32_t st_submitted = 0, st_prefilling = 0, st_decode = 0, st_other = 0;
    std::int32_t rej_prefill = 0, rej_firstchunk = 0, rej_decode = 0, rej_unmatched = 0;
    // roy-ad's facd44a3: admissions this step whose prefix matched nothing.
    // They duplicate each other's prefill work, because they all probe before
    // any of them publishes. Carried here so ONE installed extension serves
    // both branches -- a .so that two branches write to is a shared mutable
    // global, and it broke both of us today, once in each direction.
    std::int32_t cold_prefills = 0;
    std::int32_t rej_cold_cap = 0;

    auto push_operation = [&](auto operation) {
        scheduled_this_step.insert(operation.request_id);
        if (recovery_barrier_ && operation.request_id == *recovery_barrier_) {
            recovery_barrier_.reset();
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(operation)>, PrefillOperation>) {
            if (config_.role != Role::kD || operation.local_prefill) {
                token_budget -= operation.input_length;
            }
            pushed_prefill = true;
        } else if (config_.role != Role::kD) {
            token_budget -= operation.input_length;
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(operation)>, DecodeOperation>) {
            pushed_decode = true;
            ++trace_decode_ops;
        } else {
            ++trace_prefill_ops;
            trace_prefill_tokens += operation.input_length;
            if (trace_first_chunk_tokens < 0) {
                trace_first_chunk_tokens = operation.input_length;
            }
        }
        operations.push_back(std::move(operation));
    };
    const auto trackPendingForwardResult = [&](const Request* request) {
        // Intermediate prefill and D-side cache admission do not produce a
        // forward::ExtendResult.
        if (request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>()) {
            ++pending_forward_results_[request->Id()];
        }
    };

    // --- equal-progress chunking -------------------------------------------
    // Ported from TRT-LLM `setCtxRequestsChunkSize<kEQUAL_PROGRESS>`
    // (microBatchScheduler.cpp:126): offer every mid-prefill request one unit
    // per pass, repeatedly, until the budget is spent or no request can take
    // another unit. The result is a per-request cap for THIS step; the step's
    // total budget is unchanged, so this reorders completions without changing
    // makespan.
    //
    // Deliberately not TRT-LLM's reuse discount: their loop charges only the
    // non-cached portion, which is a second independent change. One variable
    // at a time -- charging by cached bytes is a separate experiment.
    std::unordered_map<std::string, std::int32_t> equal_progress_cap;
    if (config_.context_chunk_policy == SchedulerConfig::ContextChunkPolicy::kEqualProgress) {
        const std::int32_t unit =
            config_.context_chunk_unit > 0 ? config_.context_chunk_unit : coordinator_.CacheBlockTokens();
        std::vector<Request*> mid_prefill;
        for (Request* request : candidates) {
            if (pending_retraction_.contains(request->Id())) {
                continue;
            }
            const bool local_prefill =
                request->Is<fsm::Prefilling>() &&
                (config_.role != Role::kD || request->PrefillSource() == fsm::PrefillSource::kLocal);
            if (local_prefill || (config_.role != Role::kD && request->Is<fsm::Submitted>())) {
                mid_prefill.push_back(request);
                equal_progress_cap.emplace(request->Id(), 0);
            }
        }
        // A single prefill in flight has nobody to share with, and capping it
        // at one unit would be strictly worse than FCFS for no benefit.
        if (mid_prefill.size() > 1 && unit > 0) {
            std::int32_t spent = 0;
            bool progressed = true;
            while (spent < token_budget && progressed) {
                progressed = false;
                for (Request* request : mid_prefill) {
                    std::int32_t& cap = equal_progress_cap[request->Id()];
                    // `UnscheduledPrefillSize()` returns -1 for a Submitted
                    // request -- it has no prefill window yet. Feeding that to
                    // the round-robin made `cap >= want` true on the first
                    // pass for every request, so nobody was ever granted a
                    // unit, every cap stayed 0, and the step scheduled NOTHING.
                    // The unit test caught it as 0 prefill ops; on a card it
                    // would have been a stalled engine reported as a policy
                    // that "made throughput worse".
                    const std::int32_t want = request->Is<fsm::Submitted>()
                                                  ? request->PrefillSize()
                                                  : request->UnscheduledPrefillSize();
                    if (want <= 0) {
                        continue;
                    }
                    if (cap >= want) {
                        continue;  // this request already has its whole prompt
                    }
                    // Grants must stay page-aligned. Hybrid models refuse an
                    // unaligned chunk, and clamping by the leftover budget
                    // (`token_budget - spent`) can produce a partial page --
                    // which then propagates into the block geometry the
                    // executor maps. That is the difference between a chunk
                    // the scheduler declines and one it hands downstream in a
                    // shape nothing expects: measured as SIGBUS inside
                    // next_execution_plan() after several healthy shared steps.
                    //
                    // A partial grant is legitimate in exactly one case: it
                    // completes the request's remaining prompt, which is the
                    // prompt's own tail and already the last chunk.
                    const std::int32_t remaining_want = want - cap;
                    const std::int32_t budget_left = token_budget - spent;
                    std::int32_t grant = std::min(unit, remaining_want);
                    if (grant > budget_left) {
                        // Only whole units may come out of the leftover budget.
                        grant = (budget_left / unit) * unit;
                    }
                    if (grant > 0 && grant < unit && grant != remaining_want) {
                        grant = 0;  // partial page that is not a prompt tail
                    }
                    if (grant <= 0) {
                        continue;
                    }
                    cap += grant;
                    spent += grant;
                    progressed = true;
                    if (spent >= token_budget) {
                        break;
                    }
                }
            }
        } else {
            equal_progress_cap.clear();  // fall back to FCFS for this step
        }
    }

    for (Request* request : candidates) {
        if (token_budget <= 0 || operations.size() == static_cast<std::size_t>(config_.max_batch_size)) {
            trace_stop = token_budget <= 0 ? "token budget exhausted" : "max_batch_size reached";
            break;
        }
        // A marked victim must stop growing immediately, or the pages the
        // retraction is meant to free keep accumulating while it waits for its
        // outstanding result to land.
        if (pending_retraction_.contains(request->Id())) {
            ++trace_skip_retract;
            continue;
        }

        if (request->Is<fsm::Prefilling>() &&
            (config_.role != Role::kD || request->PrefillSource() == fsm::PrefillSource::kLocal)) {
            if (config_.role == Role::kD && pushed_decode) {
                trace_stop = "D-role: decode already pushed";
                break;
            }
            const std::int32_t reserve = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            ++st_prefilling;
            // Under equal progress a request may take only its share of this
            // step. An empty map means the policy is off, or it declined to
            // engage because fewer than two requests were mid-prefill.
            std::int32_t offer = token_budget;
            if (const auto it = equal_progress_cap.find(request->Id()); it != equal_progress_cap.end()) {
                offer = std::min(offer, it->second);
            }
            if (offer <= 0) {
                ++trace_no_event;
                continue;
            }
            // Forward progress is not optional. A capped offer can be smaller
            // than one aligned chunk -- AlignPrefillChunk refuses an unaligned
            // chunk on hybrid models and returns nullopt -- and if every
            // request in the step is capped that way, the step schedules
            // nothing, forever. Measured as a live hang: 16 sequences sharing
            // the step correctly, then 20 minutes of zero GPU utilisation with
            // the pool still held.
            //
            // So: if the capped offer yields no event AND this step has
            // scheduled nothing at all, fall back to the uncapped budget for
            // this one request. Equal progress is a policy about fairness
            // among requests that CAN run; it must never be the reason none do.
            auto event = schedulePrefill(context, request, offer, reserve);
            if (!event && offer < token_budget && operations.empty()) {
                event = schedulePrefill(context, request, token_budget, reserve);
            }
            if (event) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event)));
                // P-side pages stay pinned until the PD transfer completes.
                // D-side local recovery has no corresponding PD ACK; its
                // Host/Device lifetime is already owned by the L2 load ticket.
                if (config_.enable_pd_cache && config_.role != Role::kD) {
                    pd_transfer_pins_.insert(request->Id());
                }
                trackPendingForwardResult(request);
                if (config_.role == Role::kD) {
                    // Decode-side recovery prefill runs in its own batch.
                    trace_stop = "D-role: recovery prefill is its own batch";
                    break;
                }
                if (request->Is<fsm::Prefilling>()) {
                    // Admission reserves only this chunk, not the request's
                    // remaining prompt. Keep one incomplete prefill as the
                    // head of line so another request cannot strand it by
                    // consuming the capacity it needs to finish.
                    //
                    // That premise no longer holds when admission reserves the
                    // whole prompt: the capacity this request needs to finish
                    // is already charged to it, so nothing another request
                    // admits can strand it, and holding the line costs far
                    // more than it protects. A prompt longer than the token
                    // budget takes ceil(prompt / budget) steps, and while it
                    // holds the line every other request waits for all of
                    // them -- on the 35B agent shape (43k-token prompts, 8192
                    // budget, 16 concurrent) that measured 1.8 requests
                    // resident against vLLM's 12, TTFT p50 18.1 s against
                    // 3.4 s, while our TPOT was 4.4x better because almost
                    // nothing was sharing the decode.
                    //
                    // Holding the line is about admission, so it closes
                    // admission. Ending the step was a much bigger hammer: the
                    // decodes sorted after this request lost their slot too,
                    // and they were not competing for the capacity it needs.
                    if (!config_.reserve_full_prefill_at_admission) {
                        admissions_closed = true;
                    }
                }
            } else if (context.admission_failed) {
                // Still ends the step, because retraction is triggered on one
                // that came up empty. Letting the decodes through instead
                // needs the retract path reworked first: the victim filter,
                // the trigger, and an `outside_event_handler` that erases the
                // whole pending-result counter where it should decrement.
                // Half of that lands a crash -- "FSM transition invalid:
                // event=ExtendResultEvent; state=Submitted" -- so it is
                // deliberately not half-landed here.
                break;
            }
            continue;
        }

        if (request->Is<fsm::Retracted>() && !recovery_queue_.empty() && request->Id() == recovery_queue_.front()) {
            if (config_.role == Role::kD && pushed_decode) {
                break;
            }
            if (auto event = schedulePrefillFirstChunk(context, request, token_budget, config_.decode_input_tokens)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event), load_back_operations));
                trackPendingForwardResult(request);
                if (config_.role == Role::kD || request->Is<fsm::Prefilling>()) {
                    // Keep a Decode-side recovery batch local-only.
                    break;
                }
                continue;
            }
            break;
        }

        if (request->Is<fsm::Submitted>()) {
            if (admissions_closed) {
                ++rej_firstchunk;
                continue;
            }
            if (config_.role == Role::kD && pushed_decode) {
                trace_stop = "D-role: decode already pushed (first chunk)";
                break;
            }
            const std::int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
            std::int32_t prefill_budget = config_.role == Role::kD ? request->PrefillSize() : token_budget;
            // The FIRST chunk is where a long prompt starts monopolising, so
            // equal progress has to apply here and not only to continuations.
            // A unit test caught this: with the longest request exceeding the
            // step budget -- the regime the policy exists for -- every request
            // is still Submitted on the first plan, so patching only the
            // continuation path left the policy inert exactly where it matters.
            //
            // Not applied to the recovery path above: a retracted request is
            // pinned through recovery on purpose, because two of them can
            // otherwise evict each other forever. Sharing its step would
            // reintroduce that.
            if (const auto it = equal_progress_cap.find(request->Id());
                it != equal_progress_cap.end() && config_.role != Role::kD) {
                prefill_budget = std::min(prefill_budget, it->second);
            }
            if (prefill_budget <= 0) {
                ++trace_no_event;
                continue;
            }
            if (config_.max_cold_prefills > 0 && cold_prefills >= config_.max_cold_prefills) {
                ++rej_cold_cap;
                continue;
            }
            ++st_submitted;
            const std::int32_t uncapped_budget =
                config_.role == Role::kD ? request->PrefillSize() : token_budget;
            std::int32_t hit_tokens = 0;
            auto event =
                schedulePrefillFirstChunk(context, request, prefill_budget, decode_input_tokens, &hit_tokens);
            if (!event && prefill_budget < uncapped_budget && operations.empty()) {
                // Same forward-progress guarantee as the continuation path.
                event = schedulePrefillFirstChunk(context, request, uncapped_budget, decode_input_tokens,
                                                  &hit_tokens);
            }
            if (event) {
                if (hit_tokens == 0) {
                    ++cold_prefills;
                }
                push_operation(applyEventAndBuildOperation(request, std::move(*event), load_back_operations));
                if (config_.enable_pd_cache) {
                    pd_transfer_pins_.insert(request->Id());
                }
                trackPendingForwardResult(request);
                if (request->Is<fsm::Prefilling>()) {
                    // Same head-of-line rule as the continuation branch above,
                    // for a request whose first chunk did not finish its
                    // prompt. Reserving the whole prompt at admission removes
                    // the stranding risk it guards against.
                    if (!config_.reserve_full_prefill_at_admission) {
                        admissions_closed = true;
                    }
                }
            } else {
                ++rej_firstchunk;
            }
            continue;
        }

        if (request->Is<fsm::PrefillDone>() || (request->Is<fsm::Decoding>() && config_.role != Role::kP)) {
            if ((config_.role == Role::kD || !config_.enable_mixed_prefill_decode) && pushed_prefill) {
                trace_stop = "mixed prefill+decode disabled and a prefill is already in the step";
                break;
            }
            if (token_budget < state_prefill_reserve + config_.decode_input_tokens) {
                ++trace_skip_reserve;
                continue;
            }
            ++st_decode;
            if (auto event = scheduleDecode(context, request)) {
                push_operation(applyEventAndBuildOperation(request, std::move(*event)));
                trackPendingForwardResult(request);
            } else {
                ++rej_decode;
            }
            continue;
        }

        // Matched none of the branches above: neither prefill nor decode was
        // even attempted for this candidate.
        ++st_other;
        ++rej_unmatched;
    }

    if (plan_trace) {
        static std::FILE* trace_file = std::fopen(plan_trace_path, "a");
        if (trace_file != nullptr) {
            std::fprintf(trace_file,
                         "budget %d->%d used=%d candidates=%zu prefill_ops=%d first_chunk=%d "
                         "decode_ops=%d waiting=%zu skipped(retract=%d reserve=%d) "
                         "state(sub=%d pre=%d dec=%d other=%d) "
                         "rejected(firstchunk=%d prefill=%d decode=%d unmatched=%d cold_cap=%d) cold=%d "
                         "stop=\"%s\"\n",
                         trace_budget_start, token_budget, trace_budget_start - token_budget,
                         trace_candidates, trace_prefill_ops, trace_first_chunk_tokens,
                         trace_decode_ops,
                         trace_candidates - static_cast<std::size_t>(trace_prefill_ops + trace_decode_ops),
                         trace_skip_retract, trace_skip_reserve,
                         st_submitted, st_prefilling, st_decode, st_other,
                         rej_firstchunk, rej_prefill, rej_decode, rej_unmatched, rej_cold_cap,
                         cold_prefills, trace_stop);
            std::fflush(trace_file);
        }
    }

    // vLLM preempts for the ONE request that could not get blocks, inside its
    // running loop. Waiting for the whole step to come up empty means a single
    // request with room in its tail page suppresses preemption for everyone --
    // and once a blocked prefill stopped ending the step, the step was never
    // empty again, so this condition suppressed it always. Admission failing
    // is the trigger; whether anything else got scheduled is unrelated to it.
    if (context.admission_failed && (operations.empty() || config_.preempt_on_blocked_request)) {
        retractForCapacity(context, candidates, write_back_operations, scheduled_this_step);
    }
    return {std::move(operations), std::move(load_back_operations)};
}

}  // namespace tokenspeed
