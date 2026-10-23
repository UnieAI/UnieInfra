// Copyright (c) 2026 UnieAI
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

// Token-level prefix hits end to end through the Scheduler: requests publish
// their partial pages at prompt end and finish, later prompts resume at the
// longest published boundary that is a prefix of theirs, and the admission
// carries the copy of the block containing that boundary.

// Two prompts that share pages past the cached prefix, admitted while the
// first is still computing them, would both compute those pages: none is
// published before its owner's chunk ends. The later prompt waits a round
// instead (SchedulerConfig::inflight_prefix_defer_pages) and then resumes on
// the pages the first one published.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

class InflightPrefixDedupSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kP = 4;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = 257;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 256;
        cfg.max_batch_size = 16;
        cfg.inflight_prefix_defer_pages = DeferPages();
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = kP,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
        });
        if (WithSlidingGroup()) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = "swa",
                .block_granularity = kP,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = AttnKind::kSlidingWindow,
                .sliding_window_tokens = 8,
            });
        }
        if (WithStateGroup()) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = "linear_attention",
                .block_granularity = 0,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = AttnKind::kState,
            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
    virtual std::int32_t DeferPages() const { return 4; }
    virtual bool WithSlidingGroup() const { return false; }
    virtual bool WithStateGroup() const { return false; }

    // `shared` tokens common to every prompt, then a private tail of `tail`.
    static std::vector<std::int32_t> Prompt(std::int32_t shared, std::int32_t tail, std::int32_t salt) {
        std::vector<std::int32_t> tokens;
        for (std::int32_t i = 0; i < shared; ++i) {
            tokens.push_back(1000 + i);
        }
        for (std::int32_t i = 0; i < tail; ++i) {
            tokens.push_back(100000 * (salt + 1) + i);
        }
        return tokens;
    }

    static bool Holds(const ForwardBatch* batch, const std::string& id) {
        return batch != nullptr && std::ranges::find(batch->request_ids, id) != batch->request_ids.end();
    }

    static std::int32_t PrefixOf(const ForwardBatch* batch, const std::string& id) {
        const auto it = std::ranges::find(batch->request_ids, id);
        return batch->extend_prefix_lens.at(static_cast<std::size_t>(it - batch->request_ids.begin()));
    }
};

TEST_F(InflightPrefixDedupSuite, TheLaterPromptWaitsAndResumesOnThePublishedPages) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 6, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(32, 6, 1), .max_new_tokens = 4});

    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_TRUE(Holds(batch, "a"));
    EXPECT_FALSE(Holds(batch, "b")) << "b would compute the 8 pages a is computing";

    // b is admitted once a's prompt pages are published, and hits all of them.
    SendForwardDone("a", {7});
    std::int32_t b_prefix = -1;
    for (std::int32_t round = 0; round < 4 && b_prefix < 0; ++round) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* next = FindForwardBatch(plan);
        if (Holds(next, "a")) {
            SendForwardDone("a", {8 + round});
        }
        if (Holds(next, "b")) {
            b_prefix = PrefixOf(next, "b");
        }
    }
    EXPECT_EQ(b_prefix, 32);
    EXPECT_EQ(scheduler_->PrefixDedup().deferred_requests, 1) << "b counts once however many rounds it waited";
    EXPECT_GE(scheduler_->PrefixDedup().deferred_rounds, 1);
    EXPECT_EQ(scheduler_->PrefixDedup().cut_chunks, 0) << "full-history groups take no cut";
}

TEST_F(InflightPrefixDedupSuite, APromptSharingFewerPagesIsNotHeldBack) {
    // Three shared pages lead b's uncached suffix: under the threshold.
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(12, 20, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(12, 20, 1), .max_new_tokens = 4});

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    EXPECT_TRUE(Holds(batch, "a"));
    EXPECT_TRUE(Holds(batch, "b"));
}

TEST_F(InflightPrefixDedupSuite, UnrelatedPromptsAreAdmittedTogether) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(0, 40, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(0, 40, 1), .max_new_tokens = 4});

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    EXPECT_TRUE(Holds(batch, "a"));
    EXPECT_TRUE(Holds(batch, "b"));
}

// A sliding group keeps only the window behind a chunk end, so the shared
// pages are reusable only if the first prompt's chunk ends where the sharing
// ends: its first chunk is cut there, and the second prompt resumes on it.
class InflightPrefixDedupSlidingSuite : public InflightPrefixDedupSuite {
protected:
    bool WithSlidingGroup() const override { return true; }
};

TEST_F(InflightPrefixDedupSlidingSuite, TheFirstChunkEndsWhereTheSharingEnds) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 20, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(32, 20, 1), .max_new_tokens = 4});

    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_TRUE(Holds(batch, "a"));
    EXPECT_FALSE(Holds(batch, "b"));
    EXPECT_EQ(batch->input_lengths.at(0), 32) << "a's first chunk stops at the end of the shared pages";
    EXPECT_EQ(scheduler_->PrefixDedup().cut_chunks, 1);

    SendForwardDone("a");
    const ExecutionPlan second = PlanOnce();
    const ForwardBatch* next = FindForwardBatch(second);
    ASSERT_TRUE(Holds(next, "a"));
    ASSERT_TRUE(Holds(next, "b"));
    EXPECT_EQ(PrefixOf(next, "b"), 32);
}

TEST_F(InflightPrefixDedupSlidingSuite, TheFirstChunkEndsAtTheNearestBranchPoint) {
    // a and b share 8 pages; a and c (a later turn of a) share all of a's.
    // The nearest branch point is b's, after page 8.
    std::vector<std::int32_t> c_tokens = Prompt(32, 20, 0);
    c_tokens.insert(c_tokens.end(), {7, 7, 7, 7, 7, 7, 7, 7});
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 20, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(32, 20, 1), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "c", .tokens = c_tokens, .max_new_tokens = 4});

    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_TRUE(Holds(batch, "a"));
    EXPECT_FALSE(Holds(batch, "b"));
    EXPECT_FALSE(Holds(batch, "c"));
    EXPECT_EQ(batch->input_lengths.at(0), 32);

    SendForwardDone("a");
    const ExecutionPlan second = PlanOnce();
    const ForwardBatch* next = FindForwardBatch(second);
    ASSERT_TRUE(Holds(next, "b"));
    EXPECT_EQ(PrefixOf(next, "b"), 32);
    EXPECT_FALSE(Holds(next, "c")) << "c waits for the rest of a's prompt";
}

TEST_F(InflightPrefixDedupSuite, APromptThatMayNotHitTheSharedPagesIsNotHeldBack) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 6, 0), .max_new_tokens = 4});
    Submit(
        RequestSpec{.request_id = "b", .tokens = Prompt(32, 6, 1), .max_new_tokens = 4, .max_cached_prefix_tokens = 8});

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    EXPECT_TRUE(Holds(batch, "a"));
    EXPECT_TRUE(Holds(batch, "b")) << "b may take 2 pages from the cache, fewer than the threshold";
    EXPECT_EQ(scheduler_->PrefixDedup().deferred_requests, 0);
}

class InflightPrefixDedupNoPrefixCacheSuite : public InflightPrefixDedupSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = InflightPrefixDedupSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(InflightPrefixDedupNoPrefixCacheSuite, WithoutPrefixMatchingNothingWaits) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 6, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(32, 6, 1), .max_new_tokens = 4});

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    EXPECT_TRUE(Holds(batch, "a"));
    EXPECT_TRUE(Holds(batch, "b")) << "b could never hit a's pages, so waiting for them gains nothing";
    EXPECT_EQ(scheduler_->PrefixDedup().deferred_requests, 0);
}

class InflightPrefixDedupStateSuite : public InflightPrefixDedupSuite {
protected:
    bool WithStateGroup() const override { return true; }
};

// A State group publishes its checkpoint only at a junction or the prompt end,
// so the branch-point cut is a's junction: b resumes from the state it leaves
// there, not from KV alone it could not use.
TEST_F(InflightPrefixDedupStateSuite, TheCutPublishesAStateCheckpointTheWaitingPromptResumesFrom) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 20, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(32, 20, 1), .max_new_tokens = 4});

    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_TRUE(Holds(batch, "a"));
    EXPECT_FALSE(Holds(batch, "b"));
    EXPECT_EQ(batch->input_lengths.at(0), 32);

    SendForwardDone("a");
    std::int32_t b_prefix = -1;
    for (std::int32_t round = 0; round < 4 && b_prefix < 0; ++round) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* next = FindForwardBatch(plan);
        if (Holds(next, "b")) {
            b_prefix = PrefixOf(next, "b");
            const auto it = std::ranges::find(next->request_ids, "b");
            const auto row = static_cast<std::size_t>(it - next->request_ids.begin());
            EXPECT_GT(next->block_tables.at("linear_attention").at(row).at(0), 0)
                << "b reads a's checkpoint, not a zero state";
        }
        if (Holds(next, "a")) {
            SendForwardDone("a", {9});
        }
    }
    EXPECT_EQ(b_prefix, 32);
}

class InflightPrefixDedupOffSuite : public InflightPrefixDedupSuite {
protected:
    std::int32_t DeferPages() const override { return 0; }
};

TEST_F(InflightPrefixDedupOffSuite, ZeroAdmitsBothAndComputesTheSharedPagesTwice) {
    Submit(RequestSpec{.request_id = "a", .tokens = Prompt(32, 6, 0), .max_new_tokens = 4});
    Submit(RequestSpec{.request_id = "b", .tokens = Prompt(32, 6, 1), .max_new_tokens = 4});

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_TRUE(Holds(batch, "a"));
    ASSERT_TRUE(Holds(batch, "b"));
    EXPECT_EQ(PrefixOf(batch, "b"), 0);
    EXPECT_EQ(scheduler_->PrefixDedup().deferred_requests, 0);
    EXPECT_EQ(scheduler_->PrefixDedup().cut_chunks, 0);
}

}  // namespace
}  // namespace tokenspeed::test
