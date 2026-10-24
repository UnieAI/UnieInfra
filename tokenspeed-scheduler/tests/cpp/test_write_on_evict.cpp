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
// Host L2 write-on-evict: a Device entry is copied to Host only on its way out
// of the Device -- ahead of time by the cleaner, or in the very round that
// evicts it -- and a Device cache that never fills copies nothing.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

struct WriteBacks {
    int pinned{0};
    int stream_ordered{0};
};

class WriteOnEvictSuite : public SchedulerTestSuite, public ::testing::WithParamInterface<double> {
protected:
    static constexpr std::int32_t kPagesPerPrompt = 6;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 17;
        cfg.host_allocator.total_pages = 65;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.max_batch_size = 1;
        cfg.disable_l2_cache = false;
        cfg.host_write_on_evict = true;
        cfg.host_clean_reserve = GetParam();
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Counts and acknowledges every write-back of a plan.
    void Collect(const ExecutionPlan& plan) {
        for (const auto& op : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            const auto& batch = std::get<WriteBackBatch>(op);
            for (std::size_t i = 0; i < batch.op_ids.size(); ++i) {
                (batch.source_pinned[i] ? write_backs_.pinned : write_backs_.stream_ordered) +=
                    static_cast<int>(batch.group_ids[i].size());
                SendWriteBackDone(batch.op_ids[i]);
            }
        }
    }

    // Prefills one prompt, finishes it, and returns whether its admission
    // loaded any page back from Host.
    bool Serve(const std::string& id, std::int32_t first_token) {
        Submit(MakeRequestSpec(id, kPagesPerPrompt, first_token));
        ExecutionPlan plan = PlanOnce();
        const bool loaded = !ExtractCacheOpsOfKind<LoadBackBatch>(plan).empty();
        Collect(plan);
        for (const auto& op : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
            for (std::uint32_t op_id : std::get<LoadBackBatch>(op).op_ids) {
                SendLoadBackDone(op_id);
            }
        }
        SendForwardDone(id, {9000});
        SendFinish(id);
        Collect(PlanOnce());
        return loaded;
    }

    WriteBacks write_backs_;
};

TEST_P(WriteOnEvictSuite, ADeviceCacheThatNeverFillsCopiesNothing) {
    Serve("a", 1);
    Collect(PlanOnce());
    EXPECT_EQ(write_backs_.pinned + write_backs_.stream_ordered, 0);
}

TEST_P(WriteOnEvictSuite, AnEvictedPromptComesBackFromHost) {
    Serve("a", 1);
    // Distinct prompts push the first out of the 16-page Device pool.
    for (std::int32_t i = 1; i <= 4; ++i) {
        Serve("fill" + std::to_string(i), 1000 * i);
    }
    EXPECT_GT(write_backs_.pinned + write_backs_.stream_ordered, 0);
    if (GetParam() == 0.0) {
        EXPECT_EQ(write_backs_.pinned, 0) << "no cleaner: every copy happens at eviction";
    } else {
        EXPECT_GT(write_backs_.pinned, 0) << "the cleaner copies victims ahead of admission";
    }
    EXPECT_TRUE(Serve("a-again", 1)) << "the first prompt's pages were written back when evicted";
}

INSTANTIATE_TEST_SUITE_P(CleanReserve, WriteOnEvictSuite, ::testing::Values(0.0, 0.5));

}  // namespace
}  // namespace tokenspeed::test
