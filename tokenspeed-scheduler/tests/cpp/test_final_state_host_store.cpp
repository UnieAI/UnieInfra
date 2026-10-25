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

// A finishing request on a hybrid model publishes the recurrent state at its
// final token. With Host L2 it is copied there only with
// store_final_state_to_host; its KV pages and tail are copied either way.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

class FinalStateHostStoreSuite : public SchedulerTestSuite, public ::testing::WithParamInterface<bool> {
protected:
    static constexpr std::int32_t kP = 4;
    static constexpr std::uint32_t kStateGroup = 1;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = 65;
        cfg.host_allocator.total_pages = 65;
        cfg.disable_l2_cache = false;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 4;
        cfg.store_final_state_to_host = GetParam();
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = kP,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
        });
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "linear_attention",
            .block_granularity = 0,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kState,
        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // State and KV blocks across every write-back op of a plan, acknowledged.
    std::pair<int, int> StoredBlocks(const ExecutionPlan& plan) {
        int state = 0;
        int kv = 0;
        for (const auto& op : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
            const auto& batch = std::get<WriteBackBatch>(op);
            for (std::size_t i = 0; i < batch.op_ids.size(); ++i) {
                for (std::uint32_t group : batch.group_ids[i]) {
                    (group == kStateGroup ? state : kv) += 1;
                }
                SendWriteBackDone(batch.op_ids[i]);
            }
        }
        return {state, kv};
    }
};

TEST_P(FinalStateHostStoreSuite, FinishCopiesTheFinalStateOnlyWhenAsked) {
    std::vector<std::int32_t> prompt(10);
    for (std::int32_t i = 0; i < 10; ++i) {
        prompt[static_cast<std::size_t>(i)] = 100 + i;
    }
    Submit(RequestSpec{.request_id = "r", .tokens = prompt, .max_new_tokens = 8});

    int state_before_finish = 0;
    ExecutionPlan plan = PlanOnce();  // the whole prompt in one chunk
    state_before_finish += StoredBlocks(plan).first;
    SendForwardDone("r", {500});
    for (std::int32_t token = 501; token < 507; ++token) {
        plan = PlanOnce();
        state_before_finish += StoredBlocks(plan).first;
        SendForwardDone("r", {token});
    }
    EXPECT_EQ(state_before_finish, 1) << "the prompt-end checkpoint streams when it is published";

    SendFinish("r");
    const auto [state_at_finish, kv_at_finish] = StoredBlocks(PlanOnce());
    EXPECT_GT(kv_at_finish, 0) << "the decoded KV pages stream at finish either way";
    EXPECT_EQ(state_at_finish, GetParam() ? 1 : 0);
}

INSTANTIATE_TEST_SUITE_P(StoreFinalState, FinalStateHostStoreSuite, ::testing::Bool());

}  // namespace
}  // namespace tokenspeed::test
