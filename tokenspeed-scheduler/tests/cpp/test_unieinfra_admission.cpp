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

// UnieInfra fork additions on upstream's scheduler: the per-plan cap on cold
// prompt admissions (SchedulerConfig::max_cold_prefills) and the per-group
// bound-parent diagnostic (Scheduler::CacheGroupBoundParents).

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

class ColdPrefillCapSuite : public SchedulerTestSuite {
protected:
    std::int32_t cap_{0};
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_cold_prefills = cap_;
        return cfg;
    }
    void Rebuild(std::int32_t cap) {
        cap_ = cap;
        SetUp();
    }
};

TEST_F(ColdPrefillCapSuite, UnlimitedByDefaultAdmitsEveryColdPrompt) {
    Rebuild(0);
    Submit({MakeRequestSpec("a", 2, 1), MakeRequestSpec("b", 2, 101), MakeRequestSpec("c", 2, 201)});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids.size(), 3u);
}

TEST_F(ColdPrefillCapSuite, CapBoundsColdAdmissionsPerPlan) {
    Rebuild(1);
    Submit({MakeRequestSpec("a", 2, 1), MakeRequestSpec("b", 2, 101), MakeRequestSpec("c", 2, 201)});
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids.size(), 1u);
    EXPECT_EQ(batch->request_ids[0], "a");
    // The next plan admits the next cold prompt beside the running one.
    SendForwardDone("a", {7});
    const ExecutionPlan next = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(next);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(std::find(second->request_ids.begin(), second->request_ids.end(), "b"), second->request_ids.end());
    EXPECT_EQ(std::find(second->request_ids.begin(), second->request_ids.end(), "c"), second->request_ids.end());
}

TEST_F(ColdPrefillCapSuite, NegativeCapIsRejected) {
    SchedulerConfig cfg = MakeConfig();
    EXPECT_NO_THROW(cfg.Validate());
    cfg.max_cold_prefills = -1;
    EXPECT_THROW(cfg.Validate(), std::invalid_argument);
}

TEST_F(SchedulerTestSuite, BoundParentsCountWhatTheGroupHolds) {
    EXPECT_EQ(scheduler_->CacheGroupBoundParents("full_attention"), 0);
    Submit(MakeRequestSpec("r1", 3));
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_GT(scheduler_->CacheGroupBoundParents("full_attention"), 0);
    EXPECT_LE(scheduler_->CacheGroupBoundParents("full_attention"), scheduler_->CacheGroupTotalPages("full_attention"));
}

}  // namespace
}  // namespace tokenspeed::test
