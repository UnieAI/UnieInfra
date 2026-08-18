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

// SchedulerConfig::max_partial_prefills -- the cap on requests that may end a
// step with an unfinished prefill.
//
// Without it, buildForwardOperations stops admitting only on `token_budget <= 0`
// or `max_batch_size`, so a step admits every request that fits and hands each
// a fraction of the budget. Measured on gemma-4-31B / RAG at c=64: 11 new
// sequences in one 8,192-token step against ~1,149-token prompts, none of them
// finishing, 21 still queued.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

// Prompts are deliberately larger than the per-step token budget, so every
// admitted request stays Prefilling and counts against the cap. With prompts
// that fit in one step the cap is unobservable, which is the point: it bounds
// UNFINISHED prefills, not admissions.
class PartialPrefillCapSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kPagesPerPrompt = 8;   // 16 tokens at block_size 2
    std::int32_t cap_ = 0;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_scheduled_tokens = 8;   // < prompt, so no first chunk completes
        cfg.max_batch_size = 8;
        cfg.device_allocator.total_pages = 256;
        cfg.host_allocator.total_pages = 256;
        cfg.paged_cache_groups[0].total_pages = cfg.device_allocator.total_pages;
        cfg.max_partial_prefills = cap_;
        return cfg;
    }

    void Rebuild(std::int32_t cap) {
        cap_ = cap;
        SetUp();
    }

    std::size_t PrefillOpsInNextPlan() {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr) {
            return 0;
        }
        std::size_t prefills = 0;
        for (const auto& op : batch->operations) {
            if (std::holds_alternative<PrefillOperation>(op)) {
                ++prefills;
            }
        }
        return prefills;
    }

    void SubmitMany(int count) {
        std::vector<RequestSpec> specs;
        specs.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            specs.push_back(MakeRequestSpec("r" + std::to_string(i), kPagesPerPrompt, /*start=*/1 + i * 100));
        }
        Submit(specs);
    }
};

// 0 keeps the behaviour that existed before the field, so an unset config is
// not silently a policy change.
TEST_F(PartialPrefillCapSuite, ZeroMeansUnlimited) {
    Rebuild(0);
    SubmitMany(6);
    EXPECT_GT(PrefillOpsInNextPlan(), 1u);
}

TEST_F(PartialPrefillCapSuite, CapOfOneAdmitsOnePerStep) {
    Rebuild(1);
    SubmitMany(6);
    EXPECT_EQ(PrefillOpsInNextPlan(), 1u);
}

TEST_F(PartialPrefillCapSuite, CapBoundsConcurrentPartials) {
    Rebuild(3);
    SubmitMany(6);
    EXPECT_LE(PrefillOpsInNextPlan(), 3u);
}

// A cap above demand must not throttle: the failure this guards against is an
// off-by-one that makes every cap behave like cap-1.
TEST_F(PartialPrefillCapSuite, CapAboveDemandDoesNotThrottle) {
    Rebuild(0);
    SubmitMany(4);
    const std::size_t uncapped = PrefillOpsInNextPlan();
    Rebuild(64);
    SubmitMany(4);
    EXPECT_EQ(PrefillOpsInNextPlan(), uncapped);
}

}  // namespace
}  // namespace tokenspeed::test
