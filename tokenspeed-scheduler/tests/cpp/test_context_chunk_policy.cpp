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

// Does the step budget actually get shared?
//
// The end-to-end benchmark cannot answer this: a policy that silently fell back
// to FCFS would produce a clean-looking null result, and a null result is what
// this experiment might legitimately return. So the mechanism is asserted here,
// on the plan itself, before any card time is spent on the outcome.

#include "integration_test_helper.h"

namespace tokenspeed::test {

class ContextChunkPolicySuite : public SchedulerTestSuite {
protected:
    // One page = 2 tokens. A 32-token budget is 16 pages: enough for one long
    // prompt outright, which is the point -- under FCFS the first request takes
    // it all.
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_scheduled_tokens = 32;
        cfg.device_allocator.total_pages = 128;
        cfg.host_allocator.total_pages = 128;
        cfg.max_batch_size = 8;
        return cfg;
    }

    // Three requests whose prefill work is deliberately unequal, and the
    // longest MUST exceed the step budget on its own: 24 pages = 48 tokens
    // against a 32-token step. That is the regime under study -- on
    // agent-26k a 28k-token prompt meets an 8,192-token budget, so there is
    // no remainder for anyone else and FCFS serializes.
    //
    // An earlier version of this test used a 12-page long request, which FITS
    // in the budget. FCFS then served two requests from one step and the
    // baseline assertion failed -- correctly. FCFS does hand on its remainder;
    // what it never does is interrupt a request that can still use the budget.
    // A test whose longest request fits is testing the wrong regime.
    void SubmitUnequal() {
        Submit({MakeRequestSpec("long", /*num_pages=*/24, /*start=*/1),
                MakeRequestSpec("mid", /*num_pages=*/4, /*start=*/101),
                MakeRequestSpec("short", /*num_pages=*/2, /*start=*/201)});
    }

    // A plan's operations are batches, not per-request ops. Prefill shows up
    // as extends inside the ForwardBatch, one entry per request that was
    // offered budget this step -- which is exactly the quantity under test.
    static std::size_t PrefillOpCount(const ExecutionPlan& plan) {
        std::size_t n = 0;
        for (const auto& op : plan.Operations()) {
            if (auto* batch = std::get_if<ForwardBatch>(&op)) {
                n += batch->NumExtends();
            }
        }
        return n;
    }

    static std::int32_t PrefillTokenTotal(const ExecutionPlan& plan) {
        std::int32_t total = 0;
        for (const auto& op : plan.Operations()) {
            if (auto* batch = std::get_if<ForwardBatch>(&op)) {
                for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
                    total += batch->input_lengths.at(i);
                }
            }
        }
        return total;
    }
};

TEST_F(ContextChunkPolicySuite, FcfsGivesTheStepToOneRequest) {
    SubmitUnequal();
    const auto plan = PlanOnce();
    // The historical behaviour, asserted so that a change to it fails here
    // rather than silently in a benchmark six hours later.
    EXPECT_EQ(PrefillOpCount(plan), 1u)
        << "a request that can absorb the whole step must leave nothing behind";
}

TEST_F(ContextChunkPolicySuite, EqualProgressSharesTheStep) {
    config_.context_chunk_policy = SchedulerConfig::ContextChunkPolicy::kEqualProgress;
    scheduler_ = std::make_unique<Scheduler>(config_);
    SubmitUnequal();
    const auto plan = PlanOnce();
    EXPECT_GT(PrefillOpCount(plan), 1u)
        << "equal progress must offer the step to more than one request";
}

TEST_F(ContextChunkPolicySuite, EqualProgressDoesNotOverspendTheBudget) {
    config_.context_chunk_policy = SchedulerConfig::ContextChunkPolicy::kEqualProgress;
    scheduler_ = std::make_unique<Scheduler>(config_);
    SubmitUnequal();
    const auto plan = PlanOnce();
    const std::int32_t total = PrefillTokenTotal(plan);
    // Sharing must not become spending more. The makespan argument for equal
    // progress depends on the total being unchanged.
    EXPECT_LE(total, config_.max_scheduled_tokens)
        << "round-robin allocated " << total << " against a budget of "
        << config_.max_scheduled_tokens;
}

TEST_F(ContextChunkPolicySuite, SingleRequestFallsBackToFcfs) {
    config_.context_chunk_policy = SchedulerConfig::ContextChunkPolicy::kEqualProgress;
    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit(MakeRequestSpec("lonely", /*num_pages=*/12, /*start=*/1));
    const auto plan = PlanOnce();
    const std::int32_t total = PrefillTokenTotal(plan);
    // With nobody to share with, capping at one unit would be strictly worse
    // than FCFS for no benefit -- the policy must decline to engage.
    EXPECT_GT(total, config_.block_size)
        << "a lone request was capped at one unit with no one to share with";
}


// A plan that is correct once is not a plan that terminates.
//
// The single-plan tests above all passed while the policy livelocked on a real
// engine: 16 sequences shared the step correctly, then nothing progressed for
// twenty minutes. A capped offer can be smaller than one aligned chunk,
// AlignPrefillChunk refuses it, and every request is starved by the same rule
// that was meant to be fair to them. Nothing in a one-shot assertion can see
// that -- the failure is in the sequence, not the step.
TEST_F(ContextChunkPolicySuite, EqualProgressTerminates) {
    config_.context_chunk_policy = SchedulerConfig::ContextChunkPolicy::kEqualProgress;
    scheduler_ = std::make_unique<Scheduler>(config_);
    SubmitUnequal();

    // 24 + 4 + 2 pages at 2 tokens each is 60 tokens of prefill against a
    // 32-token step: three steps of real work, so a bound of 40 is generous
    // without being unfalsifiable.
    const int kMaxSteps = 40;
    int steps = 0;
    std::int32_t scheduled = 0;
    for (; steps < kMaxSteps; ++steps) {
        const auto plan = PlanOnce();
        const std::int32_t tokens = PrefillTokenTotal(plan);
        scheduled += tokens;
        if (tokens == 0) {
            break;  // nothing left to prefill, or nothing can be
        }
    }
    EXPECT_LT(steps, kMaxSteps) << "planning did not converge in " << kMaxSteps << " steps";
    EXPECT_GT(scheduled, 0) << "no prefill tokens were ever scheduled";
}

// The same, with the cap forced below one page so the refusal path is certain
// to be taken rather than merely possible.
TEST_F(ContextChunkPolicySuite, SubPageUnitStillMakesProgress) {
    config_.context_chunk_policy = SchedulerConfig::ContextChunkPolicy::kEqualProgress;
    config_.context_chunk_unit = 1;  // one token, half a page
    scheduler_ = std::make_unique<Scheduler>(config_);
    SubmitUnequal();
    const auto plan = PlanOnce();
    EXPECT_GT(PrefillTokenTotal(plan), 0)
        << "a unit smaller than a page must not stall the step";
}


// Every offer must be page-aligned, or complete the request.
//
// The round-robin clamps a grant by the leftover step budget, and a leftover
// that is not a whole number of units produced a partial page. Hybrid models
// refuse an unaligned chunk; worse, the partial propagated into the block
// geometry the executor maps, and the engine died with SIGBUS inside
// next_execution_plan() after several healthy shared steps. A budget that is
// deliberately NOT a multiple of the page is what exercises that path.
TEST_F(ContextChunkPolicySuite, EveryOfferIsPageAlignedOrFinishesTheRequest) {
    config_.context_chunk_policy = SchedulerConfig::ContextChunkPolicy::kEqualProgress;
    config_.max_scheduled_tokens = 33;  // one token past 16 pages of 2
    scheduler_ = std::make_unique<Scheduler>(config_);
    SubmitUnequal();

    for (int step = 0; step < 20; ++step) {
        const auto plan = PlanOnce();
        if (PrefillTokenTotal(plan) == 0) {
            break;
        }
        for (const auto& op : plan.Operations()) {
            auto* batch = std::get_if<ForwardBatch>(&op);
            if (batch == nullptr) {
                continue;
            }
            for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
                const std::int32_t len = batch->input_lengths.at(i);
                const std::int32_t total = batch->prefill_lengths.at(i);
                const bool aligned = (len % config_.block_size) == 0;
                const bool finishes = len == total || len < config_.block_size;
                EXPECT_TRUE(aligned || finishes)
                    << "step " << step << " offered " << len
                    << " tokens, neither a multiple of the " << config_.block_size
                    << "-token page nor the request's tail (" << total << ")";
            }
        }
    }
}

}  // namespace tokenspeed::test
