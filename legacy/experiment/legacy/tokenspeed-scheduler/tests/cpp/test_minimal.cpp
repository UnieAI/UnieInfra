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

#include "integration_test_helper.h"
#include <iostream>

namespace tokenspeed::test {

class MinimalTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 2;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        return cfg;
    }
};

TEST_F(MinimalTestSuite, BasicPrefill) {
    Submit(MakeRequestSpec("r1", 1, 1));
    std::cerr << "Before PlanOnce\n";
    auto plan = PlanOnce();
    std::cerr << "After PlanOnce\n";

    // Check if request is in forward
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<FlatForwardOperation>(&op)) {
            std::cerr << "Forward op with " << fwd->request_ids.size() << " requests\n";
            for (const auto& id : fwd->request_ids) {
                std::cerr << "  request: " << id << "\n";
            }
        }
    }

    // This should work if request is in PrefillDone
    SendForwardDone("r1", {42});
    std::cerr << "After SendForwardDone\n";
}

}  // namespace tokenspeed::test
