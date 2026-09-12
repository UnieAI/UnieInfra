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

#include <gtest/gtest.h>
#include "cache/coordinator/kv_cache_coordinator.h"

namespace tokenspeed {
namespace {
class SparseHostAdmission : public ::testing::TestWithParam<int> {};

TEST_P(SparseHostAdmission, ReservesBothMatchedStateAndNewLiveOutput) {
    // Ten full-attention host pages + one host state + two new live pages.
    // BlockPool takes usable parents (the null sentinel is separate): 13
    // fits exactly and 12 must reject without changing the destination tables.
    BlockPool device(GetParam()), host(32);
    std::vector<KvCacheSpec> specs{
        {.kind = AttnKind::kFull},
        {.kind = AttnKind::kMambaState, .checkpoint_stride_blocks = 4}};
    auto coordinator = MakeCoordinator(specs, 4, device, &host);
    std::vector<std::string> hashes;
    for (int i = 0; i < 10; ++i) hashes.push_back("page-" + std::to_string(i));
    for (int group = 0; group < 2; ++group) {
        auto& manager = coordinator.GroupManager(group);
        for (int i = group == 0 ? 0 : 9; i < 10; ++i) {
            auto ref = host.AcquireBlock(group, 1);
            ASSERT_TRUE(ref);
            manager.RegisterCachedBlock(host, ref,
                CacheKey{.group_id = static_cast<std::uint32_t>(group), .content_hash = hashes[i]},
                1, i, CacheBoundaryKind::kEndpoint);
        }
    }
    auto prefix = coordinator.ProbePrefix(hashes);
    ASSERT_EQ(prefix.device.num_common_tokens, 0);
    ASSERT_EQ(prefix.host.num_common_tokens, 40);
    std::vector<BlockTable> tables(2);
    std::vector<GroupDemand> demands{
        {.table = &tables[0], .num_tokens = 1},
        {.table = &tables[1], .num_tokens = 1}};
    auto result = coordinator.Admit(std::move(prefix), demands);
    if (GetParam() == 12) {
        EXPECT_FALSE(result);
        EXPECT_EQ(tables[0].NumBlocks(), 0);
        EXPECT_EQ(tables[1].NumBlocks(), 0);
        EXPECT_EQ(device.NumEmptyLcmBlocks(), 12);
        return;
    }
    ASSERT_TRUE(result);
    EXPECT_EQ(result->load_pairs.size(), 11u);
    ASSERT_EQ(tables[1].NumBlocks(), 11);
    EXPECT_TRUE(tables[1].Blocks()[9]);
    EXPECT_TRUE(tables[1].Blocks()[10]);
    auto ids = coordinator.GroupManager(1).BlockTablePageIds(tables[1]);
    EXPECT_NE(ids[9], ids[10]);
}

INSTANTIATE_TEST_SUITE_P(CapacityBoundary, SparseHostAdmission, ::testing::Values(12, 13));
}  // namespace
}  // namespace tokenspeed
