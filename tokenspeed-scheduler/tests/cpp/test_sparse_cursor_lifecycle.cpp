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
#include "cache/manager/swa_manager.h"

namespace tokenspeed {
namespace {

TEST(SparseCursorLifecycle, RestoredSparsePrefixAndSharedTableCopy) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    std::vector<CacheBlockRef> slots(10);
    auto refs = pool.AcquireBlocks(0, 1, 3);
    slots[3] = std::move(refs[0]);
    slots[7] = std::move(refs[1]);
    slots[9] = std::move(refs[2]);
    auto table = BlockTable::FromBlocks(std::move(slots), 0);
    auto shared = table;
    EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, true), 0);
    manager.ReclaimExpired(pool, table, 40);
    EXPECT_EQ(table.ReclaimedPrefixBlocks(), 9);
    EXPECT_EQ(shared.ReclaimedPrefixBlocks(), 0);
    EXPECT_TRUE(shared.Blocks()[3]);
    EXPECT_EQ(manager.BlocksReclaimableAt(shared, 40, true), 2);
    manager.ReclaimExpired(pool, shared, 40);
    EXPECT_FALSE(shared.Blocks()[3]);
    EXPECT_TRUE(table.Blocks()[9]);
}

TEST(SparseCursorLifecycle, MaterializedSuffixAfterReclaimedNullPrefix) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    auto table = BlockTable::FromBlocks(std::vector<CacheBlockRef>(8), 0);
    table.ReclaimPrefix(8);
    GroupDemand demand{.num_tokens = 40, .materialized_suffix_start = 8};
    ASSERT_TRUE(static_cast<KvCacheManager&>(manager).Acquire(pool, table, demand));
    ASSERT_EQ(table.NumBlocks(), 10);
    EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, true), 1);
    manager.ReclaimExpired(pool, table, 40);
    EXPECT_FALSE(table.Blocks()[8]);
    EXPECT_TRUE(table.Blocks()[9]);
}

TEST(SparseCursorLifecycle, FreedTableCanClaimAnotherSparsePrefix) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, 40));
    manager.ReclaimExpired(pool, table, 40);
    manager.Free(table);
    PrefixMatch hit;
    hit.blocks.resize(4);
    auto refs = pool.AcquireBlocks(0, 1, 2);
    hit.blocks[1] = std::move(refs[0]);
    hit.blocks[3] = std::move(refs[1]);
    manager.ClaimHitBlocks(table, std::move(hit));
    EXPECT_EQ(table.ReclaimedPrefixBlocks(), 0);
    manager.ReclaimExpired(pool, table, 16);
    EXPECT_FALSE(table.Blocks()[1]);
    EXPECT_TRUE(table.Blocks()[3]);
}

TEST(SparseCursorLifecycle, LiveInputIsRetainedAtEveryDecodePosition) {
    BlockPool pool(64);
    SwaManager manager(128, 1, 2);
    manager.SetCheckpointStrideBlocks(512);
    BlockTable table;
    // Match the live server's resolved 8192-token scheduling budget, then
    // cross an entire page with one-token decode increments.
    int computed = 0;
    for (int step = 0; step < 9 + 130; ++step) {
        int tokens = step < 9 ? 8192 : 1;
        ASSERT_TRUE(manager.Acquire(pool, table, tokens));
        computed += tokens;
        manager.ReclaimExpired(pool, table, computed);
        const int live_slot = (computed - 1) / 128;
        EXPECT_TRUE(table.Blocks()[live_slot]);
        EXPECT_EQ(table.ReclaimedPrefixBlocks(), live_slot);
        int backed = 0;
        for (const auto& ref : table.Blocks()) backed += static_cast<bool>(ref);
        EXPECT_EQ(backed, 1);
        manager.ReclaimExpired(pool, table, computed);
        EXPECT_EQ(manager.BlocksReclaimableAt(table, computed, true), 0);
    }
}

}  // namespace
}  // namespace tokenspeed
