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

TEST(SparseReclamation, ReleasesExpiredCheckpointsAcrossHoles) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    manager.SetCheckpointStrideBlocks(4);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, 40));
    ASSERT_TRUE(table.Blocks()[3]);
    ASSERT_TRUE(table.Blocks()[7]);
    ASSERT_FALSE(table.Blocks()[8]);
    ASSERT_TRUE(table.Blocks()[9]);
    EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, true), 2);
    EXPECT_EQ(manager.ReclaimableBlockLocationsAt(table, 40, {}).size(), 2u);
    manager.ReclaimExpired(pool, table, 40);
    for (int i = 0; i < 9; ++i) {
        EXPECT_FALSE(table.Blocks()[i]) << "expired logical page " << i;
    }
    EXPECT_TRUE(table.Blocks()[9]);
}

TEST(SparseReclamation, CachedSnapshotsSurviveTableRelease) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    manager.SetCheckpointStrideBlocks(4);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, 40));
    std::vector<CacheKey> keys;
    for (int i = 0; i < 10; ++i) {
        keys.push_back(CacheKey{0, 0, std::to_string(i), 0});
    }
    manager.CacheFullBlocks(pool, table, keys, 1);
    EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, false), 2);
    manager.ReclaimExpired(pool, table, 40);
    EXPECT_FALSE(table.Blocks()[3]);
    EXPECT_FALSE(table.Blocks()[7]);
    EXPECT_TRUE(table.Blocks()[9]);
    EXPECT_TRUE(manager.ContainsCachedBlock(pool, keys[3]));
    EXPECT_TRUE(manager.ContainsCachedBlock(pool, keys[7]));
    EXPECT_EQ(manager.Probe(pool, keys, 0, 8).hits.size(), 8u);
}

TEST(SparseReclamation, DenseControlReleasesAllExpiredPages) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, 40));
    EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, true), 9);
    manager.ReclaimExpired(pool, table, 40);
    for (int i = 0; i < 9; ++i) {
        EXPECT_FALSE(table.Blocks()[i]);
    }
    EXPECT_TRUE(table.Blocks()[9]);
}

TEST(SparseReclamation, QwenGeometryDoesNotAccumulateChunkTails) {
    BlockPool pool(64);
    SwaManager manager(128, 1, 2);
    manager.SetCheckpointStrideBlocks(512);
    BlockTable table;
    for (int step = 1; step <= 10; ++step) {
        const int computed = step * 6656;
        // Acquire consumes an incremental token demand, not a total length.
        ASSERT_TRUE(manager.Acquire(pool, table, 6656));
        ASSERT_EQ(table.NumBlocks(), computed / 128);
        manager.ReclaimExpired(pool, table, computed);
        int retained = 0;
        for (const auto& block : table.Blocks()) {
            retained += static_cast<bool>(block);
        }
        EXPECT_EQ(retained, 1) << "computed tokens=" << computed;
        ASSERT_TRUE(table.Blocks().back());
    }
}

TEST(SparseReclamation, FreeAndReuseRestartsReclamation) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    manager.SetCheckpointStrideBlocks(4);
    BlockTable table;
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_TRUE(manager.Acquire(pool, table, 40));
        manager.ReclaimExpired(pool, table, 40);
        manager.ReclaimExpired(pool, table, 40);
        EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, true), 0);
        EXPECT_FALSE(table.Blocks()[3]);
        EXPECT_FALSE(table.Blocks()[7]);
        EXPECT_TRUE(table.Blocks()[9]);
        manager.Free(table);
    }
}

TEST(SparseReclamation, ExternalOwnerSurvivesButIsNotForecastAsFree) {
    BlockPool pool(32);
    SwaManager manager(4, 1, 2);
    manager.SetCheckpointStrideBlocks(4);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, 40));
    auto external = table.Blocks()[3];
    EXPECT_EQ(manager.BlocksReclaimableAt(table, 40, true), 1);
    EXPECT_EQ(manager.ReclaimableBlockLocationsAt(table, 40, {}).size(), 1u);
    manager.ReclaimExpired(pool, table, 40);
    EXPECT_TRUE(external);
    EXPECT_TRUE(external.unique());
    EXPECT_FALSE(table.Blocks()[3]);
}

}  // namespace
}  // namespace tokenspeed
