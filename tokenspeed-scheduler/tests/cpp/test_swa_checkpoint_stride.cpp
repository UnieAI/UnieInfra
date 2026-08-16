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

// A windowed group may keep periodic checkpoints instead of one per page.
//
// The recurrent-state groups of a hybrid model store a full snapshot in every
// cache block, and on Qwen3.6-35B-A3B a snapshot is 20.5 MB against a 40 MB LCM
// parent -- so a 6,656-token chunk claimed 52 parents in each of three groups,
// admission refused every new request, and concurrency sat at one while vLLM
// held twelve. Every reference engine charges O(1) state per request here
// (unieinfra/xref/46 6.1), so the pages between checkpoints must be able to
// stay unbacked.
//
// The stride defaults to 1, which is a block per page and byte-for-byte the old
// behaviour; these cover what changes when it does not.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "cache/core/block_pool.h"
#include "cache/manager/swa_manager.h"

namespace tokenspeed {
namespace {

constexpr std::int32_t kBlockTokens = 4;
constexpr std::int32_t kWindow = 10;

std::int32_t BackedBlocks(const BlockTable& table) {
    std::int32_t backed = 0;
    for (const CacheBlockRef& block : table.Blocks()) {
        if (block) {
            ++backed;
        }
    }
    return backed;
}

TEST(SwaCheckpointStrideTest, DefaultStrideBacksEveryPage) {
    BlockPool pool(32);
    SwaManager manager(kBlockTokens, /*cache_blocks_per_lcm_block=*/1, kWindow);
    ASSERT_EQ(manager.CheckpointStrideBlocks(), 1);

    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/16));

    EXPECT_EQ(table.Blocks().size(), 4u);
    EXPECT_EQ(BackedBlocks(table), 4) << "stride 1 must stay byte-for-byte the old behaviour";
}

TEST(SwaCheckpointStrideTest, StrideLeavesGapsUnbackedButKeepsPositions) {
    BlockPool pool(32);
    SwaManager manager(kBlockTokens, /*cache_blocks_per_lcm_block=*/1, kWindow);
    manager.SetCheckpointStrideBlocks(4);

    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/32));

    // The table still spans the sequence -- logical block indices have to keep
    // lining up with the page hashes -- but only checkpoints and the live tail
    // hold a block.
    ASSERT_EQ(table.Blocks().size(), 8u);
    EXPECT_TRUE(table.Blocks()[3]) << "index 3 closes the first stride of 4";
    EXPECT_TRUE(table.Blocks()[7]) << "index 7 closes the second, and is the live tail";
    for (std::size_t i : {0u, 1u, 2u, 4u, 5u, 6u}) {
        EXPECT_FALSE(table.Blocks()[i]) << "page " << i << " sits between checkpoints";
    }
    EXPECT_EQ(BackedBlocks(table), 2);
}

TEST(SwaCheckpointStrideTest, LiveTailIsAlwaysBacked) {
    // A chunk that ends before a checkpoint boundary still has to be writable:
    // the request is producing state into that page right now.
    BlockPool pool(32);
    SwaManager manager(kBlockTokens, /*cache_blocks_per_lcm_block=*/1, kWindow);
    manager.SetCheckpointStrideBlocks(4);

    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/8));

    ASSERT_EQ(table.Blocks().size(), 2u);
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_TRUE(table.Blocks()[1]) << "the tail the request writes into cannot be a hole";
}

TEST(SwaCheckpointStrideTest, StrideCutsThePoolItConsumes) {
    // The point of the change: the same sequence costs a fraction of the pool,
    // which is what lets a second request in.
    BlockPool dense_pool(32);
    SwaManager dense(kBlockTokens, /*cache_blocks_per_lcm_block=*/1, kWindow);
    BlockTable dense_table;
    ASSERT_TRUE(dense.Acquire(dense_pool, dense_table, /*num_tokens=*/32));

    BlockPool sparse_pool(32);
    SwaManager sparse(kBlockTokens, /*cache_blocks_per_lcm_block=*/1, kWindow);
    sparse.SetCheckpointStrideBlocks(4);
    BlockTable sparse_table;
    ASSERT_TRUE(sparse.Acquire(sparse_pool, sparse_table, /*num_tokens=*/32));

    EXPECT_EQ(BackedBlocks(dense_table), 8);
    EXPECT_EQ(BackedBlocks(sparse_table), 2);
    EXPECT_EQ(dense_table.Blocks().size(), sparse_table.Blocks().size())
        << "position count must not change with the stride";
}

TEST(SwaCheckpointStrideTest, AdmissionEstimateMatchesWhatAcquireTakes) {
    // Admit() claims the device match and splices the host extension in before
    // it acquires, so a plan that counted checkpoints from index 0 charged for
    // a different set of pages than the one Acquire went on to allocate. When
    // the estimate came out low the difference surfaced as "admission plan no
    // longer fits the block pool" and took the engine down mid-run.
    //
    // Equality, not a bound in either direction: an acquisition that reaches
    // past its charge takes blocks a later group in the same Admit was already
    // charged for, which reproduces the same fatal check from the other side.
    for (std::int32_t prepended = 0; prepended < 8; ++prepended) {
        for (std::int32_t tokens = 4; tokens <= 40; tokens += 4) {
            BlockPool pool(64);
            SwaManager manager(kBlockTokens, /*cache_blocks_per_lcm_block=*/1, kWindow);
            manager.SetCheckpointStrideBlocks(4);

            BlockTable table;
            const std::int32_t estimate = manager.AdmissionBlocksNeededFor(table, tokens, prepended);
            if (prepended > 0) {
                ASSERT_TRUE(manager.Acquire(pool, table, prepended * kBlockTokens));
                ASSERT_EQ(table.Blocks().size(), static_cast<std::size_t>(prepended));
            }
            const std::int32_t before = BackedBlocks(table);
            ASSERT_TRUE(manager.Acquire(pool, table, tokens));
            EXPECT_EQ(BackedBlocks(table) - before, estimate)
                << "prepended=" << prepended << " tokens=" << tokens;
            EXPECT_TRUE(table.Blocks().back()) << "the tail this chunk writes into cannot be a hole";
        }
    }
}

}  // namespace
}  // namespace tokenspeed
