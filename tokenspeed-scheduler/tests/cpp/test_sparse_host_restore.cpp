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
class SparseHostRestore : public ::testing::TestWithParam<int> {};

TEST_P(SparseHostRestore, MatchedTailMustBeLoadedEvenOffPeriodicGrid) {
    const bool real = GetParam() == 1;
    const int block_tokens = real ? 128 : 4;
    const int stride = real ? 512 : 4;
    const int pages = real ? 551 : 10;
    BlockPool host(32), device(32);
    SwaManager manager(block_tokens, 1, 2);
    manager.SetCheckpointStrideBlocks(stride);
    std::vector<CacheBlockRef> host_refs(pages);
    auto refs = host.AcquireBlocks(0, 1, 1);
    host_refs.back() = std::move(refs[0]);
    std::vector<std::uint8_t> hits(pages, 0);
    hits.back() = 1;
    EXPECT_EQ(manager.BackedHostExtensionPages(0, hits), 1)
        << "the matched boundary state needs a destination, irrespective of publication stride";
    BlockTable table;
    std::vector<BlockTransfer> loads;
    manager.AppendHostExtension(device, table, std::move(host_refs), loads);
    EXPECT_EQ(loads.size(), 1u);
    EXPECT_TRUE(table.Blocks().back());
    ASSERT_TRUE(manager.Acquire(device, table, 1));
    auto ids = manager.BlockTablePageIds(table);
    EXPECT_GT(ids[pages - 1], 0);
    EXPECT_NE(ids[pages - 1], ids[pages])
        << "restored input state must not alias newly allocated zeroed output state";
}

INSTANTIATE_TEST_SUITE_P(Geometries, SparseHostRestore, ::testing::Values(0, 1));
}  // namespace
}  // namespace tokenspeed
