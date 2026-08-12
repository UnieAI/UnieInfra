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

#include "fsm/forward_events.h"

#include <gtest/gtest.h>

namespace {

using tokenspeed::fsm::detail::MambaCheckpointPublishDepthForGrain;

// The two grids this depth has to satisfy on Qwen3.5-27B: KV pages of 16, and
// a chunked-scan grain of 64 (max(FLA_CHUNK_SIZE, block_size)).
constexpr std::int32_t kPage = 16;
constexpr std::int32_t kGrain = 64;

// Every prompt long enough to hold a snapshot must get one.
//
// The regression this pins: the depth used to be computed on the page grid and
// then REJECTED unless it also happened to sit on the grain, which published
// nothing for roughly three prompts in four. Nothing failed -- reuse just
// collapsed to what an exact repeat could serve, 6.8% against an 84.1% ceiling.
TEST(MambaSnapshotDepth, PublishesForEveryChunkLongEnough) {
    for (std::int32_t len = kGrain + kPage; len <= 8192; ++len) {
        const std::int32_t depth = MambaCheckpointPublishDepthForGrain(0, len, kPage, kGrain);
        EXPECT_GT(depth, 0) << "no snapshot published for a chunk of " << len << " tokens";
    }
}

// A depth off the grain is silently dropped by the harvest, and a depth off the
// page grid has no radix node to attach to.
TEST(MambaSnapshotDepth, LandsOnBothGrids) {
    for (std::int32_t len = kGrain + kPage; len <= 8192; ++len) {
        const std::int32_t depth = MambaCheckpointPublishDepthForGrain(0, len, kPage, kGrain);
        ASSERT_GT(depth, 0);
        EXPECT_EQ(depth % kGrain, 0) << "chunk " << len << " published off the scan grain";
        EXPECT_EQ(depth % kPage, 0) << "chunk " << len << " published off the page grid";
    }
}

// Strictly inside the chunk, at both ends. An exact repeat must leave a token
// to compute, so a depth at the chunk's own end is one no match can ever reach;
// a depth at or below the chunk's start belongs to an earlier chunk.
TEST(MambaSnapshotDepth, StrictlyInsideTheChunk) {
    for (std::int32_t begin : {0, 16, 512, 4096}) {
        for (std::int32_t len = 1; len <= 4096; ++len) {
            const std::int32_t depth = MambaCheckpointPublishDepthForGrain(begin, len, kPage, kGrain);
            if (depth < 0) continue;
            EXPECT_GT(depth, begin) << "begin=" << begin << " len=" << len;
            EXPECT_LT(depth, begin + len) << "begin=" << begin << " len=" << len;
        }
    }
}

// Snapping moves the depth down, never up: the deepest legal snapshot is the
// most valuable one, and overshooting would file a state the scan never took.
TEST(MambaSnapshotDepth, NeverDeeperThanThePageGridCandidate) {
    for (std::int32_t len = kGrain + kPage; len <= 8192; ++len) {
        const std::int32_t depth = MambaCheckpointPublishDepthForGrain(0, len, kPage, kGrain);
        const std::int32_t page_candidate = (len / kPage) * kPage - kPage;
        EXPECT_LE(depth, page_candidate) << "chunk " << len;
        EXPECT_GT(depth, page_candidate - kGrain) << "chunk " << len << " snapped further than one grain";
    }
}

// Chunks too short to hold a snapshot say so rather than returning a depth at
// or before their own start.
TEST(MambaSnapshotDepth, RefusesChunksTooShort) {
    EXPECT_LT(MambaCheckpointPublishDepthForGrain(0, 0, kPage, kGrain), 0);
    EXPECT_LT(MambaCheckpointPublishDepthForGrain(0, 1, kPage, kGrain), 0);
    EXPECT_LT(MambaCheckpointPublishDepthForGrain(0, kPage, kPage, kGrain), 0);
    // One page past the grain is the first length that can carry one.
    EXPECT_LT(MambaCheckpointPublishDepthForGrain(0, kGrain, kPage, kGrain), 0);
    EXPECT_GT(MambaCheckpointPublishDepthForGrain(0, kGrain + kPage, kPage, kGrain), 0);
}

// The lengths this actually failed on, kept as literals so a regression names
// itself. 735 was the one chunk in sixteen that used to work.
TEST(MambaSnapshotDepth, TheMeasuredCases) {
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 873, kPage, kGrain), 832);
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 680, kPage, kGrain), 640);
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 1020, kPage, kGrain), 960);
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 627, kPage, kGrain), 576);
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 735, kPage, kGrain), 704);
}

// A grain of 1 (or 0, meaning "no grain") degenerates to the page grid alone,
// which is what a non-hybrid model wants.
TEST(MambaSnapshotDepth, NoGrainIsThePageGrid) {
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 873, kPage, 0), 848);
    EXPECT_EQ(MambaCheckpointPublishDepthForGrain(0, 873, kPage, 1), 848);
}

}  // namespace
