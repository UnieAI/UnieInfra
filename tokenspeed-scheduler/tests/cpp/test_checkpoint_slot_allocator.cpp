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

#include <gtest/gtest.h>

#include <set>
#include <utility>
#include <vector>

#include "resource/allocator/checkpoint_slot_allocator.h"

namespace tokenspeed::test {

TEST(CheckpointSlotAllocatorTest, RAIIReleasesSlotOnDestruction) {
    CheckpointSlotAllocator allocator(1);
    {
        auto slot = allocator.Allocate();
        EXPECT_TRUE(slot.valid());
        EXPECT_EQ(allocator.AvailableSlots(), 0);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 1);
}

// The whole point of the second index space: it starts at 0, because the
// runtime addresses this pool separately and marks absence with -1 rather than
// with a reserved null row.
TEST(CheckpointSlotAllocatorTest, IndexSpaceStartsAtZeroAndCoversTheWholePool) {
    constexpr std::int32_t kSize = 4;
    CheckpointSlotAllocator allocator(kSize);
    std::vector<CheckpointSlot> held;
    std::set<std::int32_t> seen;
    for (std::int32_t i = 0; i < kSize; ++i) {
        held.push_back(allocator.Allocate());
        ASSERT_TRUE(held.back().valid());
        EXPECT_TRUE(seen.insert(held.back().Index()).second) << "slot handed out twice";
    }
    EXPECT_EQ(*seen.begin(), 0);
    EXPECT_EQ(*seen.rbegin(), kSize - 1);
    EXPECT_EQ(allocator.AvailableSlots(), 0);
}

// Exhaustion must not throw. A checkpoint is cache: not taking one costs a
// later prefix hit, and the runtime already drops -1 rows in
// `store_from_active_indexed`. Throwing would turn a full cache into a dead
// engine, which is the failure `ReqPoolAllocator` is allowed to have and this
// one is not.
TEST(CheckpointSlotAllocatorTest, ExhaustionYieldsTheNoSlotSentinelInsteadOfThrowing) {
    CheckpointSlotAllocator allocator(1);
    auto first = allocator.Allocate();
    ASSERT_TRUE(first.valid());

    CheckpointSlot second;
    EXPECT_NO_THROW(second = allocator.Allocate());
    EXPECT_FALSE(second.valid());
    EXPECT_EQ(second.Index(), CheckpointSlot::kNoSlot);
    EXPECT_EQ(allocator.AvailableSlots(), 0);
}

// An invalid slot must not return a live index to the forward operation, and
// must not release anything it never owned.
TEST(CheckpointSlotAllocatorTest, DefaultConstructedSlotIsInertAndReportsNoSlot) {
    CheckpointSlotAllocator allocator(2);
    {
        CheckpointSlot empty;
        EXPECT_FALSE(empty.valid());
        EXPECT_EQ(empty.Index(), CheckpointSlot::kNoSlot);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 2);
}

TEST(CheckpointSlotAllocatorTest, MoveAssignmentReleasesPreviouslyOwnedSlot) {
    CheckpointSlotAllocator allocator(2);
    {
        auto dst = allocator.Allocate();
        auto src = allocator.Allocate();
        ASSERT_EQ(allocator.AvailableSlots(), 0);
        const std::int32_t moved = src.Index();

        dst = std::move(src);

        // dst's original slot went back; src is now inert.
        EXPECT_EQ(allocator.AvailableSlots(), 1);
        EXPECT_EQ(dst.Index(), moved);
        EXPECT_FALSE(src.valid());
        EXPECT_EQ(src.Index(), CheckpointSlot::kNoSlot);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 2);
}

TEST(CheckpointSlotAllocatorTest, MoveConstructionTransfersOwnershipExactlyOnce) {
    CheckpointSlotAllocator allocator(1);
    {
        auto src = allocator.Allocate();
        const std::int32_t index = src.Index();
        auto dst = std::move(src);
        EXPECT_EQ(dst.Index(), index);
        EXPECT_FALSE(src.valid());
        EXPECT_EQ(allocator.AvailableSlots(), 0) << "moved-from slot released early";
    }
    EXPECT_EQ(allocator.AvailableSlots(), 1) << "slot released twice or not at all";
}

// A pool that is configured off must be inert rather than hand out slot 0,
// which would alias the first real checkpoint row if the tensor were absent.
TEST(CheckpointSlotAllocatorTest, ZeroAndNegativeSizesHandOutNothing) {
    for (std::int32_t size : {0, -1}) {
        CheckpointSlotAllocator allocator(size);
        EXPECT_EQ(allocator.Size(), 0);
        EXPECT_EQ(allocator.AvailableSlots(), 0);
        auto slot = allocator.Allocate();
        EXPECT_FALSE(slot.valid());
        EXPECT_EQ(slot.Index(), CheckpointSlot::kNoSlot);
    }
}

TEST(CheckpointSlotAllocatorTest, ReleasedSlotsAreReusedAndNeverDuplicated) {
    constexpr std::int32_t kSize = 3;
    CheckpointSlotAllocator allocator(kSize);
    for (int round = 0; round < 4; ++round) {
        std::vector<CheckpointSlot> held;
        std::set<std::int32_t> seen;
        for (std::int32_t i = 0; i < kSize; ++i) {
            held.push_back(allocator.Allocate());
            ASSERT_TRUE(held.back().valid()) << "round " << round;
            EXPECT_TRUE(seen.insert(held.back().Index()).second) << "round " << round;
        }
        EXPECT_EQ(allocator.AvailableSlots(), 0);
    }
    EXPECT_EQ(allocator.AvailableSlots(), kSize);
}

}  // namespace tokenspeed::test
