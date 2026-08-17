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

#include <vector>

#include "resource/allocator/checkpoint_slot_allocator.h"
#include "scheduler/operations/forward.h"

namespace tokenspeed::test {
namespace {

PrefillOperation MakePrefill(const char* id, std::int32_t ckpt) {
    PrefillOperation op;
    op.request_id = id;
    op.mamba_checkpoint_dst_idx = ckpt;
    return op;
}

DecodeOperation MakeDecode(const char* id, std::int32_t ckpt) {
    DecodeOperation op;
    op.request_id = id;
    op.mamba_checkpoint_dst_idx = ckpt;
    return op;
}

}  // namespace

// Default must be the runtime's "no slot" sentinel, so a build that never
// allocates a checkpoint keeps today's behaviour: every row is dropped by
// `store_from_active_indexed` and nothing is snapshotted.
TEST(ForwardCheckpointChannelTest, DefaultsToNoSlotForBothOperationKinds) {
    EXPECT_EQ(PrefillOperation{}.mamba_checkpoint_dst_idx, CheckpointSlot::kNoSlot);
    EXPECT_EQ(DecodeOperation{}.mamba_checkpoint_dst_idx, CheckpointSlot::kNoSlot);

    std::vector<ForwardOperation> ops;
    ops.emplace_back(PrefillOperation{});
    ops.emplace_back(DecodeOperation{});
    ForwardBatch batch{std::move(ops)};
    ASSERT_EQ(batch.mamba_checkpoint_dst_indices.size(), 2u);
    for (std::int32_t idx : batch.mamba_checkpoint_dst_indices) {
        EXPECT_EQ(idx, CheckpointSlot::kNoSlot);
    }
}

// The row order the runtime indexes by is `request_ids`, and prefills are
// stably partitioned ahead of decodes. The checkpoint column has to ride that
// same reordering or a snapshot is filed against another request's state.
TEST(ForwardCheckpointChannelTest, StaysAlignedWithRequestIdsAcrossThePrefillPartition) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakeDecode("d0", 10));
    ops.emplace_back(MakePrefill("p0", 20));
    ops.emplace_back(MakeDecode("d1", 30));
    ops.emplace_back(MakePrefill("p1", 40));

    ForwardBatch batch{std::move(ops)};

    ASSERT_EQ(batch.request_ids.size(), 4u);
    ASSERT_EQ(batch.mamba_checkpoint_dst_indices.size(), batch.request_ids.size());

    // Prefills first, decodes after, each group in its original order.
    EXPECT_EQ(batch.request_ids[0], "p0");
    EXPECT_EQ(batch.request_ids[1], "p1");
    EXPECT_EQ(batch.request_ids[2], "d0");
    EXPECT_EQ(batch.request_ids[3], "d1");

    EXPECT_EQ(batch.mamba_checkpoint_dst_indices[0], 20);
    EXPECT_EQ(batch.mamba_checkpoint_dst_indices[1], 40);
    EXPECT_EQ(batch.mamba_checkpoint_dst_indices[2], 10);
    EXPECT_EQ(batch.mamba_checkpoint_dst_indices[3], 30);
}

// Slot 0 is a real checkpoint in this index space, unlike an arena page id
// where 0 is the null hole. A truthiness test anywhere on this path would
// silently drop the first row of the pool.
TEST(ForwardCheckpointChannelTest, SlotZeroSurvivesAsARealDestination) {
    std::vector<ForwardOperation> ops;
    ops.emplace_back(MakePrefill("p0", 0));
    ForwardBatch batch{std::move(ops)};
    ASSERT_EQ(batch.mamba_checkpoint_dst_indices.size(), 1u);
    EXPECT_EQ(batch.mamba_checkpoint_dst_indices[0], 0);
    EXPECT_NE(batch.mamba_checkpoint_dst_indices[0], CheckpointSlot::kNoSlot);
}

}  // namespace tokenspeed::test
