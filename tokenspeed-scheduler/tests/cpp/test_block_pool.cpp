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

#include <memory>
#include <new>
#include <optional>
#include <stdexcept>

#include <spdlog/sinks/stdout_color_sinks.h>

#include "cache/core/block_pool.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasCacheIndex = requires(T& value) { value.ContainsCachedBlock("key"); };

static_assert(!HasCacheIndex<BlockPool>);

TEST(BlockPoolTest, ConstructsExactlyRequestedLcmBlocks) {
    BlockPool pool(8);
    EXPECT_EQ(pool.NumLcmBlocks(), 8);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
}

TEST(BlockPoolTest, DestroyWithLiveReferenceReportsFatalInvariant) {
    EXPECT_DEATH(
        {
            spdlog::set_default_logger(spdlog::stderr_color_mt("fatal-check-test"));
            auto pool = std::make_unique<BlockPool>(1);
            CacheBlockRef ref = pool->AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
            pool.reset();
        },
        "BlockPool destroyed with live block references");
}

TEST(BlockPoolTest, KOneBatchAcquireIsAllOrNothing) {
    BlockPool pool(3);
    auto blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/4);
    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);

    blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/3);
    EXPECT_EQ(blocks.size(), 3u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, EmptyParentBindsToGroupOnFirstChild) {
    BlockPool pool(3);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);

    ASSERT_TRUE(first);
    EXPECT_EQ(first->Location(), (CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}));
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{7});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
}

TEST(BlockPoolLcmPlacementTest, RejectsPackingChangeWhileParentIsOccupied) {
    BlockPool pool(1);
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);

    EXPECT_THROW((void)pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/4), std::runtime_error);
}

TEST(BlockPoolLcmPlacementTest, NormalCapacityShortfallDoesNotMutatePartialParent) {
    BlockPool pool(1);
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);
    ASSERT_TRUE(existing);
    const CacheBlockLocation location = existing->Location();

    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2, /*num=*/2);

    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{7});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
    EXPECT_TRUE(pool.IsOccupied(location));
}

TEST(BlockPoolLcmPlacementTest, ParentRebindsOnlyAfterLastChildReleases) {
    BlockPool pool(1);
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/2);

    EXPECT_FALSE(pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/8));
    child.reset();
    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);

    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/8);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(rebound->Location().slot_index, 0);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{2});
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsRestoreBatchCapacity) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);
    ASSERT_EQ(blocks.size(), 2u);
    blocks[0].reset();
    blocks[1].reset();

    std::vector<CacheBlockRef> reused = pool.AcquireBlocks(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);

    EXPECT_EQ(reused.size(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsAreReusedInReleaseOrder) {
    BlockPool pool(4);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/1, /*num=*/4);
    ASSERT_EQ(blocks.size(), 4u);
    const CacheBlockLocation first_released = blocks[0]->Location();
    const CacheBlockLocation second_released = blocks[2]->Location();
    blocks[0].reset();
    blocks[2].reset();

    CacheBlockRef first_reused = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1);
    CacheBlockRef second_reused = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1);

    ASSERT_TRUE(first_reused);
    ASSERT_TRUE(second_reused);
    EXPECT_EQ(first_reused->Location(), first_released);
    EXPECT_EQ(second_reused->Location(), second_released);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, IndependentChildrenShareOneParent) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->Location().lcm_block_id, second->Location().lcm_block_id);
    EXPECT_NE(first->Location().slot_index, second->Location().slot_index);
    EXPECT_EQ(pool.OccupiedCount(1), 2);
}

TEST(BlockPoolLcmPlacementTest, ReleasingOneChildKeepsSiblingAndReusesOnlyItsSlot) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef sibling = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    const CacheBlockLocation released = first->Location();
    const CacheBlockLocation retained = sibling->Location();

    first.reset();
    EXPECT_FALSE(pool.IsOccupied(released));
    EXPECT_TRUE(pool.IsOccupied(retained));

    CacheBlockRef replacement = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location(), released);
    EXPECT_TRUE(pool.IsOccupied(retained));
}

TEST(BlockPoolLcmPlacementTest, FillsMostOccupiedPartialParentBeforeFreeParent) {
    BlockPool pool(3);
    auto first_parent = pool.AcquireBlocks(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3, /*num=*/3);
    ASSERT_EQ(first_parent.size(), 3u);
    CacheBlockRef second_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(second_parent);
    first_parent.back().reset();

    CacheBlockRef fills_more_occupied_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_more_occupied_parent);
    EXPECT_EQ(fills_more_occupied_parent->Location().lcm_block_id, first_parent.front()->Location().lcm_block_id);

    CacheBlockRef fills_second_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_second_parent);
    CacheBlockRef fills_second_parent_again = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_second_parent_again);
    CacheBlockRef uses_free_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(uses_free_parent);
    EXPECT_EQ(uses_free_parent->Location().lcm_block_id, 3);
}

TEST(BlockPoolLcmPlacementTest, LastReleaseImmediatelyClearsParentBinding) {
    BlockPool pool(1);
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/4, /*cache_blocks_per_lcm_block=*/8);
    child.reset();

    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}


// A shrinking arena stops handing out the parents it is giving up, and says
// when they have drained. The primitive a movable arena boundary needs: the
// right split between the attention arena and the recurrent-state one depends
// on the workload, so the boundary has to move after traffic is seen.
TEST(BlockPoolUsableParents, CapExcludesParentsFromAllocationAndCount) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    EXPECT_EQ(pool.UsableParents(), 8);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
    EXPECT_TRUE(pool.DrainedAboveUsable());

    pool.SetUsableParents(5);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 5);
    EXPECT_TRUE(pool.DrainedAboveUsable());

    // Every block the pool hands out now sits at or below the cap.
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 5);
    ASSERT_EQ(blocks.size(), 5u);
    for (const CacheBlockRef& block : blocks) {
        EXPECT_LE(block->Location().lcm_block_id, 5);
    }
    // And it cannot reach above the cap even though three parents are free.
    EXPECT_TRUE(pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 1).empty());
}

TEST(BlockPoolUsableParents, DrainedReportsFalseWhileGivenUpParentsAreStillBound) {
    BlockPool pool(/*num_lcm_blocks=*/4);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 4);
    ASSERT_EQ(blocks.size(), 4u);

    pool.SetUsableParents(2);
    // Parents 3 and 4 are bound, so the boundary must not move yet.
    EXPECT_FALSE(pool.DrainedAboveUsable());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);

    blocks.clear();  // every block released
    EXPECT_TRUE(pool.DrainedAboveUsable());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
}

// The mirror. The recurrent-state pool keeps its HIGHEST parents, so its free
// rows fall next to the attention parents where a moved boundary can reach
// them. Both regions reach the kernels as strided tensor views and torch has
// no negative stride, so the layout cannot count down from the far end -- the
// allocator is what runs backwards instead.
TEST(BlockPoolUsableTopParents, KeepsTheHighParentsAndGivesUpTheLowOnes) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    pool.SetUsableTopParents(3);
    EXPECT_EQ(pool.UsableParents(), 3);
    EXPECT_EQ(pool.UsableFrom(), 6);
    EXPECT_EQ(pool.UsableTo(), 8);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);

    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 3);
    ASSERT_EQ(blocks.size(), 3u);
    for (const CacheBlockRef& block : blocks) {
        EXPECT_GE(block->Location().lcm_block_id, 6);
    }
    // Five parents are free and none of them may be handed out.
    EXPECT_TRUE(pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 1).empty());
}

TEST(BlockPoolUsableTopParents, DrainWaitsForTheGivenUpLowParents) {
    BlockPool pool(/*num_lcm_blocks=*/4);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 4);
    ASSERT_EQ(blocks.size(), 4u);

    pool.SetUsableTopParents(2);
    // Parents 1 and 2 are the ones given up, and they are still bound. A cap
    // that only ever looked above itself would call this drained and let the
    // boundary move over live pages.
    EXPECT_FALSE(pool.DrainedOutsideUsable());

    blocks.clear();
    EXPECT_TRUE(pool.DrainedOutsideUsable());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
}

// The packed path plans locations separately, and it used to reach straight
// into the free list without consulting the cap at all -- so a group with more
// than one block per parent could take a parent the pool had already given to
// the other arena.
TEST(BlockPoolUsableParents, APackedGroupCannotTakeAGivenUpParent) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    pool.SetUsableParents(2);

    // Two parents usable, four blocks each: eight blocks is everything there
    // is, and a ninth must fail rather than reaching past the cap.
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/4, 8);
    ASSERT_EQ(blocks.size(), 8u);
    for (const CacheBlockRef& block : blocks) {
        EXPECT_LE(block->Location().lcm_block_id, 2);
    }
    EXPECT_TRUE(pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/4, 1).empty());
}

// The free list is a deque that release appends to, so the parent a capped
// pool picks is not always its front. Releasing 7 then 3 leaves [7, 3]; under
// a cap of 5 the pool must hand out 3. `occupy` used to assert the front and
// turned that placement into a crash -- and no existing test reached it,
// because a cap applied to a fresh pool always picks the front.
TEST(BlockPoolUsableParents, PlacesAFreeParentThatIsNotTheFrontOfTheList) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheBlockRef> all = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 8);
    ASSERT_EQ(all.size(), 8u);

    std::vector<CacheBlockRef> keep;
    for (CacheBlockRef& block : all) {
        const std::int32_t id = block->Location().lcm_block_id;
        if (id != 7 && id != 3) {
            keep.push_back(std::move(block));
        }
    }
    all.clear();  // 7 released first, then 3

    pool.SetUsableParents(5);
    std::vector<CacheBlockRef> got = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 1);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got.front()->Location().lcm_block_id, 3);
}

// Capping the range is not enough on its own. While the whole pool is usable
// the acquire loop takes the front of the free list, so a state pool fills
// from its LOW end -- and a later shrink to the top S then has to wait for
// pages scattered across the low ids to drain, which under steady traffic may
// never happen. Filling from the top makes the shrink immediate.
TEST(BlockPoolPreferHighParents, FillsFromTheTopSoAShrinkCanHappenAtOnce) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    pool.PreferHighParents(true);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 3);
    ASSERT_EQ(blocks.size(), 3u);
    for (const CacheBlockRef& block : blocks) {
        EXPECT_GE(block->Location().lcm_block_id, 6);
    }
    pool.SetUsableTopParents(3);
    EXPECT_TRUE(pool.DrainedOutsideUsable());
}

TEST(BlockPoolPreferHighParents, WithoutItTheSameShrinkIsBlocked) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, 3);
    ASSERT_EQ(blocks.size(), 3u);
    for (const CacheBlockRef& block : blocks) {
        EXPECT_LE(block->Location().lcm_block_id, 3);
    }
    pool.SetUsableTopParents(3);
    // 1..3 are bound and now outside the range: the boundary cannot move.
    EXPECT_FALSE(pool.DrainedOutsideUsable());
}

}  // namespace
}  // namespace tokenspeed::test
