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
#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "unit_test_helper.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"

namespace tokenspeed::test {

// ============================================================
//  OwnedPages unit tests — RAII, move semantics, split/merge
// ============================================================

class OwnedPagesTestSuite : public ::testing::Test {
protected:
    PageAllocator alloc_{2, 10};  // page_size=2, 9 usable pages (1..9)
};

TEST_F(OwnedPagesTestSuite, DefaultConstructed_IsEmpty) {
    OwnedPages empty;
    EXPECT_TRUE(empty.Empty());
    EXPECT_EQ(empty.Size(), 0);
    EXPECT_TRUE(empty.Ids().empty());
}

TEST_F(OwnedPagesTestSuite, Allocate_OwnsPages) {
    OwnedPages pages = alloc_.Allocate(3);
    EXPECT_EQ(pages.Size(), 3);
    EXPECT_EQ(alloc_.AvailablePages(), 6);
}

TEST_F(OwnedPagesTestSuite, DestructorDeallocates) {
    {
        OwnedPages pages = alloc_.Allocate(4);
        EXPECT_EQ(alloc_.AvailablePages(), 5);
    }
    EXPECT_EQ(alloc_.AvailablePages(), 9);
}

TEST_F(OwnedPagesTestSuite, MoveConstructor_TransfersOwnership) {
    OwnedPages a = alloc_.Allocate(3);
    auto ids = a.Ids();
    OwnedPages b(std::move(a));

    EXPECT_EQ(b.Size(), 3);
    EXPECT_EQ(b.Ids(), ids);
    EXPECT_TRUE(a.Empty());  // NOLINT: testing moved-from state
    EXPECT_EQ(alloc_.AvailablePages(), 6);
}

TEST_F(OwnedPagesTestSuite, MoveConstructor_SourceDestructorIsNoop) {
    OwnedPages a = alloc_.Allocate(3);
    {
        OwnedPages b(std::move(a));
        // b holds the pages
    }
    // b destroyed → pages returned
    EXPECT_EQ(alloc_.AvailablePages(), 9);
}

TEST_F(OwnedPagesTestSuite, MoveAssignment_TransfersOwnership) {
    OwnedPages a = alloc_.Allocate(2);
    OwnedPages b = alloc_.Allocate(3);
    EXPECT_EQ(alloc_.AvailablePages(), 4);

    b = std::move(a);
    // b's old 3 pages deallocated, b now holds a's 2 pages
    EXPECT_EQ(b.Size(), 2);
    EXPECT_EQ(alloc_.AvailablePages(), 7);
}

TEST_F(OwnedPagesTestSuite, MoveAssignment_SelfAssign) {
    OwnedPages a = alloc_.Allocate(3);
    auto* ptr = &a;
    a = std::move(*ptr);  // self-assignment
    EXPECT_EQ(a.Size(), 3);
    EXPECT_EQ(alloc_.AvailablePages(), 6);
}

TEST_F(OwnedPagesTestSuite, TakeFirst_SplitsCorrectly) {
    OwnedPages pages = alloc_.Allocate(5);
    auto all_ids = pages.Ids();

    OwnedPages first = pages.TakeFirst(2);
    EXPECT_EQ(first.Size(), 2);
    EXPECT_EQ(pages.Size(), 3);
    EXPECT_EQ(first.Ids()[0], all_ids[0]);
    EXPECT_EQ(first.Ids()[1], all_ids[1]);
    EXPECT_EQ(pages.Ids()[0], all_ids[2]);
    EXPECT_EQ(alloc_.AvailablePages(), 4);  // still 5 pages held total
}

TEST_F(OwnedPagesTestSuite, TakeFirst_DestroyBoth_AllReturned) {
    {
        OwnedPages pages = alloc_.Allocate(5);
        OwnedPages first = pages.TakeFirst(2);
        // both destroyed at end of scope
    }
    EXPECT_EQ(alloc_.AvailablePages(), 9);
}

TEST_F(OwnedPagesTestSuite, TakeLast_SplitsCorrectly) {
    OwnedPages pages = alloc_.Allocate(5);
    auto all_ids = pages.Ids();

    OwnedPages last = pages.TakeLast(2);
    EXPECT_EQ(last.Size(), 2);
    EXPECT_EQ(pages.Size(), 3);
    EXPECT_EQ(last.Ids()[0], all_ids[3]);
    EXPECT_EQ(last.Ids()[1], all_ids[4]);
    EXPECT_EQ(pages.Ids()[0], all_ids[0]);
    EXPECT_EQ(alloc_.AvailablePages(), 4);
}

TEST_F(OwnedPagesTestSuite, TakeLast_DestroyBoth_AllReturned) {
    {
        OwnedPages pages = alloc_.Allocate(5);
        OwnedPages last = pages.TakeLast(3);
    }
    EXPECT_EQ(alloc_.AvailablePages(), 9);
}

TEST_F(OwnedPagesTestSuite, Append_MergesPages) {
    OwnedPages a = alloc_.Allocate(2);
    OwnedPages b = alloc_.Allocate(3);
    auto a_ids = a.Ids();
    auto b_ids = b.Ids();

    a.Append(std::move(b));
    EXPECT_EQ(a.Size(), 5);
    EXPECT_TRUE(b.Empty());  // NOLINT
    // Verify order: a's original pages first, then b's
    EXPECT_EQ(a.Ids()[0], a_ids[0]);
    EXPECT_EQ(a.Ids()[1], a_ids[1]);
    EXPECT_EQ(a.Ids()[2], b_ids[0]);
    EXPECT_EQ(alloc_.AvailablePages(), 4);
}

TEST_F(OwnedPagesTestSuite, Append_DestroyMerged_AllReturned) {
    {
        OwnedPages a = alloc_.Allocate(2);
        OwnedPages b = alloc_.Allocate(3);
        a.Append(std::move(b));
    }
    EXPECT_EQ(alloc_.AvailablePages(), 9);
}

TEST_F(OwnedPagesTestSuite, Append_ToDefaultConstructed) {
    OwnedPages a;
    OwnedPages b = alloc_.Allocate(3);
    a.Append(std::move(b));
    EXPECT_EQ(a.Size(), 3);
    EXPECT_EQ(alloc_.AvailablePages(), 6);
}

TEST_F(OwnedPagesTestSuite, ChainedSplitAndMerge) {
    OwnedPages pages = alloc_.Allocate(6);
    OwnedPages first2 = pages.TakeFirst(2);
    OwnedPages last2 = pages.TakeLast(2);
    // pages now has 2 (middle), first2 has 2, last2 has 2
    EXPECT_EQ(pages.Size(), 2);

    first2.Append(std::move(last2));
    EXPECT_EQ(first2.Size(), 4);

    first2.Append(std::move(pages));
    EXPECT_EQ(first2.Size(), 6);
    EXPECT_EQ(alloc_.AvailablePages(), 3);
}

TEST_F(OwnedPagesTestSuite, UnownedPages_NoDeallocationOnDestroy) {
    std::int32_t before = alloc_.AvailablePages();
    {
        OwnedPages unowned{nullptr, {100, 200, 300}};
        EXPECT_EQ(unowned.Size(), 3);
    }
    // No crash, no deallocation
    EXPECT_EQ(alloc_.AvailablePages(), before);
}

// ============================================================
//  Prefix cache + OwnedPages: Insert ownership transfer
// ============================================================

class PrefixCacheOwnedPagesTestSuite : public ::testing::Test {
protected:
    PageAllocator device_alloc_{2, 20};  // page_size=2, 19 usable
    PageAllocator host_alloc_{2, 20};
    KVPrefixCache cache_{&device_alloc_, &host_alloc_, false};

    // Helper: collect all page IDs held in tree for a resource type + free pages in allocator.
    // The sum should always equal total_pages - 1 (page 0 is reserved).
    template <ResourceType RType>
    void VerifyPageAccounting(PageAllocator& alloc) {
        auto tree_pages = cache_.CollectAllPages<RType>();
        std::int32_t tree_total = 0;
        for (auto& [id, count] : tree_pages) {
            EXPECT_EQ(count, 1) << "page " << id << " appears " << count << " times in tree";
            tree_total += count;
        }
        // tree_total + free == total - 1  (page 0 reserved)
        EXPECT_EQ(tree_total + alloc.AvailablePages(), alloc.TotalPages() - 1)
            << "page accounting mismatch: tree=" << tree_total << " free=" << alloc.AvailablePages()
            << " total=" << alloc.TotalPages();
    }
};

// Insert transfers OwnedPages to tree; allocator pages not leaked.
TEST_F(PrefixCacheOwnedPagesTestSuite, Insert_TransfersOwnershipToTree) {
    OwnedPages dev_owned = device_alloc_.Allocate(2);
    EXPECT_EQ(device_alloc_.AvailablePages(), 17);

    token_vec_t tokens = {1, 2, 3, 4};
    cache_.Insert<ResourceType::Device>(tokens, {}, std::move(dev_owned));

    // Pages are now in the tree, not free
    EXPECT_EQ(device_alloc_.AvailablePages(), 17);
    VerifyPageAccounting<ResourceType::Device>(device_alloc_);
}

// Insert with partial skip: some nodes already have pages → those allocator pages are auto-freed.
TEST_F(PrefixCacheOwnedPagesTestSuite, Insert_UnconsumedPagesReturnedToPool) {
    // First insert: [1,2,3,4] → 2 pages
    OwnedPages dev1 = device_alloc_.Allocate(2);
    cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev1));

    // Second insert same tokens: all nodes already have pages → allocator pages freed on return
    OwnedPages dev2 = device_alloc_.Allocate(2);
    std::int32_t before = device_alloc_.AvailablePages();

    auto result = cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev2));
    EXPECT_EQ(result.inserted_num_pages, 0);
    EXPECT_EQ(device_alloc_.AvailablePages(), before + 2);
    VerifyPageAccounting<ResourceType::Device>(device_alloc_);
}

// Insert with partially new tokens: prefix matches, suffix gets new pages.
TEST_F(PrefixCacheOwnedPagesTestSuite, Insert_PartialMatch_CorrectOwnership) {
    // Insert [1,2,3,4]
    OwnedPages dev1 = device_alloc_.Allocate(2);
    cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev1));

    // Insert [1,2,5,6]: prefix [1,2] matches, [5,6] is new
    OwnedPages dev2 = device_alloc_.Allocate(2);
    auto result = cache_.Insert<ResourceType::Device>({1, 2, 5, 6}, {}, std::move(dev2));

    // Only 1 page inserted (suffix [5,6]), prefix [1,2] already existed
    EXPECT_EQ(result.inserted_num_pages, 1);
    VerifyPageAccounting<ResourceType::Device>(device_alloc_);
}

// alloc_nodes: successful allocation attaches pages to nodes.
TEST_F(PrefixCacheOwnedPagesTestSuite, AllocNodes_Success) {
    // Insert device-only for [1,2,3,4]
    OwnedPages dev = device_alloc_.Allocate(2);
    cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev));

    // Match and get nodes without host
    MatchResult match = cache_.Match({1, 2, 3, 4});
    EXPECT_EQ(match.device.DepthInPage(), 2);
    EXPECT_EQ(match.host.DepthInPage(), 0);

    auto nodes = match.NodesWithout<ResourceType::Host>();
    EXPECT_FALSE(nodes.empty());

    // alloc_nodes<Host> should allocate host pages for those nodes
    bool ok = cache_.AllocateResourceOfType<ResourceType::Host>(nodes);
    EXPECT_TRUE(ok);

    MatchResult match2 = cache_.Match({1, 2, 3, 4});
    EXPECT_EQ(match2.host.DepthInPage(), 2);
    VerifyPageAccounting<ResourceType::Host>(host_alloc_);
}

// alloc_nodes: partial allocation failure returns pages to pool (leak fix).
TEST_F(PrefixCacheOwnedPagesTestSuite, AllocNodes_PartialFailure_NoLeak) {
    // Create a small host allocator: only 1 usable page (total=2, page 0 reserved)
    PageAllocator small_host{2, 2};
    KVPrefixCache small_cache{&device_alloc_, &small_host, false};

    // Insert 2 pages on device
    OwnedPages dev = device_alloc_.Allocate(2);
    small_cache.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev));

    MatchResult match = small_cache.Match({1, 2, 3, 4});
    auto nodes = match.NodesWithout<ResourceType::Host>();

    // Need 2 host pages but only 1 available → allocation fails
    std::int32_t before = small_host.AvailablePages();
    bool ok = small_cache.AllocateResourceOfType<ResourceType::Host>(nodes);
    EXPECT_FALSE(ok);
    // All pages returned to pool — no leak
    EXPECT_EQ(small_host.AvailablePages(), before);
}

// Eviction: evicted pages are auto-deallocated by OwnedPages destructor.
TEST_F(PrefixCacheOwnedPagesTestSuite, Eviction_PagesReturnToPool) {
    // Insert 3 separate token sequences to fill device
    OwnedPages d1 = device_alloc_.Allocate(1);
    cache_.Insert<ResourceType::Device>({1, 2}, {}, std::move(d1));

    OwnedPages d2 = device_alloc_.Allocate(1);
    cache_.Insert<ResourceType::Device>({3, 4}, {}, std::move(d2));

    OwnedPages d3 = device_alloc_.Allocate(1);
    cache_.Insert<ResourceType::Device>({5, 6}, {}, std::move(d3));

    std::int32_t free_before = device_alloc_.AvailablePages();
    // Evict enough for 2 pages
    bool ok = cache_.EnsureCapacityByEvict<ResourceType::Device>(free_before + 2);
    EXPECT_TRUE(ok);
    EXPECT_GE(device_alloc_.AvailablePages(), free_before + 2);

    VerifyPageAccounting<ResourceType::Device>(device_alloc_);
}

// SplitSelfInto: after split, both prefix and suffix hold correct pages.
TEST_F(PrefixCacheOwnedPagesTestSuite, Split_PreservesPageOwnership) {
    // Insert [1,2,3,4] as one node with 2 pages
    OwnedPages dev = device_alloc_.Allocate(2);
    cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev));

    // Now insert [1,2,5,6]: this triggers splitChild on node [1,2,3,4]
    // → prefix [1,2] (1 page) + suffix [3,4] (1 page), plus new [5,6] (1 page)
    OwnedPages dev2 = device_alloc_.Allocate(2);
    cache_.Insert<ResourceType::Device>({1, 2, 5, 6}, {}, std::move(dev2));

    // Verify all three nodes have pages
    MatchResult m1 = cache_.Match({1, 2, 3, 4});
    EXPECT_EQ(m1.device.DepthInPage(), 2);

    MatchResult m2 = cache_.Match({1, 2, 5, 6});
    EXPECT_EQ(m2.device.DepthInPage(), 2);

    VerifyPageAccounting<ResourceType::Device>(device_alloc_);
}

// Full lifecycle: Insert → Evict → pages fully accounted.
TEST_F(PrefixCacheOwnedPagesTestSuite, FullLifecycle_InsertEvictAccounting) {
    // Insert many nodes
    for (int i = 0; i < 8; i++) {
        token_t start = static_cast<token_t>(i * 2 + 1);
        OwnedPages dev = device_alloc_.Allocate(1);
        if (dev.Empty()) break;
        cache_.Insert<ResourceType::Device>({start, static_cast<token_t>(start + 1)}, {}, std::move(dev));
    }

    VerifyPageAccounting<ResourceType::Device>(device_alloc_);

    // Evict everything
    cache_.EnsureCapacityByEvict<ResourceType::Device>(device_alloc_.TotalPages());

    // All pages should be free now (except reserved page 0)
    EXPECT_EQ(device_alloc_.AvailablePages(), device_alloc_.TotalPages() - 1);
}

// Unused allocator pages are auto-deallocated when Insert returns.
TEST_F(PrefixCacheOwnedPagesTestSuite, InsertResult_UnconsumedAutoDeallocates) {
    // Insert [1,2,3,4]
    OwnedPages dev1 = device_alloc_.Allocate(2);
    cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev1));

    // Re-insert same tokens — allocator pages freed inside Insert
    OwnedPages dev2 = device_alloc_.Allocate(2);
    std::int32_t before = device_alloc_.AvailablePages();

    cache_.Insert<ResourceType::Device>({1, 2, 3, 4}, {}, std::move(dev2));
    EXPECT_EQ(device_alloc_.AvailablePages(), before + 2);
}

// Empty OwnedPages: Insert with no allocator pages (e.g., Insert<Host> path).
TEST_F(PrefixCacheOwnedPagesTestSuite, Insert_WithEmptyOwnedPages) {
    // First set up a device node
    OwnedPages dev = device_alloc_.Allocate(1);
    cache_.Insert<ResourceType::Device>({1, 2}, {}, std::move(dev));

    // Now Insert<Host> with manually allocated host pages
    OwnedPages host_owned = host_alloc_.Allocate(1);
    cache_.Insert<ResourceType::Host>({1, 2}, {}, std::move(host_owned));

    MatchResult match = cache_.Match({1, 2});
    EXPECT_EQ(match.device.DepthInPage(), 1);
    EXPECT_EQ(match.host.DepthInPage(), 1);
}

}  // namespace tokenspeed::test
