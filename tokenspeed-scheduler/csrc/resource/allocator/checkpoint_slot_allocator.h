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

#pragma once

#include <cstdint>
#include <deque>

namespace tokenspeed {

class CheckpointSlotAllocator;

// A slot in the recurrent-state checkpoint pool, which is a tensor of its own
// on the runtime side (`MambaCheckpointPool`) and NOT part of the LCM arena.
//
// Why a second index space at all: today a checkpoint is an arena block that
// the state was written into, so a retained snapshot costs a whole parent. A
// `linear_attention` group packs one block per parent, so three groups take
// three parents per request and every one of them must be fully empty -- the
// measured cause of 100% of admission refusals (`xref/35` §10.6). A checkpoint
// tensor has no parent granularity to waste, and the runtime copies a snapshot
// out of the working slot into it, so the GDN decode kernel never reads from
// this space and its contiguous-page ABI is untouched.
struct CheckpointSlot {
    // The allocator must outlive every slot it creates.
    CheckpointSlot() = default;

    CheckpointSlot(const CheckpointSlot&) = delete;
    CheckpointSlot& operator=(const CheckpointSlot&) = delete;

    CheckpointSlot(CheckpointSlot&& other) noexcept;
    CheckpointSlot& operator=(CheckpointSlot&& other) noexcept;

    ~CheckpointSlot();

    bool valid() const;

    // The runtime's sentinel for "the scheduler had no slot to give", which
    // `store_from_active_indexed` and `load_to_active_indexed` already drop.
    static constexpr std::int32_t kNoSlot = -1;

    // -1 when default-constructed or moved from, so this can be handed to the
    // forward operation without a validity branch at every call site.
    std::int32_t Index() const { return valid() ? slot_ : kNoSlot; }

    std::int32_t slot_{kNoSlot};

private:
    friend class CheckpointSlotAllocator;

    CheckpointSlot(std::int32_t slot, CheckpointSlotAllocator* allocator);
    CheckpointSlotAllocator* allocator_{nullptr};
};

class CheckpointSlotAllocator {
public:
    explicit CheckpointSlotAllocator(std::int32_t size);

    CheckpointSlotAllocator(const CheckpointSlotAllocator&) = delete;
    CheckpointSlotAllocator& operator=(const CheckpointSlotAllocator&) = delete;
    CheckpointSlotAllocator(CheckpointSlotAllocator&&) = delete;
    CheckpointSlotAllocator& operator=(CheckpointSlotAllocator&&) = delete;

    // Returns an invalid slot when the pool is exhausted rather than throwing,
    // unlike `ReqPoolAllocator`. A checkpoint is cache: failing to take one
    // costs a later prefix hit, never correctness, and the runtime contract
    // already carries -1 for it. Throwing here would turn a full cache into a
    // dead engine.
    CheckpointSlot Allocate();

    std::int32_t Size() const;
    std::int32_t AvailableSlots() const;

private:
    void deallocate(std::int32_t slot);

    friend struct CheckpointSlot;

    std::int32_t size_{};
    std::deque<std::int32_t> free_slots_;
};

}  // namespace tokenspeed
