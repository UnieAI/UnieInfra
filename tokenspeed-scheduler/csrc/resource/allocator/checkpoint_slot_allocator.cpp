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

#include "checkpoint_slot_allocator.h"

namespace tokenspeed {

CheckpointSlot::CheckpointSlot(std::int32_t slot, CheckpointSlotAllocator* allocator)
    : slot_(slot), allocator_(allocator) {}

CheckpointSlot::CheckpointSlot(CheckpointSlot&& other) noexcept : slot_(other.slot_), allocator_(other.allocator_) {
    other.allocator_ = nullptr;
    other.slot_ = kNoSlot;
}

CheckpointSlot& CheckpointSlot::operator=(CheckpointSlot&& other) noexcept {
    if (this != &other) {
        if (allocator_) {
            allocator_->deallocate(slot_);
        }
        slot_ = other.slot_;
        allocator_ = other.allocator_;
        other.slot_ = kNoSlot;
        other.allocator_ = nullptr;
    }
    return *this;
}

CheckpointSlot::~CheckpointSlot() {
    if (allocator_) {
        allocator_->deallocate(slot_);
    }
}

bool CheckpointSlot::valid() const {
    return allocator_ != nullptr;
}

// Slot 0 is a usable checkpoint here, unlike `ReqPoolAllocator` where it is
// reserved: the runtime addresses this pool with its own index space and marks
// absence with -1, so there is no null row to keep clear.
CheckpointSlotAllocator::CheckpointSlotAllocator(std::int32_t size) : size_(size < 0 ? 0 : size) {
    for (std::int32_t i = 0; i < size_; ++i) {
        free_slots_.push_back(i);
    }
}

CheckpointSlot CheckpointSlotAllocator::Allocate() {
    if (free_slots_.empty()) {
        return CheckpointSlot{};
    }
    std::int32_t slot = free_slots_.front();
    free_slots_.pop_front();
    return CheckpointSlot{slot, this};
}

std::int32_t CheckpointSlotAllocator::Size() const {
    return size_;
}

std::int32_t CheckpointSlotAllocator::AvailableSlots() const {
    return static_cast<std::int32_t>(free_slots_.size());
}

// Back, not front: a checkpoint that was just released is the least likely to
// be read again, and reusing the oldest free slot keeps a recently-freed
// snapshot addressable for longer if a late match still names it.
void CheckpointSlotAllocator::deallocate(std::int32_t slot) {
    free_slots_.push_back(slot);
}

}  // namespace tokenspeed
