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

#include "resource/allocator/local_mamba_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"

namespace tokenspeed {

LocalMambaAllocator::LocalMambaAllocator(MambaChunkAllocator* allocator, MambaChunkAllocator* checkpoint_allocator)
    : allocator_{allocator}, checkpoint_allocator_{checkpoint_allocator != nullptr ? checkpoint_allocator : allocator} {}

bool LocalMambaAllocator::AllocateWorking() {
    auto slot = allocator_->Allocate();
    if (!slot.has_value()) return false;
    working_ = std::make_unique<MambaSlot>(std::move(*slot));
    return true;
}

void LocalMambaAllocator::ReleaseWorking() {
    working_.reset();
}

std::int32_t LocalMambaAllocator::WorkingIndex() const {
    return working_ ? working_->Index() : -1;
}

bool LocalMambaAllocator::AllocateCheckpoints(std::int32_t count) {
    if (count <= 0) return false;
    for (std::int32_t i = 0; i < count; ++i) {
        // The FIRST checkpoint is mandatory -- the request harvests its own
        // state into it and the caller throws if it is missing -- so it comes
        // from the active pool, which is what admission counts. Only the extras
        // retained for the radix tree come from the checkpoint pool, where a
        // short pool has to cost reuse depth rather than a request.
        //
        // Drawing the mandatory one from the checkpoint pool stalls the engine
        // outright when that pool is small: measured with a one-slot pool, the
        // run sat at 0% GPU because the allocation has no fallback and no way
        // to degrade.
        MambaChunkAllocator* from =
            checkpoints_.empty() ? allocator_ : checkpoint_allocator_;
        auto slot = from->Allocate();
        // Best-effort: stop at the first refusal and keep what we have. These
        // are cache entries, so a short pool costs reuse depth, never
        // correctness -- the caller publishes as many depths as it got slots.
        if (!slot.has_value()) break;
        checkpoints_.push_back(std::make_unique<MambaSlot>(std::move(*slot)));
    }
    return !checkpoints_.empty();
}

std::vector<std::int32_t> LocalMambaAllocator::CheckpointIndices() const {
    std::vector<std::int32_t> indices;
    indices.reserve(checkpoints_.size());
    for (const auto& slot : checkpoints_) {
        indices.push_back(slot ? slot->Index() : -1);
    }
    return indices;
}

std::vector<std::unique_ptr<MambaSlot>> LocalMambaAllocator::DetachCheckpoints() {
    return std::move(checkpoints_);
}

std::int32_t LocalMambaAllocator::CheckpointIndex() const {
    // The deepest one: publish order is ascending depth, so the last slot is
    // the one a single-snapshot caller means.
    if (checkpoints_.empty() || checkpoints_.back() == nullptr) return -1;
    return checkpoints_.back()->Index();
}

std::unique_ptr<MambaSlot> LocalMambaAllocator::DetachCheckpoint() {
    if (checkpoints_.empty()) return nullptr;
    auto slot = std::move(checkpoints_.back());
    checkpoints_.pop_back();
    return slot;
}

std::unique_ptr<MambaSlot> LocalMambaAllocator::DetachWorking() {
    return std::move(working_);
}

}  // namespace tokenspeed
