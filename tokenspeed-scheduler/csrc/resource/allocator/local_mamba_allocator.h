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
#include <memory>

#include "resource/radix_tree/mamba_slot.h"

namespace tokenspeed {

class MambaChunkAllocator;

class LocalMambaAllocator {
public:
    // Two pools, because the two things have different lifetimes: `allocator`
    // serves this request's working state, `checkpoint_allocator` serves the
    // snapshots it hands to the radix tree. Passing the same pointer for both
    // is the old single-pool arrangement and is still supported.
    explicit LocalMambaAllocator(MambaChunkAllocator* allocator, MambaChunkAllocator* checkpoint_allocator = nullptr);
    ~LocalMambaAllocator() = default;

    LocalMambaAllocator(LocalMambaAllocator&&) noexcept = default;
    LocalMambaAllocator& operator=(LocalMambaAllocator&&) noexcept = default;
    LocalMambaAllocator(const LocalMambaAllocator&) = delete;
    LocalMambaAllocator& operator=(const LocalMambaAllocator&) = delete;

    bool AllocateWorking();
    void ReleaseWorking();
    std::int32_t WorkingIndex() const;
    bool HasWorking() const { return working_ != nullptr; }

    // Recurrent snapshots retained for later prefix reuse.
    //
    // More than one, because reuse on a hybrid is min(KV depth, deepest usable
    // snapshot) and a single snapshot can only sit at one depth. Published at
    // the end of a request it serves an exact repeat and nothing else: a
    // request sharing the document but asking a different question matches KV
    // to the divergence point, which is *above* that snapshot, and gets zero.
    // See unieinfra/xref/35-hybrid-prefix-reuse.md.
    //
    // Allocation is best-effort: these are a cache, so running out of slots
    // must degrade reuse depth rather than fail the request. Callers get
    // however many were available, in ascending depth order.
    bool AllocateCheckpoints(std::int32_t count);
    std::vector<std::int32_t> CheckpointIndices() const;
    std::size_t CheckpointCount() const { return checkpoints_.size(); }
    std::vector<std::unique_ptr<MambaSlot>> DetachCheckpoints();

    // Slots this step actually attached to a radix node, ascending by depth.
    //
    // Detaching hands ownership to the tree, so by the time the forward
    // operation is built the allocator no longer knows which slots it gave
    // away -- and the operation is what tells the GPU where to write. Without
    // this the nodes would hold slots that never receive a state, and a later
    // hit would resume from uninitialised memory. Recorded at attach time, in
    // the same order as the depths.
    void RecordPublishedCheckpoint(std::int32_t index) { published_.push_back(index); }
    const std::vector<std::int32_t>& PublishedCheckpointIndices() const { return published_; }
    void ClearPublishedCheckpoints() { published_.clear(); }

    // Single-snapshot spellings, kept so callers that only ever want the
    // deepest one (decode-side tracking, retract) stay unchanged.
    bool AllocateCheckpoint() { return AllocateCheckpoints(1); }
    std::int32_t CheckpointIndex() const;
    bool HasCheckpoint() const { return !checkpoints_.empty(); }
    std::unique_ptr<MambaSlot> DetachCheckpoint();

    std::unique_ptr<MambaSlot> DetachWorking();

private:
    MambaChunkAllocator* allocator_;
    MambaChunkAllocator* checkpoint_allocator_;
    std::unique_ptr<MambaSlot> working_{};
    std::vector<std::unique_ptr<MambaSlot>> checkpoints_{};
    std::vector<std::int32_t> published_{};
};

}  // namespace tokenspeed
