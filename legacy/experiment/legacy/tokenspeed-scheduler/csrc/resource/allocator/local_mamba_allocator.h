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
    explicit LocalMambaAllocator(MambaChunkAllocator* allocator);
    ~LocalMambaAllocator() = default;

    LocalMambaAllocator(LocalMambaAllocator&&) noexcept = default;
    LocalMambaAllocator& operator=(LocalMambaAllocator&&) noexcept = default;
    LocalMambaAllocator(const LocalMambaAllocator&) = delete;
    LocalMambaAllocator& operator=(const LocalMambaAllocator&) = delete;

    bool AllocateWorking();
    void ReleaseWorking();
    std::int32_t WorkingIndex() const;
    bool HasWorking() const { return working_ != nullptr; }

    bool AllocateCheckpoint();
    std::int32_t CheckpointIndex() const;
    bool HasCheckpoint() const { return checkpoint_ != nullptr; }
    std::unique_ptr<MambaSlot> DetachCheckpoint();
    std::unique_ptr<MambaSlot> DetachWorking();

private:
    MambaChunkAllocator* allocator_;
    std::unique_ptr<MambaSlot> working_{};
    std::unique_ptr<MambaSlot> checkpoint_{};
};

}  // namespace tokenspeed
