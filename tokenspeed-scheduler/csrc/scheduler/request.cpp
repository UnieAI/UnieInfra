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

#include "scheduler/page_hasher.h"
#include "scheduler/request.h"

#include <stdexcept>

#include "fsm/forward_events.h"

namespace tokenspeed {

Request::Request(const RequestSpec& spec, std::int32_t page_size, Role role)
    : id_{spec.request_id},
      token_container_{spec.tokens},
      max_new_tokens_{spec.max_new_tokens},
      page_size_{page_size},
      state_{role == Role::kFused ? fsm::State{fsm::Submitted{&token_container_, page_size}}
                                  : fsm::State{fsm::Bootstrapping{&token_container_, page_size}}} {}

PrefillInfo Request::CurrentPrefillInfo() const {
    return std::visit(
        Overloaded{
            [](const fsm::Prefilling& state) { return state.CurrentPrefillInfo(); },
            [](const fsm::PrefillDone& state) { return state.CurrentPrefillInfo(); },
            [this](const auto&) -> PrefillInfo {
                throw std::logic_error("Request::CurrentPrefillInfo: expected Prefilling or PrefillDone; got " +
                                       StateName());
            },
        },
        state_);
}

fsm::ForwardState& Request::forwardState(const char* operation) {
    fsm::ForwardState* result = std::visit(
        []<typename State>(State& state) -> fsm::ForwardState* {
            if constexpr (std::derived_from<State, fsm::ForwardState>) {
                return &state;
            }
            return nullptr;
        },
        state_);
    if (result == nullptr) {
        throw std::logic_error(std::string{"Request::"} + operation + ": expected a forward state; got " + StateName());
    }
    return *result;
}

const fsm::ForwardState& Request::forwardState(const char* operation) const {
    const fsm::ForwardState* result = std::visit(
        []<typename State>(const State& state) -> const fsm::ForwardState* {
            if constexpr (std::derived_from<State, fsm::ForwardState>) {
                return &state;
            }
            return nullptr;
        },
        state_);
    if (result == nullptr) {
        throw std::logic_error(std::string{"Request::"} + operation + ": expected a forward state; got " + StateName());
    }
    return *result;
}

std::span<const std::string> Request::PagedHashes(std::int32_t pages) const {
    if (pages <= 0) {
        return {};
    }
    const auto want = static_cast<std::size_t>(pages);
    if (paged_hashes_.size() < want) {
        std::vector<std::span<const std::int32_t>> paged = FullPagedTokens(false);
        if (paged.size() > want) {
            paged.resize(want);
        }
        // Continue from where the chain already ends. `prior` is the last hash
        // held, or "" for a cold start -- the same seed ComputePagedHashes uses.
        const std::size_t have = std::min(paged_hashes_.size(), paged.size());
        if (have < paged.size()) {
            const std::string prior = have == 0 ? std::string{} : paged_hashes_[have - 1];
            std::vector<std::string> tail = ComputePagedHashes(
                std::span<const std::span<const std::int32_t>>(paged).subspan(have), prior);
            paged_hashes_.insert(paged_hashes_.end(), tail.begin(), tail.end());
        }
    }
    const auto have = std::min(paged_hashes_.size(), want);
    return std::span<const std::string>(paged_hashes_).first(have);
}

}  // namespace tokenspeed
