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

#include <concepts>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/token_container.h"
#include "fsm/forward_states.h"
#include "fsm/states.h"
#include "scheduler/request_spec.h"
#include "utils.h"

namespace tokenspeed {

class Request {
public:
    Request(const RequestSpec& spec, std::int32_t page_size, Role role);

    const std::string& Id() const { return id_; }

    std::int32_t MaxNewTokens() const { return max_new_tokens_; }

    template <typename Event>
    void Apply(Event&& event) {
        state_ = std::visit(
            [&event](auto&& state) -> fsm::State { return fsm::ToState(std::forward<Event>(event)(std::move(state))); },
            std::move(state_));
    }

    template <typename State>
    bool Is() const {
        return std::holds_alternative<State>(state_);
    }

    std::vector<std::span<const std::int32_t>> FullPagedTokens(bool except_last) const {
        return token_container_.FullPagedTokens(page_size_, except_last);
    }

    // Hash chain over the request's complete pages, computed once and extended.
    //
    // `matchPrefixAtAdmission` used to call ComputePagedHashes over the whole
    // prompt on every admission attempt (forward.cpp). A request that cannot be
    // admitted is retried every step, and HashPage feeds each token to
    // SHA256_Update individually plus a hex conversion and a 64-char string per
    // page (page_hasher.h) -- so a 10k-token prompt blocked for ten steps paid
    // 100k SHA256_Update calls and 3,120 allocations for an answer that had not
    // changed. vLLM keeps the chain on the request and only extends it
    // (v1/request.py:237-240).
    //
    // Safe because TokenContainer::Extend only appends (token_container.cpp:27):
    // the token values under any already-hashed page never change, so a prefix
    // of the chain stays valid. RebasePrefill moves the prefill boundary, which
    // changes how many pages are wanted, not what the earlier ones contain.
    std::span<const std::string> PagedHashes(std::int32_t pages) const;

    std::int32_t TokenSize() const { return token_container_.Size(); }
    std::int32_t LastToken() const { return token_container_.LastToken(); }
    std::int32_t PrefillSize() const { return token_container_.PrefillSize(); }
    PrefillInfo CurrentPrefillInfo() const;

    std::int32_t UnscheduledPrefillSize() const {
        return std::visit(Overloaded{
                              [](const fsm::Submitted&) -> std::int32_t { return -1; },
                              [this](const fsm::Prefilling& state) -> std::int32_t {
                                  return PrefillSize() - (state.window.begin + state.window.size);
                              },
                              [](const auto&) -> std::int32_t { return 0; },
                          },
                          state_);
    }

    std::int32_t RequestPoolIndex() const { return forwardState("RequestPoolIndex").RequestPoolIndex(); }

    std::vector<std::int32_t> ActiveLcmBlockIds() const {
        return forwardState("ActiveLcmBlockIds").ActiveLcmBlockIds();
    }

    const std::vector<BlockTable>& BlockTablesRef() const { return forwardState("BlockTablesRef").BlockTables(); }

    std::vector<BlockTable>& BlockTablesRef() { return forwardState("BlockTablesRef").BlockTables(); }

    fsm::CacheProgress CacheProgress() const { return forwardState("CacheProgress").CacheProgressRef(); }

    fsm::PrefillSource PrefillSource() const {
        if (const auto* state = std::get_if<fsm::Prefilling>(&state_)) {
            return state->Source();
        }
        throw std::logic_error("Request::PrefillSource: expected Prefilling; got " + StateName());
    }

    std::int32_t ReserveNumTokensInNextScheduleEvent() const {
        return std::visit(
            Overloaded{
                [](const fsm::PrefillDone& state) { return state.ReserveNumTokensInNextScheduleEvent(); },
                [](const fsm::Decoding& state) { return state.ReserveNumTokensInNextScheduleEvent(); },
                [this](const auto&) -> std::int32_t {
                    throw std::logic_error(
                        "Request::ReserveNumTokensInNextScheduleEvent: expected PrefillDone or Decoding; got " +
                        StateName());
                },
            },
            state_);
    }

    std::string StateName() const {
        return std::visit(Overloaded{
                              [](const fsm::Bootstrapping&) -> std::string { return "Bootstrapping"; },
                              [](const fsm::Submitted&) -> std::string { return "Submitted"; },
                              [](const fsm::Prefilling&) -> std::string { return "Prefilling"; },
                              [](const fsm::PrefillDone&) -> std::string { return "PrefillDone"; },
                              [](const fsm::Decoding&) -> std::string { return "Decoding"; },
                              [](const fsm::Retracted&) -> std::string { return "Retracted"; },
                              [](const fsm::Finished&) -> std::string { return "Finished"; },
                          },
                          state_);
    }

private:
    fsm::ForwardState& forwardState(const char* operation);
    const fsm::ForwardState& forwardState(const char* operation) const;

    std::string id_;
    TokenContainer token_container_;
    // The caller's generation budget. Admission needs it to charge what a
    // request will eventually occupy rather than only its prompt: SGLang's
    // reservation is prompt + max_new_tokens * an adaptive ratio, and without
    // this field the scheduler cannot express the second term.
    std::int32_t max_new_tokens_{0};
    std::int32_t page_size_{};
    fsm::State state_;
    // Filled lazily by PagedHashes(); mutable so the accessor stays const.
    mutable std::vector<std::string> paged_hashes_;
};

}  // namespace tokenspeed
