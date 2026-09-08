// Request::PagedHashes 與 ComputePagedHashes 的等價性,含增量延長路徑。
#include <cassert>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "scheduler/page_hasher.h"
#include "scheduler/request.h"

using namespace tokenspeed;

static std::vector<std::string> reference(const Request& r, std::int32_t pages) {
    auto paged = r.FullPagedTokens(false);
    if (paged.size() > static_cast<std::size_t>(pages)) paged.resize(pages);
    return ComputePagedHashes(paged, "");
}

// 這支不是 gtest:容器裡沒有 git,FetchContent 抓不到 googletest。
// 編法:
//   g++ -std=c++20 -Icsrc -I. -I<spdlog-include> tests/cpp/test_paged_hash_cache.cpp \
//       csrc/scheduler/request.cpp csrc/core/token_container.cpp \
//       csrc/resource/allocator/req_pool_allocator.cpp csrc/cache/core/cache_block_ref.cpp \
//       -lcrypto -o /tmp/t_hash && /tmp/t_hash
int main() {
    const std::int32_t page = 64;
    std::vector<std::int32_t> toks(page * 10);
    std::iota(toks.begin(), toks.end(), 1);

    int checked = 0;
    // 1) 冷啟動:任意頁數都要與參考一致
    for (std::int32_t n : {1, 3, 7, 10}) {
        Request r{RequestSpec{"r", toks, 128}, page, Role::kFused};
        auto got = r.PagedHashes(n);
        auto want = reference(r, n);
        assert(got.size() == want.size());
        for (std::size_t i = 0; i < want.size(); ++i) assert(got[i] == want[i]);
        ++checked;
    }
    // 2) 增量:先要 2 頁再要 9 頁,結果必須與一次要 9 頁相同
    {
        Request r{RequestSpec{"r", toks, 128}, page, Role::kFused};
        (void)r.PagedHashes(2);
        auto got = r.PagedHashes(9);
        auto want = reference(r, 9);
        assert(got.size() == want.size());
        for (std::size_t i = 0; i < want.size(); ++i) assert(got[i] == want[i]);
        ++checked;
    }
    // 3) 縮回:要過 9 頁之後再要 3 頁,必須回傳前 3 個且不重算出不同值
    {
        Request r{RequestSpec{"r", toks, 128}, page, Role::kFused};
        auto big = std::vector<std::string>(r.PagedHashes(9).begin(), r.PagedHashes(9).end());
        auto small = r.PagedHashes(3);
        assert(small.size() == 3);
        for (std::size_t i = 0; i < 3; ++i) assert(small[i] == big[i]);
        ++checked;
    }
    // 4) 邊界:0 與負數回空
    {
        Request r{RequestSpec{"r", toks, 128}, page, Role::kFused};
        assert(r.PagedHashes(0).empty());
        assert(r.PagedHashes(-5).empty());
        ++checked;
    }
    std::printf("PagedHashes equivalence: %d groups PASS\n", checked);
    return 0;
}
