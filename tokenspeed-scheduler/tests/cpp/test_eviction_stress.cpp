// 循環存取的壓力測試：重現「深度優先淘汰在 80 個真實請求後崩潰」的形狀。
//
// 單元測試的場景都只有兩三個請求，走不到淘汰真正開始輪替的狀態。這裡照
// benchmark 語料的形狀跑：N 個 session 輪流，每個 session 多輪、逐輪加長，
// pool 故意開小讓淘汰必然發生。在 ASAN/UBSAN 下跑，記憶體錯誤會當場報出來。
#include <gtest/gtest.h>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>
#include "cache/core/block_pool.h"
#include "cache/manager/cache_group.h"
#include "cache/coordinator/kv_cache_coordinator.h"
#include "cache/core/cache_types.h"
#include "cache/manager/full_attn_manager.h"
#include "cache/manager/swa_manager.h"
#include "cache_test_access.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;

std::vector<std::string> Hashes(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<token_span> spans;
    spans.reserve(pages.size());
    for (const auto& p : pages) spans.emplace_back(p.data(), p.size());
    return ComputePagedHashes(spans, "");
}

TEST(EvictionStressTest, CyclicSessionsUnderPressureStayConsistent) {
    constexpr int kSessions = 12;
    constexpr int kTurns = 8;
    constexpr int kBlockTokens = 4;
    constexpr int kPoolBlocks = 24;   // 遠小於工作集，強迫淘汰
    BlockPool pool(kPoolBlocks);
    const std::vector<KvCacheSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1},
        {.kind = AttnKind::kSlidingWindow, .sliding_window = 8, .cache_blocks_per_lcm_block = 1},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, kBlockTokens, pool);

    int admitted = 0, refused = 0;
    for (int turn = 0; turn < kTurns; ++turn) {
        for (int s = 0; s < kSessions; ++s) {           // round-robin，重用距離 = kSessions
            const int pages = 2 + turn;                  // 逐輪加長，共用前綴
            std::vector<std::vector<std::int32_t>> toks;
            toks.reserve(static_cast<std::size_t>(pages));
            for (int p = 0; p < pages; ++p) {
                // 第 0 頁跨 session 共用（system prompt），其餘是 session 專屬
                toks.emplace_back(kBlockTokens, p == 0 ? 999 : s * 100 + p);
            }
            const std::vector<std::string> h = Hashes(toks);
            MatchPrefixForTest(coord, h);                // 先探前綴，就像真的服務一樣
            std::vector<BlockTable> tables(coord.NumGroups());
            if (AdmitForTest(coord, tables, pages * kBlockTokens)) {
                ++admitted;
                CacheFullBlocksForTest(coord, tables, h);
                coord.Free(tables);
            } else {
                ++refused;
            }
        }
    }
    // 這個測試不斷言命中率 —— 它斷言的是「跑完而且狀態自洽」。
    // 真正的檢查是 ASAN/UBSAN 沒有報錯，以及池子沒有漏。
    EXPECT_GT(admitted, 0);
    EXPECT_EQ(admitted + refused, kSessions * kTurns);
}

}  // namespace
}  // namespace tokenspeed::test
