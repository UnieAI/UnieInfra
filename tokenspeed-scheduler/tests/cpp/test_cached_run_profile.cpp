// KvCacheManager::AnalyseIndexRuns —— D1(擱淺的 lookback run)的算術。
// 編法(容器內無 git,不能用 gtest):
//   g++ -std=c++20 -Icsrc -I. -I<spdlog-include> tests/cpp/test_cached_run_profile.cpp -o /tmp/t_runs && /tmp/t_runs
#include <cassert>
#include <cstdio>
#include <vector>

#include "cache/manager/kv_cache_manager.h"

using tokenspeed::KvCacheManager;

int main() {
    auto run = [](std::vector<std::int32_t> v, std::int32_t lb) {
        return KvCacheManager::AnalyseIndexRuns(v, lb);
    };
    int n = 0;
    // 完整的 16 頁 run,不擱淺
    { auto [r, s] = run({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, 16); assert(r==1 && s==0); ++n; }
    // 同樣 16 頁但從 40 起:長度剛好夠,不擱淺
    { std::vector<std::int32_t> v; for (int i=40;i<56;++i) v.push_back(i);
      auto [r, s] = run(v, 16); assert(r==1 && s==0); ++n; }
    // 從 40 起但被挖掉一頁 -> 兩段 (40..47, 49..55),各 8 和 7,都 <16 且不觸底 -> 全部擱淺
    { std::vector<std::int32_t> v; for (int i=40;i<56;++i) if (i!=48) v.push_back(i);
      auto [r, s] = run(v, 16); assert(r==2 && s==15); ++n; }
    // 觸底規則:0..7 只有 8 頁 <16,但從 0 起 -> 不擱淺
    { auto [r, s] = run({0,1,2,3,4,5,6,7}, 16); assert(r==1 && s==0); ++n; }
    // 觸底 run 與孤兒 run 並存
    { std::vector<std::int32_t> v{0,1,2,  90,91,92};
      auto [r, s] = run(v, 16); assert(r==2 && s==3); ++n; }
    // 空集合
    { auto [r, s] = run({}, 16); assert(r==0 && s==0); ++n; }
    // lookback 0 或負數退化成 1:單頁 run 也算完整
    { auto [r, s] = run({5}, 0); assert(r==1 && s==0); ++n; }
    // 單頁孤兒,lookback 2
    { auto [r, s] = run({5}, 2); assert(r==1 && s==1); ++n; }
    std::printf("AnalyseIndexRuns: %d cases PASS\n", n);
    return 0;
}
