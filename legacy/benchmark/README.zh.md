<!-- 中文版。英文正本：README.md（兩者需同步）。 -->
# UnieInfra Benchmark(基準測試)

自動化、可複現的 serving 基準測試,用 `vllm bench serve`(InferenceMAX 標準 client)
在**併發掃描(1, 16, 32, 64, 128, 256)**下量測 **TTFT、TPOT、throughput、RPS**,
可打任何 OpenAI 相容引擎(UnieInfra、vLLM、SGLang、LMDeploy)做對等比較。
**所有 raw 結果都會保存。**

## 目錄
```
benchmark/
├── run_sweep.sh          # harness:併發掃描 -> raw + summary
├── summarize.py          # raw JSON -> summary.csv(保存每一個 scalar 指標)
├── bench_and_profile.sh  # 一鍵迴圈:launch+sweep -> profile -> trajectory -> delta
├── trajectory.py         # results/**/summary.csv -> results/TRAJECTORY.md(+.csv)
├── recut_aggregate.py    # N 次重複 -> 中位數 + min/max 區間 + WIN/TIE/LOSE 判定
├── profile_hot_decode.py # 輕量 steady-state decode profile(torch.profiler / nsys)
├── configs/               # bench_and_profile.sh 用的 <label>.env 參數(見 example.env)
├── launch/               # 各引擎的 OpenAI server 啟動器
│   ├── launch_unieinfra.sh
│   └── launch_vllm.sh
├── scripts/              # offline 冒煙/吞吐測試 + sm_120 環境
│   ├── sm120_env.sh
│   ├── m1_forward_smoke.py       # 正確性(單請求)
│   ├── m1_concurrency_smoke.py   # offline 聚合吞吐
│   └── bench_vllm.py             # offline vLLM 對照
└── results/
    ├── TRAJECTORY.md / trajectory.csv   # trajectory.py 產生 —— 不要手動編輯
    └── <engine>/<label-或-timestamp>/
        ├── env.txt       # GPU、版本、掃描參數
        ├── launch.json   # 卡別/數量/serve 參數,複製自 launch/<engine>.launch.json
        ├── raw/
        │   ├── c<N>.json         # 每個併發的完整 vllm-bench 結果
        │   └── c<N>.stdout.log   # client 完整 stdout
        └── summary.csv   # 每個併發一列,所有指標
```

## 跑一次掃描
1. **起 server**(用空閒 GPU):
   ```bash
   MODEL=/workspace/models/Qwen/Qwen3-8B GPU=1 PORT=8000 benchmark/launch/launch_unieinfra.sh &
   # 或:MODEL=/workspace/models/Qwen/Qwen3-8B GPU=1 PORT=8001 benchmark/launch/launch_vllm.sh &
   ```
2. **掃描**(client 需要一個裝了 `vllm` 的 python 來跑 `vllm bench`):
   ```bash
   ENGINE=unieinfra BASE_URL=http://127.0.0.1:8000 MODEL=bench-model \
   VLLM_PY=/home/ubuntu/vllm-bench-venv/bin/python \
   benchmark/run_sweep.sh
   ```
3. 比較各引擎的 `results/*/*/summary.csv`。

## 可調參數(環境變數)
`CONCURRENCIES`(預設 `1 16 32 64 128 256`)、`ILEN`/`OLEN`(random 輸入/輸出長度)、
`PROMPTS_PER_CONC`(num_prompts = 併發 × 此值)、`ENDPOINT`(`/v1/completions`)。
raw 永不覆蓋 —— 每次 run 有自己的 timestamp 目錄。

## 指標對照
`request_throughput` = **RPS**(req/s)· `output_throughput` = decode **tok/s** ·
`mean/p99_ttft_ms` = **TTFT** · `mean/p99_tpot_ms` = **TPOT**(每輸出 token 延遲)·
`mean/p99_e2el_ms` = 端到端。summary CSV 保存 client 回報的**每一個** scalar;
逐請求陣列在 `raw/`。

## 持續迴圈:改動 → 跑一次 → profile 一次 → 知道下一步要修什麼

一個指令就能跑完一個 label 的完整循環:

```bash
benchmark/bench_and_profile.sh <label> [engine]     # engine: unieinfra(預設)| vllm
```

它會 (a) 啟動引擎並跑對齊過的掃描到 `results/<engine>/<label>/`(沿用
`run_sweep.sh` 的衛生規則:固定 `PROMPTS_PER_CONC`/`WARMUPS`/`SEED`、每個併發
層級間 `FLUSH=1`、prefix cache 保持開啟)、(b) 對 batch-1 與 batch-128 的
steady-state decode 跑一次輕量 profile 到 `profiles/<label>/`、(c) 用磁碟上
*所有*已保存的 run 重新產生 `results/TRAJECTORY.md`、(d) 印出這個 label
相對於「前一個 label」與「固定 baseline」(預設 `clean`)的差異,若任何併發
層級變差就標記 regression。

**1. 為要測的改動加一個 config。** 把環境變數參數放進
`configs/<label>.env`(複製 `configs/example.env`,裡面列了它認得的每個
參數:`MODEL`/`GPU`/`PORT`;`launch_unieinfra.sh` 的參數
`DTYPE`/`ATTN`/`MAX_NUM_SEQS`/`MAX_CGRAPH`/`DISABLE_RADIX`;`run_sweep.sh`
的衛生參數;以及 wrapper 自己的 `BASELINE`/`PROFILE`/`CONTAINER`)。
`bench_and_profile.sh` 只是把環境變數傳遞下去 —— 不會寫死 launch 腳本的參數
清單 —— 所以未來 `launch_unieinfra.sh` 新增的任何參數,只要那邊接上了,
這裡就能直接用。

**2. 跑它。**
```bash
benchmark/bench_and_profile.sh my_change
```

**3. 讀 `results/TRAJECTORY.md`。** 三張表,從磁碟上每一個
`results/**/summary.csv` 重新產生(參差不齊的 run —— 例如只掃過部分併發的
label —— 缺的欄位顯示 `-`,絕不會 crash):
1. **Trajectory** —— 每個 run × 每個併發,整個改動歷史的贏/輸一眼看穿。
2. **Delta vs baseline** —— 同樣的表,改成 % 差異,省去心算。
3. **UnieInfra vs vLLM vs SGLang** —— 每個併發並排比較並標記贏家,因為目標是
   「贏過它們」,不是「贏過昨天」(見 `benchmark/results/vllm/clean/` 與
   `.../sglang/clean/`)。

用 `benchmark/trajectory.py --metric mean_ttft_ms` 或
`--metric mean_tpot_ms` 切換指標(預設 `output_throughput`);其餘用
`--help` 查。

**4. 讀 profile 的提示。** `profiles/<label>/bs1_hotkernels.md` 與
`bs128_hotkernels.md` 各有一張 top-N 熱點 kernel 表、一個分類佔比、以及一句
「likely lever」啟發式提示(例如「gemv 主導 → DRAM 頻寬瓶頸 → 試試權重量化
或加大 batch」)。這是 kernel 時間佔比的啟發式判斷,不是硬體計數器
roofline —— 在 HW counters 解鎖後,花真正的優化力氣前請先用 `ncu` 確認
(這個環境目前被 `RmProfilingAdminOnly` 擋住,所以 profiler 預設走 server 自帶的
`/start_profile`/`/stop_profile` 搭配 `torch.profiler`,另外也提供 `nsys`
timeline 當作另一個永遠可用的備援 —— 見 `profile_hot_decode.py --help`)。

**不碰 GPU 也能驗證。**
```bash
benchmark/bench_and_profile.sh <任一已存在的 label> --dry-run   # 只跑 (c)+(d),用既有資料
benchmark/trajectory.py                                          # 只看 trajectory 視圖
```

### 實例(worked example)

`benchmark/results/TRAJECTORY.md`,由 `results/` 下已保存的 ~15 個 run 產生
(指標 = `output_throughput`,baseline = `unieinfra/clean`):

```markdown
Metric: `output_throughput` (higher is better). Baseline for deltas: `unieinfra/clean`.

## 1. Trajectory (all runs, all engines)

| run | C1 | C16 | C32 | C64 | C128 | C256 |
|---|---|---|---|---|---|---|
| unieinfra/clean | 77.4358 | 973.013 | 1551.86 | 2263.13 | 3026.91 | 2737.17 |
| vllm/clean | 82.6051 | 978.579 | 1491.56 | 2090.72 | 2577.34 | 2687.31 |
| sglang/clean | 82.7106 | 954.889 | 1480.86 | 2066.95 | 2553.91 | 2849.49 |
| unieinfra/fp8w_fixed | 109.659 | - | 2085.07 | - | 3519.37 | - |
| unieinfra/before_baseline | - | - | - | 1900.25 | 2433.96 | 2518.55 |

## 3. UnieInfra vs vLLM vs SGLang

| concurrency | unieinfra/clean | vllm/clean | sglang/clean | winner |
|---|---|---|---|---|
| C1 | 77.4358 | 82.6051 | 82.7106 | sglang beats unieinfra |
| C32 | 1551.86 | 1491.56 | 1480.86 | unieinfra WINS |
| C64 | 2263.13 | 2090.72 | 2066.95 | unieinfra WINS |
| C128 | 3026.91 | 2577.34 | 2553.91 | unieinfra WINS |
| C256 | 2737.17 | 2687.31 | 2849.49 | sglang beats unieinfra |
```

(以上為節錄,完整檔案有每個併發、每個 run;隨時用
`benchmark/trajectory.py` 重新產生。)解讀:`cmp_fa2` 是真的全部失敗
(全部是 0 —— FlashAttention-2 backend 真實的錯誤,不是 trajectory 工具的
bug);`before_baseline`/`before_triton` 在較高併發下比 `clean` 差
8–20%(這正是 `bench_and_profile.sh` 步驟 (d) 的 regression 標記要抓的東西);
`fp8w_fixed` 目前是 UnieInfra 低/中併發最好的結果(比 `clean` 快
16–42%),但只有 C=1/32/128 的資料(參差不齊 —— 那次掃描沒跑 C16/64/256);
UnieInfra 目前在中段(C32–C128)贏過 vLLM 和 SGLang,但在兩端(C1 的延遲
下限、C256 的飽和)輸 —— 這個差距正是下一次 `bench_and_profile.sh <label>`
該去追的目標,符合「贏過 vLLM/SGLang,不只是打平」的產品論點。

## 重複測量:差距有大過雜訊嗎?

每個引擎只跑一次,分不出真正的勝負與機器漂移。當一個數字要用來**做決策**
時,同一組掃描每個引擎要跑 **N>=3 次,而且引擎之間輪流跑**(unieinfra →
vllm → unieinfra → vllm,不要先把一個引擎跑完再跑另一個),讓緩慢漂移對每
個引擎的影響一致;run 的 label 用 `<prefix>_<engine>_r1|r2|r3`。然後:

```bash
benchmark/recut_aggregate.py recut_a_bf16
```

它會對每個指標,列出每個(引擎, 併發)的 `中位數 [min-max]`,並給出
**WIN / TIE / LOSE** 判定 —— 其中 **TIE 表示兩個引擎的觀測區間重疊**,也就
是這個差距無法與 run 之間的雜訊區分。**差距小於區間就不算結果。**

實例:`benchmark/results/RECUT_A_BF16_AGGREGATE.md`(E35 —— Qwen3-8B bf16,
UnieInfra vs vLLM,在沒有其他負載的 GPU 上跑 N=3;區間小於 0.2%,所以報告
中的每個差距都是真的)。

## 衛生規則(hygiene rules)

這些由 `run_sweep.sh` 強制執行(所以包住它的 `bench_and_profile.sh` 也一樣)
—— 加新 config 時不要繞過:
- **固定 `PROMPTS_PER_CONC`/`WARMUPS`/`SEED`**:要比較的 run 之間 —— 每次
  run 都是同一組 prompt;比較時中途不要改這些值。
- **`FLUSH=1`**:每個併發層級之間 flush,避免上一層的 cache 狀態滲進下一層。
- **Prefix cache 保持開啟。** 這是真實的 production 功能;層級間的隔離靠
  flush,不是靠關掉它。
- **同一台機器、背靠背跑。** 不要拿上週在別台機器跑的結果跟今天的比。
- **best-vs-best。** 不要故意讓 vLLM/SGLang 跑得比較差來讓 UnieInfra 好看
  —— 只調 UnieInfra 自己的 config(靠強預設值,不是針對這個 benchmark
  手調)。
