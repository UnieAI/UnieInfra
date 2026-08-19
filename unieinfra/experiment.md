# Experiment
# Unieinfra 的格子重測於 2026-08-17，commit 176c8344 之後（准入不再替 checkpoint 收費、
# head-of-line 只關閉准入、找不到可撤回請求改為跳過）。vllm 的格子未重測。
#
# !! agent 127k 那 20 列的語料有結構問題，引用前必讀 !!
# agent_real_full.jsonl 只有 32 列而 NREQ=64，vllm bench 的 maybe_oversample_requests
# 會用 random.choice 複製列補齊（兩邊日誌都印 "Oversampled requests to reach 64 total
# samples"）。而且那 32 列**全體共享 106,016 字元的前綴**，是 prompt 中位數的 47%。
# 這一格獎勵的重用是語料結構造出來的，不是一般工作負載。
#
# 同格 NREQ=32（不觸發複製）的對照，2026-08-17：
#   我們 321.7 tok/s / goodput 1.26 ；vLLM 241.2 tok/s / goodput 0.94  ->  1.34x
# 矩陣裡 NREQ=64 那五列記的是 5.84x，差別幾乎全部來自 vLLM 在複製後的格子上崩掉
# （它的 goodput 0.94 -> 0.25，我們的 1.26 -> 1.46）。兩個讀數各自對它自己的操作點
# 成立，比較也一直是配對的（同 seed、同請求串、兩邊都被複製），所以那五列照原樣有效
# —— 但 **5.84x 不可以當成一般工作負載的數字引用**，要連同上面的 1.34x 一起講。
#
# 另外 --num-warmups 送的是 input_requests[0] 重複 N 次（vllm/benchmarks/serve.py:823
# 與 :891），不是從語料抽樣。對這份語料，那一條的前綴正是共享的 47%，所以連 NREQ=32
# 都不是冷快取；真正的冷對照需要 --num-warmups 0 或在暖機後 /reset_prefix_cache。
#
# agent short 與 RAG 兩份語料的全體共同前綴都是 0，不受這個問題影響。
#
# !! agent 語料的合成 token 是還原手段，不是缺陷 (2026-08-19) !!
# agent_real*.jsonl 的內容是 NATO 字母加流水號、每個重複八次（12.5% distinct）。
# 這一度讓我把整個 agent 欄位標成作廢 —— 那是錯的，已收回。
#
# 來源是 InferenceMAX 的 semianalysisai/cc-traces-weka，真實 Claude Code 軌跡，
# 在 /workspace/.cache/hub/ 上。它不存 token，存 block hash id：
#   {"block_size":64, "requests":[{"in":640,"out":20,"hash_ids":[0..9]},
#                                 {"in":63744,"out":395,"hash_ids":[10..]}, ...]}
# hash_ids 記的是「這是第幾個獨特的 64-token block」，原始資料沒有 token 內容
# （那是使用者私有內容）。還原成可送出的請求時必須造 token，只要「同 id → 同
# token」就能重現快取行為 —— 那些填充就是這個手段。
#
# 從 393 個真實 session 直接算：
#   block 重用率 全體 98.8%，各 session 中位數 96.9%、p10 91.5%、p90 99.1%
#   每 session 請求數 中位數 67；提示長度 中位數 245,248 tok、p90 645,824
# 我們在 agent-127k 實測 98.6%。**合成語料忠實還原了真實的重用結構。**
#
# 為什麼真實 agent 就是九成以上：system prompt 與工具定義每輪相同，而對話逐輪
# 累積 —— 第 N+1 輪包含第 N 輪的全部。在這種負載下前綴重用不是捷徑，它就是
# 工作負載本身。
#
# 仍然成立的保留（只有一個，而且是次要的）：token 內容退化會讓 K/V 向量比真實
# 文字「平」。長度、重用結構、請求數都是真的，FLOP 也是真的；受影響的是模型
# 內部的數值行為與輸出品質。實測輸出是連貫英文，decode 路徑正常 ——
#   agent-127k: "The user has provided a large block of text containing a long
#                list of alphanumeric strings... a phonetic alphabet word..."
#   RAG:        "Input text: A blog post titled '5 Great Podcasts on Love...'"
#
# !! goodput 是階梯函數，在 SLO 界線附近取樣不可靠 (2026-08-18) !!
# 同一組態連跑三次，goodput 是 0.46 / 0.56 / 0.31（±60%），而同樣三次的
# 吞吐只差 1.3%。不是機器吵，是指標本身：TTFT p50 在 5,073-5,947 之間游走
# 而界線是 5,000，於是一個二元的過/不過把 15% 的延遲抖動放大成 1.8 倍的
# goodput 擺盪。
#
# 判斷一格可不可信，看「合格率」而不是看 TTFT：
#   合格率 <=5% 或 >=95%  -> 穩定，抖動翻不了判定           63 格
#   合格率 15-85%          -> 敏感，單次結果不能當判定       30 格
# 兩者之間 7 格。
#
# 敏感帶裡帶著判定的 15 格已就地標上 <<UNSTABLE ...>>，全部需要 n=3 中位數。
# 其中 14 個是 WIN —— 這條規則質疑的主要是我們的勝場，不是敗場。
#
# 同一天踩到的例子：Qwen3.8-27B / RAG c=64，ours TTFT p50 5,342 對 vllm 5,110，
# 延遲只差 4.5%，goodput 卻是 0.786x。當時我寫成「c=64 開始輸」，已撤回。
#
# 之前 WIN 12 / PARITY 12 / LOSS 11 / 未測 85
# 現在 WIN 20 / PARITY 17 / LOSS  3 / 未測 80
# 剩下三格 LOSS 都是 35B 高併發：RAG c=64/128 與 General c=64。
[x] vllm c=1, agent 127k , qwen3-27b  -- 4.8/min, 22 tok/s, TTFT p50 479 ms, 合格 100%
[x] vllm c=16, agent 127k , qwen3-27b  -- 18.6/min, 100 tok/s, TTFT p50 2831 ms, 合格 79%
[x] vllm c=32, agent 127k , qwen3-27b  -- 8.4/min, 114 tok/s, TTFT p50 2396 ms, 合格 31%
[x] vllm c=64, agent 127k , qwen3-27b  -- 0.0/min, 124 tok/s, TTFT p50 34625 ms, 合格 0%
[x] vllm c=128, agent 127k , qwen3-27b  -- 0.0/min, 36 tok/s, TTFT p50 193926 ms, 合格 0%
[x] Unieinfra c=1, agent 127k , qwen3-27b  -- 5.4/min, 22 tok/s, TTFT p50 235 ms, 合格 100%  [WIN 1.13x (5.4 vs 4.8)]
[x] Unieinfra c=16, agent 127k , qwen3-27b  -- 27.0/min, 133 tok/s, TTFT p50 1219 ms, 合格 87%  [WIN 1.45x (27.0 vs 18.6)]
[x] Unieinfra c=32, agent 127k , qwen3-27b  -- 23.4/min, 155 tok/s, TTFT p50 2547 ms, 合格 65%  [WIN 2.79x (23.4 vs 8.4)]  <<需 n=3 合格率 65%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=64, agent 127k , qwen3-27b  -- 1.8/min, 86 tok/s, TTFT p50 88118 ms, 合格 9%  [WIN 1.8 vs 0]
[x] Unieinfra c=128, agent 127k , qwen3-27b  -- 2.4/min, 91 tok/s, TTFT p50 82612 ms, 合格 11%  [WIN 2.4 vs 0]
[x] vllm c=1, agent 127k , qwen3.6-35b-a3b  -- 22.8/min, 97 tok/s, TTFT p50 332 ms, 合格 100%
[x] vllm c=16, agent 127k , qwen3.6-35b-a3b  -- 15.0/min, 109 tok/s, TTFT p50 6006 ms, 合格 58%
[x] vllm c=32, agent 127k , qwen3.6-35b-a3b  -- 7.8/min, 91 tok/s, TTFT p50 44428 ms, 合格 37%
[x] vllm c=64, agent 127k , qwen3.6-35b-a3b  -- 0.0/min, 62 tok/s, TTFT p50 137200 ms, 合格 0%
[x] vllm c=128, agent 127k , qwen3.6-35b-a3b  -- 1.8/min, 91 tok/s, TTFT p50 80075 ms, 合格 8%
[x] Unieinfra c=1, agent 127k , qwen3.6-35b-a3b  -- 26.4/min, 112 tok/s, TTFT p50 256 ms, 合格 100%  [WIN 1.16x (26.4 vs 22.8)]
[x] Unieinfra c=16, agent 127k , qwen3.6-35b-a3b  -- 87.6/min, 374 tok/s, TTFT p50 2319 ms, 合格 100%  [WIN 5.84x (87.6 vs 15.0)]
[x] Unieinfra c=32, agent 127k , qwen3.6-35b-a3b  -- 39.6/min, 265 tok/s, TTFT p50 6117 ms, 合格 64%  [WIN 5.08x (39.6 vs 7.8)]  <<需 n=3 合格率 64%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=64, agent 127k , qwen3.6-35b-a3b  -- 20.4/min, 290 tok/s, TTFT p50 27021 ms, 合格 30%  [WIN 20.4 vs 0]  <<需 n=3 合格率 30%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=128, agent 127k , qwen3.6-35b-a3b  -- 16.2/min, 218 tok/s, TTFT p50 38110 ms, 合格 32%  [WIN 9.00x (16.2 vs 1.8)]  <<需 n=3 合格率 32%，goodput 在階梯邊緣，此判定需 n=3>>
[x] vllm c=1, agent 127k , qwen3.5-122b-a10b-fp8  -- 21.0/min, 89 tok/s, TTFT p50 487 ms, 合格 100%
[x] vllm c=16, agent 127k , qwen3.5-122b-a10b-fp8  -- 72.0/min, 307 tok/s, TTFT p50 2787 ms, 合格 100%
[x] vllm c=32, agent 127k , qwen3.5-122b-a10b-fp8  -- 64.8/min, 348 tok/s, TTFT p50 2397 ms, 合格 79%
[x] vllm c=64, agent 127k , qwen3.5-122b-a10b-fp8  -- 21.0/min, 380 tok/s, TTFT p50 15382 ms, 合格 23%
[x] vllm c=128, agent 127k , qwen3.5-122b-a10b-fp8  -- 0.0/min, 278 tok/s, TTFT p50 34163 ms, 合格 0%
[] Unieinfra c=1, agent 127k , qwen3.5-122b-a10b-fp8
[] Unieinfra c=16, agent 127k , qwen3.5-122b-a10b-fp8
[] Unieinfra c=32, agent 127k , qwen3.5-122b-a10b-fp8
[] Unieinfra c=64, agent 127k , qwen3.5-122b-a10b-fp8
[] Unieinfra c=128, agent 127k , qwen3.5-122b-a10b-fp8

[x] vllm c=1, agent short , qwen3-27b  -- 4.8/min, 21 tok/s, TTFT p50 1117 ms, 合格 100%
[x] vllm c=16, agent short , qwen3-27b  -- 5.4/min, 73 tok/s, TTFT p50 9518 ms, 合格 31%
[x] vllm c=32, agent short , qwen3-27b  -- 0.0/min, 69 tok/s, TTFT p50 39842 ms, 合格 0%
[x] vllm c=64, agent short , qwen3-27b  -- 0.0/min, 72 tok/s, TTFT p50 68868 ms, 合格 0%
[x] vllm c=128, agent short , qwen3-27b  -- 0.0/min, 68 tok/s, TTFT p50 94573 ms, 合格 0%
[x] Unieinfra c=1, agent short , qwen3-27b  -- 4.8/min, 22 tok/s, TTFT p50 1375 ms, 合格 100%  [PARITY 1.00x (4.8 vs 4.8) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=16, agent short , qwen3-27b  -- 5.4/min, 72 tok/s, TTFT p50 16386 ms, 合格 32%  [PARITY 1.00x (5.4 vs 5.4) <- 未達 +10%，仍算待優化]  <<需 n=3 合格率 32%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=32, agent short , qwen3-27b  -- 3.0/min, 70 tok/s, TTFT p50 29583 ms, 合格 19%  [WIN 3.0 vs 0]  <<需 n=3 合格率 19%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=64, agent short , qwen3-27b  -- 3.0/min, 74 tok/s, TTFT p50 53263 ms, 合格 17%  [WIN 3.0 vs 0]  <<需 n=3 合格率 17%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=128, agent short , qwen3-27b  -- 2.4/min, 64 tok/s, TTFT p50 57726 ms, 合格 16%  [WIN 2.4 vs 0]  <<需 n=3 合格率 16%，goodput 在階梯邊緣，此判定需 n=3>>
[x] vllm c=1, agent short , qwen3.6-35b-a3b  -- 25.2/min, 108 tok/s, TTFT p50 348 ms, 合格 100%
[x] vllm c=16, agent short , qwen3.6-35b-a3b  -- 57.6/min, 246 tok/s, TTFT p50 3134 ms, 合格 100%
[x] vllm c=32, agent short , qwen3.6-35b-a3b  -- 19.8/min, 242 tok/s, TTFT p50 17656 ms, 合格 35%
[x] vllm c=64, agent short , qwen3.6-35b-a3b  -- 18.0/min, 245 tok/s, TTFT p50 24849 ms, 合格 31%
[x] vllm c=128, agent short , qwen3.6-35b-a3b  -- 18.0/min, 257 tok/s, TTFT p50 23079 ms, 合格 30%
[x] Unieinfra c=1, agent short , qwen3.6-35b-a3b  -- 27.6/min, 117 tok/s, TTFT p50 457 ms, 合格 100%  [PARITY 1.10x (27.6 vs 25.2) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=16, agent short , qwen3.6-35b-a3b  -- 59.4/min, 280 tok/s, TTFT p50 2941 ms, 合格 91%  [PARITY 1.03x (59.4 vs 57.6) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=32, agent short , qwen3.6-35b-a3b  -- 34.8/min, 264 tok/s, TTFT p50 4523 ms, 合格 56%  [WIN 1.76x (34.8 vs 19.8)]  <<需 n=3 合格率 56%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=64, agent short , qwen3.6-35b-a3b  -- 22.8/min, 281 tok/s, TTFT p50 18586 ms, 合格 35%  [WIN 1.27x (22.8 vs 18.0)]  <<需 n=3 合格率 35%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=128, agent short , qwen3.6-35b-a3b  -- 20.4/min, 263 tok/s, TTFT p50 15176 ms, 合格 33%  [WIN 1.13x (20.4 vs 18.0)]  <<需 n=3 合格率 33%，goodput 在階梯邊緣，此判定需 n=3>>
[x] vllm c=1, agent short , qwen3.5-122b-a10b-fp8  -- 21.0/min, 90 tok/s, TTFT p50 458 ms, 合格 100%
[x] vllm c=16, agent short , qwen3.5-122b-a10b-fp8  -- 46.2/min, 225 tok/s, TTFT p50 3392 ms, 合格 88%
[x] vllm c=32, agent short , qwen3.5-122b-a10b-fp8  -- 34.8/min, 243 tok/s, TTFT p50 6738 ms, 合格 61%
[x] vllm c=64, agent short , qwen3.5-122b-a10b-fp8  -- 0.0/min, 261 tok/s, TTFT p50 27223 ms, 合格 0%
[x] vllm c=128, agent short , qwen3.5-122b-a10b-fp8  -- 1.2/min, 260 tok/s, TTFT p50 27106 ms, 合格 2%
[] Unieinfra c=1, agent short , qwen3.5-122b-a10b-fp8
[] Unieinfra c=16, agent short , qwen3.5-122b-a10b-fp8
[] Unieinfra c=32, agent short , qwen3.5-122b-a10b-fp8
[] Unieinfra c=64, agent short , qwen3.5-122b-a10b-fp8
[] Unieinfra c=128, agent short , qwen3.5-122b-a10b-fp8

[x] vllm c=1, RAG , qwen3-27b  -- 6.0/min, 27 tok/s, TTFT p50 122 ms, 合格 100%
[x] vllm c=16, RAG , qwen3-27b  -- 75.6/min, 322 tok/s, TTFT p50 1186 ms, 合格 100%
[x] vllm c=32, RAG , qwen3-27b  -- 124.2/min, 530 tok/s, TTFT p50 1869 ms, 合格 100%
[x] vllm c=64, RAG , qwen3-27b  -- 83.4/min, 783 tok/s, TTFT p50 5234 ms, 合格 45%
[x] vllm c=128, RAG , qwen3-27b  -- 120.6/min, 784 tok/s, TTFT p50 4897 ms, 合格 66%
[x] Unieinfra c=1, RAG , qwen3-27b  -- 6.6/min, 27 tok/s, TTFT p50 209 ms, 合格 100%  [PARITY 1.10x (6.6 vs 6.0) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=16, RAG , qwen3-27b  -- 76.8/min, 328 tok/s, TTFT p50 706 ms, 合格 100%  [PARITY 1.02x (76.8 vs 75.6) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=32, RAG , qwen3-27b  -- 126.0/min, 539 tok/s, TTFT p50 1496 ms, 合格 100%  [PARITY 1.01x (126.0 vs 124.2) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=64, RAG , qwen3-27b  -- 128.4/min, 797 tok/s, TTFT p50 4028 ms, 合格 69%  [WIN 1.54x (128.4 vs 83.4)]  <<需 n=3 合格率 69%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=128, RAG , qwen3-27b  -- 135.6/min, 821 tok/s, TTFT p50 3616 ms, 合格 70%  [WIN 1.12x (135.6 vs 120.6)]  <<需 n=3 合格率 70%，goodput 在階梯邊緣，此判定需 n=3>>
[x] vllm c=1, RAG , qwen3.6-35b-a3b  -- 33.6/min, 144 tok/s, TTFT p50 72 ms, 合格 100%
[x] vllm c=16, RAG , qwen3.6-35b-a3b  -- 181.8/min, 777 tok/s, TTFT p50 258 ms, 合格 100%
[x] vllm c=32, RAG , qwen3.6-35b-a3b  -- 269.4/min, 1150 tok/s, TTFT p50 380 ms, 合格 100%
[x] vllm c=64, RAG , qwen3.6-35b-a3b  -- 385.8/min, 1647 tok/s, TTFT p50 959 ms, 合格 100%
[x] vllm c=128, RAG , qwen3.6-35b-a3b  -- 384.0/min, 1639 tok/s, TTFT p50 730 ms, 合格 100%
[x] Unieinfra c=1, RAG , qwen3.6-35b-a3b  -- 36.0/min, 155 tok/s, TTFT p50 116 ms, 合格 100%  [PARITY 1.07x (36.0 vs 33.6) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=16, RAG , qwen3.6-35b-a3b  -- 189.6/min, 810 tok/s, TTFT p50 364 ms, 合格 100%  [PARITY 1.04x (189.6 vs 181.8) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=32, RAG , qwen3.6-35b-a3b  -- 269.4/min, 1149 tok/s, TTFT p50 401 ms, 合格 100%  [PARITY 1.00x (269.4 vs 269.4) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=64, RAG , qwen3.6-35b-a3b  -- 255.6/min, 1089 tok/s, TTFT p50 1048 ms, 合格 100%  [LOSS 0.66x (255.6 vs 385.8) <- 待優化]
[x] Unieinfra c=128, RAG , qwen3.6-35b-a3b  -- 276.6/min, 1181 tok/s, TTFT p50 1090 ms, 合格 100%  [LOSS 0.72x (276.6 vs 384.0) <- 待優化]
[x] vllm c=1, RAG , qwen3.5-122b-a10b-fp8  -- 27.6/min, 119 tok/s, TTFT p50 113 ms, 合格 100%
[x] vllm c=16, RAG , qwen3.5-122b-a10b-fp8  -- 139.2/min, 594 tok/s, TTFT p50 590 ms, 合格 100%
[x] vllm c=32, RAG , qwen3.5-122b-a10b-fp8  -- 186.6/min, 797 tok/s, TTFT p50 1177 ms, 合格 100%
[x] vllm c=64, RAG , qwen3.5-122b-a10b-fp8  -- 254.4/min, 1085 tok/s, TTFT p50 1916 ms, 合格 100%
[x] vllm c=128, RAG , qwen3.5-122b-a10b-fp8  -- 253.8/min, 1084 tok/s, TTFT p50 1918 ms, 合格 100%
[] Unieinfra c=1, RAG , qwen3.5-122b-a10b-fp8
[] Unieinfra c=16, RAG , qwen3.5-122b-a10b-fp8
[] Unieinfra c=32, RAG , qwen3.5-122b-a10b-fp8
[] Unieinfra c=64, RAG , qwen3.5-122b-a10b-fp8
[] Unieinfra c=128, RAG , qwen3.5-122b-a10b-fp8

[x] vllm c=1, General , qwen3-27b  -- 1.8/min, 27 tok/s, TTFT p50 222 ms, 合格 100%
[x] vllm c=16, General , qwen3-27b  -- 19.8/min, 343 tok/s, TTFT p50 2375 ms, 合格 100%
[x] vllm c=32, General , qwen3-27b  -- 34.2/min, 586 tok/s, TTFT p50 4074 ms, 合格 100%
[x] vllm c=64, General , qwen3-27b  -- 15.0/min, 912 tok/s, TTFT p50 6558 ms, 合格 28%
[x] vllm c=128, General , qwen3-27b  -- 15.0/min, 911 tok/s, TTFT p50 6599 ms, 合格 28%
[x] Unieinfra c=1, General , qwen3-27b  -- 1.8/min, 27 tok/s, TTFT p50 232 ms, 合格 100%  [PARITY 1.00x (1.8 vs 1.8) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=16, General , qwen3-27b  -- 20.4/min, 344 tok/s, TTFT p50 1716 ms, 合格 100%  [PARITY 1.03x (20.4 vs 19.8) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=32, General , qwen3-27b  -- 34.2/min, 586 tok/s, TTFT p50 2974 ms, 合格 100%  [PARITY 1.00x (34.2 vs 34.2) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=64, General , qwen3-27b  -- 24.6/min, 906 tok/s, TTFT p50 6047 ms, 合格 47%  [WIN 1.64x (24.6 vs 15.0)]  <<需 n=3 合格率 47%，goodput 在階梯邊緣，此判定需 n=3>>
[x] Unieinfra c=128, General , qwen3-27b  -- 21.0/min, 907 tok/s, TTFT p50 5425 ms, 合格 39%  [WIN 1.40x (21.0 vs 15.0)]  <<需 n=3 合格率 39%，goodput 在階梯邊緣，此判定需 n=3>>
[x] vllm c=1, General , qwen3.6-35b-a3b  -- 9.0/min, 150 tok/s, TTFT p50 78 ms, 合格 100%
[x] vllm c=16, General , qwen3.6-35b-a3b  -- 45.0/min, 773 tok/s, TTFT p50 421 ms, 合格 100%
[x] vllm c=32, General , qwen3.6-35b-a3b  -- 65.4/min, 1114 tok/s, TTFT p50 664 ms, 合格 100%
[x] vllm c=64, General , qwen3.6-35b-a3b  -- 92.4/min, 1577 tok/s, TTFT p50 1308 ms, 合格 100%
[x] vllm c=128, General , qwen3.6-35b-a3b  -- 93.6/min, 1595 tok/s, TTFT p50 1306 ms, 合格 100%
[x] Unieinfra c=1, General , qwen3.6-35b-a3b  -- 9.6/min, 162 tok/s, TTFT p50 84 ms, 合格 100%  [PARITY 1.07x (9.6 vs 9.0) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=16, General , qwen3.6-35b-a3b  -- 45.6/min, 780 tok/s, TTFT p50 372 ms, 合格 100%  [PARITY 1.01x (45.6 vs 45.0) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=32, General , qwen3.6-35b-a3b  -- 65.4/min, 1116 tok/s, TTFT p50 579 ms, 合格 100%  [PARITY 1.00x (65.4 vs 65.4) <- 未達 +10%，仍算待優化]
[x] Unieinfra c=64, General , qwen3.6-35b-a3b  -- 84.0/min, 1606 tok/s, TTFT p50 1082 ms, 合格 89%  [LOSS 0.91x (84.0 vs 92.4) <- 待優化]
[x] Unieinfra c=128, General , qwen3.6-35b-a3b  -- 94.2/min, 1604 tok/s, TTFT p50 1069 ms, 合格 100%  [PARITY 1.01x (94.2 vs 93.6) <- 未達 +10%，仍算待優化]
[x] vllm c=1, General , qwen3.5-122b-a10b-fp8  -- 7.2/min, 126 tok/s, TTFT p50 111 ms, 合格 100%
[x] vllm c=16, General , qwen3.5-122b-a10b-fp8  -- 38.4/min, 656 tok/s, TTFT p50 580 ms, 合格 100%
[x] vllm c=32, General , qwen3.5-122b-a10b-fp8  -- 52.8/min, 900 tok/s, TTFT p50 1086 ms, 合格 100%
[x] vllm c=64, General , qwen3.5-122b-a10b-fp8  -- 76.2/min, 1298 tok/s, TTFT p50 2155 ms, 合格 100%
[x] vllm c=128, General , qwen3.5-122b-a10b-fp8  -- 76.2/min, 1299 tok/s, TTFT p50 2170 ms, 合格 100%
[] Unieinfra c=1, General , qwen3.5-122b-a10b-fp8
[] Unieinfra c=16, General , qwen3.5-122b-a10b-fp8
[] Unieinfra c=32, General , qwen3.5-122b-a10b-fp8
[] Unieinfra c=64, General , qwen3.5-122b-a10b-fp8
[] Unieinfra c=128, General , qwen3.5-122b-a10b-fp8



# Experiment 2
[] vllm c=1, agent 127k , Gemma4-31B-IT
[] vllm c=16, agent 127k , Gemma4-31B-IT
[] vllm c=32, agent 127k , Gemma4-31B-IT
[] vllm c=64, agent 127k , Gemma4-31B-IT
[] vllm c=128, agent 127k , Gemma4-31B-IT
[] Unieinfra c=1, agent 127k , Gemma4-31B-IT
[] Unieinfra c=16, agent 127k , Gemma4-31B-IT
[] Unieinfra c=32, agent 127k , Gemma4-31B-IT
[] Unieinfra c=64, agent 127k , Gemma4-31B-IT
[] Unieinfra c=128, agent 127k , Gemma4-31B-IT
[] vllm c=1, agent 127k , Gemma4-26B-A4B-IT
[] vllm c=16, agent 127k , Gemma4-26B-A4B-IT
[] vllm c=32, agent 127k , Gemma4-26B-A4B-IT
[] vllm c=64, agent 127k , Gemma4-26B-A4B-IT
[] vllm c=128, agent 127k , Gemma4-26B-A4B-IT
[] Unieinfra c=1, agent 127k , Gemma4-26B-A4B-IT
[] Unieinfra c=16, agent 127k , Gemma4-26B-A4B-IT
[] Unieinfra c=32, agent 127k , Gemma4-26B-A4B-IT
[] Unieinfra c=64, agent 127k , Gemma4-26B-A4B-IT
[] Unieinfra c=128, agent 127k , Gemma4-26B-A4B-IT
[] vllm c=1, agent 127k , Gemma4-E4B-IT
[] vllm c=16, agent 127k , Gemma4-E4B-IT
[] vllm c=32, agent 127k , Gemma4-E4B-IT
[] vllm c=64, agent 127k , Gemma4-E4B-IT
[] vllm c=128, agent 127k , Gemma4-E4B-IT
[] Unieinfra c=1, agent 127k , Gemma4-E4B-IT
[] Unieinfra c=16, agent 127k , Gemma4-E4B-IT
[] Unieinfra c=32, agent 127k , Gemma4-E4B-IT
[] Unieinfra c=64, agent 127k , Gemma4-E4B-IT
[] Unieinfra c=128, agent 127k , Gemma4-E4B-IT

[] vllm c=1, agent short , Gemma4-31B-IT
[] vllm c=16, agent short , Gemma4-31B-IT
[] vllm c=32, agent short , Gemma4-31B-IT
[] vllm c=64, agent short , Gemma4-31B-IT
[] vllm c=128, agent short , Gemma4-31B-IT
[] Unieinfra c=1, agent short , Gemma4-31B-IT
[] Unieinfra c=16, agent short , Gemma4-31B-IT
[] Unieinfra c=32, agent short , Gemma4-31B-IT
[] Unieinfra c=64, agent short , Gemma4-31B-IT
[] Unieinfra c=128, agent short , Gemma4-31B-IT
[] vllm c=1, agent short , Gemma4-26B-A4B-IT
[] vllm c=16, agent short , Gemma4-26B-A4B-IT
[] vllm c=32, agent short , Gemma4-26B-A4B-IT
[] vllm c=64, agent short , Gemma4-26B-A4B-IT
[] vllm c=128, agent short , Gemma4-26B-A4B-IT
[] Unieinfra c=1, agent short , Gemma4-26B-A4B-IT
[] Unieinfra c=16, agent short , Gemma4-26B-A4B-IT
[] Unieinfra c=32, agent short , Gemma4-26B-A4B-IT
[] Unieinfra c=64, agent short , Gemma4-26B-A4B-IT
[] Unieinfra c=128, agent short , Gemma4-26B-A4B-IT
[] vllm c=1, agent short , Gemma4-E4B-IT
[] vllm c=16, agent short , Gemma4-E4B-IT
[] vllm c=32, agent short , Gemma4-E4B-IT
[] vllm c=64, agent short , Gemma4-E4B-IT
[] vllm c=128, agent short , Gemma4-E4B-IT
[] Unieinfra c=1, agent short , Gemma4-E4B-IT
[] Unieinfra c=16, agent short , Gemma4-E4B-IT
[] Unieinfra c=32, agent short , Gemma4-E4B-IT
[] Unieinfra c=64, agent short , Gemma4-E4B-IT
[] Unieinfra c=128, agent short , Gemma4-E4B-IT

[] vllm c=1, RAG , Gemma4-31B-IT
[] vllm c=16, RAG , Gemma4-31B-IT
[] vllm c=32, RAG , Gemma4-31B-IT
[] vllm c=64, RAG , Gemma4-31B-IT
[] vllm c=128, RAG , Gemma4-31B-IT
[] Unieinfra c=1, RAG , Gemma4-31B-IT
[] Unieinfra c=16, RAG , Gemma4-31B-IT
[] Unieinfra c=32, RAG , Gemma4-31B-IT
[] Unieinfra c=64, RAG , Gemma4-31B-IT
[] Unieinfra c=128, RAG , Gemma4-31B-IT
[] vllm c=1, RAG , Gemma4-26B-A4B-IT
[] vllm c=16, RAG , Gemma4-26B-A4B-IT
[] vllm c=32, RAG , Gemma4-26B-A4B-IT
[] vllm c=64, RAG , Gemma4-26B-A4B-IT
[] vllm c=128, RAG , Gemma4-26B-A4B-IT
[] Unieinfra c=1, RAG , Gemma4-26B-A4B-IT
[] Unieinfra c=16, RAG , Gemma4-26B-A4B-IT
[] Unieinfra c=32, RAG , Gemma4-26B-A4B-IT
[] Unieinfra c=64, RAG , Gemma4-26B-A4B-IT
[] Unieinfra c=128, RAG , Gemma4-26B-A4B-IT
[] vllm c=1, RAG , Gemma4-E4B-IT
[] vllm c=16, RAG , Gemma4-E4B-IT
[] vllm c=32, RAG , Gemma4-E4B-IT
[] vllm c=64, RAG , Gemma4-E4B-IT
[] vllm c=128, RAG , Gemma4-E4B-IT
[] Unieinfra c=1, RAG , Gemma4-E4B-IT
[] Unieinfra c=16, RAG , Gemma4-E4B-IT
[] Unieinfra c=32, RAG , Gemma4-E4B-IT
[] Unieinfra c=64, RAG , Gemma4-E4B-IT
[] Unieinfra c=128, RAG , Gemma4-E4B-IT

[] vllm c=1, General , Gemma4-31B-IT
[] vllm c=16, General , Gemma4-31B-IT
[] vllm c=32, General , Gemma4-31B-IT
[] vllm c=64, General , Gemma4-31B-IT
[] vllm c=128, General , Gemma4-31B-IT
[] Unieinfra c=1, General , Gemma4-31B-IT
[] Unieinfra c=16, General , Gemma4-31B-IT
[] Unieinfra c=32, General , Gemma4-31B-IT
[] Unieinfra c=64, General , Gemma4-31B-IT
[] Unieinfra c=128, General , Gemma4-31B-IT
[] vllm c=1, General , Gemma4-26B-A4B-IT
[] vllm c=16, General , Gemma4-26B-A4B-IT
[] vllm c=32, General , Gemma4-26B-A4B-IT
[] vllm c=64, General , Gemma4-26B-A4B-IT
[] vllm c=128, General , Gemma4-26B-A4B-IT
[] Unieinfra c=1, General , Gemma4-26B-A4B-IT
[] Unieinfra c=16, General , Gemma4-26B-A4B-IT
[] Unieinfra c=32, General , Gemma4-26B-A4B-IT
[] Unieinfra c=64, General , Gemma4-26B-A4B-IT
[] Unieinfra c=128, General , Gemma4-26B-A4B-IT
[] vllm c=1, General , Gemma4-E4B-IT
[] vllm c=16, General , Gemma4-E4B-IT
[] vllm c=32, General , Gemma4-E4B-IT
[] vllm c=64, General , Gemma4-E4B-IT
[] vllm c=128, General , Gemma4-E4B-IT
[] Unieinfra c=1, General , Gemma4-E4B-IT
[] Unieinfra c=16, General , Gemma4-E4B-IT
[] Unieinfra c=32, General , Gemma4-E4B-IT
[] Unieinfra c=64, General , Gemma4-E4B-IT
[] Unieinfra c=128, General , Gemma4-E4B-IT
