# 回覆規則（使用者指定）

- **所有回覆一律用繁體中文**，包括進度回報、表格、說明。程式碼、指令、檔名維持原文。
- **精簡，不浪費 token**：只講結果、數字、下一步；不重複前文、不寫長篇解釋、不列不做的選項。
- 只優化 UnieInfra（unieinfra/），vLLM/SGLang 只調到能啟動。
- 每次迭代都記錄到 experiments.json 與 dashboard（https://claude.ai/artifact/L5bkHnHRbzb2thttkZhHHg）。
- 目標：接近 2× vLLM；c1–c128 不可缺數據。
