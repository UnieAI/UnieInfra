---
name: mi350p-benchmark-rules
description: "How the user wants the MI350P UnieInfra benchmarking/optimization work run: optimize only UnieInfra, no data gaps, log every iteration, and which comparisons are valid"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 550416c1-af79-45ad-94e9-d75d9ed2d978
  modified: 2026-11-09T00:52:12.673Z
---

On the 4x MI350P DeepSeek-V4.1 work (UnieInfra = the TokenSpeed fork in /home/smc/workspace/roy/unieinfra):

- Only optimize UnieInfra/TokenSpeed. vLLM and SGLang are baselines: change only their launch config so they start on MI350P, never their code. The user also clarified that "port from CUDA" means reading the repo's CUDA kernel/dataflow designs and porting them to ROCm, not tuning vLLM.
- No missing data: every engine needs results at c1, c4, c16, c32, c64, c128 before moving on. A failed cell is fixed and rerun, not left blank.
- Record every iteration (each experiment, including rejected ones) in /home/smc/workspace/roy/experiments.json (id, date, area, experiment, result, decision, commit) and push it to the dashboard https://claude.ai/artifact/L5bkHnHRbzb2thttkZhHHg (dataset "log"). Another Claude session writes to the same file and GPUs: re-read before writing, never overwrite.
- Dashboard results come from scripts/export_dashboard.py (dataset "agentx"); refresh it when new cells finish.

**Comparisons that are valid (learned the hard way, Nov 2026):**
- AgentX requests carry temperature 1.0. On AMD the default `--sampling-backend` is `greedy`, which ignores it, so greedy results are NOT comparable with vLLM (which samples). Use `--sampling-backend triton_full` (plus TOKENSPEED_DSPARK_COUPLED_DRAFT=1) for any "vs vLLM" claim.
- The official InferenceX AgentX compares engines at matched interactivity (1000 / p90 inter-token latency), not at peak throughput; concurrency means live agent clients. vLLM's recipe caps running requests at CONC/4 (high interactivity, low throughput, long queueing). Compare on a throughput-vs-interactivity curve; "best vLLM RPM per cell" is only a throughput bound at lower interactivity.
- `--ignore-trace-delays` (STRESS=1) and 300 s runs are unofficial smoke tests (official needs recorded gaps and >= 900 s).
- Compare runs only on the same build and the same prefill chunk size; measure the hit rate on the profiling phase only (warmup requests are cold).
- The earlier H200 "2x" (dashboard "DSv4.1 H200 AgentX Results") used a different workload: 256 short requests (~9.7K tokens, exactly 128 output tokens), cold prefix cache per cell, TP4+EP4 on NVLink; it is not the InferenceX AgentX corpus (~130K-token contexts).

**Why:** the user said "你不要去優化sglang vllm", "我不希望有缺數據的狀況 快跑完", "每次迭代都要有記錄", and asked for charts like their H200 dashboard (bars by concurrency, throughput vs TTFT) plus the InferenceX per-GPU vs interactivity chart. The comparison rules above come from mistakes found during the work (greedy vs sampling, baseline mismatch, contaminated cells).

**How to apply:** before starting optimization work, check the result matrix for gaps; after every experiment append a log row and update the dashboard; before quoting a ratio vs vLLM, check sampling mode, interactivity point, build and chunk size.
