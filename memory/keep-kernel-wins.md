---
name: keep-kernel-wins
description: Keep (and default-enable) optimizations with verified micro-benchmark gains even when end-to-end A/B is within noise
metadata:
  type: feedback
---

When a change has a verified isolated benefit (e.g. kernel −8% per layer) and is correct, keep it and enable by default even if the end-to-end A/B can't resolve it from noise. Reject only when it's measurably worse.

**Why:** User (2026-11-15): "我建議保留有好處的東西 就算被其他蓋掉" after the prefill MoE rewrite A/B came back ±4%.

**How to apply:** Decide by kernel/microbench evidence + correctness; e2e A/B only guards against regressions. Related: [[mi350p-benchmark-rules]].
