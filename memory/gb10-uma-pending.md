---
name: gb10-uma-pending
description: GB10 UMA support (L3 phase 2) is blocked until the user provides a GB10 machine
metadata:
  node_type: memory
  type: project
  originSessionId: 4ac055d4-cd27-41ff-96a7-d0aebbe01373
  modified: 2026-11-18T06:16:05.871Z
---

GB10 UMA mode for UnieInfra KV tiering is on hold: on 2026-11-18 the user said there is no GB10 machine yet ("現在沒有機器 之後再用").
Phase 1, the local-disk L3 backend, is PR #49 (UnieAI/UnieInfra, branch roy/l3-file-backend), not merged yet.

**Why:** The UMA work needs real hardware: integrated GPU detection, skipping or shrinking the L2 host pool, SSD reads into mapped-host buffers, and a KV budget that accounts for the shared memory.
**How to apply:** Don't start GB10 work until the user provides access. Possible next step: an end-to-end test of the `file` L3 backend on the current machine.
