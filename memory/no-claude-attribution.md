---
name: no-claude-attribution
description: "Commits and PRs in the unieinfra repo must not mention Claude (no Co-Authored-By, Claude-Session, or \"Generated with Claude Code\" lines)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4ac055d4-cd27-41ff-96a7-d0aebbe01373
  modified: 2026-11-13T01:53:39.191Z
---

All commits and PR descriptions for unieinfra (github.com/unieai/unieinfra) must have no Claude attribution: drop `Co-Authored-By: Claude ...`, `Claude-Session: ...` and the "🤖 Generated with Claude Code" footer. Keep `Signed-off-by` (AGENTS.md requires sign-off).

**Why:** the user said "所有commit 都要把claude拿掉 pr內容也是" (Nov 13, 2026); this overrides the session's default attribution reminder.

**How to apply:** write commit messages without those lines from the start; when briefing subagents that commit, tell them the same. The first PR branch was rewritten (mi350p-dsv41-perf-clean) to strip them. Related: [[mi350p-benchmark-rules]].
