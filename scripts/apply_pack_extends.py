#!/usr/bin/env python3
"""Add the TOKENSPEED_PACK_FINISHING_EXTENDS lever to the scheduler, or check it.

Why (see experiments 140/141): every closing extend of a request costs ~1.2 s
of prefill round while a full 4096-token chunk costs 152 ms. The scheduler is
the reason there is exactly one extend per round: scheduleLocalPrefillWork
(forward.cpp) returns as soon as the request it just scheduled is still
Prefilling (holdsHeadOfLine), so no second extend is ever admitted -- and
PrefillSize() being the whole prompt means scheduleShortPromptsAhead can never
pack one either. The batch machinery below it already supports several extends
per forward (that is what the short-prompt pass builds, and PlanBuild caps
itself on token_budget and max_batch_size), so the fix is to let the pass
continue when the flag is on and the budget still has room.

Usage:
    apply_pack_extends.py --check     # report whether each edit would apply
    apply_pack_extends.py --apply     # back the three files up to /tmp, then edit
    apply_pack_extends.py --restore   # put the newest backup back

Rebuild after --apply, inside the dev container:

    docker exec -u root ts-dev bash -c \
      'cd /workspace && python -m pip install -e tokenspeed-scheduler/ --no-deps --force-reinstall'

then serve with TOKENSPEED_PACK_FINISHING_EXTENDS=1 (serve_unieinfra.sh already
forwards the whole TOKENSPEED_* allow-list -- check that name is on it) and
restore with --restore when the cell is done.

forward.cpp is dirty in the checkout (someone else's uncommitted work), so this
never reverts anything: it edits the current text and --restore puts back the
byte-identical copy it saved.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/smc/workspace/roy/unieinfra")
SCHED = ROOT / "tokenspeed-scheduler/csrc/scheduler"
HEADER = SCHED / "scheduler.h"
CTOR = SCHED / "scheduler.cpp"
FORWARD = SCHED / "operations/forward.cpp"

# (path, anchor, replacement, label) -- each anchor must occur exactly once.
H_ANCHOR = """    std::int32_t short_prompt_tokens_{0};
    std::int32_t long_prefill_floor_{0};
"""
H_NEW = """    std::int32_t short_prompt_tokens_{0};
    std::int32_t long_prefill_floor_{0};
    // TOKENSPEED_PACK_FINISHING_EXTENDS: keep admitting prefill work after a
    // chunk left its request still Prefilling. That request used to end the
    // pass (holdsHeadOfLine), which is why a round ever carried one extend;
    // with the tokens left over from its chunk a second extend joins the same
    // forward, and the round's fixed cost is paid once for two requests.
    // Bounded by token_budget and max_batch_size, so 0 (default) is the old
    // behaviour and 1 packs whatever fits.
    bool pack_finishing_extends_{false};
"""

C_ANCHOR = """        long_prefill_floor_ = std::max(0, env_int("TOKENSPEED_LONG_PREFILL_FLOOR", mixed_ref_chunk_ / 4));
"""
C_NEW = """        long_prefill_floor_ = std::max(0, env_int("TOKENSPEED_LONG_PREFILL_FLOOR", mixed_ref_chunk_ / 4));
        pack_finishing_extends_ = env_int("TOKENSPEED_PACK_FINISHING_EXTENDS", 0) != 0;
"""

F_ANCHOR = """                pushOperation(build, *request, std::move(*admitted->operation));
                request->TrackScheduledForward();
                if (holdsHeadOfLine(*request)) {
                    return;
                }
"""
F_NEW = """                pushOperation(build, *request, std::move(*admitted->operation));
                request->TrackScheduledForward();
                if (holdsHeadOfLine(*request) && !pack_finishing_extends_) {
                    return;
                }
"""

EDITS = [(HEADER, H_ANCHOR, H_NEW, "scheduler.h member"), (CTOR, C_ANCHOR, C_NEW, "scheduler.cpp env"),
         (FORWARD, F_ANCHOR, F_NEW, "forward.cpp holdsHeadOfLine gate")]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        newest = max(Path("/tmp").glob("pack_extends_*"), key=lambda p: p.stat().st_mtime, default=None)
        if newest is None:
            print("no backup under /tmp/pack_extends_*", file=sys.stderr)
            return 1
        for path, _, _, label in EDITS:
            saved = newest / path.name
            if not saved.exists():
                print(f"backup lacks {path}", file=sys.stderr)
                return 1
            shutil.copy2(saved, path)
            print(f"restored {label} <- {saved}")
        return 0

    ok = True
    for path, anchor, new, label in EDITS:
        text = path.read_text()
        n = text.count(anchor)
        applied = new in text
        state = "already applied" if applied else ("1 match" if n == 1 else f"{n} matches -- FIX ANCHOR")
        print(f"{label}: {state}")
        ok = ok and (applied or n == 1)
    if args.check:
        return 0 if ok else 1
    if not ok:
        print("refusing to apply with a bad anchor", file=sys.stderr)
        return 1

    backup = Path(f"/tmp/pack_extends_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    backup.mkdir(parents=True, exist_ok=True)
    for path, anchor, new, label in EDITS:
        shutil.copy2(path, backup / path.name)
        text = path.read_text()
        if new not in text:
            if text.count(anchor) != 1:
                print(f"{label}: anchor drifted after backup -- restoring", file=sys.stderr)
                for p2, _, _, _ in EDITS:
                    if (backup / p2.name).exists():
                        shutil.copy2(backup / p2.name, p2)
                return 1
            path.write_text(text.replace(anchor, new))
        print(f"applied {label}")
    print(f"backup in {backup}; rebuild with pip install -e tokenspeed-scheduler/ --no-deps --force-reinstall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
