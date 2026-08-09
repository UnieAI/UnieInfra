#!/usr/bin/env python3
"""Verify every `path/file.ext:LINE symbol` citation in the xref documents.

A citation is only useful if the line still says what the table claims. This
script resolves each citation against the pinned checkouts listed in
00-METHOD.md and checks that the cited line (or a small window around it)
actually contains the cited symbol.

    python3 unieinfra/xref/check_xref.py            # check every 10-..90- doc
    python3 unieinfra/xref/check_xref.py 10-api.md  # check one

Exit code 0 = every citation resolved and matched.

Why a window rather than an exact line: upstream inserts a line above a
function far more often than it changes the function, and a citation that is
two lines stale is still a correct pointer. A citation whose symbol is gone
from the window is not, and that is the failure this catches.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIEINFRA = HERE.parent.parent  # /workspace/roy/UnieInfra
THIRD_PARTY = UNIEINFRA / "legacy" / "third_party"

# Citation roots, tried in order. The first root under which the file exists
# AND the symbol matches wins, so two engines sharing a relative path
# (lmdeploy `serve/openai/...` vs TRT-LLM `serve/openai_server.py`) still
# resolve unambiguously.
ROOTS: list[tuple[str, Path]] = [
    ("tokenspeed", UNIEINFRA / "python" / "tokenspeed"),
    ("tokenspeed-scheduler", UNIEINFRA / "tokenspeed-scheduler"),
    ("tokenspeed-kernel", UNIEINFRA / "tokenspeed-kernel" / "python"),
    (
        "tokenspeed-kernel-pkg",
        UNIEINFRA / "tokenspeed-kernel" / "python" / "tokenspeed_kernel",
    ),
    ("unieinfra", UNIEINFRA),
    ("vllm", Path("/workspace/roy/vllm/vllm")),
    ("vllm-root", Path("/workspace/roy/vllm")),
    ("sglang", THIRD_PARTY / "sglang" / "python" / "sglang"),
    ("trtllm", THIRD_PARTY / "TensorRT-LLM" / "tensorrt_llm"),
    ("trtllm-root", THIRD_PARTY / "TensorRT-LLM"),
    ("lmdeploy", THIRD_PARTY / "lmdeploy" / "lmdeploy"),
    ("lmdeploy-root", THIRD_PARTY / "lmdeploy"),
    # The gRPC servicer ships as a wheel, not as repo source, and it is
    # installed in a different place on the host than inside the dev container
    # -- list both so this script verifies the same 316 citations either way.
    ("smg-servicer", Path("/home/ubuntu/sglang-venv/lib/python3.12/site-packages")),
    ("smg-servicer-container", Path("/usr/local/lib/python3.12/dist-packages")),
]

# `some/path/file.py:1234 symbol` inside a backtick span. The symbol is
# optional; without one we only check that the file and line exist.
CITE = re.compile(
    r"`([A-Za-z0-9_./+-]+\.(?:py|cc|cpp|cu|cuh|h|hpp|rs|toml|md)):(\d+)"
    r"(?:\s+([A-Za-z_][A-Za-z0-9_:.<>~]*))?"
)

WINDOW = 6  # lines either side of the cited line to accept the symbol on


def symbol_needle(symbol: str) -> str:
    """The part of a symbol that must appear literally in the source.

    `Scheduler::NextExecutionPlan` is written with the class in C++ but a
    citation may point at the definition line where only the bare name and the
    class both appear -- keep the last component, which is what a rename
    actually changes.
    """
    if "::" in symbol:
        return symbol.split("::")[-1]
    if "." in symbol:
        return symbol.split(".")[-1]
    return symbol


def resolve(path: str, line: int, symbol: str | None) -> tuple[bool, str]:
    """Return (ok, detail). Tries every root; symbol match breaks ties."""
    candidates: list[tuple[str, Path]] = []
    for name, root in ROOTS:
        full = root / path
        if full.is_file():
            candidates.append((name, full))

    if not candidates:
        return False, "no root contains this path"

    misses = []
    for name, full in candidates:
        try:
            lines = full.read_text(errors="replace").splitlines()
        except OSError as exc:  # unreadable is a failure, not a skip
            misses.append(f"{name}: {exc}")
            continue
        if line > len(lines):
            misses.append(f"{name}: file has {len(lines)} lines")
            continue
        if symbol is None:
            return True, f"{name} (line exists, no symbol asserted)"
        lo = max(0, line - 1 - WINDOW)
        hi = min(len(lines), line + WINDOW)
        needle = symbol_needle(symbol)
        for idx in range(lo, hi):
            if needle in lines[idx]:
                off = idx + 1 - line
                where = "exact" if off == 0 else f"{off:+d}"
                return True, f"{name} ({where})"
        misses.append(f"{name}: {symbol!r} not within ±{WINDOW}")

    return False, "; ".join(misses)


def check(doc: Path) -> tuple[int, int, list[str]]:
    text = doc.read_text()
    total = 0
    ok = 0
    failures: list[str] = []
    for docline_no, docline in enumerate(text.splitlines(), 1):
        for m in CITE.finditer(docline):
            path, line_s, symbol = m.group(1), m.group(2), m.group(3)
            total += 1
            good, detail = resolve(path, int(line_s), symbol)
            if good:
                ok += 1
            else:
                failures.append(
                    f"{doc.name}:{docline_no}  `{path}:{line_s}"
                    f"{' ' + symbol if symbol else ''}`  -> {detail}"
                )
    return total, ok, failures


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        docs = [HERE / a for a in argv[1:]]
    else:
        docs = sorted(p for p in HERE.glob("*.md") if p.name[0].isdigit())

    grand_total = grand_ok = 0
    all_failures: list[str] = []
    for doc in docs:
        if not doc.is_file():
            print(f"missing document: {doc}")
            return 2
        total, ok, failures = check(doc)
        grand_total += total
        grand_ok += ok
        all_failures += failures
        status = "OK" if not failures else f"{len(failures)} FAILED"
        print(f"{doc.name:24s} {ok}/{total} citations  {status}")

    if all_failures:
        print()
        for f in all_failures:
            print("  " + f)
    print(f"\ntotal: {grand_ok}/{grand_total} citations verified")
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
