#!/usr/bin/env python3
"""#785 P1 kernel-trace classifier.

Contract (Researcher bee0):
  A (default/on): trace must contain PagedAttnPrefillSharedKWmma<2,8,16,32,false>
  B (WMMA=0): that kernel absent; scalar PagedAttnPrefillSharedK<2,8,...> present

WMMA's mangled/demangled name contains the SharedK prefix, so scalar
detection ignores any line that also contains SharedKWmma.

Exit 0 prints `arm=A|B|UNKNOWN` plus marker hits.
Exit 2 = unreadable/empty input (fail closed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WMMA_MARKERS = (
    "PagedAttnPrefillSharedKWmma<2, 8, 16, 32, false>",
    "PagedAttnPrefillSharedKWmma<2,8,16,32,false>",
    "PagedAttnPrefillSharedKWmmaILi2ELi8ELi16ELi32ELb0E",
)
SCALAR_MARKERS = (
    "PagedAttnPrefillSharedK<2, 8,",
    "PagedAttnPrefillSharedK<2,8,",
    "PagedAttnPrefillSharedKILi2ELi8E",
)


def collect_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(errors="replace")
    chunks: list[str] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".json", ".txt", ".log", ".out", ""}:
            if p.stat().st_size > 8_000_000:
                continue
        try:
            chunks.append(p.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def classify(text: str) -> tuple[str, list[str], list[str]]:
    wmma_hits = [m for m in WMMA_MARKERS if m in text]
    scalar_blob_lines = []
    for line in text.splitlines():
        if "SharedKWmma" in line:
            continue
        scalar_blob_lines.append(line)
    scalar_blob = "\n".join(scalar_blob_lines)
    scalar_hits = [m for m in SCALAR_MARKERS if m in scalar_blob]
    if wmma_hits and not scalar_hits:
        arm = "A"
    elif scalar_hits and not wmma_hits:
        arm = "B"
    else:
        arm = "UNKNOWN"
    return arm, wmma_hits, scalar_hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="rocprofv3 output file or directory")
    ap.add_argument("--expect", choices=("A", "B"), help="fail closed if arm mismatches")
    args = ap.parse_args(argv)
    src = Path(args.path)
    if not src.exists():
        print("ERROR: missing trace path", src, file=sys.stderr)
        return 2
    text = collect_text(src)
    if not text.strip():
        print("ERROR: empty trace", src, file=sys.stderr)
        return 2
    arm, wmma_hits, scalar_hits = classify(text)
    print(f"arm={arm}")
    print("wmma_hits=" + (",".join(wmma_hits) if wmma_hits else "-"))
    print("scalar_hits=" + (",".join(scalar_hits) if scalar_hits else "-"))
    if args.expect and arm != args.expect:
        print(f"ERROR: expected arm={args.expect} got {arm}", file=sys.stderr)
        return 1
    if args.expect == "A" and not wmma_hits:
        print("ERROR: A missing WMMA kernel identity", file=sys.stderr)
        return 1
    if args.expect == "B" and (wmma_hits or not scalar_hits):
        print("ERROR: B must be scalar-only", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
