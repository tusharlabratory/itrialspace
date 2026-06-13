#!/usr/bin/env python3
"""Insert the iTrialSpace PolyForm Noncommercial header into author-owned source files.

Idempotent: skips any file that already carries a copyright/SPDX header (so the
MONAI/diffusers Apache-2.0 files under src/itrialspace/synthesis/scripts/ are never
touched, and re-running is safe). The header is inserted *after* a shebang and/or
encoding line, before the rest of the file (a comment block is valid before a module
docstring or ``from __future__`` import).

Usage:
    python tools/add_license_headers.py            # apply
    python tools/add_license_headers.py --check     # report files that WOULD change, exit 1 if any
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Author-owned trees to header.  synthesis/scripts/** is excluded (MONAI/diffusers, Apache-2.0).
INCLUDE = [
    ("src/itrialspace", "*.py"),
    ("infra/bash", "*.sh"),
    ("tools", "*.py"),
]
EXCLUDE_DIRS = {
    REPO / "src/itrialspace/synthesis/scripts",
}
# Markers that mean "already has a license header" -> skip.
SKIP_MARKERS = ("SPDX-License-Identifier", "Copyright (c)", "Copyright (C)", "MONAI Consortium")

HEADER_LINES = [
    "# Copyright (c) 2026 Fakrul Islam Tushar",
    "# Department of Radiology and Imaging Sciences, University of Arizona",
    "# Email: fitushar@arizona.edu",
    "#",
    "# This file is part of iTrialSpace — a virtual clinical trial engine",
    "# for controlled evaluation of lung CT AI models.",
    "#",
    "# If you use this software or the NoduleIndex dataset, please cite:",
    "#",
    "#   @article{tushar2026itrialspace,",
    "#     title   = {iTRIALSPACE: Programmable Virtual Lesion Trials for",
    "#                Controlled Evaluation of Lung CT Models},",
    "#     author  = {Tushar, Fakrul Islam and Momy, Umme Hafsa and",
    "#                Lo, Joseph Y and Rubin, Geoffrey D},",
    "#     journal = {arXiv preprint arXiv:2605.05761},",
    "#     year    = {2026}",
    "#   }",
    "#",
    "# Licensed under the PolyForm Noncommercial License 1.0.0.",
    "# Free to use, copy, modify, and share for NONCOMMERCIAL purposes —",
    "# including academic research and teaching. Commercial use requires",
    "# a separate license.",
    "# Full terms: LICENSE file in the project root, or",
    "# https://polyformproject.org/licenses/noncommercial/1.0.0/",
    "#",
    "# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0",
]


def is_excluded(path: Path) -> bool:
    return any(ex in path.parents for ex in EXCLUDE_DIRS)


def iter_targets():
    seen = set()
    for sub, pat in INCLUDE:
        for p in sorted((REPO / sub).rglob(pat)):
            if p.is_file() and not is_excluded(p) and p not in seen:
                seen.add(p)
                yield p


def needs_header(text: str) -> bool:
    head = text[:1500]
    return not any(m in head for m in SKIP_MARKERS)


def insert_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    i = 0
    # Preserve a shebang and an optional encoding/coding line at the very top.
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
        if i < len(lines) and "coding" in lines[i] and lines[i].lstrip().startswith("#"):
            i += 1
    block = "\n".join(HEADER_LINES) + "\n\n"
    return "".join(lines[:i]) + block + "".join(lines[i:])


def main() -> int:
    check = "--check" in sys.argv
    changed = []
    for p in iter_targets():
        text = p.read_text(encoding="utf-8")
        if not needs_header(text):
            continue
        changed.append(p)
        if not check:
            p.write_text(insert_header(text), encoding="utf-8")
    rel = [str(p.relative_to(REPO)) for p in changed]
    if check:
        for r in rel:
            print("would header:", r)
        print(f"{len(rel)} file(s) need headers")
        return 1 if rel else 0
    for r in rel:
        print("headered:", r)
    print(f"Done: headered {len(rel)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
