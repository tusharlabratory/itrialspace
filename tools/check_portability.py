#!/usr/bin/env python3
# Copyright (c) 2026 Fakrul Islam Tushar
# Department of Radiology and Imaging Sciences, University of Arizona
# Email: fitushar@arizona.edu
#
# This file is part of iTrialSpace — a virtual clinical trial engine
# for controlled evaluation of lung CT AI models.
#
# If you use this software or the NoduleIndex dataset, please cite:
#
#   @article{tushar2026itrialspace,
#     title   = {iTRIALSPACE: Programmable Virtual Lesion Trials for
#                Controlled Evaluation of Lung CT Models},
#     author  = {Tushar, Fakrul Islam and Momy, Umme Hafsa and
#                Lo, Joseph Y and Rubin, Geoffrey D},
#     journal = {arXiv preprint arXiv:2605.05761},
#     year    = {2026}
#   }
#
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Free to use, copy, modify, and share for NONCOMMERCIAL purposes —
# including academic research and teaching. Commercial use requires
# a separate license.
# Full terms: LICENSE file in the project root, or
# https://polyformproject.org/licenses/noncommercial/1.0.0/
#
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0

"""Portability guard: fail if any tracked file hardcodes a machine-specific path.

Run by pre-commit and CI. Mirrors the spirit of the project's existing dataset-purge
guard: a single scan that keeps the repo copy-and-run portable on any machine.

A finding is any occurrence of a forbidden absolute-path/env token outside the
explicitly allowed locations (docs, example configs, this checker itself).

Usage:
    python tools/check_portability.py            # scan repo, exit 1 on findings
    python tools/check_portability.py --list      # also print every match
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Tokens that must never appear in shipped code/config/scripts.
FORBIDDEN = [
    r"/home/ft42",
    r"/scratch/railabs",
    r"/scratch/[A-Za-z0-9_]+/VLST",
    r"monai-auto3dseg",  # old hardcoded conda env name
]

# Paths where these tokens are allowed (documentation, history).
ALLOWED_PREFIXES = (
    "docs/",
    "CHANGELOG.md",
    "tools/check_portability.py",
)
ALLOWED_SUFFIXES = (
    ".example.yaml",
    ".example.yml",
    ".md",  # prose/docs may reference original paths illustratively
)

# Only scan these kinds of files.
SCAN_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".sh", ".sub", ".cfg", ".toml", ".txt")

PATTERN = re.compile("|".join(FORBIDDEN))


def tracked_files(root: Path) -> list[Path]:
    """Prefer git-tracked files; fall back to a filesystem walk."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        files = [root / line for line in out.splitlines() if line.strip()]
        if files:
            return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return [p for p in root.rglob("*") if p.is_file()]


def is_allowed(rel: str) -> bool:
    if rel.startswith(ALLOWED_PREFIXES):
        return True
    return rel.endswith(ALLOWED_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every matching line")
    ap.add_argument("--root", default=None, help="repo root (default: auto)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    findings: list[tuple[str, int, str]] = []

    for path in tracked_files(root):
        if path.suffix not in SCAN_SUFFIXES:
            continue
        rel = str(path.relative_to(root))
        if is_allowed(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                findings.append((rel, i, line.strip()))

    if findings:
        print(f"✗ portability check FAILED — {len(findings)} hardcoded path(s):\n")
        for rel, ln, line in findings:
            print(f"  {rel}:{ln}: {line[:120]}")
        print(
            "\nResolve paths via itrialspace.config.settings or env vars "
            "(see docs/configuration.md)."
        )
        return 1

    print("✓ portability check passed — no hardcoded machine-specific paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
