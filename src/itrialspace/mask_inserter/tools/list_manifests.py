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

"""
list_manifests.py — Recursively find manifest CSV/JSON files in a directory
and write a manifest list file (one path per line) for use with SLURM array jobs.

Usage:
    python list_manifests.py --dir $ITRIALSPACE_DATA_DIR/.../manifests/mode2_size_detection_curve \
                             --output manifests_mode2.txt

    python list_manifests.py --dir $ITRIALSPACE_DATA_DIR/.../manifests/mode7_bootstrap_confidence \
                             --output manifests_mode7.txt \
                             --pattern "*.csv"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_manifests(
    directory: str | Path,
    pattern: str = "*.csv",
    also_json: bool = True,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """
    Recursively find manifest files in *directory*.

    Parameters
    ----------
    directory : str or Path
        Root directory to search.
    pattern : str
        Glob pattern for manifest files (default: ``*.csv``).
    also_json : bool
        Also search for ``*.json`` in addition to *pattern*.
    exclude_patterns : list of str, optional
        Skip files whose name contains any of these substrings
        (e.g., ``["audit", "log", "metadata"]``).

    Returns
    -------
    list of Path
        Sorted list of absolute paths to discovered manifest files.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        print(f"ERROR: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    exclude = exclude_patterns or [
        "audit",
        "metadata",
        "log",
        "README",
        "_run_config",
    ]

    results: list[Path] = []

    # Collect matching files
    for pat in [pattern] + (["*.json"] if also_json else []):
        for p in directory.rglob(pat):
            if not p.is_file():
                continue
            # Skip known non-manifest files
            if any(ex.lower() in p.name.lower() for ex in exclude):
                continue
            results.append(p)

    # Deduplicate (in case pattern already includes json) and sort
    results = sorted(set(results))

    # When both CSV and JSON exist for the same stem, keep only the CSV to
    # prevent duplicate array tasks that race on the same output directory.
    if also_json:
        stems_with_csv = {p.stem for p in results if p.suffix == ".csv"}
        results = [p for p in results if not (p.suffix == ".json" and p.stem in stems_with_csv)]

    return results


def write_manifest_list(manifests: list[Path], output_path: str | Path) -> None:
    """Write one manifest path per line to *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for m in manifests:
            f.write(str(m) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find cohort manifest CSV/JSON files and write a list file "
        "for use with SLURM array jobs.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        help="Directory to search recursively for manifest files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path (one manifest path per line).",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for manifest files (default: *.csv).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not also search for *.json files.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Substrings to exclude from filenames (default: audit, metadata, log, README).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print found manifests but do not write the output file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress informational output (still prints errors/warnings).",
    )

    args = parser.parse_args()

    manifests = find_manifests(
        directory=args.dir,
        pattern=args.pattern,
        also_json=not args.no_json,
        exclude_patterns=args.exclude,
    )

    print(f"Directory: {os.path.abspath(args.dir)}")
    print(f"Pattern:   {args.pattern}" + (" + *.json" if not args.no_json else ""))
    print(f"Found:     {len(manifests)} manifest(s)")
    print()

    if len(manifests) == 0:
        print("WARNING: No manifests found. Check the directory path.", file=sys.stderr)
        sys.exit(0)

    if not args.quiet:
        for i, m in enumerate(manifests):
            print(f"  [{i:3d}] {m}")

    if args.dry_run:
        if not args.quiet:
            print(f"\n(dry-run) Would write to: {args.output}")
    else:
        write_manifest_list(manifests, args.output)
        if not args.quiet:
            print(f"\nWritten to: {args.output}")
            print("\nTo submit as a SLURM array job:")
            print(f"  sbatch --array=0-{len(manifests) - 1} modeX_insert_masks_array.sub")


if __name__ == "__main__":
    main()
