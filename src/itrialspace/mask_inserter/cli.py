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

# -*- coding: utf-8 -*-
"""
cli.py — Command-line interface for the iTrialSpace Mask Insertion Engine.

Usage::

    # Full batch run
    python -m itrialspace.mask_inserter run \\
        --manifest cohort_manifest.csv \\
        --output-dir $ITRIALSPACE_OUTPUT_DIR/trial_01 \\
        --config my_overrides.yaml \\
        --trial-name trial_01 \\
        --seed 42

    # Dry-run (placement only, no files written)
    python -m itrialspace.mask_inserter run \\
        --manifest cohort_manifest.csv \\
        --output-dir $ITRIALSPACE_OUTPUT_DIR/trial_01 \\
        --dry-run

    # Verify existing outputs
    python -m itrialspace.mask_inserter verify \\
        --output-dir $ITRIALSPACE_OUTPUT_DIR/trial_01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m itrialspace.mask_inserter``."""
    parser = argparse.ArgumentParser(
        prog="itrialspace.mask_inserter",
        description="iTrialSpace Mask Insertion Engine — insert donor nodule "
        "masks into host CT space according to a CohortManifest.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run the insertion pipeline.")
    p_run.add_argument(
        "--manifest",
        "-m",
        required=True,
        help="Path to CohortManifest CSV or JSON.",
    )
    p_run.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Root output directory.",
    )
    p_run.add_argument(
        "--config",
        "-c",
        default=None,
        help="YAML config overrides (merged with defaults).",
    )
    p_run.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for resolving relative paths in the manifest.",
    )
    p_run.add_argument(
        "--trial-name",
        default="unnamed",
        help="Trial name (used for deterministic seeding).",
    )
    p_run.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute placement without writing masks.",
    )
    p_run.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers (across cases).",
    )
    p_run.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )

    # ── verify ────────────────────────────────────────────────────────────
    p_verify = sub.add_parser(
        "verify",
        help="Verify outputs from a previous run.",
    )
    p_verify.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory to verify.",
    )
    p_verify.add_argument(
        "--manifest",
        "-m",
        default=None,
        help="Original manifest for cross-referencing (optional).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Set up logging
    level = logging.WARNING
    if hasattr(args, "verbose"):
        if args.verbose == 1:
            level = logging.INFO
        elif args.verbose >= 2:
            level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "verify":
        return _cmd_verify(args)
    else:
        parser.print_help()
        return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the insertion pipeline."""
    from itrialspace.mask_inserter.inserter import insert_manifest

    records = insert_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config_path=args.config,
        base_dir=args.base_dir,
        trial_name=args.trial_name,
        seed=args.seed,
        dry_run=args.dry_run,
        n_jobs=args.n_jobs,
    )

    n_ok = sum(1 for r in records if r.status == "success")
    n_fail = sum(1 for r in records if r.status == "failed")
    n_dry = sum(1 for r in records if r.status == "dry_run")

    print(f"\niTrialSpace Mask Inserter — {'DRY RUN' if args.dry_run else 'COMPLETE'}")
    print(f"  Total rows:  {len(records)}")
    print(f"  Success:     {n_ok}")
    print(f"  Failed:      {n_fail}")
    if n_dry:
        print(f"  Dry-run:     {n_dry}")
    print(f"  Output dir:  {args.output_dir}")

    if n_fail > 0:
        print("\nFailed cases:")
        for r in records:
            if r.status == "failed":
                print(f"  {r.case_id}/nodule_{r.nodule_idx}: {r.reason}")

    return 1 if n_fail > 0 and n_ok == 0 else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Verify outputs from a previous run."""
    output_dir = args.output_dir
    audit_path = os.path.join(output_dir, "audit.json")

    if not os.path.isfile(audit_path):
        print(f"ERROR: No audit.json found in {output_dir}")
        return 1

    with open(audit_path) as f:
        audit = json.load(f)

    records = audit.get("records", [])
    print(f"Verifying {len(records)} insertion records from {audit_path}")

    missing_nodule_masks = 0
    missing_combined = 0
    total_success = 0
    seen_combined = set()

    for rec in records:
        if rec.get("status") != "success":
            continue
        total_success += 1

        # Per-nodule mask (optional — may not be saved)
        mask_path = rec.get("output_mask_path", "")
        if mask_path and not os.path.isfile(mask_path):
            missing_nodule_masks += 1

        # Combined mask (one per case)
        combined_path = rec.get("output_combined_path", "")
        if combined_path and combined_path not in seen_combined:
            seen_combined.add(combined_path)
            if not os.path.isfile(combined_path):
                missing_combined += 1
                print(f"  MISSING combined: {combined_path}")

    print("\nVerification:")
    print(f"  Successful insertions: {total_success}")
    print(
        f"  Combined masks:        {len(seen_combined)} expected, "
        f"{len(seen_combined) - missing_combined} found"
    )
    if missing_combined:
        print(f"  Missing combined:      {missing_combined}")

    if args.manifest:
        import pandas as pd

        df = pd.read_csv(args.manifest)
        n_cases = df["case_id"].nunique() if "case_id" in df.columns else len(df)
        print(f"  Manifest rows:         {len(df)} ({n_cases} unique cases)")
        print(
            f"  Coverage:              {total_success}/{len(df)} "
            f"({100*total_success/max(len(df),1):.1f}%)"
        )

    return 1 if missing_combined > 0 else 0


def _get_version() -> str:
    try:
        from itrialspace.mask_inserter import __version__

        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
