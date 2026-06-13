#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
make_test_manifest.py — Generate a minimal test manifest (1–3 rows)
pointing to real host CTs and donor nodule masks on the server.

Usage:
    python itrialspace_mask_inserter/tools/make_test_manifest.py \
        --output $ITRIALSPACE_DATA_DIR/test_manifest.csv \
        [--base-dir $ITRIALSPACE_DATA_DIR] \
        [--n-rows 2] \
        [--dataset LUNA25]

The script uses iTrialSpace's PathResolver to discover real data paths,
picks 1 host CT + 1 donor nodule from the specified (or first available)
dataset, and writes a CSV with absolute paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root() -> str:
    """Walk up from this script to find the itrialspace package root."""
    here = Path(__file__).resolve().parent
    for candidate in [here.parent.parent, here.parent.parent.parent]:
        if (candidate / "itrialspace" / "site" / "path_resolver.py").exists():
            return str(candidate)
    return str(here.parent.parent)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a minimal test manifest for full-volume integration tests.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="iTrialSpace data base directory. " "Default: $ITRIALSPACE_DATA_DIR",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to pick host/donor from (e.g. LUNA25, DLCS24). "
        "Default: first dataset with available profile CSV.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=2,
        help="Number of manifest rows to generate (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir or (
        os.environ.get("ITRIALSPACE_DATA_DIR") or os.path.expanduser("~/.itrialspace/data")
    )

    # Try to import PathResolver
    project_root = _find_project_root()
    sys.path.insert(0, project_root)

    try:
        from itrialspace.site.path_resolver import PathResolver

        resolver = PathResolver(base_dir=base_dir)
        print(f"PathResolver loaded. Base dir: {resolver.base_dir}")
        print(f"Available datasets: {resolver.available_datasets}")
    except Exception as e:
        print(f"WARNING: Could not load PathResolver: {e}")
        print("Falling back to manual path construction.")
        resolver = None

    # Pick dataset
    dataset = args.dataset
    if dataset is None and resolver is not None:
        dataset = _pick_dataset(resolver)
    if dataset is None:
        dataset = "LUNA25"
    print(f"Using dataset: {dataset}")

    # Load profile CSV to find real nodules
    rng = np.random.default_rng(args.seed)
    rows = _build_rows(resolver, base_dir, dataset, args.n_rows, rng)

    if not rows:
        print("ERROR: Could not build any manifest rows. Check data paths.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print(df[["case_id", "host_ct_path", "donor_nodule_mask_path", "insertion_lobe"]].to_string())


def _pick_dataset(resolver) -> str | None:
    """Pick the first dataset that has a readable profile CSV."""
    for ds in resolver.available_datasets:
        csv_path = resolver.resolve_profile_csv_path(ds)
        if os.path.isfile(csv_path):
            return ds
    return None


def _build_rows(
    resolver,
    base_dir: str,
    dataset: str,
    n_rows: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Build manifest rows from real data."""
    # Load profile to find nodule info
    if resolver is not None:
        profile_path = resolver.resolve_profile_csv_path(dataset)
    else:
        profile_path = os.path.join(base_dir, "profiles", f"{dataset}_nodule_profiles.csv")

    if not os.path.isfile(profile_path):
        print(f"WARNING: Profile CSV not found at {profile_path}")
        return _build_rows_manual(base_dir, dataset, n_rows, rng)

    print(f"Loading profile: {profile_path}")
    prof = pd.read_csv(profile_path)
    print(f"  {len(prof)} nodules in {dataset}")

    # Filter to nodules with valid lobe data
    valid = prof.dropna(subset=["ct_path"])
    if "reinsertion_lobe" in valid.columns:
        valid = valid[valid["reinsertion_lobe"].notna()]
    if len(valid) == 0:
        print("WARNING: No valid nodules after filtering.")
        return []

    # Sample
    n = min(n_rows, len(valid))
    sample = valid.sample(n=n, random_state=int(rng.integers(0, 2**31)))

    rows = []
    for idx, (_, nod) in enumerate(sample.iterrows()):
        ct_path_rel = str(nod["ct_path"])
        ct_id = os.path.basename(ct_path_rel).replace(".nii.gz", "")

        # Resolve paths
        if resolver is not None:
            host_ct = resolver.resolve_ct_path(dataset, ct_path_rel)
            host_seg = resolver.resolve_organ_seg_path(dataset, ct_id)
            ann_id = str(nod.get("annotation_id", ct_id)) if "annotation_id" in nod.index else ct_id
            donor_mask = resolver.resolve_nodule_mask_path(dataset, ann_id, ct_id)
        else:
            host_ct = os.path.join(base_dir, "raw_ct", ct_path_rel)
            host_seg = os.path.join(
                base_dir, "masks", dataset, "refined_seg", f"{ct_id}_seg.nii.gz"
            )
            donor_mask = os.path.join(base_dir, "masks", dataset, "nodule_seg", f"{ct_id}.nii.gz")

        lobe = str(nod.get("reinsertion_lobe", nod.get("lobe_name", "right_lung_upper_lobe")))
        cc_pct = float(nod.get("reinsertion_lobe_cc_pct", nod.get("lobe_cc_pct", 50.0)))
        ml_pct = float(nod.get("reinsertion_lobe_ml_pct", nod.get("lobe_ml_pct", 50.0)))
        ap_pct = float(nod.get("reinsertion_lobe_ap_pct", nod.get("lobe_ap_pct", 50.0)))

        # Clamp percentiles to [0, 100]
        cc_pct = max(0.0, min(100.0, cc_pct))
        ml_pct = max(0.0, min(100.0, ml_pct))
        ap_pct = max(0.0, min(100.0, ap_pct))

        rows.append(
            {
                "case_id": f"test_case_{idx:03d}",
                "nodule_idx": 0,
                "is_primary_nodule": True,
                "companion_group_id": "",
                "trial_name": "integration_test",
                "host_dataset": dataset,
                "host_patient_id": ct_id,
                "host_ct_path": host_ct,
                "host_organ_seg_path": host_seg,
                "donor_dataset": dataset,
                "donor_annotation_id": (
                    str(nod.get("annotation_id", ct_id)) if "annotation_id" in nod.index else ct_id
                ),
                "donor_patient_id": ct_id,
                "donor_ct_path": host_ct,
                "donor_nodule_mask_path": donor_mask,
                "donor_refined_seg_path": "",
                "insertion_lobe": lobe,
                "insertion_lobe_cc_pct": cc_pct,
                "insertion_lobe_ml_pct": ml_pct,
                "insertion_lobe_ap_pct": ap_pct,
                "insertion_mode": "profile_faithful",
                "scale_factor": 1.0,
                "warp_applied": "none",
                "effective_diam_mm": float(
                    nod.get("reinsertion_nodule_diam_mm", nod.get("nodule_mean_diam_mm", 8.0))
                ),
                "nodule_diam_mm": float(nod.get("nodule_mean_diam_mm", 8.0)),
                "label": (
                    int(nod["label"]) if "label" in nod.index and pd.notna(nod.get("label")) else 0
                ),
            }
        )

    return rows


def _build_rows_manual(
    base_dir: str,
    dataset: str,
    n_rows: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Fallback: scan directories directly."""
    ct_dir = os.path.join(base_dir, "raw_ct", dataset)
    if not os.path.isdir(ct_dir):
        print(f"WARNING: CT directory not found: {ct_dir}")
        return []

    cts = [f for f in os.listdir(ct_dir) if f.endswith(".nii.gz")]
    if not cts:
        return []

    rng.shuffle(cts)  # type: ignore[arg-type]
    rows = []
    for idx, ct_file in enumerate(cts[:n_rows]):
        ct_id = ct_file.replace(".nii.gz", "")
        rows.append(
            {
                "case_id": f"test_case_{idx:03d}",
                "nodule_idx": 0,
                "is_primary_nodule": True,
                "companion_group_id": "",
                "trial_name": "integration_test",
                "host_dataset": dataset,
                "host_patient_id": ct_id,
                "host_ct_path": os.path.join(ct_dir, ct_file),
                "host_organ_seg_path": os.path.join(
                    base_dir,
                    "masks",
                    dataset,
                    "refined_seg",
                    f"{ct_id}_seg.nii.gz",
                ),
                "donor_dataset": dataset,
                "donor_annotation_id": ct_id,
                "donor_patient_id": ct_id,
                "donor_ct_path": os.path.join(ct_dir, ct_file),
                "donor_nodule_mask_path": os.path.join(
                    base_dir,
                    "masks",
                    dataset,
                    "nodule_seg",
                    f"{ct_id}.nii.gz",
                ),
                "donor_refined_seg_path": "",
                "insertion_lobe": "right_lung_upper_lobe",
                "insertion_lobe_cc_pct": 50.0,
                "insertion_lobe_ml_pct": 50.0,
                "insertion_lobe_ap_pct": 50.0,
                "insertion_mode": "profile_faithful",
                "scale_factor": 1.0,
                "warp_applied": "none",
                "effective_diam_mm": 8.0,
                "nodule_diam_mm": 8.0,
                "label": 0,
            }
        )
    return rows


if __name__ == "__main__":
    main()
