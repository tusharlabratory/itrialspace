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
Build nodule-free negative slices for VLM presence detection evaluation.

For each unique CT in an existing eval_dataset.csv, this script:
1. Loads the CT volume (RAS+ canonical)
2. Identifies all nodule z-locations (from the mask) to avoid
3. Picks a random z-index that is:
   - Far from any nodule (>= MIN_Z_DISTANCE slices away)
   - Within the lung field (>= MIN_LUNG_FRAC voxels in lung HU range)
4. Extracts the slice using the same profile/windowing as the positive pipeline
5. Outputs eval_dataset_negatives.csv with ground_truth_presence = "absent"

Usage:
    python -m itrialspace.evaluation.vlm_eval.build_negative_dataset \\
        --eval-csv vlm_eval/lung_axial/eval_dataset.csv \\
        --profile lung_axial \\
        --output-dir vlm_eval/lung_axial/negatives \\
        [--workers 8] [--seed 42] [--negatives-per-ct 1]

    # For MedGemma profile:
    python -m itrialspace.evaluation.vlm_eval.build_negative_dataset \\
        --eval-csv vlm_eval/lung_axial_medgemma/eval_dataset.csv \\
        --profile lung_axial_medgemma \\
        --output-dir vlm_eval/lung_axial_medgemma/negatives
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from itrialspace.evaluation.vlm_eval.preprocess_profiles import get_profile
from itrialspace.evaluation.vlm_eval.slice_extractor import (
    NODULE_LABEL,
    _nodule_centroid_from_mask,
    extract_slice_at_z,
)

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
MIN_Z_DISTANCE = 30  # minimum slices away from any nodule centroid z
MIN_LUNG_FRAC = 0.05  # ≥5% of voxels in lung HU range (-1024 to -200)
LUNG_HU_LO = -1024.0
LUNG_HU_HI = -200.0
MAX_CANDIDATES = 20  # max candidate slices to evaluate per CT


def _get_all_nodule_z(
    mask_paths: List[str],
    nodule_label: Optional[int],
) -> List[int]:
    """Collect all nodule centroid z-indices from a list of mask paths."""
    z_list = []
    for mp in mask_paths:
        if not mp or not os.path.isfile(mp):
            continue
        centroid = _nodule_centroid_from_mask(mp, nodule_label=nodule_label)
        if centroid is not None:
            z_list.append(centroid[2])  # z-index (axis 2 = axial)
    return z_list


def _is_lung_slice(ct_data: np.ndarray, z: int, threshold: float = MIN_LUNG_FRAC) -> bool:
    """Check if axial slice z has enough voxels in lung HU range."""
    slc = ct_data[:, :, z]
    lung_voxels = np.sum((slc >= LUNG_HU_LO) & (slc <= LUNG_HU_HI))
    total = slc.size
    return (lung_voxels / total) >= threshold


def _pick_negative_z(
    ct_data: np.ndarray,
    nodule_z_list: List[int],
    rng: np.random.Generator,
    n: int = 1,
    min_dist: int = MIN_Z_DISTANCE,
) -> List[int]:
    """Pick up to *n* negative z-indices far from all nodule locations.

    Returns a list of z-indices (may be shorter than *n* if not enough
    valid candidates exist).
    """
    max_z = ct_data.shape[2] - 1
    # Build set of excluded z-indices (nodule ± min_dist)
    excluded = set()
    for nz in nodule_z_list:
        for offset in range(-min_dist, min_dist + 1):
            excluded.add(nz + offset)

    # Candidate z-indices: within volume, not excluded
    candidates = [z for z in range(0, max_z + 1) if z not in excluded]
    if not candidates:
        # Relax distance to min_dist // 2
        half_dist = max(min_dist // 2, 5)
        excluded_relaxed = set()
        for nz in nodule_z_list:
            for offset in range(-half_dist, half_dist + 1):
                excluded_relaxed.add(nz + offset)
        candidates = [z for z in range(0, max_z + 1) if z not in excluded_relaxed]
        if not candidates:
            return []

    # Shuffle and pick candidates that pass the lung-field check
    rng.shuffle(candidates)
    picked = []
    for z in candidates[: MAX_CANDIDATES * 3]:  # check a generous number
        if _is_lung_slice(ct_data, z):
            picked.append(z)
            if len(picked) >= n:
                break
    return picked


def _process_one_ct(args: Tuple) -> List[dict]:
    """Worker function: extract negative slices for one CT.

    Returns a list of row dicts for the negatives CSV.
    """
    (
        ct_path,
        mask_paths,
        mode,
        trial_name,
        dataset_name,
        donor_dataset,
        population_type,
        insertion_mode,
        profile_name,
        output_dir,
        seed,
        n_neg,
        nodule_label,
        source,
    ) = args

    rng = np.random.default_rng(seed)
    profile = get_profile(profile_name)

    # Collect nodule z-locations to avoid
    nodule_z_list = _get_all_nodule_z(mask_paths, nodule_label=nodule_label)
    if not nodule_z_list:
        logger.warning("No nodule z found for %s — skipping", ct_path)
        return []

    # Load CT volume
    try:
        nii = nib.load(ct_path)
        nii = nib.as_closest_canonical(nii)
        ct_data = nii.get_fdata(dtype=np.float32)
    except Exception as e:
        logger.warning("Failed to load CT %s: %s", ct_path, e)
        return []

    # Pick negative z-indices
    neg_z_list = _pick_negative_z(ct_data, nodule_z_list, rng, n=n_neg)
    if not neg_z_list:
        logger.warning("No valid negative z found for %s", ct_path)
        return []

    rows = []
    for i, neg_z in enumerate(neg_z_list):
        # Build output filename
        ct_basename = os.path.basename(os.path.dirname(ct_path))
        if source == "real":
            # Real CTs: use ct filename stem
            ct_basename = os.path.splitext(os.path.splitext(os.path.basename(ct_path))[0])[0]

        suffix = f"_neg{i}" if n_neg > 1 else "_neg"
        ext = ".png"
        if profile and profile.output_format == "npy":
            ext = ".npy"

        out_name = f"{ct_basename}{suffix}{ext}"
        mode_dir = mode if source == "synthetic" else dataset_name
        out_path = os.path.join(output_dir, "slices_negative", mode_dir, out_name)

        try:
            extract_slice_at_z(ct_path, neg_z, out_path, profile=profile)
        except Exception as e:
            logger.warning("Failed extracting neg slice z=%d from %s: %s", neg_z, ct_path, e)
            continue

        row = {
            "ct_path": ct_path,
            "mode": mode,
            "trial_name": trial_name,
            "dataset_name": dataset_name,
            "donor_dataset": donor_dataset,
            "png_path": out_path,
            "slice_z": neg_z,
            "slice_method": "negative_random",
            "preprocess_profile": profile_name,
            "ground_truth_presence": "absent",
            "ground_truth_lobe": "N/A",
            "ground_truth_size_bucket": "N/A",
            "effective_diam_mm": 0.0,
            "population_type": population_type,
            "insertion_mode": insertion_mode,
            "source": source,
            "nodule_z_avoided": str(nodule_z_list),
        }
        rows.append(row)

    return rows


def build_synthetic_negatives(
    eval_csv: str,
    profile_name: str,
    output_dir: str,
    workers: int = 8,
    seed: int = 42,
    n_neg: int = 1,
) -> pd.DataFrame:
    """Build negative slices from the synthetic eval_dataset.csv."""
    df = pd.read_csv(eval_csv)
    logger.info(
        "Loaded %d rows from %s (%d unique CTs)", len(df), eval_csv, df["ct_path"].nunique()
    )

    # Group by CT path: collect all mask paths per CT
    tasks = []
    for ct_path, grp in df.groupby("ct_path"):
        mask_paths = []
        for _, row in grp.iterrows():
            if pd.notna(row.get("input_mask_path")):
                mask_paths.append(row["input_mask_path"])
            if pd.notna(row.get("inserted_mask_path")):
                mask_paths.append(row["inserted_mask_path"])
        mask_paths = list(set(mask_paths))

        first = grp.iloc[0]
        # Deterministic per-CT seed
        ct_seed = seed + hash(ct_path) % (2**31)

        tasks.append(
            (
                ct_path,
                mask_paths,
                first.get("mode", ""),
                first.get("trial_name", ""),
                first.get("dataset_name", ""),
                first.get("donor_dataset", ""),
                first.get("population_type", ""),
                first.get("insertion_mode", ""),
                profile_name,
                output_dir,
                ct_seed,
                n_neg,
                NODULE_LABEL,
                "synthetic",
            )
        )

    logger.info("Processing %d CTs with %d workers...", len(tasks), workers)
    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_ct, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                logger.info("  %d / %d CTs done", done, len(tasks))
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("Worker error for %s: %s", futures[fut], e)

    neg_df = pd.DataFrame(all_rows)
    return neg_df


def build_real_negatives(
    eval_csv: str,
    profile_name: str,
    output_dir: str,
    workers: int = 8,
    seed: int = 42,
    n_neg: int = 1,
) -> pd.DataFrame:
    """Build negative slices from the real eval_dataset.csv."""
    df = pd.read_csv(eval_csv)
    logger.info(
        "Loaded %d rows from %s (%d unique CTs)", len(df), eval_csv, df["ct_path"].nunique()
    )

    # Group by CT path: collect all nodule mask paths per CT
    tasks = []
    for ct_path, grp in df.groupby("ct_path"):
        mask_paths = []
        mask_col = "nodule_mask_path" if "nodule_mask_path" in grp.columns else "input_mask_path"
        for _, row in grp.iterrows():
            if pd.notna(row.get(mask_col)):
                mask_paths.append(row[mask_col])
        mask_paths = list(set(mask_paths))

        first = grp.iloc[0]
        ct_seed = seed + hash(ct_path) % (2**31)

        dataset = first.get("dataset", first.get("dataset_name", ""))

        tasks.append(
            (
                ct_path,
                mask_paths,
                "",  # mode (N/A for real)
                "",  # trial_name
                dataset,
                "",  # donor_dataset
                "",  # population_type
                "",  # insertion_mode
                profile_name,
                output_dir,
                ct_seed,
                n_neg,
                None,  # nodule_label = None (binary masks for real)
                "real",
            )
        )

    logger.info("Processing %d unique real CTs with %d workers...", len(tasks), workers)
    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_ct, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                logger.info("  %d / %d CTs done", done, len(tasks))
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("Worker error for %s: %s", futures[fut], e)

    neg_df = pd.DataFrame(all_rows)
    return neg_df


def main():
    parser = argparse.ArgumentParser(
        description="Build nodule-free negative slices for VLM presence evaluation."
    )
    parser.add_argument(
        "--eval-csv", required=True, help="Path to existing eval_dataset.csv (positive slices)"
    )
    parser.add_argument("--profile", default="lung_axial", help="Preprocessing profile name")
    parser.add_argument("--output-dir", required=True, help="Root output directory for negatives")
    parser.add_argument(
        "--source",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Whether the eval CSV is synthetic or real",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--negatives-per-ct", type=int, default=1, help="Number of negative slices per CT"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.source == "real":
        neg_df = build_real_negatives(
            args.eval_csv,
            args.profile,
            args.output_dir,
            workers=args.workers,
            seed=args.seed,
            n_neg=args.negatives_per_ct,
        )
    else:
        neg_df = build_synthetic_negatives(
            args.eval_csv,
            args.profile,
            args.output_dir,
            workers=args.workers,
            seed=args.seed,
            n_neg=args.negatives_per_ct,
        )

    if neg_df.empty:
        logger.error("No negative slices generated!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "eval_dataset_negatives.csv")
    neg_df.to_csv(out_csv, index=False)
    logger.info("Saved %d negative slices → %s", len(neg_df), out_csv)

    # Summary
    print(f"\n{'='*60}")
    print("Negative slice extraction complete")
    print(f"  Source:     {args.source}")
    print(f"  Profile:    {args.profile}")
    print(f"  Total:      {len(neg_df)} negative slices")
    if "mode" in neg_df.columns and neg_df["mode"].str.len().sum() > 0:
        print("  By mode:")
        for mode, cnt in neg_df["mode"].value_counts().items():
            if mode:
                print(f"    {mode}: {cnt}")
    if "dataset_name" in neg_df.columns:
        print("  By dataset:")
        for ds, cnt in neg_df["dataset_name"].value_counts().items():
            if ds:
                print(f"    {ds}: {cnt}")
    print(f"  CSV:        {out_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
