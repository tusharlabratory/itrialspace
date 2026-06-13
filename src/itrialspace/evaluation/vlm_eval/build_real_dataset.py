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
Build a VLM evaluation dataset from real CT images and nodule profiles.

Reads per-dataset nodule profile CSVs (from iTrialSpace ``profiles/``),
resolves the corresponding raw CT NIfTI and per-nodule segmentation mask,
extracts a lesion-centred 2D axial slice, and produces a clean
``eval_dataset.csv`` in the same schema as the synthetic pipeline.

Differences from the synthetic pipeline (build_dataset.py):
- No synthetic CT / NodMAISI – uses raw CTs under ``raw_ct/``
- Per-nodule binary masks (label > 0) under ``masks/{dataset}/nodule_seg/``
- Ground truth from profile CSV columns (``lobe_name``, ``nodule_mean_diam_mm``)
- Column normalisation across heterogeneous datasets
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from itrialspace.evaluation.vlm_eval.overlay import generate_overlays, generate_qc_grid
from itrialspace.evaluation.vlm_eval.preprocess_profiles import (
    DEFAULT_PROFILE,
    PreprocessProfile,
    get_profile,
    list_profiles,
)
from itrialspace.evaluation.vlm_eval.prompt_templates import (
    diameter_to_size_bucket,
    normalise_lobe,
)
from itrialspace.evaluation.vlm_eval.slice_extractor import extract_slice

logger = logging.getLogger(__name__)

# The 7 public datasets of the iTrialSpace core. NLST3D is intentionally excluded — it is
# permanently purged from the project (see the core's NLST3D guard + purge tests); do not
# re-add it to this default list.
ALL_DATASETS = [
    "DLCS24",
    "IMDCT",
    "LNDbv4",
    "LUNA16",
    "LUNA25",
    "LUNGx",
    "NSCLCR",
]


def _resolve_nodule_mask(
    row: pd.Series,
    dataset: str,
    masks_base: str,
) -> Optional[str]:
    """Resolve the per-nodule segmentation mask path for a profile row.

    Returns the absolute path to the NIfTI mask, or None if not found.

    Naming conventions vary by dataset:
    - Most: ``masks/{dataset}/nodule_seg/{AnnotationID}.nii.gz``
    - IMDCT: ``masks/{dataset}/nodule_seg/{PatientID}_seg.nii.gz``
    """
    mask_dir = os.path.join(masks_base, dataset, "nodule_seg")

    if dataset == "IMDCT":
        pid = str(row.get("PatientID", ""))
        if not pid:
            return None
        mask_path = os.path.join(mask_dir, f"{pid}_seg.nii.gz")
    else:
        ann_id = str(row.get("AnnotationID", ""))
        if not ann_id or ann_id == "nan":
            return None
        mask_path = os.path.join(mask_dir, f"{ann_id}.nii.gz")

    return mask_path if os.path.isfile(mask_path) else None


def _resolve_raw_ct(
    row: pd.Series,
    raw_ct_base: str,
) -> Optional[str]:
    """Resolve the raw CT NIfTI path from the profile row's ct_path column.

    ``ct_path`` already contains ``{DATASET}/{filename}.nii.gz``.
    """
    ct_rel = str(row.get("ct_path", ""))
    if not ct_rel or ct_rel == "nan":
        return None
    ct_full = os.path.join(raw_ct_base, ct_rel)
    return ct_full if os.path.isfile(ct_full) else None


def _case_id_for_row(row: pd.Series, dataset: str) -> str:
    """Build a unique case identifier for a profile row."""
    if dataset == "IMDCT":
        return str(row.get("PatientID", ""))
    return str(row.get("AnnotationID", ""))


def build_real_dataset(
    data_base: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    max_cases: Optional[int] = None,
    profile: Optional[PreprocessProfile] = None,
    generate_overlay_conditions: bool = False,
) -> pd.DataFrame:
    """Build evaluation dataset from real CT images.

    Parameters
    ----------
    data_base : str
        Root of iTrialSpace data (contains ``raw_ct/``, ``profiles/``,
        ``masks/``).
    output_dir : str
        Root output directory.
    datasets : list of str, optional
        Which datasets to include. If None, uses all 7 core datasets.
    max_cases : int, optional
        Limit total cases (for quick testing).
    profile : PreprocessProfile, optional
        Preprocessing profile.
    generate_overlay_conditions : bool
        If True, generate bbox/contour/bbox_contour overlays.

    Returns
    -------
    pd.DataFrame
        The evaluation dataset, also saved to ``output_dir/eval_dataset.csv``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(output_dir, exist_ok=True)

    if datasets is None:
        datasets = list(ALL_DATASETS)

    raw_ct_base = os.path.join(data_base, "raw_ct")
    masks_base = os.path.join(data_base, "masks")
    profiles_dir = os.path.join(data_base, "profiles")

    rows: List[dict] = []
    total_processed = 0
    skip_count = 0
    method_counts: Dict[str, int] = {}

    for dataset in datasets:
        profile_csv = os.path.join(profiles_dir, f"{dataset}_nodule_profiles.csv")
        if not os.path.isfile(profile_csv):
            logger.warning("Profile CSV not found: %s — skipping dataset", profile_csv)
            continue

        logger.info("Reading profile: %s", profile_csv)
        df = pd.read_csv(profile_csv)

        # Per-dataset slice directory
        slice_dir = os.path.join(output_dir, "slices", dataset)
        os.makedirs(slice_dir, exist_ok=True)

        for _, row in df.iterrows():
            if max_cases and total_processed >= max_cases:
                break

            case_id = _case_id_for_row(row, dataset)

            # Resolve raw CT
            ct_path = _resolve_raw_ct(row, raw_ct_base)
            if ct_path is None:
                logger.debug("CT not found for %s — skipping", case_id)
                skip_count += 1
                continue

            # Resolve nodule mask
            mask_path = _resolve_nodule_mask(row, dataset, masks_base)
            if mask_path is None:
                logger.debug("Nodule mask not found for %s — skipping", case_id)
                skip_count += 1
                continue

            # Output PNG
            png_name = f"{case_id}.png"
            png_path = os.path.join(slice_dir, png_name)

            try:
                slice_info = extract_slice(
                    ct_path=ct_path,
                    output_png=png_path,
                    input_mask_path=mask_path,
                    inserted_mask_path=None,
                    profile=profile,
                    nodule_label=None,  # binary masks: any non-zero voxel
                )
            except Exception as e:
                logger.warning("Failed to extract slice for %s: %s", case_id, e)
                skip_count += 1
                continue

            if slice_info is None:
                skip_count += 1
                continue

            method_counts[slice_info.method] = method_counts.get(slice_info.method, 0) + 1

            # ── Generate overlay variants ─────────────────────────────────
            overlay_paths = {"bbox": "", "contour": "", "bbox_contour": ""}
            if generate_overlay_conditions and mask_path:
                try:
                    result = generate_overlays(
                        plain_png_path=png_path,
                        mask_path=mask_path,
                        slice_index=slice_info.center_z,
                        slice_plane=(profile.slice_plane if profile else "axial"),
                        nodule_label=None,  # binary per-nodule masks
                    )
                    for cond in ("bbox", "contour", "bbox_contour"):
                        if result.get(cond):
                            overlay_paths[cond] = result[cond]
                except Exception as e:
                    logger.warning("Overlay generation failed for %s: %s", case_id, e)

            # ── Ground truth labels ───────────────────────────────────────
            lobe_raw = str(row.get("lobe_name", ""))
            gt_lobe = normalise_lobe(lobe_raw)

            diam = row.get("nodule_mean_diam_mm")
            gt_size = diameter_to_size_bucket(float(diam)) if pd.notna(diam) else ""

            rows.append(
                {
                    "case_id": case_id,
                    "dataset": dataset,
                    "ct_path": ct_path,
                    "nodule_mask_path": mask_path,
                    "png_path": png_path,
                    "png_bbox_path": overlay_paths.get("bbox", ""),
                    "png_contour_path": overlay_paths.get("contour", ""),
                    "png_bbox_contour_path": overlay_paths.get("bbox_contour", ""),
                    "slice_z": slice_info.center_z,
                    "slice_method": slice_info.method,
                    "preprocess_profile": (profile.name if profile else DEFAULT_PROFILE),
                    "ground_truth_presence": "present",
                    "ground_truth_lobe": gt_lobe,
                    "ground_truth_size_bucket": gt_size,
                    "nodule_mean_diam_mm": (float(diam) if pd.notna(diam) else None),
                    "lobe_name": lobe_raw,
                }
            )
            total_processed += 1

        if max_cases and total_processed >= max_cases:
            break

    eval_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "eval_dataset.csv")
    eval_df.to_csv(out_csv, index=False)

    logger.info("Real CT eval dataset: %d cases -> %s", len(eval_df), out_csv)
    logger.info("Slice methods: %s", method_counts)
    if skip_count > 0:
        logger.warning("Skipped %d cases (missing CT/mask)", skip_count)

    # ── Per-dataset summary ───────────────────────────────────────────────
    if len(eval_df) > 0:
        for ds in eval_df["dataset"].unique():
            n = len(eval_df[eval_df["dataset"] == ds])
            logger.info("  %s: %d nodules", ds, n)

    if generate_overlay_conditions:
        n_bbox = (eval_df["png_bbox_path"] != "").sum()
        n_contour = (eval_df["png_contour_path"] != "").sum()
        logger.info(
            "Overlays: bbox=%d, contour=%d (of %d cases)",
            n_bbox,
            n_contour,
            len(eval_df),
        )

        # QC grids for first 10 cases
        qc_dir = os.path.join(output_dir, "qc_overlays")
        qc_count = 0
        for _, r in eval_df.head(min(30, len(eval_df))).iterrows():
            if r.get("png_bbox_path") and r.get("png_contour_path"):
                qc_path = os.path.join(
                    qc_dir,
                    f"qc_{r['dataset']}_{r['case_id']}.png",
                )
                generate_qc_grid(
                    plain_path=r["png_path"],
                    bbox_path=r.get("png_bbox_path"),
                    contour_path=r.get("png_contour_path"),
                    bbox_contour_path=r.get("png_bbox_contour_path"),
                    output_path=qc_path,
                )
                qc_count += 1
                if qc_count >= 10:
                    break
        if qc_count > 0:
            logger.info("QC grids saved: %d -> %s", qc_count, qc_dir)

    return eval_df


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build VLM evaluation dataset from real CT images.",
    )
    parser.add_argument(
        "--data-base",
        required=True,
        help=(
            "Root of iTrialSpace data directory containing raw_ct/, "
            "profiles/, and masks/ sub-directories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory for eval dataset and extracted slices.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=ALL_DATASETS,
        help=(
            "Which datasets to include. Default: all 7 core datasets. "
            f"Choices: {', '.join(ALL_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit total number of cases (for testing).",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=list_profiles(),
        help=("Preprocessing profile name. Default: %(default)s"),
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help=(
            "Generate bbox, contour, and bbox_contour overlay PNGs alongside " "each plain slice."
        ),
    )
    args = parser.parse_args()

    build_real_dataset(
        data_base=args.data_base,
        output_dir=args.output_dir,
        datasets=args.datasets,
        max_cases=args.max_cases,
        profile=get_profile(args.profile),
        generate_overlay_conditions=args.overlays,
    )


if __name__ == "__main__":
    main()
