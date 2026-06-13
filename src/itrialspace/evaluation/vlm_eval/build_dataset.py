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
Build an evaluation dataset from iTrialSpace manifest + generated CTs.

Reads a manifest CSV (or multiple), locates the corresponding synthetic CT
for each case, extracts a lesion-centred 2D axial slice via the co-located
input_mask.nii.gz, and produces a clean eval_dataset.csv.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
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
from itrialspace.evaluation.vlm_eval.slice_extractor import NODULE_LABEL, extract_slice

logger = logging.getLogger(__name__)


# ── Column mapping ───────────────────────────────────────────────────────────
# iTrialSpace manifest columns → eval dataset columns.

_MANIFEST_KEY_COLS = [
    "case_id",
    "trial_name",
    "host_dataset",
    "host_ct_path",
    "donor_dataset",
    "insertion_coord_x",
    "insertion_coord_y",
    "insertion_coord_z",
    "insertion_lobe",
    "effective_diam_mm",
    "nodule_diam_mm",
    "label",
    "size_bucket",
    "n_nodules_in_case",
    "insertion_mode",
    "cohort_mode",
    "population_type",
]

# These columns may appear in digital-twin manifests
_OPTIONAL_MANIFEST_COLS = [
    "mode",
    "donor_nodule_mask_path",
]


def _find_mode_dir_name(manifest_path: str) -> str:
    """Walk up from a manifest CSV path to find the ``mode*_`` ancestor.

    For flat manifests (modes 1-10) the parent directory *is* the mode
    directory.  For nested manifests (modes 11-13::

        manifests/mode11_digital_twin_isolation/DLCS24/digital_twin_isolation_DLCS24.csv

    ) the immediate parent is a dataset sub-directory; we need to go one
    more level up to reach the ``mode11_digital_twin_isolation`` name.
    """
    parts = manifest_path.replace("\\", "/").split("/")
    for part in reversed(parts):
        if part.startswith("mode"):
            return part
    # Fallback: immediate parent (original behaviour for modes 1-10)
    return os.path.basename(os.path.dirname(manifest_path))


def _resolve_synthetic_ct(
    case_id: int,
    trial_name: str,
    manifest_path: str,
    ct_base: str,
) -> Optional[str]:
    """Find the synthetic_ct.nii.gz produced by NodMAISI for this case.

    NodMAISI case directories are named using the mask inserter output
    filename stem (the iTS--...) convention. We search for a matching
    directory under generated_cts/{mode_dir}/ or, for multi-trial modes,
    under generated_cts/{mode_dir}/{trial_name}/ or
    generated_cts/{mode_dir}/{manifest_stem}/  (for bootstrap runs).
    """
    mode_dir = _find_mode_dir_name(manifest_path)
    ct_mode_dir = os.path.join(ct_base, mode_dir)

    if not os.path.isdir(ct_mode_dir):
        return None

    prefix = f"iTS--{trial_name}--C{case_id:04d}--"
    manifest_stem = os.path.splitext(os.path.basename(manifest_path))[0]

    # Search directories: trial_name sub-dir, manifest-stem sub-dir, mode root
    search_dirs = [ct_mode_dir]
    trial_subdir = os.path.join(ct_mode_dir, trial_name)
    if os.path.isdir(trial_subdir):
        search_dirs.insert(0, trial_subdir)  # prefer sub-trial dir
    stem_subdir = os.path.join(ct_mode_dir, manifest_stem)
    if os.path.isdir(stem_subdir) and stem_subdir != trial_subdir:
        search_dirs.insert(0, stem_subdir)  # prefer manifest-stem dir

    for search_dir in search_dirs:
        for entry in os.listdir(search_dir):
            if entry.startswith(prefix):
                ct_path = os.path.join(search_dir, entry, "synthetic_ct.nii.gz")
                if os.path.isfile(ct_path):
                    return ct_path

    return None


def _resolve_input_mask(ct_path: str) -> Optional[str]:
    """Find input_mask.nii.gz co-located with the synthetic CT.

    This mask is in the same NodMAISI-resampled geometry (e.g. 512x512x256)
    as the synthetic CT and contains label 23 for the inserted nodule.
    """
    ct_dir = os.path.dirname(ct_path)
    input_mask = os.path.join(ct_dir, "input_mask.nii.gz")
    if os.path.isfile(input_mask):
        return input_mask
    return None


def _resolve_mask_path(
    case_id: int,
    trial_name: str,
    manifest_path: str,
    mask_base: str,
) -> Optional[str]:
    """Find the inserted combined mask for this case."""
    mode_dir = _find_mode_dir_name(manifest_path)
    mask_mode_dir = os.path.join(mask_base, mode_dir)

    if not os.path.isdir(mask_mode_dir):
        return None

    prefix = f"iTS--{trial_name}--C{case_id:04d}--"
    manifest_stem = os.path.splitext(os.path.basename(manifest_path))[0]

    # Search: trial_name sub-dir, manifest-stem sub-dir, mode root
    search_dirs = [mask_mode_dir]
    trial_subdir = os.path.join(mask_mode_dir, trial_name)
    if os.path.isdir(trial_subdir):
        search_dirs.insert(0, trial_subdir)
    stem_subdir = os.path.join(mask_mode_dir, manifest_stem)
    if os.path.isdir(stem_subdir) and stem_subdir != trial_subdir:
        search_dirs.insert(0, stem_subdir)

    for search_dir in search_dirs:
        for entry in os.listdir(search_dir):
            if entry.startswith(prefix) and entry.endswith("_mask.nii.gz"):
                return os.path.join(search_dir, entry)

    return None


def _process_single_case(args: Tuple) -> Optional[dict]:
    """Worker function for parallel case extraction.

    Handles CT resolution, mask resolution, slice extraction, overlay
    generation, and ground-truth label construction for a single case.
    Returns a row dict on success, or None on skip/failure.
    """
    (
        case_id,
        trial_name,
        manifest_path,
        manifest_stem,
        mode_name,
        slice_dir,
        ct_base,
        mask_base,
        use_synthetic_ct,
        profile,
        generate_overlay_conditions,
        row_dict,
    ) = args

    # Resolve CT path
    ct_path = None
    if use_synthetic_ct:
        ct_path = _resolve_synthetic_ct(case_id, trial_name, manifest_path, ct_base)
    if ct_path is None:
        ct_path = str(row_dict.get("host_ct_path", ""))
        if not ct_path or not os.path.isfile(ct_path):
            return None

    # Resolve masks
    input_mask_path = _resolve_input_mask(ct_path) if use_synthetic_ct else None
    inserted_mask_path = _resolve_mask_path(case_id, trial_name, manifest_path, mask_base)

    # Extract slice
    png_name = f"{manifest_stem}__C{case_id:04d}.png"
    png_path = os.path.join(slice_dir, png_name)

    try:
        slice_info = extract_slice(
            ct_path=ct_path,
            output_png=png_path,
            input_mask_path=input_mask_path,
            inserted_mask_path=inserted_mask_path,
            save_overlay=False,
            profile=profile,
            nodule_label=NODULE_LABEL,
        )
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to extract slice for case %d: %s", case_id, e)
        return None

    if slice_info is None:
        return None

    # Generate overlay variants
    overlay_paths = {"bbox": "", "contour": "", "bbox_contour": ""}
    if generate_overlay_conditions:
        mask_for_overlay = input_mask_path or inserted_mask_path
        if mask_for_overlay:
            try:
                result = generate_overlays(
                    plain_png_path=png_path,
                    mask_path=mask_for_overlay,
                    slice_index=slice_info.center_z,
                    slice_plane=(profile.slice_plane if profile else "axial"),
                    nodule_label=NODULE_LABEL,
                )
                for cond in ("bbox", "contour", "bbox_contour"):
                    if result.get(cond):
                        overlay_paths[cond] = result[cond]
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Overlay generation failed for case %d: %s", case_id, e
                )

    # Ground truth labels
    insertion_lobe = str(row_dict.get("insertion_lobe", ""))
    gt_lobe = normalise_lobe(insertion_lobe)
    eff_diam = row_dict.get("effective_diam_mm")
    if eff_diam is None or (isinstance(eff_diam, float) and np.isnan(eff_diam)):
        eff_diam = row_dict.get("nodule_diam_mm")
    gt_size = ""
    if eff_diam is not None and not (isinstance(eff_diam, float) and np.isnan(eff_diam)):
        gt_size = diameter_to_size_bucket(float(eff_diam))

    label_val = row_dict.get("label")
    if label_val is not None and isinstance(label_val, float) and np.isnan(label_val):
        label_val = None
    elif label_val is not None:
        label_val = int(label_val)

    return {
        "case_id": case_id,
        "mode": mode_name,
        "trial_name": trial_name,
        "dataset_name": str(row_dict.get("host_dataset", "")),
        "donor_dataset": str(row_dict.get("donor_dataset", "")),
        "ct_path": ct_path,
        "input_mask_path": input_mask_path or "",
        "inserted_mask_path": inserted_mask_path or "",
        "png_path": png_path,
        "png_bbox_path": overlay_paths.get("bbox", ""),
        "png_contour_path": overlay_paths.get("contour", ""),
        "png_bbox_contour_path": overlay_paths.get("bbox_contour", ""),
        "slice_z": slice_info.center_z,
        "slice_method": slice_info.method,
        "preprocess_profile": profile.name if profile else DEFAULT_PROFILE,
        "ground_truth_presence": "present",
        "ground_truth_lobe": gt_lobe,
        "ground_truth_size_bucket": gt_size,
        "effective_diam_mm": (
            float(eff_diam)
            if eff_diam is not None and not (isinstance(eff_diam, float) and np.isnan(eff_diam))
            else None
        ),
        "insertion_lobe": insertion_lobe,
        "label": label_val,
        "population_type": str(row_dict.get("population_type", "")),
        "insertion_mode": str(row_dict.get("insertion_mode", "")),
    }


def build_dataset(
    manifest_paths: List[str],
    output_dir: str,
    ct_base: str = os.path.join(
        os.environ.get("ITRIALSPACE_OUTPUT_DIR")
        or os.environ.get("ITRIALSPACE_DATA_DIR")
        or os.path.expanduser("~/.itrialspace/data"),
        "generated_cts",
    ),
    mask_base: str = os.path.join(
        os.environ.get("ITRIALSPACE_OUTPUT_DIR")
        or os.environ.get("ITRIALSPACE_DATA_DIR")
        or os.path.expanduser("~/.itrialspace/data"),
        "inserted_masks",
    ),
    max_cases: Optional[int] = None,
    use_synthetic_ct: bool = True,
    save_overlay: bool = False,
    profile: Optional[PreprocessProfile] = None,
    generate_overlay_conditions: bool = False,
    skip_existing: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """Build the evaluation dataset from one or more manifest CSVs.

    Parameters
    ----------
    manifest_paths : list of str
        Paths to iTrialSpace manifest CSVs.
    output_dir : str
        Root output directory. PNGs go to output_dir/slices/{mode_name}/.
    ct_base : str
        Root of NodMAISI generated CTs.
    mask_base : str
        Root of inserted masks.
    max_cases : int, optional
        Limit total cases (for quick testing).
    use_synthetic_ct : bool
        If True, use NodMAISI synthetic CTs. If False, use original host CTs.
    save_overlay : bool
        If True, save overlay PNGs alongside slices for QC.
        Overridden by *profile* if provided.
    profile : PreprocessProfile, optional
        Preprocessing profile controlling windowing, slice plane, format, etc.
        If None, uses the default ``lung_axial`` profile behaviour.
    skip_existing : bool
        If True and an ``eval_dataset.csv`` already exists in *output_dir*,
        reuse rows whose plain PNG is still on disk.  Only genuinely new
        cases (not in the cache or whose PNG is missing) trigger NIfTI
        loading and slice extraction.  Dramatically speeds up incremental
        rebuilds when only a few new CTs have appeared.

    Returns
    -------
    pd.DataFrame
        The evaluation dataset, also saved to output_dir/eval_dataset.csv.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(output_dir, exist_ok=True)

    # ── Skip-existing cache ──────────────────────────────────────────────
    _cache: Dict[tuple, dict] = {}
    if skip_existing:
        _old_csv = os.path.join(output_dir, "eval_dataset.csv")
        if os.path.isfile(_old_csv):
            _old_df = pd.read_csv(_old_csv)
            for _, _r in _old_df.iterrows():
                _key = (str(_r["mode"]), str(_r["trial_name"]), int(_r["case_id"]))
                _png = str(_r.get("png_path", ""))
                if _png and os.path.isfile(_png):
                    _cache[_key] = _r.to_dict()
            logger.info(
                "skip-existing: loaded %d cached rows from %s",
                len(_cache),
                _old_csv,
            )
    cache_hits = 0

    rows = []
    total_processed = 0
    skip_count = 0
    method_counts: Dict[str, int] = {}

    # ── Phase 1: resolve cache hits and collect work items ───────────────
    work_items = []  # list of tuples for _process_single_case

    for manifest_path in manifest_paths:
        logger.info("Reading manifest: %s", manifest_path)
        df = pd.read_csv(manifest_path)

        # Infer mode name from the mode* ancestor directory
        mode_name = _find_mode_dir_name(manifest_path)
        # Use manifest stem to disambiguate when multiple CSVs per mode
        manifest_stem = os.path.splitext(os.path.basename(manifest_path))[0]

        # Per-mode slice directory
        slice_dir = os.path.join(output_dir, "slices", mode_name)
        os.makedirs(slice_dir, exist_ok=True)

        # For multi-nodule cases, take only the primary nodule row
        if "is_primary_nodule" in df.columns:
            df = df[df["is_primary_nodule"] == True].copy()

        for _, row in df.iterrows():
            if max_cases and (total_processed + len(work_items)) >= max_cases:
                break

            case_id = int(row["case_id"])
            trial_name = str(row["trial_name"])

            # ── Check skip-existing cache (fast, no I/O) ─────────────────
            _cache_key = (mode_name, trial_name, case_id)
            if _cache and _cache_key in _cache:
                # Quick CT path check — need to resolve to compare
                ct_path = None
                if use_synthetic_ct:
                    ct_path = _resolve_synthetic_ct(case_id, trial_name, manifest_path, ct_base)
                if ct_path is None:
                    ct_path = str(row.get("host_ct_path", ""))

                cached_row = _cache[_cache_key]
                cached_ct = str(cached_row.get("ct_path", ""))
                if cached_ct == ct_path:
                    rows.append(cached_row)
                    cache_hits += 1
                    total_processed += 1
                    method = str(cached_row.get("slice_method", "cached"))
                    method_counts[method] = method_counts.get(method, 0) + 1
                    continue

            # Needs extraction — collect as a work item
            work_items.append(
                (
                    case_id,
                    trial_name,
                    manifest_path,
                    manifest_stem,
                    mode_name,
                    slice_dir,
                    ct_base,
                    mask_base,
                    use_synthetic_ct,
                    profile,
                    generate_overlay_conditions,
                    row.to_dict(),
                )
            )

        if max_cases and (total_processed + len(work_items)) >= max_cases:
            break

    logger.info(
        "Phase 1 done: %d cache hits, %d cases need extraction (workers=%d)",
        cache_hits,
        len(work_items),
        workers,
    )

    # ── Phase 2: extract slices (parallel or sequential) ─────────────────
    if work_items:
        if workers > 1:
            completed = 0
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = {pool.submit(_process_single_case, item): item for item in work_items}
                for future in as_completed(futures):
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.warning("Worker failed: %s", e)
                        skip_count += 1
                        continue
                    if result is None:
                        skip_count += 1
                    else:
                        rows.append(result)
                        total_processed += 1
                        method = result.get("slice_method", "unknown")
                        method_counts[method] = method_counts.get(method, 0) + 1
                    if completed % 200 == 0:
                        logger.info("Progress: %d/%d cases processed", completed, len(work_items))
        else:
            for i, item in enumerate(work_items):
                result = _process_single_case(item)
                if result is None:
                    skip_count += 1
                else:
                    rows.append(result)
                    total_processed += 1
                    method = result.get("slice_method", "unknown")
                    method_counts[method] = method_counts.get(method, 0) + 1

    eval_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "eval_dataset.csv")
    eval_df.to_csv(out_csv, index=False)

    logger.info("Eval dataset: %d cases -> %s", len(eval_df), out_csv)
    logger.info("Slice methods: %s", method_counts)
    if cache_hits > 0:
        logger.info("skip-existing: reused %d cached rows", cache_hits)
    if skip_count > 0:
        logger.warning("Skipped %d cases (no nodule localisation)", skip_count)

    if generate_overlay_conditions:
        n_bbox = (eval_df["png_bbox_path"] != "").sum()
        n_contour = (eval_df["png_contour_path"] != "").sum()
        logger.info(
            "Overlays generated: bbox=%d, contour=%d (of %d cases)", n_bbox, n_contour, len(eval_df)
        )

        # Save QC grid for first 10 cases
        qc_dir = os.path.join(output_dir, "qc_overlays")
        qc_count = 0
        for _, r in eval_df.head(min(30, len(eval_df))).iterrows():
            if r.get("png_bbox_path") and r.get("png_contour_path"):
                qc_path = os.path.join(qc_dir, f"qc_{r['mode']}_C{r['case_id']:04d}.png")
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
        description="Build VLM evaluation dataset from iTrialSpace manifests.",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        nargs="+",
        required=True,
        help="Path(s) to iTrialSpace manifest CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory for eval dataset and extracted slices.",
    )
    parser.add_argument(
        "--ct-base",
        default=os.path.join(
            os.environ.get("ITRIALSPACE_OUTPUT_DIR")
            or os.environ.get("ITRIALSPACE_DATA_DIR")
            or os.path.expanduser("~/.itrialspace/data"),
            "generated_cts",
        ),
        help="Root directory of NodMAISI generated CTs.",
    )
    parser.add_argument(
        "--mask-base",
        default=os.path.join(
            os.environ.get("ITRIALSPACE_OUTPUT_DIR")
            or os.environ.get("ITRIALSPACE_DATA_DIR")
            or os.path.expanduser("~/.itrialspace/data"),
            "inserted_masks",
        ),
        help="Root directory of inserted masks.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit total number of cases (for testing).",
    )
    parser.add_argument(
        "--use-host-ct",
        action="store_true",
        help="Use original host CTs instead of NodMAISI synthetic CTs.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=list_profiles(),
        help=(
            "Preprocessing profile name controlling CT window, slice plane, "
            "and output format. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help=(
            "Generate bbox, contour, and bbox_contour overlay PNGs alongside "
            "each plain slice. Adds png_bbox_path, png_contour_path, "
            "png_bbox_contour_path columns to the dataset CSV."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse rows from existing eval_dataset.csv when the plain PNG "
            "is still on disk and the CT path has not changed. Only new "
            "cases trigger NIfTI loading. Dramatically speeds up incremental "
            "rebuilds."
        ),
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for slice extraction. "
            "Each worker loads one 512x512x256 CT volume at a time "
            "(~256 MB RAM). Default: 1 (sequential)."
        ),
    )
    args = parser.parse_args()

    build_dataset(
        manifest_paths=args.manifest,
        output_dir=args.output_dir,
        ct_base=args.ct_base,
        mask_base=args.mask_base,
        max_cases=args.max_cases,
        use_synthetic_ct=not args.use_host_ct,
        profile=get_profile(args.profile),
        generate_overlay_conditions=args.overlays,
        skip_existing=args.skip_existing,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
