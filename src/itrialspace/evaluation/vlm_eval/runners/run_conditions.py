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
Run VLM inference under different image presentation conditions.

Wraps the existing per-model runners (BiomedCLIP, LLaVA-Med, MedGemma) and
remaps the image path column in the dataset CSV to point to the appropriate
overlay variant before invoking the runner.

Image conditions:
  plain          — original extracted slice (default, existing behaviour)
  bbox           — plain slice + bounding-box overlay
  contour        — plain slice + segmentation contour overlay
  bbox_contour   — plain slice + both overlays

Usage
-----

Single condition:
    python -m itrialspace.evaluation.vlm_eval.runners.run_conditions \\
        --model biomedclip \\
        --dataset-csv outputs/vlm_eval/lung_axial/eval_dataset.csv \\
        --output-dir  outputs/vlm_eval/lung_axial/biomedclip \\
        --image-condition bbox

All conditions:
    python -m itrialspace.evaluation.vlm_eval.runners.run_conditions \\
        --model biomedclip \\
        --dataset-csv outputs/vlm_eval/lung_axial/eval_dataset.csv \\
        --output-dir  outputs/vlm_eval/lung_axial/biomedclip \\
        --run-all-conditions
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from itrialspace.evaluation.vlm_eval.overlay import VALID_CONDITIONS
from itrialspace.evaluation.vlm_eval.prompt_templates import ALL_TASKS

logger = logging.getLogger(__name__)

# Column mapping: condition -> column in eval_dataset.csv
_CONDITION_COLUMNS = {
    "plain": "png_path",
    "bbox": "png_bbox_path",
    "contour": "png_contour_path",
    "bbox_contour": "png_bbox_contour_path",
}


def _prepare_condition_csv(
    dataset_csv: str,
    condition: str,
    output_dir: str,
    case_ids: Optional[str] = None,
) -> Optional[str]:
    """Create a copy of the dataset CSV with png_path remapped to the condition column.

    Rows where the condition image is missing are dropped. If ``case_ids`` (a
    frozen split file) is given, the dataset is first restricted to that case set.
    Returns the path to the new CSV, or None if no valid rows.
    """
    df = pd.read_csv(dataset_csv)

    if case_ids:
        from itrialspace.evaluation.vlm_eval.splits import filter_to_split

        n0 = len(df)
        df = filter_to_split(df, case_ids)
        logger.info("Restricted to split %s: %d / %d rows", case_ids, len(df), n0)

    source_col = _CONDITION_COLUMNS.get(condition)
    if source_col is None:
        raise ValueError(
            f"Unknown condition '{condition}'. " f"Valid: {list(_CONDITION_COLUMNS.keys())}"
        )

    if condition == "plain":
        if not case_ids:
            # No remapping or filtering needed — png_path already points to the plain slice
            return dataset_csv
        # Filtered plain set — write it out so the runner reads the restricted cases
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, "eval_dataset_plain.csv")
        df.to_csv(out_csv, index=False)
        logger.info("Condition CSV: %s (%d rows)", out_csv, len(df))
        return out_csv

    if source_col not in df.columns:
        logger.error(
            "Column '%s' not found in dataset CSV. " "Did you run build_dataset with --overlays?",
            source_col,
        )
        return None

    # Filter to rows where the overlay exists
    valid_mask = df[source_col].notna() & (df[source_col].astype(str).str.len() > 0)
    valid_df = df[valid_mask].copy()

    if valid_df.empty:
        logger.warning("No valid rows for condition '%s'", condition)
        return None

    n_dropped = len(df) - len(valid_df)
    if n_dropped > 0:
        logger.info(
            "Condition '%s': dropped %d/%d rows (missing overlay)",
            condition,
            n_dropped,
            len(df),
        )

    # Remap png_path to the condition-specific path
    valid_df["png_path_original"] = valid_df["png_path"]
    valid_df["png_path"] = valid_df[source_col]
    valid_df["image_condition"] = condition

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"eval_dataset_{condition}.csv")
    valid_df.to_csv(out_csv, index=False)
    logger.info("Condition CSV: %s (%d rows)", out_csv, len(valid_df))
    return out_csv


def _run_model(
    model_name: str,
    dataset_csv: str,
    output_dir: str,
    task_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_cases: Optional[int] = None,
    skip_existing: bool = False,
    **extra_kwargs,
) -> Dict:
    """Dispatch to the appropriate model runner."""
    if model_name == "biomedclip":
        from itrialspace.evaluation.vlm_eval.run_biomedclip import run_biomedclip

        return run_biomedclip(
            dataset_csv=dataset_csv,
            output_dir=output_dir,
            task_names=task_names,
            device=device,
            cache_dir=cache_dir,
            skip_existing=skip_existing,
        )
    elif model_name == "llava_med":
        from itrialspace.evaluation.vlm_eval.runners.run_llava_med import run_llava_med

        return run_llava_med(
            dataset_csv=dataset_csv,
            output_dir=output_dir,
            task_names=task_names,
            device=device,
            cache_dir=cache_dir,
            max_cases=max_cases,
            skip_existing=skip_existing,
            **{
                k: v
                for k, v in extra_kwargs.items()
                if k in ("model_id", "torch_dtype", "max_new_tokens")
            },
        )
    elif model_name == "medgemma":
        from itrialspace.evaluation.vlm_eval.runners.run_medgemma import run_medgemma

        return run_medgemma(
            dataset_csv=dataset_csv,
            output_dir=output_dir,
            task_names=task_names,
            device=device,
            cache_dir=cache_dir,
            max_cases=max_cases,
            skip_existing=skip_existing,
            **{
                k: v
                for k, v in extra_kwargs.items()
                if k in ("model_id", "torch_dtype", "max_new_tokens")
            },
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. " f"Valid: biomedclip, llava_med, medgemma"
        )


def run_with_conditions(
    model_name: str,
    dataset_csv: str,
    output_dir: str,
    conditions: List[str],
    task_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_cases: Optional[int] = None,
    skip_existing: bool = False,
    case_ids: Optional[str] = None,
    **extra_kwargs,
) -> Dict[str, Dict]:
    """Run a model across multiple image conditions.

    Output structure:
        output_dir/
            plain/
                {mode}/  ...
                aggregate/ ...
            bbox/
                {mode}/  ...
                aggregate/ ...
            contour/
                ...

    Returns
    -------
    dict
        condition_name -> task_metrics dict
    """
    all_results = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {condition}")
        print(f"{'='*60}\n")

        cond_output_dir = os.path.join(output_dir, condition)

        cond_csv = _prepare_condition_csv(
            dataset_csv, condition, cond_output_dir, case_ids=case_ids
        )
        if cond_csv is None:
            logger.warning("Skipping condition '%s' (no valid data)", condition)
            continue

        metrics = _run_model(
            model_name=model_name,
            dataset_csv=cond_csv,
            output_dir=cond_output_dir,
            task_names=task_names,
            device=device,
            cache_dir=cache_dir,
            max_cases=max_cases,
            skip_existing=skip_existing,
            **extra_kwargs,
        )
        all_results[condition] = metrics

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run VLM inference under different image presentation conditions "
            "(plain / bbox / contour / bbox_contour)."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["biomedclip", "llava_med", "medgemma"],
        help="Which model to run.",
    )
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Path to eval_dataset.csv (must have overlay columns if not plain).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory. Sub-dirs created per condition.",
    )
    parser.add_argument(
        "--image-condition",
        nargs="+",
        choices=list(VALID_CONDITIONS),
        default=None,
        help="Image condition(s) to evaluate. Default: plain.",
    )
    parser.add_argument(
        "--run-all-conditions",
        action="store_true",
        help="Run all conditions: plain, bbox, contour, bbox_contour.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(ALL_TASKS.keys()),
        help="Task(s) to evaluate. Default: all.",
    )
    parser.add_argument("--device", default=None, help="PyTorch device.")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit total cases (smoke test).",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse inference results from prior runs for unchanged rows.",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Frozen split file (e.g. splits/release_v1.synthetic.txt): score only these cases.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.run_all_conditions:
        conditions = list(VALID_CONDITIONS)
    elif args.image_condition:
        conditions = args.image_condition
    else:
        conditions = ["plain"]

    results = run_with_conditions(
        model_name=args.model,
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        conditions=conditions,
        task_names=args.tasks,
        device=args.device,
        cache_dir=args.cache_dir,
        max_cases=args.max_cases,
        skip_existing=args.skip_existing,
        case_ids=args.case_ids,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  CONDITION SUMMARY")
    print("=" * 60)
    for cond, metrics in results.items():
        print(f"\n  {cond}:")
        for task, m in metrics.items():
            acc = m.get("accuracy", 0)
            n = m.get("n_samples", 0)
            print(f"    {task}: accuracy={acc:.4f}  (n={n})")


if __name__ == "__main__":
    main()
