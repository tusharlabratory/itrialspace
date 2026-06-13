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
Run BiomedCLIP inference on an evaluation dataset for one or more tasks.

Produces per-mode result directories:
    output_dir/
        mode1_controlled_prevalence/
            presence_results.csv
            lobe_results.csv
            size_results.csv
            summary.json
            plots/
        mode2_size_detection_curve/
            ...
        aggregate/
            summary.json
            plots/

Usage:
    python -m itrialspace.evaluation.vlm_eval.run_biomedclip \\
        --dataset-csv outputs/eval_dataset.csv \\
        --output-dir  outputs/biomedclip \\
        --tasks presence lobe size
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from itrialspace.evaluation.vlm_eval.biomedclip_model import BiomedCLIPModel
from itrialspace.evaluation.vlm_eval.inference_cache import load_cached_results, merge_results
from itrialspace.evaluation.vlm_eval.metrics import (
    compute_binary_metrics,
    compute_grouped_accuracy,
    compute_multiclass_metrics,
)
from itrialspace.evaluation.vlm_eval.plots import (
    plot_accuracy_bar,
    plot_confusion_matrix,
    plot_grouped_accuracy,
)
from itrialspace.evaluation.vlm_eval.prompt_templates import ALL_TASKS, TaskDefinition, get_task

_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


def run_task(
    model: BiomedCLIPModel,
    df: pd.DataFrame,
    task: TaskDefinition,
    output_dir: str,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Run inference for a single task across all rows in the eval dataset.

    Parameters
    ----------
    model : BiomedCLIPModel
    df : pd.DataFrame
        Must contain columns: png_path, and task.manifest_column for GT.
    task : TaskDefinition
    output_dir : str
    skip_existing : bool
        If True, reuse results from a prior run for unchanged rows.

    Returns
    -------
    pd.DataFrame
        Results with predictions, scores, and correctness.
    """
    os.makedirs(output_dir, exist_ok=True)

    cached_df = None
    if skip_existing:
        cached_df, df = load_cached_results(output_dir, task.name, df)
        if df.empty:
            logger.info("  %s: all %d rows cached, skipping inference", task.name, len(cached_df))
            return cached_df

    labels = task.labels
    texts = task.texts
    gt_col = task.manifest_column

    # Pre-filter rows with valid PNGs for batching
    valid_rows = []
    for idx, row in df.iterrows():
        png_path = row["png_path"]
        if not os.path.isfile(png_path):
            print(f"  WARN: missing PNG {png_path}, skipping")
            continue
        valid_rows.append(row)

    results = []
    t0 = time.time()

    # Process in batches
    for batch_start in range(0, len(valid_rows), _BATCH_SIZE):
        batch = valid_rows[batch_start : batch_start + _BATCH_SIZE]
        batch_paths = [row["png_path"] for row in batch]

        batch_predictions = model.predict_batch(batch_paths, labels, texts, batch_size=_BATCH_SIZE)

        for row, (best_label, scores) in zip(batch, batch_predictions):
            gt_label = str(row.get(gt_col, ""))
            result = {
                "case_id": row["case_id"],
                "mode": row.get("mode", row.get("dataset", "")),
                "image_path": row["png_path"],
                "task": task.name,
                "prediction": best_label,
                "ground_truth": gt_label,
                "correct": best_label == gt_label,
            }
            for lbl, score in zip(labels, scores):
                result[f"score_{lbl}"] = float(score)
            results.append(result)

        done = min(batch_start + _BATCH_SIZE, len(valid_rows))
        elapsed = time.time() - t0
        print(f"    [{task.name}] {done} / {len(valid_rows)} ({elapsed:.1f}s)")

    result_df = pd.DataFrame(results)
    result_df = merge_results(cached_df, result_df)
    csv_path = os.path.join(output_dir, f"{task.name}_results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"  {task.name}: {len(result_df)} results -> {csv_path}")
    return result_df


def evaluate_and_plot(
    result_df: pd.DataFrame,
    task: TaskDefinition,
    output_dir: str,
) -> Dict:
    """Compute metrics and generate plots for a completed task."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    y_true = result_df["ground_truth"].tolist()
    y_pred = result_df["prediction"].tolist()

    if task.name == "presence":
        metrics = compute_binary_metrics(y_true, y_pred, positive_label="present")
        cm_labels = ["present", "absent"]
    else:
        cm_labels = task.labels
        metrics = compute_multiclass_metrics(y_true, y_pred, labels=cm_labels)

    # Confusion matrix plot
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        labels=cm_labels,
        title=f"BiomedCLIP – {task.name} confusion matrix",
        output_path=Path(plots_dir) / f"{task.name}_confusion_matrix.png",
    )

    # Grouped accuracy by mode (if mode column exists)
    if "mode" in result_df.columns and result_df["mode"].nunique() > 1:
        grouped = compute_grouped_accuracy(result_df, "prediction", "ground_truth", "mode")
        metrics["accuracy_by_mode"] = grouped
        plot_grouped_accuracy(
            grouped,
            group_name="Mode",
            title=f"BiomedCLIP – {task.name} accuracy by mode",
            output_path=Path(plots_dir) / f"{task.name}_accuracy_by_mode.png",
        )

    return metrics


def run_biomedclip(
    dataset_csv: str,
    output_dir: str,
    task_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    skip_existing: bool = False,
    case_ids: Optional[str] = None,
) -> Dict[str, Dict]:
    """Full inference + evaluation pipeline with per-mode output.

    Parameters
    ----------
    case_ids : str, optional
        Path to a frozen split file (see ``splits.py``). If given, only cases
        whose canonical uid is in the split are scored -- use it to reproduce a
        published result on exactly its case set.

    Returns
    -------
    dict
        task_name -> aggregate metrics dict
    """
    df = pd.read_csv(dataset_csv)
    logger.info("Eval dataset: %d rows from %s", len(df), dataset_csv)
    if case_ids:
        from itrialspace.evaluation.vlm_eval.splits import filter_to_split

        n0 = len(df)
        df = filter_to_split(df, case_ids)
        logger.info("Restricted to split %s: %d / %d rows", case_ids, len(df), n0)

    if task_names is None:
        task_names = list(ALL_TASKS.keys())

    model = BiomedCLIPModel(device=device, cache_dir=cache_dir)

    # Run inference on the full dataset (tasks x all cases)
    full_results: Dict[str, pd.DataFrame] = {}
    for tname in task_names:
        task = get_task(tname)
        logger.info("Task: %s (%s)", task.name, task.description)
        result_df = run_task(model, df, task, output_dir, skip_existing=skip_existing)
        full_results[tname] = result_df

    # ── Per-mode output ──────────────────────────────────────────────────
    aggregate_metrics: Dict[str, Dict] = {}

    # result DFs always have "mode" (populated from dataset col for real CTs)
    first_results = next((v for v in full_results.values() if not v.empty), None)
    if (
        first_results is not None
        and "mode" in first_results.columns
        and first_results["mode"].str.len().any()
    ):
        modes = sorted(first_results["mode"].unique())
    else:
        modes = []

    for mode_name in modes:
        mode_dir = os.path.join(output_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        logger.info("Writing per-mode results: %s", mode_name)

        mode_metrics: Dict[str, Dict] = {}
        for tname in task_names:
            task = get_task(tname)
            mode_results = full_results[tname]
            if mode_results.empty:
                continue
            mode_slice = mode_results[mode_results["mode"] == mode_name].copy()
            if mode_slice.empty:
                continue

            # Save per-mode results CSV
            csv_path = os.path.join(mode_dir, f"{tname}_results.csv")
            mode_slice.to_csv(csv_path, index=False)

            # Per-mode metrics and plots
            metrics = evaluate_and_plot(mode_slice, task, mode_dir)
            mode_metrics[tname] = metrics

        # Per-mode summary
        summary_path = os.path.join(mode_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(mode_metrics, f, indent=2, default=str)

    # ── Aggregate output ─────────────────────────────────────────────────
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    for tname in task_names:
        task = get_task(tname)
        result_df = full_results[tname]
        if result_df.empty:
            continue

        # Full results CSV
        csv_path = os.path.join(agg_dir, f"{tname}_results.csv")
        result_df.to_csv(csv_path, index=False)

        metrics = evaluate_and_plot(result_df, task, agg_dir)
        aggregate_metrics[tname] = metrics
        logger.info("  %s aggregate accuracy: %.4f", tname, metrics["accuracy"])

    # Accuracy bar chart
    accs = {t: m["accuracy"] for t, m in aggregate_metrics.items()}
    if accs:
        plot_accuracy_bar(
            accs,
            title="BiomedCLIP – Accuracy by Task (aggregate)",
            output_path=Path(agg_dir) / "plots" / "accuracy_by_task.png",
        )

    # Aggregate summary
    summary_path = os.path.join(agg_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=2, default=str)
    logger.info("Summary -> %s", summary_path)

    return aggregate_metrics


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run BiomedCLIP inference on an iTrialSpace eval dataset.",
    )
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Path to eval_dataset.csv from build_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for result CSVs, plots, and summary.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(ALL_TASKS.keys()),
        help="Task(s) to evaluate. Default: all tasks.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (e.g. cuda, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse results from a prior run for unchanged rows.",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Frozen split file (e.g. splits/release_v1.synthetic.txt): score only these cases.",
    )
    args = parser.parse_args()
    run_biomedclip(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        task_names=args.tasks,
        device=args.device,
        cache_dir=args.cache_dir,
        skip_existing=args.skip_existing,
        case_ids=args.case_ids,
    )


if __name__ == "__main__":
    main()
