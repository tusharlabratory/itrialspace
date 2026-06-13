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
Run LLaVA-Med inference on an evaluation dataset for one or more tasks.

Reuses the shared evaluation dataset, task definitions, metrics, and plots
from the vlm_evaluation framework.  The key difference from the BiomedCLIP
runner is that LLaVA-Med is *generative*: it produces free-text answers that
are normalised into canonical labels via ``parsers.py``.

Usage
-----
    python -m itrialspace.evaluation.vlm_eval.runners.run_llava_med \\
        --dataset-csv outputs/vlm_eval/dataset/eval_dataset.csv \\
        --output-dir  outputs/vlm_eval/llava_med \\
        --tasks presence lobe size

Smoke test (50 cases):
    python -m itrialspace.evaluation.vlm_eval.runners.run_llava_med \\
        --dataset-csv outputs/vlm_eval/dataset/eval_dataset.csv \\
        --output-dir  outputs/vlm_eval/llava_med_smoke \\
        --max-cases 50
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

logger = logging.getLogger(__name__)

from itrialspace.evaluation.vlm_eval.inference_cache import load_cached_results, merge_results
from itrialspace.evaluation.vlm_eval.metrics import (
    compute_binary_metrics,
    compute_grouped_accuracy,
    compute_multiclass_metrics,
)
from itrialspace.evaluation.vlm_eval.models.llava_med import LLaVAMedModel
from itrialspace.evaluation.vlm_eval.parsers import parse_answer
from itrialspace.evaluation.vlm_eval.plots import (
    plot_accuracy_bar,
    plot_confusion_matrix,
    plot_grouped_accuracy,
)
from itrialspace.evaluation.vlm_eval.prompt_templates import (
    ALL_TASKS,
    get_generative_prompt,
    get_task,
)

# ── Per-task inference ───────────────────────────────────────────────────────


def run_task(
    model: LLaVAMedModel,
    df: pd.DataFrame,
    task_name: str,
    output_dir: str,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Run LLaVA-Med inference for a single task."""
    os.makedirs(output_dir, exist_ok=True)

    cached_df = None
    if skip_existing:
        cached_df, df = load_cached_results(output_dir, task_name, df)
        if df.empty:
            logger.info("  %s: all %d rows cached, skipping inference", task_name, len(cached_df))
            return cached_df

    task = get_task(task_name)
    question = get_generative_prompt(task_name)
    gt_col = task.manifest_column

    results: List[Dict] = []
    t0 = time.time()

    for _, row in df.iterrows():
        png_path = row["png_path"]
        gt_label = str(row.get(gt_col, ""))

        if not os.path.isfile(png_path):
            print(f"  WARN: missing PNG {png_path}, skipping")
            continue

        raw_output = model.generate(png_path, question)
        parsed = parse_answer(task_name, raw_output)

        results.append(
            {
                "case_id": row["case_id"],
                "mode": row.get("mode", row.get("dataset", "")),
                "image_path": png_path,
                "task": task_name,
                "prompt": question,
                "raw_output": raw_output,
                "prediction": parsed.label,
                "parse_status": parsed.status,
                "ground_truth": gt_label,
                "correct": parsed.label == gt_label,
            }
        )

        if len(results) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    [{task_name}] {len(results)} / {len(df)} " f"({elapsed:.1f}s)")

    result_df = pd.DataFrame(results)
    result_df = merge_results(cached_df, result_df)
    csv_path = os.path.join(output_dir, f"{task_name}_results.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"  {task_name}: {len(result_df)} results -> {csv_path}")
    return result_df


# ── Metrics + plots ──────────────────────────────────────────────────────────


def evaluate_and_plot(
    result_df: pd.DataFrame,
    task_name: str,
    output_dir: str,
) -> Dict:
    """Compute metrics and generate plots for one task."""
    task = get_task(task_name)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Filter to successfully parsed rows for metrics
    valid = result_df[result_df["parse_status"] == "ok"]
    n_unparsed = len(result_df) - len(valid)

    if valid.empty:
        return {
            "accuracy": 0.0,
            "n_samples": 0,
            "n_unparsed": n_unparsed,
        }

    y_true = valid["ground_truth"].tolist()
    y_pred = valid["prediction"].tolist()

    if task_name == "presence":
        metrics = compute_binary_metrics(y_true, y_pred, positive_label="present")
        cm_labels = ["present", "absent"]
    else:
        cm_labels = task.labels
        metrics = compute_multiclass_metrics(y_true, y_pred, labels=cm_labels)

    metrics["n_unparsed"] = n_unparsed
    metrics["parse_rate"] = float(len(valid) / len(result_df)) if len(result_df) > 0 else 0.0

    # Confusion matrix plot
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        labels=cm_labels,
        title=f"LLaVA-Med – {task_name} confusion matrix",
        output_path=Path(plots_dir) / f"{task_name}_confusion_matrix.png",
    )

    # Grouped accuracy by mode
    if "mode" in valid.columns and valid["mode"].nunique() > 1:
        grouped = compute_grouped_accuracy(valid, "prediction", "ground_truth", "mode")
        metrics["accuracy_by_mode"] = grouped
        plot_grouped_accuracy(
            grouped,
            group_name="Mode",
            title=f"LLaVA-Med – {task_name} accuracy by mode",
            output_path=Path(plots_dir) / f"{task_name}_accuracy_by_mode.png",
        )

    return metrics


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_llava_med(
    dataset_csv: str,
    output_dir: str,
    task_names: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    max_new_tokens: int = 64,
    max_cases: Optional[int] = None,
    skip_existing: bool = False,
    case_ids: Optional[str] = None,
) -> Dict[str, Dict]:
    """Full LLaVA-Med inference + evaluation pipeline with per-mode output.

    ``case_ids`` (optional): path to a frozen split file (see ``splits.py``); when
    set, only cases whose canonical uid is in the split are scored.
    """
    df = pd.read_csv(dataset_csv)
    logger.info("Eval dataset: %d rows from %s", len(df), dataset_csv)
    if case_ids:
        from itrialspace.evaluation.vlm_eval.splits import filter_to_split

        n0 = len(df)
        df = filter_to_split(df, case_ids)
        logger.info("Restricted to split %s: %d / %d rows", case_ids, len(df), n0)

    if max_cases is not None:
        df = df.head(max_cases)
        logger.info("  Limited to %d cases (--max-cases %d)", len(df), max_cases)

    if task_names is None:
        task_names = list(ALL_TASKS.keys())

    model = LLaVAMedModel(
        model_id=model_id,
        device=device,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
    )

    # Run inference on the full dataset (tasks × all cases)
    full_results: Dict[str, pd.DataFrame] = {}
    for tname in task_names:
        logger.info("Task: %s", tname)
        result_df = run_task(model, df, tname, output_dir, skip_existing=skip_existing)
        full_results[tname] = result_df

    # ── Per-mode output ──────────────────────────────────────────────────
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
            result_df = full_results[tname]
            if result_df.empty:
                continue
            mode_slice = result_df[result_df["mode"] == mode_name].copy()
            if mode_slice.empty:
                continue

            csv_path = os.path.join(mode_dir, f"{tname}_results.csv")
            mode_slice.to_csv(csv_path, index=False)

            metrics = evaluate_and_plot(mode_slice, tname, mode_dir)
            mode_metrics[tname] = metrics

        summary_path = os.path.join(mode_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(mode_metrics, f, indent=2, default=str)

    # ── Aggregate output ─────────────────────────────────────────────────
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    aggregate_metrics: Dict[str, Dict] = {}

    for tname in task_names:
        result_df = full_results[tname]
        if result_df.empty:
            continue

        csv_path = os.path.join(agg_dir, f"{tname}_results.csv")
        result_df.to_csv(csv_path, index=False)

        metrics = evaluate_and_plot(result_df, tname, agg_dir)
        aggregate_metrics[tname] = metrics
        logger.info(
            "  %s aggregate accuracy: %.4f  (parsed: %.1f%%, unparsed: %d)",
            tname,
            metrics["accuracy"],
            metrics.get("parse_rate", 0) * 100,
            metrics.get("n_unparsed", 0),
        )

    accs = {t: m["accuracy"] for t, m in aggregate_metrics.items() if m.get("n_samples", 0) > 0}
    if accs:
        plot_accuracy_bar(
            accs,
            title="LLaVA-Med – Accuracy by Task (aggregate)",
            output_path=Path(agg_dir) / "plots" / "accuracy_by_task.png",
        )

    summary = {
        "model": model.model_id,
        "max_new_tokens": max_new_tokens,
        "tasks": aggregate_metrics,
    }
    summary_path = os.path.join(agg_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary -> %s", summary_path)

    return aggregate_metrics


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaVA-Med inference on an iTrialSpace eval dataset.",
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
        help="Task(s) to evaluate. Default: all.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model ID or local path. " f"Default: {LLaVAMedModel.DEFAULT_MODEL_ID}",
    )
    parser.add_argument("--device", default=None, help="PyTorch device.")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory.")
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model precision. Default: float16 on GPU, float32 on CPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per answer.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit total cases (smoke test).",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Frozen split file (e.g. splits/release_v1.synthetic.txt): score only these cases.",
    )
    args = parser.parse_args()

    run_llava_med(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        task_names=args.tasks,
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        max_cases=args.max_cases,
        case_ids=args.case_ids,
    )


if __name__ == "__main__":
    main()
