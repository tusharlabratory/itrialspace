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
Cross-condition comparison metrics and plots.

Reads per-condition summary.json files produced by the run_conditions wrapper,
computes delta metrics (e.g. accuracy_bbox − accuracy_plain), and generates
side-by-side bar charts.

Reporting convention: accuracy is reported on the cases scored in *every*
(task x condition x model) so all numbers share one N -- use
``metrics.common_case_ids()`` on the result frames before aggregating. For the
released dataset this is the full case count; the paper's 42,382 is the same rule
applied before inference finished. See ``docs/vlm_eval.md`` (section "Reporting
convention") and ``docs/dataset_card.md``.

Usage
-----
    python -m itrialspace.evaluation.vlm_eval.comparison \\
        --results-dir outputs/vlm_eval/lung_axial/biomedclip \\
        --conditions plain bbox contour
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Load per-condition summaries ─────────────────────────────────────────────


def _load_condition_summary(
    results_dir: str,
    condition: str,
    sub_dir: str = "aggregate",
) -> Optional[Dict]:
    """Load summary.json from results_dir/{condition}/{sub_dir}/summary.json."""
    path = os.path.join(results_dir, condition, sub_dir, "summary.json")
    if not os.path.isfile(path):
        logger.warning("Summary not found: %s", path)
        return None
    with open(path) as f:
        data = json.load(f)
    # Handle both flat {task: metrics} and nested {"tasks": {task: metrics}}
    if "tasks" in data and isinstance(data["tasks"], dict):
        return data["tasks"]
    return data


def load_all_condition_summaries(
    results_dir: str,
    conditions: List[str],
    sub_dir: str = "aggregate",
) -> Dict[str, Dict]:
    """Load summaries for all conditions.

    Returns
    -------
    dict
        condition -> {task_name: metrics_dict}
    """
    summaries = {}
    for cond in conditions:
        s = _load_condition_summary(results_dir, cond, sub_dir)
        if s is not None:
            summaries[cond] = s
    return summaries


# ── Delta computation ────────────────────────────────────────────────────────


def compute_deltas(
    summaries: Dict[str, Dict],
    baseline: str = "plain",
    metric_key: str = "accuracy",
) -> pd.DataFrame:
    """Compute per-task deltas relative to the baseline condition.

    Returns a DataFrame:
        task | condition | {metric_key} | delta
    """
    baseline_data = summaries.get(baseline)
    if baseline_data is None:
        raise ValueError(f"Baseline condition '{baseline}' not found")

    rows = []
    for cond, tasks in summaries.items():
        for task_name, metrics in tasks.items():
            value = metrics.get(metric_key, 0.0)
            base_value = baseline_data.get(task_name, {}).get(metric_key, 0.0)
            delta = value - base_value if cond != baseline else 0.0
            rows.append(
                {
                    "task": task_name,
                    "condition": cond,
                    metric_key: value,
                    "delta": delta,
                }
            )

    return pd.DataFrame(rows)


# ── Summary table ────────────────────────────────────────────────────────────


def build_comparison_table(
    summaries: Dict[str, Dict],
    metric_key: str = "accuracy",
) -> pd.DataFrame:
    """Build a pivot table: rows=tasks, columns=conditions.

    Returns
    -------
    pd.DataFrame
        Index = task names, columns = condition names, values = metric.
    """
    records = []
    for cond, tasks in summaries.items():
        for task_name, metrics in tasks.items():
            records.append(
                {
                    "task": task_name,
                    "condition": cond,
                    metric_key: metrics.get(metric_key, 0.0),
                }
            )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.pivot(index="task", columns="condition", values=metric_key)


# ── Plots ────────────────────────────────────────────────────────────────────

_COND_COLORS = {
    "plain": "#4C72B0",
    "bbox": "#55A868",
    "contour": "#C44E52",
    "bbox_contour": "#8172B2",
}


def plot_condition_comparison(
    summaries: Dict[str, Dict],
    title: str,
    output_path: Path,
    metric_key: str = "accuracy",
    figsize: tuple = (10, 5),
) -> Path:
    """Grouped bar chart: tasks on x-axis, bars for each condition."""
    table = build_comparison_table(summaries, metric_key)
    if table.empty:
        return output_path

    tasks = list(table.index)
    conditions = list(table.columns)
    n_tasks = len(tasks)
    n_conds = len(conditions)

    x = np.arange(n_tasks)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(conditions):
        values = [table.loc[t, cond] if t in table.index else 0 for t in tasks]
        color = _COND_COLORS.get(cond, None)
        bars = ax.bar(x + i * width, values, width, label=cond, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_delta_bars(
    deltas_df: pd.DataFrame,
    title: str,
    output_path: Path,
    metric_key: str = "accuracy",
    figsize: tuple = (10, 5),
) -> Path:
    """Bar chart of accuracy deltas relative to plain baseline."""
    non_baseline = deltas_df[deltas_df["condition"] != "plain"].copy()
    if non_baseline.empty:
        return output_path

    tasks = sorted(non_baseline["task"].unique())
    conditions = sorted(non_baseline["condition"].unique())
    n_tasks = len(tasks)
    n_conds = len(conditions)

    x = np.arange(n_tasks)
    width = 0.8 / max(n_conds, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(conditions):
        cond_data = non_baseline[non_baseline["condition"] == cond]
        values = []
        for t in tasks:
            row = cond_data[cond_data["task"] == t]
            values.append(row["delta"].values[0] if len(row) > 0 else 0.0)
        color = _COND_COLORS.get(cond, None)
        bars = ax.bar(x + i * width, values, width, label=cond, color=color)
        for bar, val in zip(bars, values):
            offset = 0.005 if val >= 0 else -0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:+.1%}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )

    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel(f"Δ {metric_key}")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ── Full comparison pipeline ─────────────────────────────────────────────────


def run_comparison(
    results_dir: str,
    conditions: List[str],
    model_name: str = "",
    sub_dir: str = "aggregate",
    output_dir: Optional[str] = None,
) -> Dict:
    """Run the full comparison pipeline.

    Parameters
    ----------
    results_dir : str
        Root directory containing per-condition sub-directories.
    conditions : list of str
        Conditions to compare.
    model_name : str
        For plot titles.
    sub_dir : str
        Sub-directory within each condition dir to find summary.json.
    output_dir : str, optional
        Where to write comparison outputs. Default: results_dir/comparison.

    Returns
    -------
    dict
        Comparison summary with tables and deltas.
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    summaries = load_all_condition_summaries(results_dir, conditions, sub_dir)
    if not summaries:
        logger.error("No condition summaries found in %s", results_dir)
        return {}

    found = list(summaries.keys())
    logger.info("Loaded summaries for conditions: %s", found)

    # Comparison table
    table = build_comparison_table(summaries)
    table_path = os.path.join(output_dir, "accuracy_table.csv")
    table.to_csv(table_path)
    logger.info("Accuracy table -> %s", table_path)

    # Deltas
    deltas = compute_deltas(summaries, baseline="plain")
    deltas_path = os.path.join(output_dir, "deltas.csv")
    deltas.to_csv(deltas_path, index=False)
    logger.info("Deltas -> %s", deltas_path)

    # Plots
    label = f"{model_name} – " if model_name else ""
    plots_dir = os.path.join(output_dir, "plots")

    plot_condition_comparison(
        summaries,
        title=f"{label}Accuracy by Task × Condition",
        output_path=Path(plots_dir) / "accuracy_by_condition.png",
    )

    plot_delta_bars(
        deltas,
        title=f"{label}Accuracy Δ (vs. plain)",
        output_path=Path(plots_dir) / "accuracy_delta.png",
    )

    # Summary JSON
    summary = {
        "conditions": found,
        "accuracy_table": table.to_dict(),
        "deltas": deltas.to_dict(orient="records"),
    }
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Comparison summary -> %s", summary_path)

    print(f"\nComparison table:\n{table.to_string()}")
    print("\nDeltas (vs. plain):")
    non_base = deltas[deltas["condition"] != "plain"]
    if not non_base.empty:
        print(non_base.to_string(index=False))

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare VLM results across image presentation conditions.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root directory with per-condition sub-dirs (e.g. .../biomedclip).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["plain", "bbox", "contour", "bbox_contour"],
        help="Conditions to compare.",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Model name for plot titles.",
    )
    parser.add_argument(
        "--sub-dir",
        default="aggregate",
        help="Sub-directory within each condition dir for summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for comparison results.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_comparison(
        results_dir=args.results_dir,
        conditions=args.conditions,
        model_name=args.model_name,
        sub_dir=args.sub_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
