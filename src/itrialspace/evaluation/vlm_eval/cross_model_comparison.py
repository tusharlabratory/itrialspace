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
Cross-model, cross-condition comparison.

Aggregates per-model comparison results (produced by comparison.py) across
all three VLMs and generates a unified summary table, cross-model plots,
and a combined deltas report.

Reporting convention: accuracy is reported on the cases scored in *every*
(task x condition x model) so all numbers share one N. Use
``metrics.common_case_ids()`` on the result frames before aggregating. For the
released dataset this is the full case count (42,858 synthetic / 13,087 real);
the paper's 42,382 is the same rule at an earlier, pre-inference snapshot. See
``docs/vlm_eval.md`` (section "Reporting convention") and ``docs/dataset_card.md``.

Usage
-----
    python -m itrialspace.evaluation.vlm_eval.cross_model_comparison \\
        --eval-base $ITRIALSPACE_DATA_DIR/.../vlm_eval \\
        --models biomedclip llava_med medgemma \\
        --conditions plain bbox contour bbox_contour
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
import seaborn as sns

from itrialspace.evaluation.vlm_eval.comparison import (
    compute_deltas,
    load_all_condition_summaries,
)

logger = logging.getLogger(__name__)

# Model display names for plot labels
_MODEL_LABELS = {
    "biomedclip": "BiomedCLIP",
    "llava_med": "LLaVA-Med",
    "medgemma": "MedGemma",
}

# Model → profile mapping
_MODEL_PROFILES = {
    "biomedclip": "lung_axial",
    "llava_med": "lung_axial",
    "medgemma": "lung_axial_medgemma",
}

_MODEL_COLORS = {
    "biomedclip": "#4C72B0",
    "llava_med": "#55A868",
    "medgemma": "#C44E52",
}

_COND_HATCHES = {
    "plain": "",
    "bbox": "//",
    "contour": "\\\\",
    "bbox_contour": "xx",
}


def load_all_model_summaries(
    eval_base: str,
    models: List[str],
    conditions: List[str],
    sub_dir: str = "aggregate",
) -> Dict[str, Dict[str, Dict]]:
    """Load summaries for all models × conditions.

    Returns
    -------
    dict
        model -> condition -> {task: metrics}
    """
    result = {}
    for model in models:
        profile = _MODEL_PROFILES.get(model, "lung_axial")
        model_dir = os.path.join(eval_base, profile, model)
        if not os.path.isdir(model_dir):
            logger.warning("Model dir not found: %s", model_dir)
            continue
        summaries = load_all_condition_summaries(model_dir, conditions, sub_dir)
        if summaries:
            result[model] = summaries
    return result


def build_cross_model_table(
    all_summaries: Dict[str, Dict[str, Dict]],
    metric_key: str = "accuracy",
) -> pd.DataFrame:
    """Build a multi-index table: rows=model×condition, columns=tasks.

    Returns
    -------
    pd.DataFrame
        MultiIndex (model, condition), columns = tasks.
    """
    records = []
    for model, cond_summaries in all_summaries.items():
        for cond, tasks in cond_summaries.items():
            for task, metrics in tasks.items():
                records.append(
                    {
                        "model": _MODEL_LABELS.get(model, model),
                        "condition": cond,
                        "task": task,
                        metric_key: metrics.get(metric_key, 0.0),
                    }
                )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index=["model", "condition"],
        columns="task",
        values=metric_key,
    )
    return pivot


def build_cross_model_deltas(
    all_summaries: Dict[str, Dict[str, Dict]],
    baseline: str = "plain",
    metric_key: str = "accuracy",
) -> pd.DataFrame:
    """Compute deltas for all models.

    Returns
    -------
    pd.DataFrame
        model | task | condition | accuracy | delta
    """
    frames = []
    for model, cond_summaries in all_summaries.items():
        deltas = compute_deltas(cond_summaries, baseline=baseline, metric_key=metric_key)
        deltas["model"] = _MODEL_LABELS.get(model, model)
        frames.append(deltas)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_cross_model_per_task(
    all_summaries: Dict[str, Dict[str, Dict]],
    output_dir: str,
    metric_key: str = "accuracy",
) -> List[Path]:
    """One chart per task: models on x-axis, grouped bars per condition."""
    # Collect all tasks
    tasks = set()
    for cond_summaries in all_summaries.values():
        for cond, task_metrics in cond_summaries.items():
            tasks |= set(task_metrics.keys())
    tasks = sorted(tasks)

    models = list(all_summaries.keys())
    conditions = set()
    for cond_summaries in all_summaries.values():
        conditions |= set(cond_summaries.keys())
    conditions = sorted(conditions)

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for task in tasks:
        n_models = len(models)
        n_conds = len(conditions)
        x = np.arange(n_models)
        width = 0.8 / max(n_conds, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))
        for j, cond in enumerate(conditions):
            values = []
            for model in models:
                val = 0.0
                if model in all_summaries and cond in all_summaries[model]:
                    val = all_summaries[model][cond].get(task, {}).get(metric_key, 0.0)
                values.append(val)

            color = sns.color_palette("Set2", n_conds)[j]
            hatch = _COND_HATCHES.get(cond, "")
            bars = ax.bar(
                x + j * width,
                values,
                width,
                label=cond,
                color=color,
                hatch=hatch,
                edgecolor="white",
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        labels = [_MODEL_LABELS.get(m, m) for m in models]
        ax.set_xticks(x + width * (n_conds - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task.title()} – All Models × Conditions")
        ax.set_ylim(0, 1.15)
        ax.legend(title="Condition", loc="upper right")
        plt.tight_layout()

        out = Path(output_dir) / f"{task}_cross_model.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    return paths


def plot_cross_model_deltas(
    deltas_df: pd.DataFrame,
    output_dir: str,
    metric_key: str = "accuracy",
) -> List[Path]:
    """One chart per task: model × condition delta bars."""
    if deltas_df.empty:
        return []

    non_base = deltas_df[deltas_df["condition"] != "plain"]
    if non_base.empty:
        return []

    tasks = sorted(non_base["task"].unique())
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for task in tasks:
        task_data = non_base[non_base["task"] == task]
        models = sorted(task_data["model"].unique())
        conditions = sorted(task_data["condition"].unique())
        n_models = len(models)
        n_conds = len(conditions)

        x = np.arange(n_models)
        width = 0.8 / max(n_conds, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))
        for j, cond in enumerate(conditions):
            values = []
            for model in models:
                row = task_data[(task_data["model"] == model) & (task_data["condition"] == cond)]
                values.append(row["delta"].values[0] if len(row) > 0 else 0.0)

            color = sns.color_palette("Set2", n_conds)[j]
            bars = ax.bar(x + j * width, values, width, label=cond, color=color)
            for bar, val in zip(bars, values):
                offset = 0.005 if val >= 0 else -0.015
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:+.1%}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7,
                )

        ax.set_xticks(x + width * (n_conds - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel(f"Δ {metric_key}")
        ax.set_title(f"{task.title()} – Accuracy Δ (vs. plain)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(title="Condition")
        plt.tight_layout()

        out = Path(output_dir) / f"{task}_delta_cross_model.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    return paths


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    output_path: Path,
    figsize: tuple = (12, 6),
) -> Path:
    """Model×condition heatmap showing accuracy per task."""
    if table.empty:
        return output_path

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        table,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_cross_model_comparison(
    eval_base: str,
    models: List[str],
    conditions: List[str],
    output_dir: Optional[str] = None,
    sub_dir: str = "aggregate",
) -> Dict:
    """Run the full cross-model comparison pipeline.

    Parameters
    ----------
    eval_base : str
        Root of vlm_eval/ (e.g. $ITRIALSPACE_DATA_DIR/.../iTrialSpace/vlm_eval).
    models : list of str
        Model names (biomedclip, llava_med, medgemma).
    conditions : list of str
        Image conditions to compare.
    output_dir : str, optional
        Where to write outputs. Default: eval_base/cross_model_comparison.

    Returns
    -------
    dict
        Summary with tables and deltas.
    """
    if output_dir is None:
        output_dir = os.path.join(eval_base, "cross_model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    # Load all summaries
    all_summaries = load_all_model_summaries(eval_base, models, conditions, sub_dir)
    if not all_summaries:
        logger.error("No model summaries found in %s", eval_base)
        return {}

    found_models = list(all_summaries.keys())
    logger.info("Loaded summaries for models: %s", found_models)

    # Cross-model table
    table = build_cross_model_table(all_summaries)
    table_path = os.path.join(output_dir, "cross_model_accuracy.csv")
    table.to_csv(table_path)
    logger.info("Cross-model table -> %s", table_path)
    print(f"\nCross-model accuracy table:\n{table.to_string()}")

    # Cross-model deltas
    deltas = build_cross_model_deltas(all_summaries)
    deltas_path = os.path.join(output_dir, "cross_model_deltas.csv")
    deltas.to_csv(deltas_path, index=False)
    logger.info("Cross-model deltas -> %s", deltas_path)
    non_base = deltas[deltas["condition"] != "plain"]
    if not non_base.empty:
        print(f"\nCross-model deltas (vs. plain):\n{non_base.to_string(index=False)}")

    # Heatmap
    plot_heatmap(
        table,
        title="Accuracy: Model × Condition × Task",
        output_path=Path(plots_dir) / "accuracy_heatmap.png",
    )

    # Per-task cross-model bar charts
    plot_cross_model_per_task(all_summaries, plots_dir)

    # Per-task delta charts
    plot_cross_model_deltas(deltas, plots_dir)

    # Summary JSON
    summary = {
        "models": found_models,
        "conditions": conditions,
        "cross_model_table": (
            table.reset_index().to_dict(orient="records") if not table.empty else []
        ),
        "deltas": deltas.to_dict(orient="records") if not deltas.empty else [],
    }
    summary_path = os.path.join(output_dir, "cross_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary -> %s", summary_path)

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare VLM condition results across all models. "
            "Generates unified accuracy tables, delta reports, and plots."
        ),
    )
    parser.add_argument(
        "--eval-base",
        required=True,
        help="Root of vlm_eval/ directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["biomedclip", "llava_med", "medgemma"],
        help="Models to compare.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["plain", "bbox", "contour", "bbox_contour"],
        help="Conditions to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: eval_base/cross_model_comparison.",
    )
    parser.add_argument(
        "--sub-dir",
        default="aggregate",
        help="Sub-dir within condition dir for summary.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_cross_model_comparison(
        eval_base=args.eval_base,
        models=args.models,
        conditions=args.conditions,
        output_dir=args.output_dir,
        sub_dir=args.sub_dir,
    )


if __name__ == "__main__":
    main()
