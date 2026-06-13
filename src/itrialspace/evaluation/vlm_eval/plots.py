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
Plotting utilities for VLM evaluation results.

Generates confusion matrix heatmaps and accuracy bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    title: str,
    output_path: Path,
    figsize: tuple = (8, 6),
) -> Path:
    """Save a confusion matrix heatmap to disk."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_accuracy_bar(
    task_accuracies: Dict[str, float],
    title: str,
    output_path: Path,
    figsize: tuple = (8, 5),
) -> Path:
    """Save an accuracy bar chart across tasks."""
    tasks = list(task_accuracies.keys())
    accs = [task_accuracies[t] for t in tasks]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(tasks, accs, color=sns.color_palette("muted", len(tasks)))
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_grouped_accuracy(
    grouped: Dict[str, Dict],
    group_name: str,
    title: str,
    output_path: Path,
    figsize: tuple = (10, 5),
) -> Path:
    """Save a grouped accuracy bar chart (e.g. accuracy by mode or dataset)."""
    keys = sorted(grouped.keys())
    accs = [grouped[k]["accuracy"] for k in keys]
    counts = [grouped[k]["n_samples"] for k in keys]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(keys, accs, color=sns.color_palette("Set2", len(keys)))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(group_name)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    for bar, acc, n in zip(bars, accs, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
