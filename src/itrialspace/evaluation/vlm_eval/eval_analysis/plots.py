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
Figures for the VLM benchmark, drawn from the tidy tables.

Publication style: top/right spines removed, subplots share the y-axis with tick
labels only on the first panel, sequential **shades of blue** for series. Every
function takes already-aggregated tables (from ``tables.py``) and writes a PNG
(+ PDF), degrading gracefully so demo runs still produce valid (smaller) figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Sequential blues (ColorBrewer), light -> dark.
_BLUES5 = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08306b"]
# Conditions in canonical order get progressively darker blues.
_COND_ORDER = ["plain", "bbox", "contour", "bbox_contour"]
_COND_COLORS = {
    "plain": "#c6dbef",
    "bbox": "#6baed6",
    "contour": "#3182bd",
    "bbox_contour": "#08306b",
}
_CHANCE = {"presence": 0.5, "lobe": 0.2, "size": 0.25}

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def _blues(n: int) -> List[str]:
    """n evenly-spaced shades of blue."""
    if n <= 1:
        return ["#3182bd"]
    import matplotlib.cm as cm

    return [matplotlib.colors.to_hex(cm.Blues(x)) for x in np.linspace(0.35, 0.92, n)]


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _order_conditions(conds) -> List[str]:
    present = list(conds)
    return [c for c in _COND_ORDER if c in present] + [c for c in present if c not in _COND_ORDER]


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def bar_accuracy_by_condition(acc: pd.DataFrame, set_name: str, out_path: Path) -> Optional[Path]:
    """Per-model subplot; x = tasks, bars = conditions (shared y; labels on first)."""
    sub = acc[acc["set"] == set_name]
    if sub.empty:
        return None
    models = sorted(sub["model"].unique())
    tasks = sorted(sub["task"].unique())
    conds = _order_conditions(sub["condition"].unique())
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.6 * len(models), 3.6), squeeze=False, sharey=True
    )
    for ax_i, (ax, model) in enumerate(zip(axes[0], models)):
        msub = sub[sub["model"] == model]
        x = np.arange(len(tasks))
        w = 0.8 / max(len(conds), 1)
        for i, c in enumerate(conds):
            vals = [
                msub[(msub["task"] == t) & (msub["condition"] == c)]["accuracy"].mean()
                for t in tasks
            ]
            ax.bar(x + i * w, vals, w, label=c, color=_COND_COLORS.get(c, _blues(len(conds))[i]))
        ax.set_xticks(x + w * (len(conds) - 1) / 2)
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.0)
        ax.set_title(model)
        _despine(ax)
        if ax_i == 0:
            ax.set_ylabel("accuracy")
        else:
            ax.tick_params(labelleft=False)
    axes[0][-1].legend(frameon=False, fontsize=7, title="condition")
    fig.suptitle(f"{set_name}: accuracy by task × condition")
    fig.tight_layout()
    return _save(fig, out_path)


def bar_delta_vs_baseline(
    delta: pd.DataFrame, set_name: str, baseline: str, out_path: Path
) -> Optional[Path]:
    col = f"delta_vs_{baseline}"
    sub = delta[(delta["set"] == set_name) & (delta["condition"] != baseline)]
    if sub.empty or col not in sub.columns:
        return None
    models = sorted(sub["model"].unique())
    tasks = sorted(sub["task"].unique())
    conds = _order_conditions(sub["condition"].unique())
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.6 * len(models), 3.6), squeeze=False, sharey=True
    )
    for ax_i, (ax, model) in enumerate(zip(axes[0], models)):
        msub = sub[sub["model"] == model]
        x = np.arange(len(tasks))
        w = 0.8 / max(len(conds), 1)
        for i, c in enumerate(conds):
            vals = [msub[(msub["task"] == t) & (msub["condition"] == c)][col].mean() for t in tasks]
            ax.bar(x + i * w, vals, w, label=c, color=_COND_COLORS.get(c, _blues(len(conds))[i]))
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(x + w * (len(conds) - 1) / 2)
        ax.set_xticklabels(tasks)
        ax.set_title(model)
        _despine(ax)
        if ax_i == 0:
            ax.set_ylabel(f"Δ accuracy vs {baseline}")
        else:
            ax.tick_params(labelleft=False)
    axes[0][-1].legend(frameon=False, fontsize=7, title="condition")
    fig.suptitle(f"{set_name}: Δ accuracy vs {baseline}")
    fig.tight_layout()
    return _save(fig, out_path)


def heatmap_confusion(
    conf_long: pd.DataFrame, set_name: str, model: str, condition: str, task: str, out_path: Path
) -> Optional[Path]:
    sub = conf_long[
        (conf_long["set"] == set_name)
        & (conf_long["model"] == model)
        & (conf_long["condition"] == condition)
        & (conf_long["task"] == task)
    ]
    if sub.empty:
        return None
    mat = sub.pivot_table(
        index="ground_truth", columns="prediction", values="count", aggfunc="sum", fill_value=0
    )
    labels = sorted(set(mat.index) | set(mat.columns))
    mat = mat.reindex(index=labels, columns=labels, fill_value=0)
    norm = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.1 * len(labels) + 1))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{int(mat.values[i, j])}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if norm.values[i, j] < 0.5 else "white",
            )
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    ax.set_title(f"{set_name} {model} {task} ({condition})", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


def bar_breakdown(
    bd: pd.DataFrame, set_name: str, task: str, condition: str, out_path: Path
) -> Optional[Path]:
    """Accuracy across a breakdown's group values; bars grouped by model (blue shades)."""
    sub = bd[(bd["set"] == set_name) & (bd["task"] == task) & (bd["condition"] == condition)]
    if sub.empty:
        return None
    by = sub["group_by"].iloc[0]
    groups = sorted(sub["group_value"].astype(str).unique())
    models = sorted(sub["model"].unique())
    colors = _blues(len(models))
    x = np.arange(len(groups))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(groups) + 2), 3.8))
    for i, model in enumerate(models):
        msub = sub[sub["model"] == model].copy()
        msub.index = msub["group_value"].astype(str)
        vals = [float(msub.loc[g, "accuracy"]) if g in msub.index else np.nan for g in groups]
        ax.bar(x + i * w, vals, w, label=model, color=colors[i])
    ax.axhline(
        _CHANCE.get(task, 0),
        color="0.5",
        ls=":",
        lw=0.8,
        label=f"chance ({_CHANCE.get(task, 0):.0%})",
    )
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{set_name}: {task} accuracy by {by} ({condition})")
    _despine(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, out_path)
