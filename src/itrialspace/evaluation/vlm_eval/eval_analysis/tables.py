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
Tables from the tidy long results frame (see ``load.load_long``).

Every function returns a tidy ``pd.DataFrame`` and is a pure group-by, so the
same code serves the full dataset and a demo run. ``accuracy`` is
``mean(correct)``; ``n`` is the case count behind each cell.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

_GROUP = ["set", "model", "task", "condition"]


def _acc(df: pd.DataFrame, group: List[str]) -> pd.DataFrame:
    g = df.groupby(group, dropna=False)["correct"]
    out = g.agg(accuracy="mean", n="size", n_correct="sum").reset_index()
    out["accuracy"] = out["accuracy"].astype(float).round(4)
    out["n_correct"] = out["n_correct"].astype(int)
    return out


def accuracy_table(long: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per set x model x task x condition."""
    return _acc(long, _GROUP)


def delta_table(long: pd.DataFrame, baseline: str = "plain") -> pd.DataFrame:
    """Accuracy and Δ-vs-baseline per condition."""
    acc = accuracy_table(long)
    base = (
        acc[acc["condition"] == baseline]
        .set_index(["set", "model", "task"])["accuracy"]
        .rename("baseline_accuracy")
    )
    out = acc.join(base, on=["set", "model", "task"])
    out["delta_vs_" + baseline] = (out["accuracy"] - out["baseline_accuracy"]).round(4)
    return out


def breakdown_table(long: pd.DataFrame, by: str) -> pd.DataFrame:
    """Accuracy broken down by a metadata column (mode / source_dataset / lobe / size_bucket).

    Rows with a missing value for ``by`` are dropped (e.g. ``mode`` is empty for
    the real set).
    """
    if by not in long.columns:
        return pd.DataFrame()
    sub = long[long[by].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    return _acc(sub, _GROUP + [by]).rename(columns={by: "group_value"}).assign(group_by=by)


def confusion_long(long: pd.DataFrame) -> pd.DataFrame:
    """Confusion counts per set x model x condition x task (ground_truth x prediction)."""
    g = (
        long.groupby(
            ["set", "model", "condition", "task", "ground_truth", "prediction"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    return g


def per_class_accuracy(long: pd.DataFrame) -> pd.DataFrame:
    """Recall per ground-truth class (set x model x condition x task x class)."""
    g = long.groupby(["set", "model", "condition", "task", "ground_truth"], dropna=False)["correct"]
    out = g.agg(accuracy="mean", n="size", n_correct="sum").reset_index()
    out = out.rename(columns={"ground_truth": "class"})
    out["accuracy"] = out["accuracy"].astype(float).round(4)
    out["n_correct"] = out["n_correct"].astype(int)
    return out


# ── Markdown rendering ────────────────────────────────────────────────────────


def accuracy_markdown(
    acc: pd.DataFrame, set_name: str, conditions: Optional[List[str]] = None
) -> str:
    """Render a model x task table (conditions as columns) for one set."""
    sub = acc[acc["set"] == set_name]
    if sub.empty:
        return ""
    conds = conditions or sorted(sub["condition"].unique())
    pivot = sub.pivot_table(index=["model", "task"], columns="condition", values="accuracy")
    header = "| Model | Task | " + " | ".join(conds) + " | best |"
    sep = "|---|---|" + "---:|" * len(conds) + "---|"
    lines = [header, sep]
    for (model, task), row in pivot.iterrows():
        vals = {c: row.get(c) for c in conds}
        cells = [f"{vals[c]*100:.1f}" if pd.notna(vals[c]) else "–" for c in conds]
        numeric = {c: v for c, v in vals.items() if pd.notna(v)}
        best = max(numeric, key=numeric.get) if numeric else "–"
        lines.append(f"| {model} | {task} | " + " | ".join(cells) + f" | {best} |")
    return "\n".join(lines)
