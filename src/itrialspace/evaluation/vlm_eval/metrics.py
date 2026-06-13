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
Evaluation metrics for VLM tasks.

Computes accuracy, precision, recall, F1, and confusion matrices.

Reporting convention (evaluation N)
-----------------------------------
Accuracy is reported on the set of cases that have a completed prediction in
*every* (task x condition x model), so all reported numbers share one N -- the
same cases are scored for presence, lobe, and size. Use ``common_case_ids()`` to
take that intersection before aggregating.

For the released dataset this equals the full case count (42,858 synthetic /
13,087 real): every ``eval_dataset.csv`` row is complete for all three tasks. The
original paper reported a smaller synthetic N (42,382) under the *same* rule,
because at analysis time 334 mode-13 cases had slices extracted but inference not
yet run and were excluded to keep N consistent. See
``docs/vlm_eval.md`` (section "Reporting convention") and
``docs/dataset_card.md`` for the full reconciliation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def common_case_ids(
    frames: Iterable[pd.DataFrame],
    use_canonical: bool = True,
    id_col: str = "case_id",
) -> Set[str]:
    """Intersection of case ids across result frames -- the reporting N.

    Enforces the evaluation-N convention (see module docstring): given one result
    DataFrame per (task x condition x model) you want to compare, return the set
    of cases present in *all* of them. Aggregate accuracy only over these cases so
    every reported table shares the same N, instead of silently averaging a
    different case set per task.

    By default the case identity is the **canonical uid** derived from the slice
    path (``splits.canonical_uid``) -- the ``case_id`` column is a non-unique
    positional index and must not be used as a key. Pass ``use_canonical=False``
    only if you have a genuinely unique ``id_col``.

    Parameters
    ----------
    frames : iterable of pd.DataFrame
        Result frames (each with a slice-path column, or ``id_col``).
    use_canonical : bool
        Key on the canonical uid from the slice path (default, correct).
    id_col : str
        Fallback identifier column when ``use_canonical`` is False.

    Returns
    -------
    set of str
        Cases scored in every frame. Empty set if no frames are given.
    """
    from itrialspace.evaluation.vlm_eval.splits import uid_series

    common: Optional[Set[str]] = None
    for df in frames:
        if use_canonical:
            ids = set(uid_series(df))
        else:
            if id_col not in df.columns:
                raise KeyError(f"frame is missing the id column {id_col!r}")
            ids = set(df[id_col].astype(str))
        common = ids if common is None else (common & ids)
    return common or set()


def compute_binary_metrics(
    y_true: List[str],
    y_pred: List[str],
    positive_label: str = "present",
) -> Dict[str, Any]:
    """Compute binary classification metrics for the presence task."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, "absent"])

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute multiclass metrics (accuracy + confusion matrix)."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class accuracy
    per_class = {}
    for i, lbl in enumerate(labels):
        mask = [t == lbl for t in y_true]
        if sum(mask) > 0:
            class_preds = [p for p, m in zip(y_pred, mask) if m]
            class_acc = sum(1 for p in class_preds if p == lbl) / len(class_preds)
            per_class[lbl] = {"accuracy": float(class_acc), "n_samples": sum(mask)}

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": per_class,
        "n_samples": len(y_true),
    }


def compute_grouped_accuracy(
    df: pd.DataFrame,
    pred_col: str,
    gt_col: str,
    group_col: str,
) -> Dict[str, Dict[str, Any]]:
    """Compute accuracy grouped by a categorical column."""
    result = {}
    for group_val, grp in df.groupby(group_col):
        correct = (grp[pred_col] == grp[gt_col]).sum()
        total = len(grp)
        result[str(group_val)] = {
            "accuracy": float(correct / total) if total > 0 else 0.0,
            "n_samples": int(total),
            "n_correct": int(correct),
        }
    return result
