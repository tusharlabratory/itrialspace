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
Uncertainty and significance for the VLM benchmark.

- ``bootstrap_ci``: case-resampled confidence interval on accuracy per
  set x model x task x condition.
- ``mcnemar_vs_baseline``: paired McNemar test of each condition against the
  baseline (``plain``) on the cases both scored -- the right test for "did the
  overlay change accuracy?" because the same cases are compared.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

_GROUP = ["set", "model", "task", "condition"]


def bootstrap_ci(
    long: pd.DataFrame,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Percentile bootstrap CI on accuracy per set x model x task x condition."""
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for keys, grp in long.groupby(_GROUP, dropna=False):
        c = grp["correct"].to_numpy(dtype=float)
        n = len(c)
        if n == 0:
            continue
        acc = float(c.mean())
        if n_boot > 0:
            idx = rng.integers(0, n, size=(n_boot, n))
            boot = c[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            lo = hi = acc
        rows.append(
            dict(zip(_GROUP, keys))
            | {
                "accuracy": round(acc, 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def _mcnemar(b: int, c: int) -> float:
    """Two-sided McNemar p-value (exact binomial for small discordant counts)."""
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        from math import comb

        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5**n)
        return min(1.0, 2 * tail)
    # chi-square with continuity correction
    from math import erfc, sqrt

    chi = (abs(b - c) - 1) ** 2 / n
    return erfc(sqrt(chi / 2))  # survival of chi2_1 == erfc(sqrt(x/2))


def mcnemar_vs_baseline(long: pd.DataFrame, baseline: str = "plain") -> pd.DataFrame:
    """Paired McNemar test of each condition vs the baseline, per set x model x task."""
    rows: List[dict] = []
    for (s, m, t), grp in long.groupby(["set", "model", "task"], dropna=False):
        base = grp[grp["condition"] == baseline].set_index("uid")["correct"]
        if base.empty:
            continue
        for cond, cg in grp.groupby("condition"):
            if cond == baseline:
                continue
            cur = cg.set_index("uid")["correct"]
            common = base.index.intersection(cur.index)
            if len(common) == 0:
                continue
            b = int((base.loc[common] & ~cur.loc[common]).sum())  # base right, cond wrong
            c = int((~base.loc[common] & cur.loc[common]).sum())  # base wrong, cond right
            rows.append(
                {
                    "set": s,
                    "model": m,
                    "task": t,
                    "condition": cond,
                    "n_paired": len(common),
                    "base_only_correct": b,
                    "cond_only_correct": c,
                    "acc_delta": round(float(cur.loc[common].mean() - base.loc[common].mean()), 4),
                    "p_value": round(_mcnemar(b, c), 6),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant_0.05"] = out["p_value"] < 0.05
    return out
