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
LabelNormaliser — maps each dataset's native label column(s)
to a unified int | None value:
    0 = benign
    1 = malignant
    None = unknown / unlabelled
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class LabelNormaliser:
    """
    Per-dataset rules for producing a unified binary label.

    Usage:
        normaliser = LabelNormaliser("DLCS24")
        label = normaliser.normalise(row)   # row is a pandas Series
    """

    # Maps dataset → callable(row) → int | None
    _RULES: dict[str, callable] = {}

    def __init__(self, dataset: str):
        if dataset not in self._RULES:
            raise ValueError(f"Unknown dataset '{dataset}'. " f"Known: {list(self._RULES.keys())}")
        self.dataset = dataset
        self._fn = self._RULES[dataset]

    def normalise(self, row: pd.Series) -> Optional[int]:
        return self._fn(row)

    def normalise_series(self, df: pd.DataFrame) -> pd.Series:
        """Apply to entire DataFrame, return Series of normalised labels."""
        return df.apply(self._fn, axis=1)

    # ── Register all dataset rules ────────────────────────────────────────────

    @classmethod
    def _register(cls):
        def _safe_int(v) -> Optional[int]:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        cls._RULES["DLCS24"] = lambda row: _safe_int(row.get("label"))

        cls._RULES["LUNA25"] = lambda row: _safe_int(row.get("label"))

        cls._RULES["LUNA16"] = lambda row: None  # no malignancy label available

        cls._RULES["LUNGx"] = lambda row: _safe_int(row.get("CADx_label"))

        def _lndbv4(row) -> Optional[int]:
            v = row.get("Malignancy")
            if pd.isna(v):
                return None
            try:
                return 1 if float(v) >= 3.0 else 0
            except (TypeError, ValueError):
                return None

        cls._RULES["LNDbv4"] = _lndbv4

        cls._RULES["NSCLCR"] = lambda row: 1  # 100% cancer cohort

        def _imdct(row) -> Optional[int]:
            v = row.get("CADx_label")
            if pd.isna(v):
                return None
            s = str(v).strip().lower()
            if s == "cancer":
                return 1
            if s == "benign":
                return 0
            return None

        cls._RULES["IMDCT"] = _imdct

    @classmethod
    def available_datasets(cls) -> list[str]:
        return list(cls._RULES.keys())


# Register rules at import time
LabelNormaliser._register()


# ── Standalone helper ─────────────────────────────────────────────────────────


def normalise_label(row: pd.Series, dataset: str) -> Optional[int]:
    """Convenience function — normalise a single row."""
    return LabelNormaliser(dataset).normalise(row)
