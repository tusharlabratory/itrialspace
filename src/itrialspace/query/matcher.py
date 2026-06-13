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
ReinsertionMatcher — given a target anatomy, find the best matching donor nodule.

Uses the `reinsertion_*` columns (100% complete across all 8 datasets)
as the match space so no anatomical re-computation is needed at insert time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ReinsertionTarget:
    """
    Describes where you want to place a nodule in a target CT.

    Fields map directly to the reinsertion_* columns in the index.
    Only `lobe` is required — all others are optional soft constraints.
    """

    lobe: str  # required — e.g. 'right_lung_upper_lobe'
    lobe_cc_pct: Optional[float] = None  # craniocaudal % within lobe (0=apex, 100=base)
    pleural_dist_mm: Optional[float] = None  # desired pleural distance
    diameter_mm: Optional[float] = None  # desired nodule diameter
    label: Optional[int] = None  # 0=benign, 1=malignant, None=any
    lung_zone: Optional[str] = None  # 'upper_zone' | 'middle_zone' | 'lower_zone'
    lung_side: Optional[str] = None  # 'left' | 'right'
    exclude_datasets: list[str] = field(default_factory=list)
    include_datasets: list[str] = field(default_factory=list)  # empty = all


@dataclass
class MatchResult:
    """A single match result with its score."""

    annotation_id: str
    dataset: str
    ct_path: str
    score: float  # lower = better match
    lobe: str
    diameter_mm: float
    pleural_mm: Optional[float]
    lobe_cc_pct: float
    label: Optional[int]
    row: pd.Series = field(repr=False)  # full row for downstream use

    def __repr__(self) -> str:
        lbl = {0: "benign", 1: "malignant", None: "unlabelled"}[self.label]
        return (
            f"MatchResult(score={self.score:.3f} | {self.dataset} | "
            f"{self.annotation_id} | {self.diameter_mm:.1f}mm | "
            f"{self.lobe} | {lbl})"
        )


class ReinsertionMatcher:
    """
    Anatomy-based nearest-neighbour matcher over the reinsertion index.

    Match score (lower = better):

        score = w_lobe    * (0 if lobe matches else lobe_penalty)
              + w_cc      * |lobe_cc_pct - target.lobe_cc_pct| / 100
              + w_pleural * |pleural_dist_mm - target.pleural_dist_mm| / max_pleural
              + w_diam    * |diameter_mm - target.diameter_mm| / max_diam

    All weights and penalty values are configurable.
    """

    DEFAULT_WEIGHTS = {
        "w_cc": 1.0,
        "w_pleural": 0.8,
        "w_diam": 0.6,
        "lobe_penalty": 5.0,  # added to score when lobe doesn't match
    }

    def __init__(self, index, weights: Optional[dict] = None):
        """
        Args:
            index: NoduleIndex instance
            weights: override default scoring weights
        """
        from itrialspace.index.nodule_index import NoduleIndex

        if not isinstance(index, NoduleIndex):
            raise TypeError("index must be a NoduleIndex")
        self._df = index.df
        self._w = {**self.DEFAULT_WEIGHTS, **(weights or {})}

    def find_best(self, target: ReinsertionTarget) -> Optional[MatchResult]:
        """Return the single best matching nodule."""
        results = self.find_top_k(target, k=1)
        return results[0] if results else None

    def find_top_k(self, target: ReinsertionTarget, k: int = 10) -> list[MatchResult]:
        """Return the top-k matching nodules, sorted by score ascending."""
        candidates = self._filter_candidates(target)
        if candidates.empty:
            return []

        scores = self._score(candidates, target)
        candidates = candidates.copy()
        candidates["_score"] = scores
        candidates = candidates.nsmallest(k, "_score")

        results = []
        for _, row in candidates.iterrows():
            results.append(
                MatchResult(
                    annotation_id=str(row.get("annotation_id", "")),
                    dataset=str(row.get("dataset", "")),
                    ct_path=str(row.get("ct_path", "")),
                    score=float(row["_score"]),
                    lobe=str(row.get("reinsertion_lobe", "")),
                    diameter_mm=float(row.get("reinsertion_nodule_diam_mm", 0)),
                    pleural_mm=row.get("reinsertion_pleural_dist_mm"),
                    lobe_cc_pct=float(row.get("reinsertion_lobe_cc_pct", 0)),
                    label=(int(row["label"]) if pd.notna(row.get("label")) else None),
                    row=row,
                )
            )
        return results

    def _filter_candidates(self, target: ReinsertionTarget) -> pd.DataFrame:
        """Apply hard filters before scoring."""
        mask = pd.Series(True, index=self._df.index)

        # Dataset filters
        if target.exclude_datasets:
            mask &= ~self._df["dataset"].isin(target.exclude_datasets)
        if target.include_datasets:
            mask &= self._df["dataset"].isin(target.include_datasets)

        # Label filter (hard)
        if target.label is not None:
            mask &= self._df["label"] == target.label

        # Zone filter (hard if specified)
        if target.lung_zone:
            mask &= self._df["reinsertion_lung_zone"] == target.lung_zone

        # Side filter (hard if specified)
        if target.lung_side:
            mask &= self._df["reinsertion_lung_side"] == target.lung_side

        return self._df[mask].copy()

    def _score(self, df: pd.DataFrame, target: ReinsertionTarget) -> pd.Series:
        """Compute match score for each candidate row."""
        w = self._w
        score = pd.Series(0.0, index=df.index)

        # Lobe match (soft penalty if no hard filter was applied)
        lobe_match = df["reinsertion_lobe"] == target.lobe
        score += (~lobe_match).astype(float) * w["lobe_penalty"]

        # Craniocaudal position within lobe
        if target.lobe_cc_pct is not None:
            cc_diff = (df["reinsertion_lobe_cc_pct"] - target.lobe_cc_pct).abs() / 100.0
            score += w["w_cc"] * cc_diff

        # Pleural distance
        if target.pleural_dist_mm is not None:
            max_p = max(df["reinsertion_pleural_dist_mm"].max(), 1.0)
            pl_diff = (
                df["reinsertion_pleural_dist_mm"].fillna(max_p) - target.pleural_dist_mm
            ).abs() / max_p
            score += w["w_pleural"] * pl_diff

        # Diameter
        if target.diameter_mm is not None:
            max_d = max(df["reinsertion_nodule_diam_mm"].max(), 1.0)
            d_diff = (df["reinsertion_nodule_diam_mm"] - target.diameter_mm).abs() / max_d
            score += w["w_diam"] * d_diff

        return score
