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
SimilarityEngine — query-by-example nearest-neighbour retrieval.

Given a reference nodule (by annotation_id), find the k most similar
nodules using weighted Euclidean distance over the reinsertion feature
space with optional per-dataset z-score normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from itrialspace import NoduleIndex

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# The 12 reinsertion columns used as the similarity feature vector.
# Categorical columns are one-hot-encoded; numeric columns are z-normalised.
_NUMERIC_FEATURES = [
    "reinsertion_lobe_cc_pct",
    "reinsertion_lobe_ml_pct",
    "reinsertion_lobe_ap_pct",
    "reinsertion_lung_cc_pct",
    "reinsertion_lung_ml_pct",
    "reinsertion_lung_ap_pct",
    "reinsertion_pleural_dist_mm",
    "reinsertion_airway_dist_mm",
    "reinsertion_nodule_diam_mm",
]

_CATEGORICAL_FEATURES = [
    "reinsertion_lobe",
    "reinsertion_lung_side",
    "reinsertion_lung_zone",
]

# Default weights (higher = more important in distance calc)
DEFAULT_WEIGHTS = {
    "reinsertion_lobe_cc_pct": 1.0,
    "reinsertion_lobe_ml_pct": 0.8,
    "reinsertion_lobe_ap_pct": 0.8,
    "reinsertion_lung_cc_pct": 0.6,
    "reinsertion_lung_ml_pct": 0.5,
    "reinsertion_lung_ap_pct": 0.5,
    "reinsertion_pleural_dist_mm": 1.2,
    "reinsertion_airway_dist_mm": 0.7,
    "reinsertion_nodule_diam_mm": 1.0,
    # Categorical one-hot penalty (lobe mismatch costs this much)
    "_lobe_mismatch": 3.0,
    "_side_mismatch": 2.0,
    "_zone_mismatch": 1.5,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SimilarityResult:
    """A single similar nodule with its distance."""

    annotation_id: str
    dataset: str
    distance: float
    rank: int
    feature_deltas: dict  # per-feature raw delta for explainability
    row: pd.Series = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"SimilarityResult(rank={self.rank} | d={self.distance:.4f} | "
            f"{self.dataset}/{self.annotation_id})"
        )


# ---------------------------------------------------------------------------
# Similarity engine
# ---------------------------------------------------------------------------


class SimilarityEngine:
    """
    Weighted nearest-neighbour search over reinsertion feature space.

    On construction the engine z-normalises numeric features (globally or
    per-dataset, controlled by *per_dataset_norm*) and caches the result
    matrix for O(n) distance lookups.
    """

    def __init__(
        self,
        index: NoduleIndex,
        weights: Optional[dict[str, float]] = None,
        per_dataset_norm: bool = False,
    ):
        self._index = index
        self._weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self._per_dataset = per_dataset_norm

        # Build normalised feature matrix
        self._df = index.df.copy()
        self._feat_matrix, self._feat_cols = self._build_features()

    # ── Public API ────────────────────────────────────────────────────────────

    def find_similar(
        self,
        annotation_id: str,
        k: int = 10,
        exclude_same_patient: bool = True,
        include_datasets: Optional[list[str]] = None,
        exclude_datasets: Optional[list[str]] = None,
        label: Optional[int] = None,
    ) -> list[SimilarityResult]:
        """Find the k most similar nodules to the given reference.

        Args:
            annotation_id: The reference nodule.
            k: Number of results to return.
            exclude_same_patient: Exclude nodules from the same patient.
            include_datasets: Restrict search to these datasets.
            exclude_datasets: Exclude these datasets.
            label: Filter by label (0, 1, or None for any).

        Returns:
            List of SimilarityResult sorted by distance ascending.
        """
        ref_mask = self._df["annotation_id"] == annotation_id
        if not ref_mask.any():
            raise KeyError(f"annotation_id '{annotation_id}' not found in index")

        ref_idx = ref_mask.idxmax()
        ref_row = self._df.loc[ref_idx]
        ref_vec = self._feat_matrix[ref_idx]

        # Build candidate mask
        cand_mask = ~ref_mask  # exclude self
        if exclude_same_patient:
            cand_mask &= self._df["patient_id"] != ref_row["patient_id"]
        if include_datasets:
            cand_mask &= self._df["dataset"].isin(include_datasets)
        if exclude_datasets:
            cand_mask &= ~self._df["dataset"].isin(exclude_datasets)
        if label is not None:
            cand_mask &= self._df["label"] == label

        cand_indices = self._df.index[cand_mask].to_numpy()
        if len(cand_indices) == 0:
            return []

        # Compute distances
        cand_vecs = self._feat_matrix[cand_indices]
        dists = np.linalg.norm(cand_vecs - ref_vec, axis=1)

        # Top-k
        top_k_local = np.argsort(dists)[:k]
        top_k_global = cand_indices[top_k_local]
        top_k_dists = dists[top_k_local]

        results = []
        for rank, (gi, d) in enumerate(zip(top_k_global, top_k_dists), 1):
            row = self._df.loc[gi]
            deltas = self._compute_deltas(ref_vec, self._feat_matrix[gi])
            results.append(
                SimilarityResult(
                    annotation_id=str(row["annotation_id"]),
                    dataset=str(row["dataset"]),
                    distance=float(d),
                    rank=rank,
                    feature_deltas=deltas,
                    row=row,
                )
            )
        return results

    def get_feature_vector(self, annotation_id: str) -> dict[str, float]:
        """Return the normalised feature vector for a nodule."""
        mask = self._df["annotation_id"] == annotation_id
        if not mask.any():
            raise KeyError(f"annotation_id '{annotation_id}' not found")
        idx = mask.idxmax()
        return dict(zip(self._feat_cols, self._feat_matrix[idx]))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_features(self) -> tuple[np.ndarray, list[str]]:
        """Build the normalised numeric + one-hot feature matrix."""
        df = self._df
        feat_cols: list[str] = []
        feat_arrays: list[np.ndarray] = []

        # Numeric features — z-score normalise
        for col in _NUMERIC_FEATURES:
            if col not in df.columns:
                continue
            raw = df[col].fillna(0.0).to_numpy(dtype=np.float64)
            w = self._weights.get(col, 1.0)

            if self._per_dataset:
                normed = np.zeros_like(raw)
                for ds in df["dataset"].unique():
                    ds_mask = (df["dataset"] == ds).to_numpy()
                    subset = raw[ds_mask]
                    mu, sigma = subset.mean(), subset.std()
                    sigma = max(sigma, 1e-8)
                    normed[ds_mask] = (subset - mu) / sigma
            else:
                mu, sigma = raw.mean(), raw.std()
                sigma = max(sigma, 1e-8)
                normed = (raw - mu) / sigma

            feat_arrays.append(normed * w)
            feat_cols.append(col)

        # Categorical features — one-hot with mismatch penalty
        penalty_map = {
            "reinsertion_lobe": self._weights.get("_lobe_mismatch", 3.0),
            "reinsertion_lung_side": self._weights.get("_side_mismatch", 2.0),
            "reinsertion_lung_zone": self._weights.get("_zone_mismatch", 1.5),
        }
        for col in _CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            penalty = penalty_map.get(col, 1.0)
            dummies = pd.get_dummies(df[col], prefix=col).to_numpy(dtype=np.float64)
            dummies *= penalty
            for i in range(dummies.shape[1]):
                feat_cols.append(f"{col}_{i}")
            feat_arrays.append(dummies)

        matrix = np.column_stack(feat_arrays) if feat_arrays else np.zeros((len(df), 0))
        return matrix, feat_cols

    def _compute_deltas(self, ref_vec: np.ndarray, cand_vec: np.ndarray) -> dict[str, float]:
        """Per-feature delta for explainability."""
        deltas = {}
        for i, col in enumerate(self._feat_cols):
            if i < len(ref_vec):
                deltas[col] = float(cand_vec[i] - ref_vec[i])
        return deltas
