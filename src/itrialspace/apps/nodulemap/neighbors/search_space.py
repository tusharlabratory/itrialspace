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

# -*- coding: utf-8 -*-
"""
search_space.py — similarity/search backends for NoduleMap.

The map's *layout* (2D embeddings) and its *search* (donor ranking / k-NN) are
deliberately decoupled. Search must run on a faithful, reduced-free space — never on a
2D/16D embedding. Three backends are provided:

  - ``weighted``    (default) — reproduces the clinical donor-matching metric of
                    :class:`itrialspace.query.matcher.ReinsertionMatcher` *exactly* as an
                    L1 (Manhattan) distance on a weighted feature matrix, so NoduleMap's
                    "similar donors" agree with the reinsertion pipeline.
  - ``feature_l2``  — exact L2 on the standardized full feature vector
                    (``features_<feature_set>.npy``).
  - ``embedding``   — L2 on the reduced embedding (legacy; kept for comparison only).

Why L1 for ``weighted``: the matcher score is
``w_cc·|Δcc|/100 + w_pleural·|Δpleural|/max_p + w_diam·|Δdiam|/max_d + lobe_penalty·𝟙[lobe≠]``
— a weighted sum of absolute differences plus a lobe indicator. Encoding each numeric as a
scaled column and the lobe as a one-hot scaled by ``lobe_penalty/2`` makes the **L1 distance
between two rows equal the matcher score** (a one-hot lobe mismatch differs in two positions
→ ``2·(lobe_penalty/2) = lobe_penalty``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from itrialspace.query.matcher import ReinsertionMatcher

DEFAULT_WEIGHTS = dict(ReinsertionMatcher.DEFAULT_WEIGHTS)
BACKENDS = ("weighted", "feature_l2", "embedding")


def metric_for(backend: str) -> str:
    """Distance metric a backend uses: 'l1' for weighted, 'l2' otherwise."""
    return "l1" if backend == "weighted" else "l2"


def weighted_matrix(meta: pd.DataFrame, weights: dict | None = None) -> np.ndarray:
    """Build the weighted feature matrix whose L1 distance == ReinsertionMatcher score.

    Uses the reinsertion_* columns present in NoduleMap metadata. ``max_p``/``max_d`` are
    taken over the whole set (matching the matcher's normalisation).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    cc = pd.to_numeric(meta["reinsertion_lobe_cc_pct"], errors="coerce").to_numpy(float)
    pl = pd.to_numeric(meta["reinsertion_pleural_dist_mm"], errors="coerce").to_numpy(float)
    dm = pd.to_numeric(meta["reinsertion_nodule_diam_mm"], errors="coerce").to_numpy(float)

    finite_p = pl[np.isfinite(pl)]
    finite_d = dm[np.isfinite(dm)]
    max_p = max(float(finite_p.max()) if finite_p.size else 1.0, 1.0)
    max_d = max(float(finite_d.max()) if finite_d.size else 1.0, 1.0)

    cc_fill = float(np.nanmedian(cc)) if np.isfinite(np.nanmedian(cc)) else 0.0
    cc = np.nan_to_num(cc, nan=cc_fill)
    pl = np.nan_to_num(pl, nan=max_p)
    dm = np.nan_to_num(dm, nan=0.0)

    cols = [
        (w["w_cc"] / 100.0) * cc,
        (w["w_pleural"] / max_p) * pl,
        (w["w_diam"] / max_d) * dm,
    ]
    lobe = pd.get_dummies(meta["reinsertion_lobe"].astype(str))
    lobe_mat = lobe.to_numpy(float) * (w["lobe_penalty"] / 2.0)

    return np.column_stack(cols + [lobe_mat]).astype(np.float32)


def build_search_matrix(
    backend: str,
    artifact_dir: str | Path,
    feature_set: str,
    model_name: str,
    meta: pd.DataFrame,
    weights: dict | None = None,
) -> np.ndarray:
    """Return the (n, d) matrix to run k-NN on for the given backend."""
    artifact_dir = Path(artifact_dir)
    if backend == "weighted":
        return weighted_matrix(meta, weights)
    if backend in ("feature_l2", "features"):
        p = artifact_dir / f"features_{feature_set}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Feature matrix not found: {p}. Rebuild embeddings to generate it "
                f"(its-nodulemap build ...)."
            )
        return np.load(p).astype(np.float32)
    if backend == "embedding":
        p = artifact_dir / f"embeddings_{feature_set}__{model_name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Embeddings not found: {p}")
        return np.load(p).astype(np.float32)
    raise ValueError(f"Unknown search backend '{backend}'. Choose from {BACKENDS}.")


def knn(base: np.ndarray, queries: np.ndarray, k: int, metric: str = "l2"):
    """k-NN of ``queries`` against ``base``. Returns (true_distances, indices), (m, k).

    Uses FAISS when available (exact IndexFlat), else sklearn. Distances are returned as
    true (non-squared) distances in both cases.
    """
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    k = int(min(k, base.shape[0]))
    try:
        import faiss

        d = base.shape[1]
        index = faiss.IndexFlat(d, faiss.METRIC_L1) if metric == "l1" else faiss.IndexFlatL2(d)
        index.add(base)
        dists, inds = index.search(queries, k)
        if metric != "l1":
            dists = np.sqrt(np.maximum(dists, 0.0))
        return dists, inds
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(
            n_neighbors=k, metric=("manhattan" if metric == "l1" else "euclidean")
        )
        nn.fit(base)
        return nn.kneighbors(queries)
