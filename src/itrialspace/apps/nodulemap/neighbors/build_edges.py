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
build_edges.py — compute KNN edge lists for the NoduleMap graph.

Edges encode *similarity in the SEARCH space*, not the 2D layout. The space is selectable
(see neighbors/search_space.py):

  - ``weighted``   (default) clinical donor-matching metric (L1), matches ReinsertionMatcher
  - ``feature_l2`` exact L2 on the standardized full feature vector
  - ``embedding``  L2 on the reduced embedding (legacy; comparison only)

Edge files keep the per-(feature_set, model) name the DataStore expects; only their *content*
changes with ``--space``. Uses FAISS (exact) or sklearn fallback.

Usage (CLI):
    itrialspace-nodulemap edges --model UMAP_2D --feature-set reinsertion_core \\
        --k 5 --space weighted --outdir ./nodulemap_artifacts
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from itrialspace.apps.nodulemap.neighbors import search_space


class EdgeBuilder:
    """Build KNN edge lists from saved artifacts, in a selectable search space."""

    def __init__(self, artifact_dir: str, verbose: bool = True):
        self._dir = Path(artifact_dir)
        self._verbose = verbose

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

    def build(
        self,
        model_name: str,
        feature_set: str = "reinsertion_core",
        k: int = 5,
        mode: Literal["closest", "farthest"] = "closest",
        scope: Literal["cross_dataset", "within_dataset"] = "cross_dataset",
        space: str = "weighted",
    ) -> str:
        """Build edge list and save to parquet. Returns output path."""
        tag = f"{feature_set}__{model_name}"

        # Metadata (node_id, dataset + reinsertion_* needed by the weighted space)
        meta_path = self._dir / f"metadata_{tag}.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        meta = pd.read_parquet(meta_path)

        matrix = search_space.build_search_matrix(space, self._dir, feature_set, model_name, meta)
        metric = search_space.metric_for(space)
        assert len(meta) == matrix.shape[0], "Metadata/matrix length mismatch"
        self._log(f"Search space '{space}' (metric={metric}) matrix: {matrix.shape}")

        if scope == "within_dataset":
            edges = self._build_within_dataset(matrix, meta, k, mode, metric)
        else:
            edges = self._build_cross_dataset(matrix, meta, k, mode, metric)

        out_name = f"edges_{tag}_k{k}_{mode}_{scope}.parquet"
        out_path = self._dir / out_name
        edges.to_parquet(out_path, index=False)
        self._log(f"Saved {len(edges)} edges → {out_path}")
        return str(out_path)

    def _build_cross_dataset(self, matrix, meta, k, mode, metric) -> pd.DataFrame:
        """All-vs-all KNN search."""
        n = matrix.shape[0]
        actual_k = min(k + 1, n)  # +1 because the query includes self
        self._log(f"Cross-dataset KNN (k={k}, mode={mode}, n={n}) …")
        t0 = time.time()
        if mode == "farthest":
            distances, indices = self._farthest_neighbors_brute(matrix, actual_k, metric)
        else:
            distances, indices = search_space.knn(matrix, matrix, actual_k, metric)
        self._log(f"KNN search done in {time.time() - t0:.1f}s")
        return self._to_edge_df(meta, indices, distances, k, mode)

    def _build_within_dataset(self, matrix, meta, k, mode, metric) -> pd.DataFrame:
        """Per-dataset KNN search."""
        all_edges = []
        for ds in meta["dataset"].unique():
            mask = meta["dataset"].values == ds
            idx_map = np.where(mask)[0]
            sub = matrix[mask]
            n = len(sub)
            if n <= 1:
                continue
            actual_k = min(k + 1, n)
            self._log(f"  {ds}: {n} nodes, k={min(k, n - 1)}")
            if mode == "farthest":
                dists, inds = self._farthest_neighbors_brute(sub, actual_k, metric)
            else:
                dists, inds = search_space.knn(sub, sub, actual_k, metric)
            sub_meta = meta.iloc[idx_map].reset_index(drop=True)
            all_edges.append(self._to_edge_df(sub_meta, inds, dists, min(k, n - 1), mode))
        if not all_edges:
            return pd.DataFrame(columns=["src_id", "dst_id", "distance", "similarity", "rank"])
        return pd.concat(all_edges, ignore_index=True)

    def _farthest_neighbors_brute(self, matrix, k, metric):
        """Brute-force farthest-neighbor search (true distances)."""
        from sklearn.metrics import pairwise_distances

        skm = "manhattan" if metric == "l1" else "euclidean"
        n = matrix.shape[0]
        batch = 2000
        all_d = np.zeros((n, k), dtype=np.float32)
        all_i = np.zeros((n, k), dtype=np.int64)
        for s in range(0, n, batch):
            e = min(s + batch, n)
            D = pairwise_distances(matrix[s:e], matrix, metric=skm)
            top_k = np.argpartition(-D, min(k, D.shape[1] - 1), axis=1)[:, :k]
            for i in range(e - s):
                order = top_k[i][np.argsort(-D[i, top_k[i]])]
                all_i[s + i] = order
                all_d[s + i] = D[i, order]
        return all_d, all_i

    def _to_edge_df(self, meta, indices, distances, k, mode) -> pd.DataFrame:
        """Convert KNN output (true distances) to an edge DataFrame."""
        rows = []
        node_ids = meta["node_id"].values
        for i in range(len(meta)):
            rank = 0
            for j in range(indices.shape[1]):
                nb = indices[i, j]
                if nb == i:
                    continue
                dist = float(max(distances[i, j], 0.0))
                rank += 1
                rows.append(
                    {
                        "src_id": node_ids[i],
                        "dst_id": node_ids[nb],
                        "distance": round(dist, 6),
                        "similarity": round(float(np.exp(-dist)), 6),
                        "rank": rank,
                    }
                )
                if rank >= k:
                    break
        return pd.DataFrame(rows)

    def build_multiple_k(
        self,
        model_name: str,
        feature_set: str = "reinsertion_core",
        k_values: list[int] | None = None,
        mode: str = "closest",
        space: str = "weighted",
    ) -> list[str]:
        """Build edges for multiple K values."""
        if k_values is None:
            k_values = [3, 5, 10, 25]
        return [self.build(model_name, feature_set, k=k, mode=mode, space=space) for k in k_values]
