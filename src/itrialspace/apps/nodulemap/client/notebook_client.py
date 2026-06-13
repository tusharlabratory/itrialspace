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
notebook_client.py — Jupyter helper for NoduleMap.

Supports two modes:
  A) Direct mode — loads artifacts in-process (no server needed)
  B) API mode   — queries a running NoduleMap backend

Usage:
  from itrialspace.apps.nodulemap.client.notebook_client import NoduleMapClient

  # Direct mode (load artifacts locally)
  client = NoduleMapClient(artifact_dir="./nodulemap_artifacts")

  # API mode
  client = NoduleMapClient(api_url="http://localhost:8422")

  # Explore
  client.summary()
  client.scatter_2d()
  client.neighbors("DLCS24_DLCS_0001_01", k=10)
  client.scatter_neighbors("DLCS24_DLCS_0001_01", k=10)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NoduleMapClient:
    """Jupyter-friendly NoduleMap client."""

    def __init__(
        self,
        artifact_dir: str | None = None,
        api_url: str | None = None,
        feature_set: str = "reinsertion_core",
        model: str = "UMAP_2D",
    ):
        self._api_url = api_url
        self._artifact_dir = artifact_dir
        self._feature_set = feature_set
        self._model = model
        self._store = None

        if artifact_dir and not api_url:
            from itrialspace.apps.nodulemap.backend.data_store import DataStore

            self._store = DataStore(artifact_dir, verbose=False)

    def _api_get(self, path: str, params: dict | None = None):
        """Make GET request to API."""
        import requests

        url = f"{self._api_url}{path}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _api_post(self, path: str, data: dict):
        """Make POST request to API."""
        import requests

        url = f"{self._api_url}{path}"
        resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Summary ──────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Show summary of available models and node counts."""
        if self._store:
            models = self._store.available_models
            rows = []
            for m in models:
                meta = self._store.get_metadata(m["model_name"], m["feature_set"])
                emb = self._store.get_embeddings(m["model_name"], m["feature_set"])
                rows.append(
                    {
                        "model": m["model_name"],
                        "feature_set": m["feature_set"],
                        "n_nodes": len(meta),
                        "n_dims": emb.shape[1],
                        "datasets": ", ".join(sorted(meta["dataset"].unique())),
                    }
                )
            return pd.DataFrame(rows)
        else:
            data = self._api_get("/api/models")
            return pd.DataFrame(data["models"])

    # ── Metadata ──────────────────────────────────────────────────────────
    def metadata(self, model: str | None = None, feature_set: str | None = None) -> pd.DataFrame:
        """Get the metadata DataFrame."""
        m = model or self._model
        fs = feature_set or self._feature_set
        if self._store:
            return self._store.get_metadata(m, fs)
        else:
            # API doesn't serve full metadata directly; use graph endpoint
            data = self._api_post(
                "/api/graph",
                {
                    "vis_model": m,
                    "search_model": m,
                    "feature_set": fs,
                    "k": 0,
                    "mode": "closest",
                },
            )
            return pd.DataFrame(data["nodes"])

    # ── Node detail ──────────────────────────────────────────────────────
    def node(self, node_id: str) -> dict:
        """Get full metadata for a node."""
        if self._store:
            return self._store.get_node(node_id, self._model, self._feature_set)
        else:
            return self._api_get(
                f"/api/node/{node_id}", {"model": self._model, "feature_set": self._feature_set}
            )

    # ── Neighbors ────────────────────────────────────────────────────────
    def neighbors(
        self,
        node_id: str,
        k: int = 10,
        model: str | None = None,
        scope: str = "cross_dataset",
        datasets: list[str] | None = None,
        balance_by_dataset: bool = False,
    ) -> pd.DataFrame:
        """Query K nearest neighbors, return as DataFrame."""
        m = model or self._model
        if self._store:
            results = self._store.query_neighbors(
                node_id=node_id,
                model_name=m,
                feature_set=self._feature_set,
                k=k,
                scope=scope,
                datasets=datasets,
                balance_by_dataset=balance_by_dataset,
            )
        else:
            params = {
                "model": m,
                "feature_set": self._feature_set,
                "k": k,
                "scope": scope,
                "balance_by_dataset": balance_by_dataset,
            }
            if datasets:
                params["datasets"] = ",".join(datasets)
            data = self._api_get(f"/api/node/{node_id}/neighbors", params)
            results = data["neighbors"]

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        cols = [
            "rank",
            "node_id",
            "dataset",
            "similarity",
            "distance",
            "reinsertion_nodule_diam_mm",
            "reinsertion_lobe",
            "label_text",
        ]
        cols = [c for c in cols if c in df.columns]
        return df[cols]

    # ── 2D Scatter Plot ──────────────────────────────────────────────────
    def scatter_2d(
        self,
        color_by: str = "dataset",
        model: str | None = None,
        figsize: tuple = (12, 8),
        alpha: float = 0.5,
        s: int = 5,
        filters: dict | None = None,
    ):
        """
        Plot 2D embedding scatter (matplotlib).

        Args:
            color_by: Column to color nodes by (dataset, label_text, reinsertion_lobe, etc.)
            model: 2D model to use (default: UMAP_2D)
            figsize: Figure size
            alpha: Point opacity
            s: Point size
            filters: Optional filter dict
        """
        import matplotlib.pyplot as plt

        m = model or self._model

        if self._store:
            meta = self._store.get_metadata(m, self._feature_set)
            pos = self._store.get_positions(m, self._feature_set)

            if filters:
                mask = self._store._apply_filters(meta, filters)
                meta = meta[mask].reset_index(drop=True)
                pos = pos[mask]
        else:
            data = self._api_post(
                "/api/graph",
                {
                    "vis_model": m,
                    "search_model": m,
                    "feature_set": self._feature_set,
                    "k": 0,
                    "color_by": color_by,
                    "filters": filters,
                },
            )
            nodes = data["nodes"]
            meta = pd.DataFrame(nodes)
            pos = np.column_stack([meta["x"].values, meta["y"].values])

        # Color mapping
        PALETTES = {
            "dataset": {
                "DLCS24": "#4a9eff",
                "LUNA25": "#f97316",
                "LUNA16": "#10b981",
                "LUNGx": "#ef4444",
                "LNDbv4": "#a855f7",
                "NSCLCR": "#ec4899",
                "IMDCT": "#06b6d4",
            },
            "label_text": {"malignant": "#ef4444", "benign": "#10b981", "unlabelled": "#888888"},
        }
        pal = PALETTES.get(color_by, {})

        if color_by in meta.columns:
            groups = meta[color_by].fillna("unknown").values
        else:
            groups = np.full(len(meta), "unknown")

        unique_groups = sorted(set(groups))
        cmap = plt.cm.get_cmap("tab20", len(unique_groups))

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
        ax.set_facecolor("#0f1117")

        for i, g in enumerate(unique_groups):
            mask = groups == g
            color = pal.get(g, cmap(i))
            ax.scatter(
                pos[mask, 0],
                pos[mask, 1],
                c=[color],
                label=str(g),
                s=s,
                alpha=alpha,
                edgecolors="none",
            )

        ax.legend(
            fontsize=8, loc="upper right", framealpha=0.5, labelcolor="white", facecolor="#22262f"
        )
        ax.set_title(f"NoduleMap — {m} colored by {color_by}", color="white", fontsize=12)
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
        plt.show()
        return fig

    # ── Neighbor Scatter ─────────────────────────────────────────────────
    def scatter_neighbors(
        self,
        node_id: str,
        k: int = 10,
        model: str | None = None,
        figsize: tuple = (10, 7),
    ):
        """Plot a node and its K neighbors on the 2D map."""
        import matplotlib.pyplot as plt

        m = model or self._model
        nb_df = self.neighbors(node_id, k=k, model=m)

        if self._store:
            meta = self._store.get_metadata(m, self._feature_set)
            pos = self._store.get_positions(m, self._feature_set)
        else:
            data = self._api_post(
                "/api/graph",
                {
                    "vis_model": m,
                    "search_model": m,
                    "feature_set": self._feature_set,
                    "k": 0,
                },
            )
            nodes = data["nodes"]
            meta = pd.DataFrame(nodes)
            pos = np.column_stack([meta["x"].values, meta["y"].values])

        # Find indices
        node_ids = meta["node_id"].values if "node_id" in meta.columns else meta["id"].values
        query_idx = np.where(node_ids == node_id)[0]
        nb_ids = set(nb_df["node_id"].values) if "node_id" in nb_df.columns else set()
        nb_mask = np.isin(node_ids, list(nb_ids))

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
        ax.set_facecolor("#0f1117")

        # All nodes (faded)
        ax.scatter(pos[:, 0], pos[:, 1], c="#333", s=2, alpha=0.2)
        # Neighbors (highlighted)
        ax.scatter(
            pos[nb_mask, 0],
            pos[nb_mask, 1],
            c="#4a9eff",
            s=30,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="Neighbors",
        )
        # Query node (star)
        if len(query_idx) > 0:
            qi = query_idx[0]
            ax.scatter(
                pos[qi, 0],
                pos[qi, 1],
                c="#ef4444",
                s=100,
                marker="*",
                edgecolors="white",
                linewidths=1,
                zorder=4,
                label=node_id,
            )
            # Draw lines to neighbors
            for ni in np.where(nb_mask)[0]:
                ax.plot(
                    [pos[qi, 0], pos[ni, 0]],
                    [pos[qi, 1], pos[ni, 1]],
                    c="#4a9eff",
                    alpha=0.3,
                    linewidth=0.5,
                )

        ax.legend(fontsize=8, labelcolor="white", facecolor="#22262f", framealpha=0.7)
        ax.set_title(f"Neighbors of {node_id} (k={k})", color="white", fontsize=11)
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
        plt.show()
        return fig

    # ── Filters ──────────────────────────────────────────────────────────
    def filters(self) -> dict:
        """Get available filter values."""
        if self._api_url:
            return self._api_get("/api/filters")
        elif self._store:
            meta = self._store.get_metadata(self._model, self._feature_set)
            return {
                "datasets": sorted(meta["dataset"].unique().tolist()),
                "lobes": sorted(meta["reinsertion_lobe"].dropna().unique().tolist()),
                "zones": sorted(meta["reinsertion_lung_zone"].dropna().unique().tolist()),
                "sides": sorted(meta["reinsertion_lung_side"].dropna().unique().tolist()),
            }
        return {}

    # ── Export ──────────────────────────────────────────────────────────
    def export_neighbors(
        self,
        node_id: str,
        k: int = 10,
        fmt: str = "csv",
        output_dir: str = "./nodulemap_exports",
    ) -> str:
        """Export a node's neighbors to file."""
        if self._api_url:
            data = self._api_post(
                "/api/export",
                {
                    "node_id": node_id,
                    "model": self._model,
                    "feature_set": self._feature_set,
                    "k": k,
                    "format": fmt,
                },
            )
            return data.get("path", "")
        else:
            from itrialspace.apps.nodulemap.backend.export import Exporter

            exporter = Exporter(output_dir)
            query_node = self._store.get_node(node_id, self._model, self._feature_set)
            nbs = self._store.query_neighbors(
                node_id=node_id,
                model_name=self._model,
                feature_set=self._feature_set,
                k=k,
            )
            return exporter.export_neighbors(
                query_node,
                nbs,
                self._model,
                self._feature_set,
                k,
                "cross_dataset",
                fmt,
            )
