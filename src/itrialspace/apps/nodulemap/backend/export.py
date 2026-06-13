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
export.py — export selected nodes / neighbor sets to CSV/JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


class Exporter:
    """Export neighbor sets and selected nodes."""

    def __init__(self, export_dir: str = "./exports"):
        self._dir = Path(export_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def export_neighbors(
        self,
        query_node: dict,
        neighbors: list[dict],
        model_name: str,
        feature_set: str,
        k: int,
        scope: str,
        fmt: str = "csv",
    ) -> str:
        """
        Export a node + its neighbors.

        Returns path to the exported file.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        qid = query_node.get("node_id", "unknown")

        rows = []
        for nb in neighbors:
            rows.append(
                {
                    "query_node_id": qid,
                    "query_dataset": query_node.get("dataset", ""),
                    "query_annotation_id": query_node.get("annotation_id", ""),
                    "neighbor_node_id": nb.get("node_id", ""),
                    "neighbor_dataset": nb.get("dataset", ""),
                    "neighbor_annotation_id": nb.get("annotation_id", ""),
                    "distance": nb.get("distance", 0),
                    "similarity": nb.get("similarity", 0),
                    "rank": nb.get("rank", 0),
                    "neighbor_diameter_mm": nb.get("reinsertion_nodule_diam_mm", None),
                    "neighbor_lobe": nb.get("reinsertion_lobe", ""),
                    "neighbor_label": nb.get("label_text", ""),
                    "model": model_name,
                    "feature_set": feature_set,
                    "k": k,
                    "scope": scope,
                }
            )

        df = pd.DataFrame(rows)

        if fmt == "json":
            out_path = self._dir / f"neighbors_{qid}_{ts}.json"
            payload = {
                "query": query_node,
                "neighbors": neighbors,
                "meta": {
                    "model": model_name,
                    "feature_set": feature_set,
                    "k": k,
                    "scope": scope,
                    "exported_at": ts,
                },
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        else:
            out_path = self._dir / f"neighbors_{qid}_{ts}.csv"
            df.to_csv(out_path, index=False)

        return str(out_path)

    def export_candidate_donors(
        self,
        neighbors: list[dict],
        path_resolver=None,
    ) -> str:
        """
        Export compatible with iTrialSpace Retriever manifest format.

        Columns: donor_annotation_id, donor_dataset, ct_path, mask_path
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        rows = []
        for nb in neighbors:
            row = {
                "donor_annotation_id": nb.get("annotation_id", ""),
                "donor_dataset": nb.get("dataset", ""),
                "ct_path": nb.get("ct_path", ""),
            }
            # Resolve mask path if PathResolver available
            if path_resolver and nb.get("annotation_id"):
                try:
                    mask = path_resolver.resolve_nodule_mask_path(
                        nb["dataset"], nb["annotation_id"]
                    )
                    row["mask_path"] = mask
                except Exception:
                    row["mask_path"] = ""
            else:
                row["mask_path"] = ""

            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self._dir / f"candidate_donors_{ts}.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    def export_selected_nodes(
        self,
        node_ids: list[str],
        metadata: pd.DataFrame,
        fmt: str = "csv",
    ) -> str:
        """Export metadata for selected nodes."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        selected = metadata[metadata["node_id"].isin(node_ids)]

        if fmt == "json":
            out_path = self._dir / f"selected_nodes_{ts}.json"
            selected.to_json(out_path, orient="records", indent=2)
        else:
            out_path = self._dir / f"selected_nodes_{ts}.csv"
            selected.to_csv(out_path, index=False)

        return str(out_path)
