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
resolver_bridge.py — Thin adapter between iTrialSpace's PathResolver
and the mask insertion engine.

The CohortManifest carries *relative* or *pre-resolved* paths in columns:

    host_ct_path, host_organ_seg_path,
    donor_ct_path, donor_nodule_mask_path, donor_refined_seg_path

If paths are already absolute and exist on disk, we use them directly.
Otherwise, we fall back to PathResolver for resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ResolvedPaths:
    """All paths needed for a single insertion row."""

    host_ct: str
    host_organ_seg: str
    donor_ct: str
    donor_nodule_mask: str
    donor_refined_seg: str = ""

    def validate(self) -> list[str]:
        """Return list of missing files (empty if all OK)."""
        missing = []
        for label, p in [
            ("host_ct", self.host_ct),
            ("host_organ_seg", self.host_organ_seg),
            ("donor_nodule_mask", self.donor_nodule_mask),
        ]:
            if not p or not os.path.isfile(p):
                missing.append(f"{label}: {p}")
        # donor_ct is optional (only needed for CT-level insertion)
        if self.donor_ct and not os.path.isfile(self.donor_ct):
            missing.append(f"donor_ct: {self.donor_ct}")
        return missing


class ResolverBridge:
    """Resolve manifest row paths, using PathResolver when needed.

    Usage::

        bridge = ResolverBridge(path_resolver=my_resolver)
        # or without PathResolver (paths must all be absolute already):
        bridge = ResolverBridge()

        resolved = bridge.resolve(manifest_row)
    """

    def __init__(
        self,
        path_resolver: Optional[object] = None,
        base_dir: Optional[str] = None,
    ):
        """
        Args:
            path_resolver: An iTrialSpace ``PathResolver`` instance (optional).
            base_dir: Fallback base directory for relative-path resolution.
        """
        self._resolver = path_resolver
        self._base_dir = base_dir or ""

    def resolve(self, row: pd.Series) -> ResolvedPaths:
        """Resolve all paths for a manifest row.

        Logic per column:
        1. If the column value is an absolute path and exists → use as-is.
        2. If a PathResolver is available → delegate resolution.
        3. Else prepend base_dir and hope for the best.
        """
        return ResolvedPaths(
            host_ct=self._resolve_one(
                row.get("host_ct_path", ""),
                "ct",
                row.get("host_dataset", ""),
                row,
            ),
            host_organ_seg=self._resolve_one(
                row.get("host_organ_seg_path", ""),
                "organ_seg",
                row.get("host_dataset", ""),
                row,
            ),
            donor_ct=self._resolve_one(
                row.get("donor_ct_path", ""),
                "ct",
                row.get("donor_dataset", ""),
                row,
            ),
            donor_nodule_mask=self._resolve_one(
                row.get("donor_nodule_mask_path", ""),
                "nodule_mask",
                row.get("donor_dataset", ""),
                row,
            ),
            donor_refined_seg=self._resolve_one(
                row.get("donor_refined_seg_path", ""),
                "refined_seg",
                row.get("donor_dataset", ""),
                row,
            ),
        )

    def _resolve_one(
        self,
        raw_value: str,
        kind: str,
        dataset: str,
        row: pd.Series,
    ) -> str:
        """Resolve a single path column value."""
        raw_value = str(raw_value).strip() if pd.notna(raw_value) else ""
        if not raw_value:
            return ""

        # 1. Already absolute and exists
        if os.path.isabs(raw_value) and os.path.exists(raw_value):
            return raw_value

        # 2. PathResolver delegation
        if self._resolver is not None and dataset:
            try:
                resolved = self._try_path_resolver(raw_value, kind, dataset, row)
                if resolved and os.path.exists(resolved):
                    return resolved
            except (KeyError, AttributeError, TypeError):
                pass  # fall through

        # 3. base_dir join
        if self._base_dir:
            joined = os.path.join(self._base_dir, raw_value)
            if os.path.exists(joined):
                return joined

        # 4. Return raw (caller will validate)
        return raw_value

    def _try_path_resolver(
        self,
        raw_value: str,
        kind: str,
        dataset: str,
        row: pd.Series,
    ) -> str:
        """Attempt to resolve via PathResolver API."""
        r = self._resolver
        if kind == "ct":
            return r.resolve_ct_path(dataset, raw_value)  # type: ignore[union-attr]
        elif kind == "organ_seg":
            ct_path = str(
                row.get("host_ct_path", "") if "host" in kind else row.get("donor_ct_path", "")
            )
            ct_id = _extract_ct_id(ct_path)
            return r.resolve_organ_seg_path(dataset, ct_id)  # type: ignore[union-attr]
        elif kind == "nodule_mask":
            ann_id = str(row.get("donor_annotation_id", ""))
            ct_path = str(row.get("donor_ct_path", ""))
            ct_id = _extract_ct_id(ct_path)
            return r.resolve_nodule_mask_path(dataset, ann_id, ct_id)  # type: ignore[union-attr]
        elif kind == "refined_seg":
            ct_path = str(row.get("donor_ct_path", ""))
            ct_id = _extract_ct_id(ct_path)
            return r.resolve_refined_seg_path(dataset, ct_id)  # type: ignore[union-attr]
        return raw_value


def _extract_ct_id(ct_path: str) -> str:
    """Extract ct_id from a path — basename without .nii.gz."""
    basename = os.path.basename(ct_path)
    if basename.endswith(".nii.gz"):
        return basename[:-7]
    return Path(basename).stem
