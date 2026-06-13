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
FacetedSearch — multi-dimensional filtering over the NoduleIndex.

Wraps itrialspace.NoduleQuery with a typed filter dataclass and
result-summary logic for the retriever UI layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from itrialspace import (
    DATASET_NAMES,
    LOBE_NAMES,
    SIDE_NAMES,
    ZONE_NAMES,
    NoduleIndex,
    NoduleQuery,
)

# ---------------------------------------------------------------------------
# Filter specification
# ---------------------------------------------------------------------------


@dataclass
class SearchFilters:
    """Typed, serialisable filter spec for faceted search."""

    # Dataset selection
    datasets: Optional[list[str]] = None  # None = all
    exclude_datasets: Optional[list[str]] = None

    # Label
    label: Optional[int] = None  # 0, 1, or None = any

    # Anatomy
    lobe: Optional[list[str]] = None
    lung_zone: Optional[list[str]] = None
    lung_side: Optional[str] = None
    central_peripheral: Optional[str] = None

    # Size (mm)
    diameter_min: Optional[float] = None
    diameter_max: Optional[float] = None

    # Distance (mm)
    pleural_distance_min: Optional[float] = None
    pleural_distance_max: Optional[float] = None
    airway_distance_min: Optional[float] = None
    airway_distance_max: Optional[float] = None

    # Reinsertion space (preferred for retrieval)
    reinsertion_lobe: Optional[list[str]] = None
    reinsertion_zone: Optional[list[str]] = None
    reinsertion_side: Optional[str] = None
    reinsertion_diameter_min: Optional[float] = None
    reinsertion_diameter_max: Optional[float] = None
    reinsertion_pleural_min: Optional[float] = None
    reinsertion_pleural_max: Optional[float] = None
    reinsertion_cc_pct_min: Optional[float] = None
    reinsertion_cc_pct_max: Optional[float] = None

    # Multi-nodule
    n_nodules_min: Optional[int] = None
    n_nodules_max: Optional[int] = None

    # Population / label source
    population_type: Optional[str] = None  # "screening" | "diagnostic"
    label_source: Optional[str] = None  # "histopathology" | "radiology"

    # Pagination / sampling
    limit: Optional[int] = None
    offset: int = 0
    sample_n: Optional[int] = None
    sample_seed: Optional[int] = None

    # Sort
    sort_by: Optional[str] = None
    sort_ascending: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Search result container
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Wraps a DataFrame of matches + metadata."""

    df: pd.DataFrame
    total_matching: int
    filters_applied: SearchFilters
    facet_counts: Optional[dict] = None

    def __repr__(self) -> str:
        shown = len(self.df)
        return f"SearchResult({shown}/{self.total_matching} shown)"


# ---------------------------------------------------------------------------
# Faceted search engine
# ---------------------------------------------------------------------------

# Label-source / population-type dataset maps (from spec.py)
_LABEL_SOURCE_MAP = {
    "histopathology": ["DLCS24", "LUNA25", "NSCLCR", "IMDCT", "LUNGx"],
    "radiology": ["LUNA16", "LNDbv4"],
}
_POPULATION_TYPE_MAP = {
    "screening": ["DLCS24", "LUNA16", "LUNA25", "LNDbv4"],
    "diagnostic": ["NSCLCR", "IMDCT", "LUNGx"],
}


class FacetedSearch:
    """
    Apply SearchFilters to a NoduleIndex and return SearchResult.

    Builds a NoduleQuery chain from the filter dataclass, runs it, and
    computes facet counts for the UI sidebar.
    """

    def __init__(self, index: NoduleIndex):
        self._index = index

    @property
    def index(self) -> NoduleIndex:
        return self._index

    # ── Main entry ────────────────────────────────────────────────────────────

    def search(self, filters: SearchFilters) -> SearchResult:
        """Apply *filters* and return a SearchResult."""
        q = self._build_query(filters)
        total = q.count()

        # Fetch (optionally sampled) results
        if filters.sample_n is not None:
            q = q.sample(filters.sample_n, seed=filters.sample_seed)

        df = q.fetch()

        # Sort
        if filters.sort_by and filters.sort_by in df.columns:
            df = df.sort_values(filters.sort_by, ascending=filters.sort_ascending).reset_index(
                drop=True
            )

        # Paginate
        if filters.limit is not None:
            start = filters.offset
            df = df.iloc[start : start + filters.limit].reset_index(drop=True)

        facets = self._compute_facets(filters)

        return SearchResult(
            df=df,
            total_matching=total,
            filters_applied=filters,
            facet_counts=facets,
        )

    # ── Facet counts ──────────────────────────────────────────────────────────

    def _compute_facets(self, filters: SearchFilters) -> dict:
        """Compute value counts for each facet dimension.

        Uses the *unfiltered* index so users can see options outside
        their current selection (like an e-commerce sidebar).
        """
        df = self._index.df
        facets: dict = {}

        facets["datasets"] = df["dataset"].value_counts().to_dict()

        if "lobe_name" in df.columns:
            facets["lobes"] = df["lobe_name"].value_counts().to_dict()

        if "lung_zone" in df.columns:
            facets["zones"] = df["lung_zone"].value_counts().to_dict()

        if "lung_side" in df.columns:
            facets["sides"] = df["lung_side"].value_counts().to_dict()

        if "central_peripheral" in df.columns:
            facets["central_peripheral"] = df["central_peripheral"].value_counts().to_dict()

        if "label" in df.columns:
            label_counts: dict = {}
            label_counts["malignant"] = int((df["label"] == 1).sum())
            label_counts["benign"] = int((df["label"] == 0).sum())
            label_counts["unlabelled"] = int(df["label"].isna().sum())
            facets["labels"] = label_counts

        # Size histogram bins
        if "nodule_mean_diam_mm" in df.columns:
            bins = [0, 5, 10, 15, 20, 30, 200]
            lbls = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-30mm", ">30mm"]
            facets["size_buckets"] = (
                pd.cut(df["nodule_mean_diam_mm"], bins=bins, labels=lbls).value_counts().to_dict()
            )

        return facets

    # ── Query builder ─────────────────────────────────────────────────────────

    def _build_query(self, f: SearchFilters) -> NoduleQuery:
        """Translate SearchFilters into a NoduleQuery chain."""
        q = self._index.query()

        # Dataset selection (label-source / population-type → dataset list)
        effective_datasets = self._resolve_datasets(f)
        if effective_datasets is not None:
            q = q.datasets(effective_datasets)
        if f.exclude_datasets:
            q = q.exclude_datasets(f.exclude_datasets)

        # Label
        if f.label is not None:
            q = q.label(f.label)

        # Anatomy
        if f.lobe:
            q = q.lobe(f.lobe)
        if f.lung_zone:
            q = q.lung_zone(f.lung_zone)
        if f.lung_side:
            q = q.lung_side(f.lung_side)
        if f.central_peripheral:
            q = q.central_peripheral(f.central_peripheral)

        # Size
        q = q.diameter(min=f.diameter_min, max=f.diameter_max)

        # Distances
        q = q.pleural_distance(min=f.pleural_distance_min, max=f.pleural_distance_max)
        q = q.airway_distance(min=f.airway_distance_min, max=f.airway_distance_max)

        # Reinsertion-space filters
        if f.reinsertion_lobe:
            q = q.reinsertion_lobe(f.reinsertion_lobe)
        if f.reinsertion_zone:
            q = q.reinsertion_zone(f.reinsertion_zone)
        if f.reinsertion_side:
            q = q.reinsertion_side(f.reinsertion_side)
        q = q.reinsertion_diameter(min=f.reinsertion_diameter_min, max=f.reinsertion_diameter_max)
        q = q.reinsertion_pleural(min=f.reinsertion_pleural_min, max=f.reinsertion_pleural_max)
        q = q.reinsertion_cc_pct(min=f.reinsertion_cc_pct_min, max=f.reinsertion_cc_pct_max)

        # Multi-nodule
        q = q.n_nodules_in_patient(min=f.n_nodules_min, max=f.n_nodules_max)

        return q

    @staticmethod
    def _resolve_datasets(f: SearchFilters) -> Optional[list[str]]:
        """Merge explicit datasets with label-source/population-type filters."""
        explicit = set(f.datasets) if f.datasets else None

        # Label source filter
        ls_set = None
        if f.label_source and f.label_source in _LABEL_SOURCE_MAP:
            ls_set = set(_LABEL_SOURCE_MAP[f.label_source])

        # Population type filter
        pt_set = None
        if f.population_type and f.population_type in _POPULATION_TYPE_MAP:
            pt_set = set(_POPULATION_TYPE_MAP[f.population_type])

        # Intersect all non-None sets
        result = None
        for s in [explicit, ls_set, pt_set]:
            if s is not None:
                result = s if result is None else result & s

        return list(result) if result is not None else None

    # ── Convenience: available options ────────────────────────────────────────

    @property
    def available_filters(self) -> dict:
        """Return the valid values for each categorical filter."""
        return {
            "datasets": DATASET_NAMES,
            "lobes": LOBE_NAMES,
            "zones": ZONE_NAMES,
            "sides": SIDE_NAMES,
            "labels": [0, 1, None],
            "central_peripheral": ["central", "peripheral"],
            "population_types": ["screening", "diagnostic", "all"],
            "label_sources": ["histopathology", "radiology", "all"],
        }
