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
RetrieverEngine — top-level façade for the iTrialSpace Retriever.

Composes FacetedSearch, SimilarityEngine, ReinsertionMatcher, PathResolver,
and NIfTISlicer into a single object for all three deployment modes.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from itrialspace import (
    MatchResult,
    NoduleIndex,
    ReinsertionMatcher,
    ReinsertionTarget,
)
from itrialspace.apps.retriever.search import FacetedSearch, SearchFilters, SearchResult
from itrialspace.apps.retriever.similarity import SimilarityEngine, SimilarityResult
from itrialspace.apps.retriever.slicer import NIfTISlicer, SliceAxis, SliceResult
from itrialspace.io.registry import DatasetRegistry
from itrialspace.site.path_resolver import PathResolver


class RetrieverEngine:
    """
    Unified façade: search + similarity + matching + NIfTI viewing + export.

    Usage:
        engine = RetrieverEngine.from_defaults()
        results = engine.search(SearchFilters(label=1, diameter_min=5))
        similar = engine.find_similar("DLCS24_n0001", k=10)
        match   = engine.find_reinsertion_match(lobe="right_lung_upper_lobe", ...)
        sl      = engine.get_slice("DLCS24", "DLCS_0001", axis="axial", index=128)
    """

    def __init__(
        self,
        index: NoduleIndex,
        path_resolver: PathResolver,
        similarity_weights: Optional[dict] = None,
        matcher_weights: Optional[dict] = None,
        per_dataset_norm: bool = False,
    ):
        self._index = index
        self._resolver = path_resolver
        self._search = FacetedSearch(index)
        self._similarity = SimilarityEngine(
            index, weights=similarity_weights, per_dataset_norm=per_dataset_norm
        )
        self._matcher = ReinsertionMatcher(index, weights=matcher_weights)
        self._slicer = NIfTISlicer()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_defaults(
        cls,
        datasets_yaml: Optional[str] = None,
        paths_yaml: Optional[str] = None,
        base_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> "RetrieverEngine":
        """Load the full NoduleIndex from the default registry.

        Paths are resolved portably via ``itrialspace.config.settings`` when the optional
        ``datasets_yaml`` / ``paths_yaml`` overrides are not given.
        """
        registry = DatasetRegistry.from_yaml(datasets_yaml)
        if verbose:
            print("Loading NoduleIndex …")
        index = NoduleIndex.from_registry(registry, verbose=verbose)
        resolver = PathResolver(config_path=paths_yaml, base_dir=base_dir)

        return cls(index=index, path_resolver=resolver)

    @classmethod
    def from_index(
        cls,
        index: NoduleIndex,
        paths_yaml: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> "RetrieverEngine":
        """Build from a pre-loaded NoduleIndex."""
        resolver = PathResolver(config_path=paths_yaml, base_dir=base_dir)
        return cls(index=index, path_resolver=resolver)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def index(self) -> NoduleIndex:
        return self._index

    @property
    def resolver(self) -> PathResolver:
        return self._resolver

    @property
    def n_nodules(self) -> int:
        return len(self._index)

    @property
    def datasets(self) -> list[str]:
        return self._index.datasets

    # ── A. Faceted search ─────────────────────────────────────────────────────

    def search(self, filters: Optional[SearchFilters] = None, **kwargs) -> SearchResult:
        """Run a faceted search.

        Accepts either a SearchFilters object or keyword args matching
        SearchFilters fields.  Returns SearchResult with matching DataFrame.
        """
        if filters is None:
            filters = SearchFilters(**kwargs)
        return self._search.search(filters)

    @property
    def available_filters(self) -> dict:
        return self._search.available_filters

    @property
    def facet_counts(self) -> dict:
        """Return facet counts for the full (unfiltered) index."""
        return self._search._compute_facets(SearchFilters())

    # ── B. Query-by-example similarity ────────────────────────────────────────

    def find_similar(
        self,
        annotation_id: str,
        k: int = 10,
        exclude_same_patient: bool = True,
        include_datasets: Optional[list[str]] = None,
        exclude_datasets: Optional[list[str]] = None,
        label: Optional[int] = None,
    ) -> list[SimilarityResult]:
        """Find the k most similar nodules to a reference."""
        return self._similarity.find_similar(
            annotation_id=annotation_id,
            k=k,
            exclude_same_patient=exclude_same_patient,
            include_datasets=include_datasets,
            exclude_datasets=exclude_datasets,
            label=label,
        )

    def get_feature_vector(self, annotation_id: str) -> dict[str, float]:
        """Return the normalised feature vector for a nodule."""
        return self._similarity.get_feature_vector(annotation_id)

    # ── C. Anatomy-aware reinsertion matching ─────────────────────────────────

    def find_reinsertion_match(
        self,
        lobe: str,
        lobe_cc_pct: Optional[float] = None,
        pleural_dist_mm: Optional[float] = None,
        diameter_mm: Optional[float] = None,
        label: Optional[int] = None,
        lung_zone: Optional[str] = None,
        lung_side: Optional[str] = None,
        include_datasets: Optional[list[str]] = None,
        exclude_datasets: Optional[list[str]] = None,
        k: int = 10,
    ) -> list[MatchResult]:
        """Find the best reinsertion-matched donor nodules."""
        target = ReinsertionTarget(
            lobe=lobe,
            lobe_cc_pct=lobe_cc_pct,
            pleural_dist_mm=pleural_dist_mm,
            diameter_mm=diameter_mm,
            label=label,
            lung_zone=lung_zone,
            lung_side=lung_side,
            include_datasets=include_datasets or [],
            exclude_datasets=exclude_datasets or [],
        )
        return self._matcher.find_top_k(target, k=k)

    # ── D. CT viewer / NIfTI slicing ──────────────────────────────────────────

    def resolve_paths(self, annotation_id: str) -> dict[str, str]:
        """Resolve all file paths for a nodule."""
        mask = self._index.df["annotation_id"] == annotation_id
        if not mask.any():
            raise KeyError(f"annotation_id '{annotation_id}' not found")
        row = self._index.df.loc[mask.idxmax()]
        return self._resolver.resolve_all_paths(row)

    def get_slice(
        self,
        ct_path: str,
        axis: str = "axial",
        index: Optional[int] = None,
        mask_path: Optional[str] = None,
        window: str = "lung",
    ) -> SliceResult:
        """Extract a 2-D slice from a CT or mask volume."""
        return self._slicer.get_slice(
            ct_path=ct_path,
            axis=SliceAxis(axis),
            index=index,
            mask_path=mask_path,
            window=window,
        )

    def get_nodule_view(
        self,
        annotation_id: str,
        axis: str = "axial",
        window: str = "lung",
        show_mask: bool = True,
    ) -> SliceResult:
        """Convenience: resolve paths and get the slice at the nodule centre."""
        mask = self._index.df["annotation_id"] == annotation_id
        if not mask.any():
            raise KeyError(f"annotation_id '{annotation_id}' not found")
        row = self._index.df.loc[mask.idxmax()]
        paths = self._resolver.resolve_all_paths(row)

        ct_path = paths["ct_path"]
        mask_path = paths.get("nodule_mask_path") if show_mask else None
        coord_x = float(row.get("coordX", 0))
        coord_y = float(row.get("coordY", 0))
        coord_z = float(row.get("coordZ", 0))

        return self._slicer.get_nodule_slice(
            ct_path=ct_path,
            mask_path=mask_path or "",
            coord_x=coord_x,
            coord_y=coord_y,
            coord_z=coord_z,
            axis=SliceAxis(axis),
            window=window,
        )

    def volume_shape(self, ct_path: str) -> tuple[int, int, int]:
        """Return the 3-D shape of a NIfTI volume."""
        return self._slicer.volume_shape(ct_path)

    # ── E. Export to CohortManifest format ────────────────────────────────────

    def export_search_csv(
        self,
        filters: SearchFilters,
        output_path: str,
        include_paths: bool = True,
    ) -> str:
        """Run search and export results to CSV.

        If include_paths=True, resolved absolute paths are added.
        Returns the output path.
        """
        result = self.search(filters)
        df = result.df.copy()

        if include_paths:
            path_rows = []
            for _, row in df.iterrows():
                try:
                    p = self._resolver.resolve_all_paths(row)
                    path_rows.append(p)
                except Exception:
                    path_rows.append({})
            paths_df = pd.DataFrame(path_rows)
            df = pd.concat([df, paths_df], axis=1)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def export_search_json(
        self,
        filters: SearchFilters,
        output_path: str,
        include_paths: bool = True,
    ) -> str:
        """Run search and export results to JSON."""
        import json

        result = self.search(filters)
        df = result.df.copy()

        if include_paths:
            path_rows = []
            for _, row in df.iterrows():
                try:
                    p = self._resolver.resolve_all_paths(row)
                    path_rows.append(p)
                except Exception:
                    path_rows.append({})
            paths_df = pd.DataFrame(path_rows)
            df = pd.concat([df, paths_df], axis=1)

        records = df.to_dict(orient="records")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "n_results": len(records),
                    "filters": filters.to_dict(),
                    "results": records,
                },
                f,
                indent=2,
                default=str,
            )
        return output_path

    # ── Nodule detail ─────────────────────────────────────────────────────────

    def get_nodule_detail(self, annotation_id: str) -> dict:
        """Return full detail dict for a nodule (for UI detail panel)."""
        mask = self._index.df["annotation_id"] == annotation_id
        if not mask.any():
            raise KeyError(f"annotation_id '{annotation_id}' not found")

        row = self._index.df.loc[mask.idxmax()]
        detail = row.to_dict()

        # Add resolved paths
        try:
            detail["_paths"] = self._resolver.resolve_all_paths(row)
        except Exception:
            detail["_paths"] = {}

        # Add path existence flags
        for key, path in detail.get("_paths", {}).items():
            detail[f"_exists_{key}"] = os.path.isfile(str(path)) if path else False

        return detail

    # ── Summary stats ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return summary statistics about the index."""
        return {
            "n_nodules": len(self._index),
            "datasets": self._index.datasets,
            "stats": self._index.stats().to_dict(orient="index"),
        }
