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
Pydantic request / response models for the FastAPI backend.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """POST /search request body."""

    datasets: Optional[list[str]] = None
    exclude_datasets: Optional[list[str]] = None
    label: Optional[int] = Field(None, ge=0, le=1)
    lobe: Optional[list[str]] = None
    lung_zone: Optional[list[str]] = None
    lung_side: Optional[str] = None
    central_peripheral: Optional[str] = None
    diameter_min: Optional[float] = None
    diameter_max: Optional[float] = None
    pleural_distance_min: Optional[float] = None
    pleural_distance_max: Optional[float] = None
    airway_distance_min: Optional[float] = None
    airway_distance_max: Optional[float] = None
    reinsertion_lobe: Optional[list[str]] = None
    reinsertion_zone: Optional[list[str]] = None
    reinsertion_side: Optional[str] = None
    reinsertion_diameter_min: Optional[float] = None
    reinsertion_diameter_max: Optional[float] = None
    reinsertion_pleural_min: Optional[float] = None
    reinsertion_pleural_max: Optional[float] = None
    reinsertion_cc_pct_min: Optional[float] = None
    reinsertion_cc_pct_max: Optional[float] = None
    n_nodules_min: Optional[int] = None
    n_nodules_max: Optional[int] = None
    population_type: Optional[str] = None
    label_source: Optional[str] = None
    limit: Optional[int] = Field(50, ge=1, le=5000)
    offset: int = Field(0, ge=0)
    sample_n: Optional[int] = None
    sample_seed: Optional[int] = None
    sort_by: Optional[str] = None
    sort_ascending: bool = True


class NoduleSummary(BaseModel):
    """Compact nodule representation for search results."""

    annotation_id: str
    dataset: str
    patient_id: str
    ct_path: str
    label: Optional[int] = None
    lobe_name: str = ""
    lung_side: str = ""
    lung_zone: str = ""
    central_peripheral: str = ""
    nodule_mean_diam_mm: float = 0.0
    pleural_distance_mm: Optional[float] = None
    reinsertion_lobe: str = ""
    reinsertion_nodule_diam_mm: float = 0.0
    reinsertion_pleural_dist_mm: Optional[float] = None
    reinsertion_lobe_cc_pct: float = 0.0


class SearchResponse(BaseModel):
    """POST /search response."""

    total_matching: int
    returned: int
    results: list[NoduleSummary]
    facet_counts: Optional[dict] = None


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


class SimilarityRequest(BaseModel):
    """POST /similar request body."""

    annotation_id: str
    k: int = Field(10, ge=1, le=500)
    exclude_same_patient: bool = True
    include_datasets: Optional[list[str]] = None
    exclude_datasets: Optional[list[str]] = None
    label: Optional[int] = None


class SimilarNodule(BaseModel):
    """Single similar nodule result."""

    annotation_id: str
    dataset: str
    distance: float
    rank: int
    lobe_name: str = ""
    nodule_mean_diam_mm: float = 0.0
    label: Optional[int] = None
    feature_deltas: dict = {}


class SimilarityResponse(BaseModel):
    """POST /similar response."""

    reference_id: str
    k: int
    results: list[SimilarNodule]


# ---------------------------------------------------------------------------
# Reinsertion matcher
# ---------------------------------------------------------------------------


class MatcherRequest(BaseModel):
    """POST /matcher request body."""

    lobe: str
    lobe_cc_pct: Optional[float] = None
    pleural_dist_mm: Optional[float] = None
    diameter_mm: Optional[float] = None
    label: Optional[int] = None
    lung_zone: Optional[str] = None
    lung_side: Optional[str] = None
    include_datasets: Optional[list[str]] = None
    exclude_datasets: Optional[list[str]] = None
    k: int = Field(10, ge=1, le=500)


class MatchedNodule(BaseModel):
    """A single match result."""

    annotation_id: str
    dataset: str
    ct_path: str
    score: float
    lobe: str
    diameter_mm: float
    pleural_mm: Optional[float] = None
    lobe_cc_pct: float = 0.0
    label: Optional[int] = None


class MatcherResponse(BaseModel):
    """POST /matcher response."""

    target_lobe: str
    k: int
    results: list[MatchedNodule]


# ---------------------------------------------------------------------------
# CT slice viewer
# ---------------------------------------------------------------------------


class SliceRequest(BaseModel):
    """GET /ct/slice query params."""

    ct_path: str
    axis: str = "axial"
    index: Optional[int] = None
    mask_path: Optional[str] = None
    window: str = "lung"


class NoduleViewRequest(BaseModel):
    """GET /ct/nodule-view query params."""

    annotation_id: str
    axis: str = "axial"
    window: str = "lung"


class VolumeInfoResponse(BaseModel):
    """GET /ct/info response."""

    shape: list[int]
    spacing_mm: list[float]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    """POST /export request body."""

    filters: SearchRequest
    format: str = "csv"  # "csv" or "json"
    include_paths: bool = True
    filename: Optional[str] = None


# ---------------------------------------------------------------------------
# Nodule detail
# ---------------------------------------------------------------------------


class NoduleDetailResponse(BaseModel):
    """GET /nodule/{annotation_id} response."""

    annotation_id: str
    dataset: str
    patient_id: str
    label: Optional[int] = None
    detail: dict
    paths: dict = {}
    paths_exist: dict = {}


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    version: str
    n_nodules: int
    datasets: list[str]


class FiltersInfoResponse(BaseModel):
    """GET /filters response — available facet values."""

    datasets: list[str]
    lobes: list[str]
    zones: list[str]
    sides: list[str]
    labels: list
    central_peripheral: list[str]
    population_types: list[str]
    label_sources: list[str]
