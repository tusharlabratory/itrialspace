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
FastAPI backend for the iTrialSpace Retriever.

Endpoints:
    GET  /health              — service health + stats
    GET  /filters             — available filter values
    POST /search              — faceted search
    POST /similar             — query-by-example similarity
    POST /matcher             — reinsertion anatomy matching
    GET  /nodule/{id}         — full nodule detail
    GET  /ct/slice            — extract a 2-D CT slice (PNG)
    GET  /ct/nodule-view      — CT slice at nodule centre (PNG)
    GET  /ct/info             — volume metadata
    POST /export              — export search results (CSV/JSON download)

Launch:
    uvicorn retriever.api.app:create_app --factory --host 0.0.0.0 --port 8421
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from itrialspace.apps.retriever.api.models import (
    ExportRequest,
    FiltersInfoResponse,
    HealthResponse,
    MatchedNodule,
    MatcherRequest,
    MatcherResponse,
    NoduleDetailResponse,
    NoduleSummary,
    SearchRequest,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse,
    SimilarNodule,
    VolumeInfoResponse,
)
from itrialspace.apps.retriever.engine import RetrieverEngine
from itrialspace.apps.retriever.search import SearchFilters

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_engine: Optional[RetrieverEngine] = None


def get_engine() -> RetrieverEngine:
    """Lazy singleton engine loader."""
    global _engine
    if _engine is None:
        print("[Retriever API] Loading NoduleIndex …")
        _engine = RetrieverEngine.from_defaults(verbose=True)
        print(f"[Retriever API] Ready — {_engine.n_nodules:,} nodules loaded")
    return _engine


def create_app(
    engine: Optional[RetrieverEngine] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        engine: optional pre-built engine (for testing or Jupyter embedding).
    """
    global _engine
    if engine is not None:
        _engine = engine

    app = FastAPI(
        title="iTrialSpace Retriever",
        description="Interactive retrieval + visualization for 13k+ lung nodules across 7 datasets.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    def health():
        eng = get_engine()
        return HealthResponse(
            version="0.1.0",
            n_nodules=eng.n_nodules,
            datasets=eng.datasets,
        )

    # ── Filters info ──────────────────────────────────────────────────────────

    @app.get("/filters", response_model=FiltersInfoResponse)
    def filters_info():
        eng = get_engine()
        af = eng.available_filters
        return FiltersInfoResponse(**af)

    # ── Faceted search ────────────────────────────────────────────────────────

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        eng = get_engine()
        sf = SearchFilters(**req.model_dump())
        result = eng.search(sf)
        summaries = _df_to_summaries(result.df)
        return SearchResponse(
            total_matching=result.total_matching,
            returned=len(summaries),
            results=summaries,
            facet_counts=result.facet_counts,
        )

    # ── Similarity ────────────────────────────────────────────────────────────

    @app.post("/similar", response_model=SimilarityResponse)
    def similar(req: SimilarityRequest):
        eng = get_engine()
        try:
            results = eng.find_similar(
                annotation_id=req.annotation_id,
                k=req.k,
                exclude_same_patient=req.exclude_same_patient,
                include_datasets=req.include_datasets,
                exclude_datasets=req.exclude_datasets,
                label=req.label,
            )
        except KeyError as e:
            raise HTTPException(404, detail=str(e))

        items = []
        for r in results:
            row = r.row
            items.append(
                SimilarNodule(
                    annotation_id=r.annotation_id,
                    dataset=r.dataset,
                    distance=round(r.distance, 4),
                    rank=r.rank,
                    lobe_name=str(row.get("lobe_name", "")),
                    nodule_mean_diam_mm=float(row.get("nodule_mean_diam_mm", 0)),
                    label=int(row["label"]) if pd.notna(row.get("label")) else None,
                    feature_deltas={k: round(v, 4) for k, v in r.feature_deltas.items()},
                )
            )
        return SimilarityResponse(
            reference_id=req.annotation_id,
            k=req.k,
            results=items,
        )

    # ── Reinsertion matcher ───────────────────────────────────────────────────

    @app.post("/matcher", response_model=MatcherResponse)
    def matcher(req: MatcherRequest):
        eng = get_engine()
        results = eng.find_reinsertion_match(
            lobe=req.lobe,
            lobe_cc_pct=req.lobe_cc_pct,
            pleural_dist_mm=req.pleural_dist_mm,
            diameter_mm=req.diameter_mm,
            label=req.label,
            lung_zone=req.lung_zone,
            lung_side=req.lung_side,
            include_datasets=req.include_datasets,
            exclude_datasets=req.exclude_datasets,
            k=req.k,
        )
        items = [
            MatchedNodule(
                annotation_id=m.annotation_id,
                dataset=m.dataset,
                ct_path=m.ct_path,
                score=round(m.score, 4),
                lobe=m.lobe,
                diameter_mm=round(m.diameter_mm, 2),
                pleural_mm=round(m.pleural_mm, 2) if m.pleural_mm is not None else None,
                lobe_cc_pct=round(m.lobe_cc_pct, 2),
                label=m.label,
            )
            for m in results
        ]
        return MatcherResponse(
            target_lobe=req.lobe,
            k=req.k,
            results=items,
        )

    # ── Nodule detail ─────────────────────────────────────────────────────────

    @app.get("/nodule/{annotation_id}", response_model=NoduleDetailResponse)
    def nodule_detail(annotation_id: str):
        eng = get_engine()
        try:
            detail = eng.get_nodule_detail(annotation_id)
        except KeyError as e:
            raise HTTPException(404, detail=str(e))

        paths = detail.pop("_paths", {})
        paths_exist = {
            k.replace("_exists_", ""): v for k, v in detail.items() if k.startswith("_exists_")
        }
        for k in list(detail.keys()):
            if k.startswith("_exists_"):
                del detail[k]

        # Convert NaN/NaT to None for JSON serialisation
        cleaned = {}
        for k, v in detail.items():
            if isinstance(v, float) and np.isnan(v):
                cleaned[k] = None
            elif pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v

        return NoduleDetailResponse(
            annotation_id=str(cleaned.get("annotation_id", annotation_id)),
            dataset=str(cleaned.get("dataset", "")),
            patient_id=str(cleaned.get("patient_id", "")),
            label=cleaned.get("label"),
            detail=cleaned,
            paths=paths,
            paths_exist=paths_exist,
        )

    # ── CT slice viewer ───────────────────────────────────────────────────────

    @app.get("/ct/slice")
    def ct_slice(
        ct_path: str,
        axis: str = "axial",
        index: Optional[int] = None,
        mask_path: Optional[str] = None,
        window: str = "lung",
    ):
        eng = get_engine()
        try:
            sl = eng.get_slice(
                ct_path=ct_path,
                axis=axis,
                index=index,
                mask_path=mask_path,
                window=window,
            )
            png = sl.to_png_bytes()
        except FileNotFoundError as e:
            raise HTTPException(404, detail=str(e))
        except Exception as e:
            raise HTTPException(500, detail=str(e))

        return Response(content=png, media_type="image/png")

    @app.get("/ct/nodule-view")
    def ct_nodule_view(
        annotation_id: str,
        axis: str = "axial",
        window: str = "lung",
        show_mask: bool = True,
    ):
        eng = get_engine()
        try:
            sl = eng.get_nodule_view(annotation_id, axis=axis, window=window, show_mask=show_mask)
            png = sl.to_png_bytes()
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, detail=str(e))
        except Exception as e:
            raise HTTPException(500, detail=str(e))

        return Response(content=png, media_type="image/png")

    @app.get("/ct/info", response_model=VolumeInfoResponse)
    def ct_info(ct_path: str):
        eng = get_engine()
        try:
            shape = eng.volume_shape(ct_path)
            spacing = eng._slicer.voxel_spacing(ct_path)
        except FileNotFoundError as e:
            raise HTTPException(404, detail=str(e))
        return VolumeInfoResponse(
            shape=list(shape),
            spacing_mm=[round(s, 3) for s in spacing],
        )

    # ── Export ────────────────────────────────────────────────────────────────

    @app.post("/export")
    def export(req: ExportRequest):
        eng = get_engine()
        sf = SearchFilters(**req.filters.model_dump())

        ext = "csv" if req.format == "csv" else "json"
        fname = req.filename or f"itrialspace_export.{ext}"

        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False, prefix="retriever_")
        tmp.close()

        try:
            if req.format == "csv":
                eng.export_search_csv(sf, tmp.name, include_paths=req.include_paths)
            else:
                eng.export_search_json(sf, tmp.name, include_paths=req.include_paths)
        except Exception as e:
            os.unlink(tmp.name)
            raise HTTPException(500, detail=str(e))

        media = "text/csv" if ext == "csv" else "application/json"
        return FileResponse(
            tmp.name,
            media_type=media,
            filename=fname,
            background=None,
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_to_summaries(df: pd.DataFrame) -> list[NoduleSummary]:
    """Convert a results DataFrame to NoduleSummary list."""
    summaries = []
    for _, row in df.iterrows():
        summaries.append(
            NoduleSummary(
                annotation_id=str(row.get("annotation_id", "")),
                dataset=str(row.get("dataset", "")),
                patient_id=str(row.get("patient_id", "")),
                ct_path=str(row.get("ct_path", "")),
                label=int(row["label"]) if pd.notna(row.get("label")) else None,
                lobe_name=str(row.get("lobe_name", "")),
                lung_side=str(row.get("lung_side", "")),
                lung_zone=str(row.get("lung_zone", "")),
                central_peripheral=str(row.get("central_peripheral", "")),
                nodule_mean_diam_mm=float(row.get("nodule_mean_diam_mm", 0)),
                pleural_distance_mm=(
                    float(row["pleural_distance_mm"])
                    if pd.notna(row.get("pleural_distance_mm"))
                    else None
                ),
                reinsertion_lobe=str(row.get("reinsertion_lobe", "")),
                reinsertion_nodule_diam_mm=float(row.get("reinsertion_nodule_diam_mm", 0)),
                reinsertion_pleural_dist_mm=(
                    float(row["reinsertion_pleural_dist_mm"])
                    if pd.notna(row.get("reinsertion_pleural_dist_mm"))
                    else None
                ),
                reinsertion_lobe_cc_pct=float(row.get("reinsertion_lobe_cc_pct", 0)),
            )
        )
    return summaries
