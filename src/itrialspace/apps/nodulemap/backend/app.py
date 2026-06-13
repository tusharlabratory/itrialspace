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
app.py — FastAPI backend for NoduleMap.

Endpoints:
  GET  /                         → serves static frontend
  GET  /api/models               → available embedding models
  GET  /api/graph                → nodes + edges for map view
  GET  /api/node/{node_id}       → full metadata for a node
  GET  /api/node/{node_id}/neighbors → ranked neighbor list
  POST /api/export               → export selection to file
  GET  /api/filters              → available filter values
  GET  /health                   → health check

Start:
  uvicorn nodulemap.backend.app:create_app --factory --host 0.0.0.0 --port 8422
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from itrialspace.apps.nodulemap.backend.data_store import DataStore
from itrialspace.apps.nodulemap.backend.export import Exporter
from itrialspace.config import settings

# Artifact directory (configurable via env var; default resolved by settings:
# $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)
_DEFAULT_ARTIFACT_DIR = str(settings.nodulemap_artifacts_dir())
_EXPORT_DIR = os.environ.get(
    "NODULEMAP_EXPORTS",
    str(settings.output_dir() / "nodulemap_exports"),
)
_STATIC_DIR = Path(__file__).parent.parent / "frontend" / "static"


# ── Pydantic models ──────────────────────────────────────────────────────


class GraphRequest(BaseModel):
    vis_model: str = "UMAP_2D"
    search_model: str = "UMAP_2D"
    feature_set: str = "reinsertion_core"
    k: int = 5
    mode: str = "closest"
    scope: str = "cross_dataset"
    color_by: str = "dataset"
    filters: dict | None = None


class NeighborRequest(BaseModel):
    model: str = "UMAP_2D"
    feature_set: str = "reinsertion_core"
    k: int = 10
    scope: str = "cross_dataset"
    datasets: list[str] | None = None
    balance_by_dataset: bool = False


class ExportRequest(BaseModel):
    node_id: str
    model: str = "UMAP_2D"
    feature_set: str = "reinsertion_core"
    k: int = 10
    scope: str = "cross_dataset"
    format: str = "csv"  # csv | json | donors


# ── App factory ──────────────────────────────────────────────────────────


def create_app(artifact_dir: str | None = None) -> FastAPI:
    """Create the NoduleMap FastAPI app."""
    art_dir = artifact_dir or _DEFAULT_ARTIFACT_DIR

    app = FastAPI(
        title="iTrialSpace NoduleMap",
        version="0.1.0",
        description="Embedding-based interactive nodule similarity explorer",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy init data store (on first request)
    store: dict = {"instance": None}
    exporter = Exporter(_EXPORT_DIR)

    def get_store() -> DataStore:
        if store["instance"] is None:
            print(f"[NoduleMap] Loading artifacts from {art_dir}")
            store["instance"] = DataStore(art_dir)
        return store["instance"]

    # ── Health ──────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        ds = get_store()
        return {
            "status": "ok",
            "version": "0.1.0",
            "models": ds.available_models,
            "artifact_dir": art_dir,
        }

    # ── Models ──────────────────────────────────────────────────────────
    @app.get("/api/models")
    def list_models():
        ds = get_store()
        # Probe readiness: each model has embeddings + metadata + edge files
        model_health: list[dict] = []
        for m in ds.available_models:
            mn, fs = m["model_name"], m["feature_set"]
            ok = True
            try:
                meta = ds.get_metadata(mn, fs)
                emb = ds.get_embeddings(mn, fs)
                ok = len(meta) == emb.shape[0] > 0
            except Exception:
                ok = False
            model_health.append({**m, "ready": ok})
        return {
            "models": ds.available_models,
            "layout_models": ds.layout_models,
            "search_models": ds.search_models,
            "health": model_health,
            "version": "0.2.0",
        }

    # ── Filters ─────────────────────────────────────────────────────────
    @app.get("/api/filters")
    def get_filters(
        model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
    ):
        """Return all unique values for filterable columns."""
        ds = get_store()
        meta = ds.get_metadata(model, feature_set)

        def unique_list(col):
            if col in meta.columns:
                return sorted([str(v) for v in meta[col].dropna().unique()])
            return []

        return {
            "datasets": unique_list("dataset"),
            "labels": ["benign", "malignant", "unlabelled"],
            "lobes": unique_list("reinsertion_lobe"),
            "zones": unique_list("reinsertion_lung_zone"),
            "sides": unique_list("reinsertion_lung_side"),
            "size_buckets": unique_list("size_bucket"),
            "label_sources": unique_list("label_source") if "label_source" in meta.columns else [],
            "population_types": (
                unique_list("population_type") if "population_type" in meta.columns else []
            ),
            "diameter_range": {
                "min": (
                    float(meta["reinsertion_nodule_diam_mm"].min())
                    if "reinsertion_nodule_diam_mm" in meta.columns
                    else 0
                ),
                "max": (
                    float(meta["reinsertion_nodule_diam_mm"].max())
                    if "reinsertion_nodule_diam_mm" in meta.columns
                    else 100
                ),
            },
        }

    # ── Graph ───────────────────────────────────────────────────────────
    @app.post("/api/graph")
    def get_graph(req: GraphRequest):
        ds = get_store()
        try:
            data = ds.get_graph_data(
                vis_model=req.vis_model,
                search_model=req.search_model,
                feature_set=req.feature_set,
                k=req.k,
                mode=req.mode,
                scope=req.scope,
                color_by=req.color_by,
                filters=req.filters,
            )
            return data
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # Also support GET for simpler access
    @app.get("/api/graph")
    def get_graph_get(
        vis_model: str = "UMAP_2D",
        search_model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
        k: int = 5,
        mode: str = "closest",
        scope: str = "cross_dataset",
        color_by: str = "dataset",
        datasets: str | None = None,
        label: str | None = None,
        lobe: str | None = None,
        zone: str | None = None,
        side: str | None = None,
        size_bucket: str | None = None,
        diameter_min: float | None = None,
        diameter_max: float | None = None,
    ):
        filters = {}
        if datasets:
            filters["datasets"] = datasets.split(",")
        if label:
            filters["label"] = label
        if lobe:
            filters["lobe"] = lobe.split(",")
        if zone:
            filters["zone"] = zone
        if side:
            filters["side"] = side
        if size_bucket:
            filters["size_bucket"] = size_bucket.split(",")
        if diameter_min is not None:
            filters["diameter_min"] = diameter_min
        if diameter_max is not None:
            filters["diameter_max"] = diameter_max

        req = GraphRequest(
            vis_model=vis_model,
            search_model=search_model,
            feature_set=feature_set,
            k=k,
            mode=mode,
            scope=scope,
            color_by=color_by,
            filters=filters if filters else None,
        )
        return get_graph(req)

    # ── Node detail ─────────────────────────────────────────────────────
    @app.get("/api/node/{node_id}")
    def get_node(
        node_id: str,
        model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
    ):
        ds = get_store()
        try:
            return ds.get_node_detail(node_id, model, feature_set)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    # ── Axial thumbnail ────────────────────────────────────────────────
    @app.get("/api/node/{node_id}/axial_thumbnail")
    def get_axial_thumbnail(
        node_id: str,
        model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
    ):
        from fastapi.responses import Response

        ds = get_store()
        try:
            png_bytes = ds.get_axial_thumbnail(node_id, model, feature_set)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        if png_bytes is None:
            raise HTTPException(
                status_code=404, detail="Thumbnail unavailable: CT or mask not found"
            )
        return Response(content=png_bytes, media_type="image/png")

    # ── Neighbors ───────────────────────────────────────────────────────
    @app.get("/api/node/{node_id}/neighbors")
    def get_neighbors(
        node_id: str,
        model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
        k: int = 10,
        scope: str = "cross_dataset",
        datasets: str | None = None,
        balance_by_dataset: bool = False,
        search_backend: str = "weighted",
        debug: int = 0,
    ):
        import time as _time

        ds = get_store()
        canon_model = ds._canon(model)
        ds_list = datasets.split(",") if datasets else None
        t0 = _time.perf_counter()
        try:
            results = ds.query_neighbors(
                node_id=node_id,
                model_name=model,
                feature_set=feature_set,
                k=k,
                scope=scope,
                datasets=ds_list,
                balance_by_dataset=balance_by_dataset,
                search_backend=search_backend,
            )
            elapsed = _time.perf_counter() - t0
            resp: dict = {
                "query_node_id": node_id,
                "k": k,
                "scope": scope,
                "neighbors": results,
            }
            if debug:
                emb = ds.get_embeddings(canon_model, feature_set)
                resp["_debug"] = {
                    "model_requested": model,
                    "model_canonical": canon_model,
                    "feature_set": feature_set,
                    "embedding_dim": int(emb.shape[1]),
                    "n_embeddings": int(emb.shape[0]),
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "n_results": len(results),
                }
            return resp
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    # ── Export ──────────────────────────────────────────────────────────
    @app.post("/api/export")
    def export_data(req: ExportRequest):
        ds = get_store()
        try:
            query_node = ds.get_node(req.node_id, req.model, req.feature_set)
            neighbors = ds.query_neighbors(
                node_id=req.node_id,
                model_name=req.model,
                feature_set=req.feature_set,
                k=req.k,
                scope=req.scope,
            )

            if req.format == "donors":
                path = exporter.export_candidate_donors(neighbors)
            else:
                path = exporter.export_neighbors(
                    query_node=query_node,
                    neighbors=neighbors,
                    model_name=req.model,
                    feature_set=req.feature_set,
                    k=req.k,
                    scope=req.scope,
                    fmt=req.format,
                )
            return {"status": "ok", "path": path}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Node not found: {req.node_id}")

    # ── Debug endpoints ───────────────────────────────────────────────
    @app.get("/debug/layout")
    def debug_layout(
        model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
        limit: int = 5,
    ):
        """Return first *limit* 2D positions + node_ids for quick sanity-check."""
        ds = get_store()
        try:
            pos = ds.get_positions(model, feature_set)
            meta = ds.get_metadata(model, feature_set)
            ids = meta["node_id"].values[:limit].tolist()
            coords = pos[:limit].tolist()
            return {
                "model": model,
                "feature_set": feature_set,
                "n_total": len(pos),
                "samples": [{"node_id": i, "x": c[0], "y": c[1]} for i, c in zip(ids, coords)],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/debug/search")
    def debug_search(
        node_id: str = "",
        model: str = "UMAP_2D",
        feature_set: str = "reinsertion_core",
        k: int = 3,
    ):
        """Quick neighbor probe with debug info (embedding dim, index type, timing)."""
        import time as _time

        ds = get_store()
        if not node_id:
            # return first available node_id so the user can try
            meta = ds.get_metadata(model, feature_set)
            node_id = meta["node_id"].values[0]
        try:
            emb = ds.get_embeddings(model, feature_set)
            t0 = _time.perf_counter()
            results = ds.query_neighbors(
                node_id=node_id,
                model_name=model,
                feature_set=feature_set,
                k=k,
            )
            elapsed = _time.perf_counter() - t0
            return {
                "query_node_id": node_id,
                "model": model,
                "embedding_dim": int(emb.shape[1]),
                "n_embeddings": int(emb.shape[0]),
                "k": k,
                "elapsed_ms": round(elapsed * 1000, 2),
                "neighbors": [
                    {
                        "node_id": r["node_id"],
                        "distance": r["distance"],
                        "similarity": r["similarity"],
                    }
                    for r in results
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/debug/alignment")
    def debug_alignment(feature_set: str = "reinsertion_core"):
        """Check node_id alignment across all models for a given feature set."""
        ds = get_store()
        report: dict = {"feature_set": feature_set, "models": {}}
        ref_ids = None
        ref_name = None
        for m in ds.available_models:
            if m["feature_set"] != feature_set:
                continue
            mn = m["model_name"]
            try:
                meta = ds.get_metadata(mn, feature_set)
                ids = meta["node_id"].values
                entry = {"n_nodes": len(ids), "first_5": ids[:5].tolist(), "aligned": True}
                if ref_ids is None:
                    ref_ids = ids
                    ref_name = mn
                elif len(ids) != len(ref_ids) or not (ids == ref_ids).all():
                    entry["aligned"] = False
                    entry["ref_model"] = ref_name
                report["models"][mn] = entry
            except Exception as e:
                report["models"][mn] = {"error": str(e)}
        return report

    # ── Static frontend ─────────────────────────────────────────────────
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def serve_frontend():
            index = _STATIC_DIR / "index.html"
            if index.exists():
                content = index.read_text()
                return HTMLResponse(
                    content=content,
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "Pragma": "no-cache",
                        "Expires": "0",
                    },
                )
            return "<h1>NoduleMap frontend not found</h1>"

    else:

        @app.get("/", response_class=HTMLResponse)
        def serve_placeholder():
            return "<h1>NoduleMap — frontend not built. Static dir: {}</h1>".format(_STATIC_DIR)

    return app
