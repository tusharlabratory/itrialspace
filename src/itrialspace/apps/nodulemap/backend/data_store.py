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
data_store.py -- Load and serve precomputed artifacts (embeddings, metadata, edges).

Provides fast in-memory access to:
  - Node metadata (parquet -> DataFrame)
  - 2D positions (embeddings .npy files)
  - Edge lists (parquet)
  - KNN index (FAISS / sklearn) for on-demand neighbor queries
  - Full node detail with demographics and resolved paths
  - Axial slice thumbnails with mask overlay
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from itrialspace.apps.nodulemap.neighbors import search_space

try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# Demographics column mapping per dataset (matches CohortBuilder)
_DEMO_COL_MAP: dict[str, dict[str, str]] = {
    "DLCS24": {
        "Age": "patient_age",
        "Sex": "patient_sex",
        "Smoking Status (Current/ Former/ Never/ Unknown)": "smoking_status",
    },
    "LUNA25": {"age": "patient_age", "gender": "patient_sex"},
    "NSCLCR": {"age": "patient_age", "gender": "patient_sex"},
    "IMDCT": {"age": "patient_age", "gender": "patient_sex", "smoke": "smoking_status"},
    "LUNGx": {},
    "LNDbv4": {},
    "LUNA16": {},
}


def _extract_demographics(row: pd.Series, dataset: str) -> dict:
    """Extract demographics from a full NoduleIndex row (best-effort)."""
    col_map = _DEMO_COL_MAP.get(dataset, {})
    result = {
        "patient_age": "NA",
        "patient_sex": "NA",
        "smoking_status": "NA",
        "pack_years": "NA",
    }
    for src_col, tgt in col_map.items():
        val = row.get(src_col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if tgt == "patient_age":
            try:
                result[tgt] = str(int(float(val)))
            except (ValueError, TypeError):
                pass
        elif tgt == "pack_years":
            try:
                result[tgt] = str(round(float(val), 1))
            except (ValueError, TypeError):
                pass
        else:
            result[tgt] = str(val)
    return result


# ── Canonical axial thumbnail renderer ─────────────────────────────────────

# LRU cache for canonical NIfTI volumes (path -> (data_ras, affine_ras))
_NIFTI_CANON_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_NIFTI_CANON_ORDER: list[str] = []
_NIFTI_CANON_MAX = 8


def _load_canonical(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI, reorient to RAS+ canonical, return (data, affine)."""
    if path in _NIFTI_CANON_CACHE:
        return _NIFTI_CANON_CACHE[path]

    import nibabel as nib

    img = nib.load(path)
    canon = nib.as_closest_canonical(img)
    data = np.asarray(canon.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    affine = canon.affine.copy()

    # LRU eviction
    if len(_NIFTI_CANON_CACHE) >= _NIFTI_CANON_MAX:
        evict = _NIFTI_CANON_ORDER.pop(0)
        _NIFTI_CANON_CACHE.pop(evict, None)
    _NIFTI_CANON_CACHE[path] = (data, affine)
    _NIFTI_CANON_ORDER.append(path)
    return data, affine


def _render_canonical_axial_thumbnail(
    ct_path: str,
    mask_path: Optional[str],
    coord_x: Optional[float],
    coord_y: Optional[float],
    coord_z: Optional[float],
    window_center: float = -600.0,
    window_width: float = 1500.0,
    overlay_alpha: float = 0.35,
) -> Optional[bytes]:
    """Render an axial thumbnail in standard radiology orientation.

    1. Load CT & mask, reorient both to RAS+ canonical.
    2. Compute axial slice index from world coords (or mask COM fallback).
    3. Extract axial slice [:, :, z] in canonical space.
    4. Transpose + fliplr for radiology display (patient-R on image-left).
    5. Overlay mask, draw crosshair, burn R/L/A/P labels.
    6. Return PNG bytes.
    """
    import io

    from PIL import Image, ImageDraw, ImageFont

    # ── Load canonical CT ──────────────────────────────────────────────
    ct_data, ct_affine = _load_canonical(ct_path)
    ct_shape = ct_data.shape  # (X, Y, Z) in RAS+

    # ── Load canonical mask (if available) ─────────────────────────────
    mask_data = None
    if mask_path:
        m_data, m_affine = _load_canonical(mask_path)
        if m_data.shape == ct_shape:
            mask_data = (m_data > 0).astype(np.uint8)
        else:
            # Shape mismatch after canonicalization -- cannot overlay safely
            mask_data = None

    # ── Determine axial slice index in canonical space ─────────────────
    has_coords = (
        coord_x is not None
        and coord_y is not None
        and coord_z is not None
        and not any(isinstance(c, float) and pd.isna(c) for c in (coord_x, coord_y, coord_z))
    )

    voxel_i, voxel_j = None, None  # for crosshair

    if has_coords:
        inv_aff = np.linalg.inv(ct_affine)
        world = np.array([float(coord_x), float(coord_y), float(coord_z), 1.0])
        ijk = inv_aff @ world
        z_idx = int(round(ijk[2]))
        voxel_i = int(round(ijk[0]))
        voxel_j = int(round(ijk[1]))
    elif mask_data is not None and mask_data.any():
        from scipy import ndimage

        com = ndimage.center_of_mass(mask_data)
        z_idx = int(round(com[2]))
        voxel_i = int(round(com[0]))
        voxel_j = int(round(com[1]))
    else:
        z_idx = ct_shape[2] // 2

    z_idx = max(0, min(z_idx, ct_shape[2] - 1))

    # ── Extract axial slices ───────────────────────────────────────────
    ct_slice = ct_data[:, :, z_idx]  # shape (X_ras, Y_ras)
    mask_slice = mask_data[:, :, z_idx] if mask_data is not None else None

    # ── Radiology display transform ────────────────────────────────────
    # In RAS+ canonical: axis-0=R->L (X), axis-1=A->P (Y), axis-2=I->S (Z)
    # Transpose so rows=Y (A->P), cols=X (R->L), then fliplr so
    # patient-Right is on image-Left (radiological convention).
    ct_disp = np.fliplr(ct_slice.T)  # shape (Y, X), radiology view
    mask_disp = np.fliplr(mask_slice.T) if mask_slice is not None else None

    # ── Compute crosshair in display coords ────────────────────────────
    disp_row, disp_col = None, None
    if voxel_i is not None and voxel_j is not None:
        disp_row = voxel_j  # row = j (Y index)
        disp_col = ct_slice.shape[0] - 1 - voxel_i  # col = (X-1-i) due to fliplr

    # ── Window CT to uint8 ─────────────────────────────────────────────
    lo = window_center - window_width / 2
    hi = window_center + window_width / 2
    img_u8 = np.clip(ct_disp, lo, hi)
    img_u8 = ((img_u8 - lo) / (hi - lo) * 255).astype(np.uint8)

    # ── Build RGB with mask overlay ────────────────────────────────────
    rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

    if mask_disp is not None and mask_disp.any():
        overlay = np.zeros_like(rgb)
        overlay[mask_disp > 0] = [255, 50, 50]  # red
        blended = (
            rgb.astype(float) * (1 - overlay_alpha) + overlay.astype(float) * overlay_alpha
        ).astype(np.uint8)
        # Only blend where mask is active
        rgb[mask_disp > 0] = blended[mask_disp > 0]

    # ── Convert to PIL image (origin="lower" effect via flipud) ────────
    # imshow origin="lower" means row-0 is at bottom.
    # In our RAS+ transposed view, row-0 = Anterior, which should be at
    # the top in standard radiology. So we flipud to place row-0 at bottom
    # (Anterior at top when rendered top-down in PNG).
    rgb = np.flipud(rgb)
    pil_img = Image.fromarray(rgb)

    # ── Draw crosshair ─────────────────────────────────────────────────
    draw = ImageDraw.Draw(pil_img)
    h_img, w_img = rgb.shape[0], rgb.shape[1]

    if disp_row is not None and disp_col is not None:
        # After flipud: display_row_flipped = (H-1) - disp_row
        cr = (h_img - 1) - disp_row
        cc = disp_col
        cross_color = (0, 255, 0)  # green
        arm = 8
        draw.line([(cc - arm, cr), (cc + arm, cr)], fill=cross_color, width=1)
        draw.line([(cc, cr - arm), (cc, cr + arm)], fill=cross_color, width=1)

    # ── Orientation labels ─────────────────────────────────────────────
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    label_color = (255, 255, 0)  # yellow
    shadow_color = (0, 0, 0)
    margin = 4

    def _draw_label(x, y, text, anchor=None):
        # Shadow for readability
        draw.text((x + 1, y + 1), text, fill=shadow_color, font=font, anchor=anchor)
        draw.text((x, y), text, fill=label_color, font=font, anchor=anchor)

    # R on left (radiological: patient-Right on viewer-Left)
    _draw_label(margin, h_img // 2, "R", anchor="lm")
    # L on right
    _draw_label(w_img - margin, h_img // 2, "L", anchor="rm")
    # A at top
    _draw_label(w_img // 2, margin, "A", anchor="mt")
    # P at bottom
    _draw_label(w_img // 2, h_img - margin, "P", anchor="mb")

    # ── Encode PNG ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class DataStore:
    """In-memory store for NoduleMap artifacts."""

    # ── Model name canonicalization ────────────────────────────────────────
    _NAME_ALIASES: dict[str, str] = {
        "tsne_2d": "TSNE_2D",
        "t-sne_2d": "TSNE_2D",
        "tSNE_2D": "TSNE_2D",
        "pca_2d": "PCA_2D",
        "pca_32d": "PCA_32",
        "pca_32": "PCA_32",
        "umap_2d": "UMAP_2D",
        "umap_16d": "UMAP_16",
        "umap_16": "UMAP_16",
        "umap_32d": "UMAP_32",
        "umap_32": "UMAP_32",
    }

    @classmethod
    def _canon(cls, name: str) -> str:
        """Canonicalize a model name.  E.g. 'tSNE_2D' → 'TSNE_2D'."""
        return cls._NAME_ALIASES.get(name.lower().strip(), name)

    def __init__(self, artifact_dir: str, verbose: bool = True):
        self._dir = Path(artifact_dir)
        self._verbose = verbose

        # Caches
        self._metadata: dict[str, pd.DataFrame] = {}  # tag -> metadata df
        self._embeddings: dict[str, np.ndarray] = {}  # tag -> (n, d) array
        self._positions: dict[str, np.ndarray] = {}  # tag -> (n, 2) array
        self._edges: dict[str, pd.DataFrame] = {}  # edge_key -> edge df
        self._knn_indices: dict[str, object] = {}  # tag -> faiss/sklearn index
        self._search_matrices: dict[str, np.ndarray] = {}  # (fs,backend,model) -> matrix
        self._available_models: list[dict] = []

        # Lazy-loaded helpers for detail/thumbnail
        self._full_index: Optional[pd.DataFrame] = None
        self._path_resolver = None  # type: ignore
        self._slicer = None  # type: ignore

        self._scan_artifacts()

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[DataStore] {msg}")

    def _scan_artifacts(self) -> None:
        """Scan artifact directory and catalog available models.

        Classifies each model as *layout* (2D — suitable for ``vis_model``)
        or *search* (any D — suitable for ``search_model``).
        Also validates that node_id ordering is consistent across all models.
        """
        models: dict[tuple[str, str], int] = {}  # (fs, model) → n_dims
        for f in self._dir.glob("embeddings_*.npy"):
            stem = f.stem.replace("embeddings_", "")
            parts = stem.split("__")
            if len(parts) == 2:
                fs, mn = parts
                # quick-probe dimensionality without loading full array
                arr = np.load(f, mmap_mode="r")
                n_dims = arr.shape[1] if arr.ndim == 2 else 0
                models[(fs, mn)] = n_dims

        layout_models: list[dict] = []
        search_models: list[dict] = []
        for (fs, mn), nd in sorted(models.items()):
            entry = {"feature_set": fs, "model_name": mn, "n_dims": nd}
            if nd == 2:
                layout_models.append(entry)
            search_models.append(entry)  # all models can be used for search

        self._layout_models = layout_models
        self._search_models = search_models
        self._available_models = [
            {"feature_set": fs, "model_name": mn, "n_dims": nd}
            for (fs, mn), nd in sorted(models.items())
        ]

        if self._verbose:
            self._log(
                f"Found {len(self._available_models)} model(s): "
                f"{len(layout_models)} layout, {len(search_models)} search"
            )

        # ── Validate node_id alignment across all models ───────────────
        ref_ids: Optional[np.ndarray] = None
        ref_tag: Optional[str] = None
        for fs, mn in sorted(models.keys()):
            meta_path = self._dir / f"metadata_{fs}__{mn}.parquet"
            if not meta_path.exists():
                continue
            ids = pd.read_parquet(meta_path, columns=["node_id"])["node_id"].values
            if ref_ids is None:
                ref_ids = ids
                ref_tag = f"{fs}__{mn}"
            else:
                if len(ids) != len(ref_ids) or not (ids == ref_ids).all():
                    self._log(
                        f"WARNING: node_id mismatch between {ref_tag} and {fs}__{mn}! "
                        f"len {len(ref_ids)} vs {len(ids)}"
                    )
        if ref_ids is not None and self._verbose:
            self._log(f"node_id alignment OK ({len(ref_ids)} nodes, {len(models)} models)")

    @property
    def available_models(self) -> list[dict]:
        return self._available_models

    @property
    def layout_models(self) -> list[dict]:
        """Models with 2D embeddings — suitable as vis_model / position model."""
        return self._layout_models

    @property
    def search_models(self) -> list[dict]:
        """All models — suitable for similarity search."""
        return self._search_models

    def _tag(self, model_name: str, feature_set: str) -> str:
        return f"{feature_set}__{model_name}"

    # ── Loaders (lazy, cached) ─────────────────────────────────────────────

    def get_metadata(self, model_name: str, feature_set: str) -> pd.DataFrame:
        """Load node metadata for a specific model+feature_set."""
        model_name = self._canon(model_name)
        tag = self._tag(model_name, feature_set)
        if tag not in self._metadata:
            path = self._dir / f"metadata_{tag}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Metadata not found: {path}")
            self._metadata[tag] = pd.read_parquet(path)
            self._log(f"Loaded metadata: {len(self._metadata[tag])} nodes")
        return self._metadata[tag]

    def get_embeddings(self, model_name: str, feature_set: str) -> np.ndarray:
        """Load embedding matrix."""
        model_name = self._canon(model_name)
        tag = self._tag(model_name, feature_set)
        if tag not in self._embeddings:
            path = self._dir / f"embeddings_{tag}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Embeddings not found: {path}")
            self._embeddings[tag] = np.load(path).astype(np.float32)
            self._log(f"Loaded embeddings: {self._embeddings[tag].shape}")
        return self._embeddings[tag]

    def get_positions(self, vis_model: str, feature_set: str) -> np.ndarray:
        """Load 2D positions for visualization (must be a 2D model)."""
        vis_model = self._canon(vis_model)
        tag = self._tag(vis_model, feature_set)
        if tag not in self._positions:
            emb = self.get_embeddings(vis_model, feature_set)
            if emb.shape[1] < 2:
                raise ValueError(f"Model {vis_model} has <2 dimensions")
            self._positions[tag] = emb[:, :2]
        return self._positions[tag]

    def get_edges(
        self,
        model_name: str,
        feature_set: str,
        k: int = 5,
        mode: str = "closest",
        scope: str = "cross_dataset",
    ) -> pd.DataFrame:
        """Load precomputed edge list."""
        model_name = self._canon(model_name)
        tag = self._tag(model_name, feature_set)
        edge_key = f"{tag}_k{k}_{mode}_{scope}"
        if edge_key not in self._edges:
            path = self._dir / f"edges_{tag}_k{k}_{mode}_{scope}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Edge file not found: {path}. "
                    f"Run: itrialspace-nodulemap edges --model {model_name} "
                    f"--feature-set {feature_set} --k {k} --mode {mode}"
                )
            self._edges[edge_key] = pd.read_parquet(path)
            self._log(f"Loaded edges: {len(self._edges[edge_key])} ({edge_key})")
        return self._edges[edge_key]

    # ── KNN index (for on-demand queries) ──────────────────────────────────

    def _get_knn_index(self, model_name: str, feature_set: str):
        """Build or retrieve a FAISS/sklearn index for on-demand neighbor queries."""
        model_name = self._canon(model_name)
        tag = self._tag(model_name, feature_set)
        if tag not in self._knn_indices:
            emb = self.get_embeddings(model_name, feature_set)
            if _HAS_FAISS:
                index = faiss.IndexFlatL2(emb.shape[1])
                index.add(emb)
                self._knn_indices[tag] = ("faiss", index)
            else:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(
                    n_neighbors=min(50, len(emb)), metric="euclidean", algorithm="auto"
                )
                nn.fit(emb)
                self._knn_indices[tag] = ("sklearn", nn)
            self._log(f"Built KNN index for {tag}")
        return self._knn_indices[tag]

    def _get_search_matrix(self, model_name: str, feature_set: str, backend: str) -> np.ndarray:
        """Build/cache the matrix that similarity SEARCH runs on for a backend."""
        key = f"{feature_set}|{backend}|{model_name if backend == 'embedding' else '-'}"
        if key not in self._search_matrices:
            meta = self.get_metadata(model_name, feature_set)
            self._search_matrices[key] = search_space.build_search_matrix(
                backend, self._dir, feature_set, model_name, meta
            )
            self._log(f"Built search matrix [{backend}]: {self._search_matrices[key].shape}")
        return self._search_matrices[key]

    def query_neighbors(
        self,
        node_id: str,
        model_name: str,
        feature_set: str,
        k: int = 10,
        scope: Literal["cross_dataset", "within_dataset", "selected_datasets"] = "cross_dataset",
        datasets: list[str] | None = None,
        balance_by_dataset: bool = False,
        search_backend: str = "weighted",
    ) -> list[dict]:
        """
        On-demand neighbor (donor-matching) query for a specific node.

        ``search_backend``: 'weighted' (default, clinical ReinsertionMatcher metric),
        'feature_l2' (standardized full feature space), or 'embedding' (reduced; legacy).
        Returns list of dicts with: node_id, distance, similarity, rank, + metadata.
        """
        model_name = self._canon(model_name)
        meta = self.get_metadata(model_name, feature_set)
        matrix = self._get_search_matrix(model_name, feature_set, search_backend)
        metric = search_space.metric_for(search_backend)

        # Find query node index
        node_mask = meta["node_id"].values == node_id
        if not node_mask.any():
            raise KeyError(f"Node not found: {node_id}")
        query_idx = int(np.where(node_mask)[0][0])
        query_vec = matrix[query_idx : query_idx + 1]  # (1, d)

        # Determine candidate set
        if scope == "within_dataset":
            query_ds = meta.iloc[query_idx]["dataset"]
            candidate_mask = meta["dataset"].values == query_ds
        elif scope == "selected_datasets" and datasets:
            candidate_mask = meta["dataset"].isin(datasets).values
        else:
            candidate_mask = np.ones(len(meta), dtype=bool)

        candidate_indices = np.where(candidate_mask)[0]
        candidate_mat = matrix[candidate_mask].astype(np.float32)

        # KNN on candidates (true distances)
        search_k = min(k + 1, len(candidate_mat))
        dists, inds = search_space.knn(candidate_mat, query_vec, search_k, metric)

        # Build result list
        results = []
        rank = 0
        for j in range(inds.shape[1]):
            global_idx = candidate_indices[inds[0, j]]
            if global_idx == query_idx:
                continue
            dist = float(max(dists[0, j], 0.0))
            sim = float(np.exp(-dist))
            row = meta.iloc[global_idx].to_dict()
            row["distance"] = round(dist, 6)
            row["similarity"] = round(sim, 6)
            rank += 1
            row["rank"] = rank
            results.append(row)
            if rank >= k:
                break

        # Optional: balance by dataset
        if balance_by_dataset and results:
            results = self._balance_results(results, k)

        return results

    def _balance_results(self, results: list[dict], k: int) -> list[dict]:
        """Re-rank results to ensure balanced dataset representation."""
        from collections import defaultdict

        by_ds = defaultdict(list)
        for r in results:
            by_ds[r["dataset"]].append(r)
        n_ds = len(by_ds)
        per_ds = max(1, k // n_ds)
        balanced = []
        for ds_results in by_ds.values():
            balanced.extend(ds_results[:per_ds])
        # Fill remaining slots
        remaining = k - len(balanced)
        if remaining > 0:
            used = {r["node_id"] for r in balanced}
            for r in results:
                if r["node_id"] not in used:
                    balanced.append(r)
                    remaining -= 1
                    if remaining <= 0:
                        break
        # Re-rank
        balanced.sort(key=lambda x: x["distance"])
        for i, r in enumerate(balanced):
            r["rank"] = i + 1
        return balanced[:k]

    def get_node(self, node_id: str, model_name: str, feature_set: str) -> dict:
        """Get full metadata for a single node."""
        model_name = self._canon(model_name)
        meta = self.get_metadata(model_name, feature_set)
        row = meta[meta["node_id"] == node_id]
        if row.empty:
            raise KeyError(f"Node not found: {node_id}")
        return row.iloc[0].to_dict()

    # ── Full index / path resolver / slicer (lazy) ─────────────────────────

    def _get_full_index(self) -> pd.DataFrame:
        """Load the full NoduleIndex DataFrame (with all extra columns)."""
        if self._full_index is None:
            try:
                from itrialspace.index.nodule_index import NoduleIndex
                from itrialspace.io.registry import DatasetRegistry

                # Portable registry resolution via itrialspace.config.settings.
                registry = DatasetRegistry.from_yaml()
                idx = NoduleIndex.from_registry(registry, verbose=False)
                self._full_index = idx.df
                self._log(
                    f"Loaded full NoduleIndex: {len(self._full_index)} rows, "
                    f"{len(self._full_index.columns)} cols"
                )
            except Exception as e:
                self._log(f"WARNING: Could not load NoduleIndex: {e}")
                self._full_index = pd.DataFrame()
        return self._full_index

    def _get_path_resolver(self):
        """Load PathResolver for resolving CT/mask paths."""
        if self._path_resolver is None:
            try:
                from itrialspace.site.path_resolver import PathResolver

                self._path_resolver = PathResolver()
                self._log("PathResolver loaded")
            except Exception as e:
                self._log(f"WARNING: Could not load PathResolver: {e}")
                self._path_resolver = False  # sentinel: tried, failed
        return self._path_resolver if self._path_resolver is not False else None

    def _get_slicer(self):
        """Load NIfTISlicer for thumbnail generation."""
        if self._slicer is None:
            try:
                from itrialspace.apps.retriever.slicer import NIfTISlicer

                self._slicer = NIfTISlicer(cache_size=8)
                self._log("NIfTISlicer loaded")
            except Exception as e:
                self._log(f"WARNING: Could not load NIfTISlicer: {e}")
                self._slicer = False  # sentinel
        return self._slicer if self._slicer is not False else None

    def _resolve_paths_for_row(self, row: pd.Series) -> dict:
        """Resolve CT/mask paths for a NoduleIndex row."""
        resolver = self._get_path_resolver()
        if resolver is None:
            return {}
        try:
            paths = resolver.resolve_all_paths(row)
            # Only return paths that actually exist on disk
            result = {}
            for key in ("ct_path", "nodule_mask_path", "organ_seg_path"):
                p = paths.get(key, "")
                result[f"{key}_abs"] = p if p and os.path.isfile(p) else ""
            return result
        except Exception:
            return {}

    def get_node_detail(self, node_id: str, model_name: str, feature_set: str) -> dict:
        """Get comprehensive node detail: metadata + demographics + resolved paths."""
        model_name = self._canon(model_name)
        # Start with embedding metadata
        base = self.get_node(node_id, model_name, feature_set)

        # Look up full index row for demographics and extra fields
        full_idx = self._get_full_index()
        demographics = {
            "patient_age": "NA",
            "patient_sex": "NA",
            "smoking_status": "NA",
            "pack_years": "NA",
        }
        resolved_paths = {}

        if not full_idx.empty:
            # Match by annotation_id (more reliable than node_id)
            ann_id = base.get("annotation_id", node_id)
            match = full_idx[full_idx["annotation_id"] == ann_id]
            if match.empty:
                # Try node_id
                if "node_id" in full_idx.columns:
                    match = full_idx[full_idx["node_id"] == node_id]

            if not match.empty:
                full_row = match.iloc[0]
                dataset = base.get("dataset", full_row.get("dataset", ""))

                # Extract demographics
                demographics = _extract_demographics(full_row, dataset)

                # Add patient_id from full index if not already present
                if "patient_id" not in base or pd.isna(base.get("patient_id")):
                    pid = full_row.get("patient_id")
                    if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                        base["patient_id"] = str(pid)

                # Resolve paths
                resolved_paths = self._resolve_paths_for_row(full_row)

                # Ensure key donor fields are present
                for col in (
                    "annotation_id",
                    "ct_path",
                    "coordX",
                    "coordY",
                    "coordZ",
                    "label",
                    "nodule_mean_diam_mm",
                ):
                    if col not in base or pd.isna(base.get(col)):
                        val = full_row.get(col)
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            base[col] = val

        base["demographics"] = demographics
        base.update(resolved_paths)

        # Clean NaN/NaT values for JSON serialization
        for k, v in list(base.items()):
            if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
                base[k] = None
            elif isinstance(v, (np.integer,)):
                base[k] = int(v)
            elif isinstance(v, (np.floating,)):
                base[k] = float(v) if not np.isnan(v) else None

        return base

    def get_axial_thumbnail(
        self, node_id: str, model_name: str, feature_set: str
    ) -> Optional[bytes]:
        """Generate an axial slice thumbnail (PNG) for a nodule.

        Uses canonical (RAS+) reorientation for consistent radiology display.
        Returns PNG bytes or None if CT/mask is unavailable.
        """
        model_name = self._canon(model_name)
        # Get node detail with resolved paths
        detail = self.get_node_detail(node_id, model_name, feature_set)
        ct_path = detail.get("ct_path_abs", "")
        mask_path = detail.get("nodule_mask_path_abs", "")

        if not ct_path or not os.path.isfile(ct_path):
            return None

        try:
            return _render_canonical_axial_thumbnail(
                ct_path=ct_path,
                mask_path=mask_path if mask_path and os.path.isfile(mask_path) else None,
                coord_x=detail.get("coordX"),
                coord_y=detail.get("coordY"),
                coord_z=detail.get("coordZ"),
            )
        except Exception as e:
            self._log(f"Thumbnail error for {node_id}: {e}")
            return None

    def get_graph_data(
        self,
        vis_model: str,
        search_model: str,
        feature_set: str,
        k: int = 5,
        mode: str = "closest",
        scope: str = "cross_dataset",
        color_by: str = "dataset",
        filters: dict | None = None,
    ) -> dict:
        """
        Get nodes + edges for the map view.

        Returns: {"nodes": [...], "edges": [...], "stats": {...}}
        """
        vis_model = self._canon(vis_model)
        search_model = self._canon(search_model)
        meta = self.get_metadata(search_model, feature_set)
        pos = self.get_positions(vis_model, feature_set)

        # Apply filters
        mask = np.ones(len(meta), dtype=bool)
        if filters:
            mask = self._apply_filters(meta, filters)

        filtered_meta = meta[mask].copy()
        filtered_pos = pos[mask]
        filtered_ids = set(filtered_meta["node_id"].values)

        # Load edges and filter to visible nodes
        try:
            edges_df = self.get_edges(search_model, feature_set, k, mode, scope)
            edges_df = edges_df[
                edges_df["src_id"].isin(filtered_ids) & edges_df["dst_id"].isin(filtered_ids)
            ]
        except FileNotFoundError:
            edges_df = pd.DataFrame(columns=["src_id", "dst_id", "distance", "similarity", "rank"])

        # Build node list
        nodes = []
        for i, (_, row) in enumerate(filtered_meta.iterrows()):
            node = {
                "id": row["node_id"],
                "x": float(filtered_pos[i, 0]),
                "y": float(filtered_pos[i, 1]),
                "dataset": row.get("dataset", ""),
                "label_text": row.get("label_text", "unlabelled"),
                "size_bucket": row.get("size_bucket", ""),
            }
            # Add color_by value
            if color_by in row.index:
                val = row[color_by]
                node["color_value"] = str(val) if pd.notna(val) else "unknown"
            else:
                node["color_value"] = "unknown"
            # Add hover tooltip fields
            for col in [
                "reinsertion_nodule_diam_mm",
                "reinsertion_lobe",
                "reinsertion_lung_zone",
                "annotation_id",
            ]:
                if col in row.index:
                    val = row[col]
                    node[col] = val if pd.notna(val) else None
            nodes.append(node)

        # Build edge list
        edges = []
        for _, erow in edges_df.iterrows():
            edges.append(
                {
                    "source": erow["src_id"],
                    "target": erow["dst_id"],
                    "similarity": erow.get("similarity", 0),
                    "distance": erow.get("distance", 0),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "n_total": len(meta),
                "n_filtered": len(meta) - len(filtered_meta),
            },
        }

    def _apply_filters(self, meta: pd.DataFrame, filters: dict) -> np.ndarray:
        """Apply filter dict to metadata, return boolean mask."""
        mask = np.ones(len(meta), dtype=bool)

        # Dataset filter
        if "datasets" in filters and filters["datasets"]:
            mask &= meta["dataset"].isin(filters["datasets"]).values

        # Label filter
        if "label" in filters:
            lbl = filters["label"]
            if lbl == "unlabelled":
                mask &= meta["label"].isna().values
            elif lbl in ("0", "1"):
                mask &= (meta["label"] == int(lbl)).values
            elif lbl == "labelled":
                mask &= meta["label"].notna().values

        # Label source
        if "label_source" in filters and filters["label_source"] != "all":
            if "label_source" in meta.columns:
                mask &= (meta["label_source"] == filters["label_source"]).values

        # Population type
        if "population_type" in filters and filters["population_type"] != "all":
            if "population_type" in meta.columns:
                mask &= (meta["population_type"] == filters["population_type"]).values

        # Lobe
        if "lobe" in filters and filters["lobe"]:
            col = "reinsertion_lobe" if "reinsertion_lobe" in meta.columns else "lobe_name"
            if col in meta.columns:
                if isinstance(filters["lobe"], list):
                    mask &= meta[col].isin(filters["lobe"]).values
                else:
                    mask &= (meta[col] == filters["lobe"]).values

        # Zone
        if "zone" in filters and filters["zone"]:
            col = (
                "reinsertion_lung_zone" if "reinsertion_lung_zone" in meta.columns else "lung_zone"
            )
            if col in meta.columns:
                mask &= (meta[col] == filters["zone"]).values

        # Side
        if "side" in filters and filters["side"]:
            col = (
                "reinsertion_lung_side" if "reinsertion_lung_side" in meta.columns else "lung_side"
            )
            if col in meta.columns:
                mask &= (meta[col] == filters["side"]).values

        # Diameter range
        diam_col = "reinsertion_nodule_diam_mm"
        if diam_col not in meta.columns:
            diam_col = "nodule_mean_diam_mm"
        if "diameter_min" in filters and filters["diameter_min"] is not None:
            mask &= (meta[diam_col] >= filters["diameter_min"]).values
        if "diameter_max" in filters and filters["diameter_max"] is not None:
            mask &= (meta[diam_col] <= filters["diameter_max"]).values

        # Size bucket
        if "size_bucket" in filters and filters["size_bucket"]:
            if "size_bucket" in meta.columns:
                if isinstance(filters["size_bucket"], list):
                    mask &= meta["size_bucket"].isin(filters["size_bucket"]).values
                else:
                    mask &= (meta["size_bucket"] == filters["size_bucket"]).values

        # Pleural distance range
        if "pleural_min" in filters and filters["pleural_min"] is not None:
            col = "reinsertion_pleural_dist_mm"
            if col in meta.columns:
                mask &= (meta[col] >= filters["pleural_min"]).values
        if "pleural_max" in filters and filters["pleural_max"] is not None:
            col = "reinsertion_pleural_dist_mm"
            if col in meta.columns:
                mask &= (meta[col] <= filters["pleural_max"]).values

        # Airway distance range
        if "airway_min" in filters and filters["airway_min"] is not None:
            col = "reinsertion_airway_dist_mm"
            if col in meta.columns:
                mask &= (meta[col] >= filters["airway_min"]).values
        if "airway_max" in filters and filters["airway_max"] is not None:
            col = "reinsertion_airway_dist_mm"
            if col in meta.columns:
                mask &= (meta[col] <= filters["airway_max"]).values

        return mask
