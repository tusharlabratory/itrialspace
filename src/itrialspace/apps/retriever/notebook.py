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
Jupyter notebook client for the iTrialSpace Retriever.

Two modes:
    1. **Direct mode** (default): Loads the engine in-process.
    2. **API mode**: Connects to a running FastAPI backend.

Usage (direct):
    from itrialspace.apps.retriever.notebook import RetrieverClient
    client = RetrieverClient()
    client.search(label=1, lobe=["right_lung_upper_lobe"])
    client.find_similar("DLCS24_n0001", k=10)
    client.show_nodule("DLCS24_n0001")

Usage (API mode):
    client = RetrieverClient(api_url="http://localhost:8421")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from IPython.display import HTML, display
    from IPython.display import Image as IPImage

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class RetrieverClient:
    """
    Interactive Jupyter client for the iTrialSpace Retriever.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        datasets_yaml: Optional[str] = None,
        paths_yaml: Optional[str] = None,
        verbose: bool = True,
    ):
        self._api_url = api_url
        self._engine = None

        if api_url is None:
            # Direct mode — load engine
            from itrialspace.apps.retriever.engine import RetrieverEngine

            self._engine = RetrieverEngine.from_defaults(
                datasets_yaml=datasets_yaml,
                paths_yaml=paths_yaml,
                verbose=verbose,
            )
            if verbose:
                print(f"RetrieverClient ready — {self._engine.n_nodules:,} nodules")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_nodules(self) -> int:
        if self._engine:
            return self._engine.n_nodules
        return self._api_get("/health")["n_nodules"]

    @property
    def datasets(self) -> list[str]:
        if self._engine:
            return self._engine.datasets
        return self._api_get("/health")["datasets"]

    # ── A. Search ─────────────────────────────────────────────────────────────

    def search(self, display_results: bool = True, **kwargs) -> pd.DataFrame:
        """Run a faceted search.

        All kwargs map to SearchFilters fields:
            datasets, label, lobe, lung_zone, lung_side, diameter_min, diameter_max,
            pleural_distance_min, pleural_distance_max, population_type, label_source,
            limit, sort_by, ...

        Returns:
            DataFrame of results.
        """
        if self._engine:
            from itrialspace.apps.retriever.search import SearchFilters

            sf = SearchFilters(**kwargs)
            result = self._engine.search(sf)
            df = result.df
            total = result.total_matching
        else:
            resp = self._api_post("/search", kwargs)
            df = pd.DataFrame(resp["results"])
            total = resp["total_matching"]

        if display_results and HAS_IPYTHON:
            display(HTML(f"<b>{total:,}</b> matching, showing <b>{len(df)}</b>"))
            display(self._format_results(df))
        return df

    # ── B. Similarity ─────────────────────────────────────────────────────────

    def find_similar(
        self,
        annotation_id: str,
        k: int = 10,
        display_results: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Find the k most similar nodules to a reference.

        Extra kwargs: exclude_same_patient, include_datasets, exclude_datasets, label.
        """
        if self._engine:
            results = self._engine.find_similar(annotation_id=annotation_id, k=k, **kwargs)
            rows = [
                {
                    "rank": r.rank,
                    "annotation_id": r.annotation_id,
                    "dataset": r.dataset,
                    "distance": round(r.distance, 4),
                }
                for r in results
            ]
            df = pd.DataFrame(rows)
        else:
            body = {"annotation_id": annotation_id, "k": k, **kwargs}
            resp = self._api_post("/similar", body)
            df = pd.DataFrame(resp["results"])

        if display_results and HAS_IPYTHON:
            display(HTML(f"<b>Top-{k}</b> similar to <code>{annotation_id}</code>"))
            display(df)
        return df

    # ── C. Reinsertion matching ───────────────────────────────────────────────

    def find_match(
        self,
        lobe: str,
        k: int = 10,
        display_results: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Find best reinsertion-matched donor nodules.

        Extra kwargs: lobe_cc_pct, pleural_dist_mm, diameter_mm, label, lung_zone, ...
        """
        if self._engine:
            results = self._engine.find_reinsertion_match(lobe=lobe, k=k, **kwargs)
            rows = [
                {
                    "annotation_id": m.annotation_id,
                    "dataset": m.dataset,
                    "score": round(m.score, 4),
                    "lobe": m.lobe,
                    "diameter_mm": round(m.diameter_mm, 2),
                    "label": m.label,
                }
                for m in results
            ]
            df = pd.DataFrame(rows)
        else:
            body = {"lobe": lobe, "k": k, **kwargs}
            resp = self._api_post("/matcher", body)
            df = pd.DataFrame(resp["results"])

        if display_results and HAS_IPYTHON:
            display(HTML(f"<b>Top-{k}</b> matches for <code>{lobe}</code>"))
            display(df)
        return df

    # ── D. CT viewer ──────────────────────────────────────────────────────────

    def show_nodule(
        self,
        annotation_id: str,
        axis: str = "axial",
        window: str = "lung",
        figsize: tuple = (8, 8),
    ):
        """Display the CT slice at the nodule centre with mask overlay.

        Works in both direct mode (nibabel) and API mode (PNG download).
        """
        if self._engine:
            try:
                sl = self._engine.get_nodule_view(annotation_id, axis=axis, window=window)
                self._plot_slice(sl, title=f"{annotation_id} ({axis}, {window})", figsize=figsize)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            resp = self._api_get_bytes(
                f"/ct/nodule-view?annotation_id={annotation_id}&axis={axis}&window={window}"
            )
            if resp and HAS_IPYTHON:
                display(IPImage(data=resp))

    def show_slice(
        self,
        ct_path: str,
        axis: str = "axial",
        index: Optional[int] = None,
        mask_path: Optional[str] = None,
        window: str = "lung",
        figsize: tuple = (8, 8),
    ):
        """Display an arbitrary CT slice."""
        if self._engine:
            sl = self._engine.get_slice(
                ct_path=ct_path,
                axis=axis,
                index=index,
                mask_path=mask_path,
                window=window,
            )
            self._plot_slice(sl, title=f"{ct_path} (z={sl.slice_index})", figsize=figsize)
        else:
            params = f"ct_path={ct_path}&axis={axis}&window={window}"
            if index is not None:
                params += f"&index={index}"
            if mask_path:
                params += f"&mask_path={mask_path}"
            resp = self._api_get_bytes(f"/ct/slice?{params}")
            if resp and HAS_IPYTHON:
                display(IPImage(data=resp))

    # ── E. Export ─────────────────────────────────────────────────────────────

    def export_csv(self, output_path: str, include_paths: bool = True, **kwargs) -> str:
        """Export search results to CSV."""
        if self._engine:
            from itrialspace.apps.retriever.search import SearchFilters

            sf = SearchFilters(**kwargs)
            return self._engine.export_search_csv(sf, output_path, include_paths=include_paths)
        else:
            # Download from API
            body = {"filters": kwargs, "format": "csv", "include_paths": include_paths}
            import requests

            resp = requests.post(f"{self._api_url}/export", json=body, timeout=60)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path

    # ── F. Nodule detail ──────────────────────────────────────────────────────

    def detail(self, annotation_id: str, display_detail: bool = True) -> dict:
        """Get full detail for a nodule."""
        if self._engine:
            d = self._engine.get_nodule_detail(annotation_id)
        else:
            d = self._api_get(f"/nodule/{annotation_id}")

        if display_detail and HAS_IPYTHON:
            info = d.get("detail", d) if isinstance(d, dict) else d
            rows = [(k, v) for k, v in info.items() if not str(k).startswith("_")]
            display(pd.DataFrame(rows, columns=["Field", "Value"]))
        return d

    # ── G. Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if self._engine:
            return self._engine.summary()
        return self._api_get("/health")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _plot_slice(self, sl, title: str = "", figsize: tuple = (8, 8)):
        """Render a SliceResult with matplotlib."""
        if not HAS_MPL:
            print("matplotlib required for inline plotting")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(sl.windowed, cmap="gray", aspect="auto")

        if sl.mask is not None and sl.mask.any():
            mask_rgba = np.zeros((*sl.mask.shape, 4), dtype=np.float32)
            mask_rgba[sl.mask > 0] = [1, 0.2, 0.2, 0.4]
            ax.imshow(mask_rgba, aspect="auto")

        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _format_results(df: pd.DataFrame) -> pd.DataFrame:
        """Pick display columns for nice notebook rendering."""
        cols = [
            c
            for c in [
                "annotation_id",
                "dataset",
                "label",
                "lobe_name",
                "nodule_mean_diam_mm",
                "pleural_distance_mm",
                "reinsertion_lobe",
                "reinsertion_nodule_diam_mm",
            ]
            if c in df.columns
        ]
        return df[cols] if cols else df

    def _api_get(self, path: str) -> dict:
        import requests

        resp = requests.get(f"{self._api_url}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _api_post(self, path: str, body: dict) -> dict:
        import requests

        resp = requests.post(f"{self._api_url}{path}", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _api_get_bytes(self, path: str) -> Optional[bytes]:
        import requests

        resp = requests.get(f"{self._api_url}{path}", timeout=30)
        if resp.ok:
            return resp.content
        return None
