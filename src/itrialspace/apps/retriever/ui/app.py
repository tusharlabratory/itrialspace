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
Streamlit frontend for the iTrialSpace Retriever.

Launch:
    streamlit run retriever/ui/app.py -- [--api-url http://localhost:8421]

Two modes:
    1. **Direct mode** (default):  Loads the NoduleIndex in-process.
    2. **API mode** (--api-url):   Calls the FastAPI backend via requests.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Mode detection  (direct vs API)
# ---------------------------------------------------------------------------


def _parse_api_url() -> Optional[str]:
    """Check for --api-url flag in sys.argv (Streamlit passes args after --)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--api-url" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


API_URL = _parse_api_url()

# ---------------------------------------------------------------------------
# Engine / API wrappers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading 13k+ nodules …")
def _load_engine():
    """Load the retriever engine in direct mode."""
    from itrialspace.apps.retriever.engine import RetrieverEngine

    return RetrieverEngine.from_defaults(verbose=True)


def _search(filters: dict) -> dict:
    """Run a search, returning {total_matching, results (list[dict]), facet_counts}."""
    if API_URL:
        import requests

        resp = requests.post(f"{API_URL}/search", json=filters, timeout=30)
        resp.raise_for_status()
        return resp.json()
    else:
        from itrialspace.apps.retriever.search import SearchFilters

        engine = _load_engine()
        sf = SearchFilters(**{k: v for k, v in filters.items() if v is not None})
        result = engine.search(sf)
        rows = result.df.to_dict(orient="records")
        # Serialise NaN
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = None
        return {
            "total_matching": result.total_matching,
            "results": rows,
            "facet_counts": result.facet_counts,
        }


def _find_similar(annotation_id: str, k: int, **kwargs) -> list[dict]:
    if API_URL:
        import requests

        body = {"annotation_id": annotation_id, "k": k, **kwargs}
        resp = requests.post(f"{API_URL}/similar", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["results"]
    else:
        engine = _load_engine()
        results = engine.find_similar(annotation_id=annotation_id, k=k, **kwargs)
        return [
            {
                "annotation_id": r.annotation_id,
                "dataset": r.dataset,
                "distance": round(r.distance, 4),
                "rank": r.rank,
            }
            for r in results
        ]


def _find_match(lobe: str, k: int = 10, **kwargs) -> list[dict]:
    if API_URL:
        import requests

        body = {"lobe": lobe, "k": k, **kwargs}
        resp = requests.post(f"{API_URL}/matcher", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["results"]
    else:
        engine = _load_engine()
        results = engine.find_reinsertion_match(lobe=lobe, k=k, **kwargs)
        return [
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


def _get_nodule_detail(annotation_id: str) -> dict:
    if API_URL:
        import requests

        resp = requests.get(f"{API_URL}/nodule/{annotation_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    else:
        engine = _load_engine()
        return engine.get_nodule_detail(annotation_id)


def _get_slice_png(
    ct_path: str, axis: str, index: int, mask_path: str = None, window: str = "lung"
) -> Optional[bytes]:
    """Return PNG bytes for a CT slice."""
    if API_URL:
        import requests

        params = {"ct_path": ct_path, "axis": axis, "index": index, "window": window}
        if mask_path:
            params["mask_path"] = mask_path
        resp = requests.get(f"{API_URL}/ct/slice", params=params, timeout=30)
        if resp.ok:
            return resp.content
        return None
    else:
        try:
            engine = _load_engine()
            sl = engine.get_slice(
                ct_path=ct_path, axis=axis, index=index, mask_path=mask_path, window=window
            )
            return sl.to_png_bytes()
        except Exception:
            return None


def _get_nodule_view_png(
    annotation_id: str, axis: str = "axial", window: str = "lung", show_mask: bool = True
) -> Optional[bytes]:
    """Return PNG bytes for the CT slice at the nodule centre."""
    if API_URL:
        import requests

        params = {
            "annotation_id": annotation_id,
            "axis": axis,
            "window": window,
            "show_mask": show_mask,
        }
        resp = requests.get(f"{API_URL}/ct/nodule-view", params=params, timeout=30)
        if resp.ok:
            return resp.content
        return None
    else:
        try:
            engine = _load_engine()
            sl = engine.get_nodule_view(
                annotation_id, axis=axis, window=window, show_mask=show_mask
            )
            return sl.to_png_bytes()
        except Exception:
            return None


def _get_volume_info(ct_path: str) -> Optional[dict]:
    """Return {shape, spacing_mm} for a CT volume."""
    if API_URL:
        import requests

        resp = requests.get(f"{API_URL}/ct/info", params={"ct_path": ct_path}, timeout=15)
        if resp.ok:
            return resp.json()
        return None
    else:
        try:
            engine = _load_engine()
            shape = engine.volume_shape(ct_path)
            spacing = engine._slicer.voxel_spacing(ct_path)
            return {"shape": list(shape), "spacing_mm": [round(s, 3) for s in spacing]}
        except Exception:
            return None


def _resolve_paths(annotation_id: str) -> dict:
    """Resolve all file paths for a nodule."""
    if API_URL:
        import requests

        resp = requests.get(f"{API_URL}/nodule/{annotation_id}", timeout=15)
        if resp.ok:
            data = resp.json()
            return data.get("paths", {})
        return {}
    else:
        try:
            engine = _load_engine()
            return engine.resolve_paths(annotation_id)
        except Exception:
            return {}


def _available_filters() -> dict:
    if API_URL:
        import requests

        resp = requests.get(f"{API_URL}/filters", timeout=10)
        resp.raise_for_status()
        return resp.json()
    else:
        engine = _load_engine()
        return engine.available_filters


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Branding (consistent with the NoduleMap viewer)
# ---------------------------------------------------------------------------

APP_VERSION = "0.1.0"
ACCENT = "#4a9eff"
GITHUB_URL = "https://github.com/tusharlabratory/itrialspace"
ORG = "Tushar Lab · Dept. of Radiology & Imaging Sciences · University of Arizona"
DISCLAIMER = "For research use only — not for clinical use."


def _render_brand_header():
    st.markdown(
        f"""
        <style>
          .its-appbar {{ display:flex; align-items:center; gap:14px;
            padding:2px 2px 10px; border-bottom:1px solid #2d3140; margin-bottom:12px; }}
          .its-appbar .mark {{ font-size:24px; line-height:1; }}
          .its-appbar .title {{ font-size:20px; font-weight:700; letter-spacing:.2px; }}
          .its-appbar .title .dot {{ color:{ACCENT}; margin:0 4px; }}
          .its-appbar .title .sub {{ color:#9ba0ad; font-weight:600; }}
          .its-appbar .org {{ font-size:11px; color:#9ba0ad; margin-top:1px; }}
          .its-appbar .spacer {{ flex:1; }}
          .its-appbar .chip {{ font-size:11px; color:#9ba0ad;
            border:1px solid #2d3140; border-radius:10px; padding:1px 8px; }}
          .its-appbar a {{ color:{ACCENT}; text-decoration:none; font-size:12px; }}
        </style>
        <div class="its-appbar">
          <span class="mark">🫁</span>
          <span>
            <div class="title">iTrialSpace<span class="dot">·</span><span class="sub">Retriever</span></div>
            <div class="org">{ORG}</div>
          </span>
          <span class="spacer"></span>
          <span class="chip">v{APP_VERSION}</span>
          <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_about():
    with st.sidebar.expander("ℹ About"):
        st.markdown(f"""
**iTrialSpace · Retriever** — browse, visualize, and export lung-nodule data across
**13,140 nodules / 7 datasets**. Faceted search, query-by-example similarity, and
reinsertion donor matching (clinical `ReinsertionMatcher` metric), with a CT-slice viewer
and CSV/JSON export.

[{GITHUB_URL.replace("https://", "")}]({GITHUB_URL})

⚠ **{DISCLAIMER}** Not a medical device.
""")


def _render_footer():
    st.markdown(
        f"""
        <div style="margin-top:18px;padding-top:8px;border-top:1px solid #2d3140;
                    font-size:11px;color:#9ba0ad;display:flex;justify-content:space-between">
          <span style="color:#f59e0b">⚠ {DISCLAIMER}</span>
          <span><a href="{GITHUB_URL}" target="_blank" rel="noopener"
                   style="color:#9ba0ad;text-decoration:none">GitHub</a></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _download_buttons(df: pd.DataFrame, stem: str):
    """CSV + JSON download buttons for a result DataFrame (consistent across modes)."""
    st.subheader("Export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            file_name=f"{stem}.csv",
            mime="text/csv",
            key=f"dl_csv_{stem}",
        )
    with c2:
        st.download_button(
            "⬇ Download JSON",
            df.to_json(orient="records", indent=2, default_handler=str),
            file_name=f"{stem}.json",
            mime="application/json",
            key=f"dl_json_{stem}",
        )


def main():
    st.set_page_config(
        page_title="iTrialSpace · Retriever",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_brand_header()
    st.caption("Interactive retrieval across 13,140 lung nodules from 7 datasets")

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        af = _available_filters()

        # Tabs for different modes
        mode = st.radio("Mode", ["Search", "Similarity", "Matcher"], horizontal=True)

        if mode == "Search":
            filters = _build_search_sidebar(af)
        elif mode == "Similarity":
            sim_params = _build_similarity_sidebar()
        else:
            match_params = _build_matcher_sidebar(af)

        _render_about()

    # Main content area
    if mode == "Search":
        _render_search(filters)
    elif mode == "Similarity":
        _render_similarity(sim_params)
    else:
        _render_matcher(match_params)

    _render_footer()


# ---------------------------------------------------------------------------
# Sidebar builders
# ---------------------------------------------------------------------------


def _build_search_sidebar(af: dict) -> dict:
    """Build the faceted search sidebar, return filter dict."""
    filters: dict = {}

    # Dataset multi-select
    ds_options = af.get("datasets", [])
    selected_ds = st.multiselect("Datasets", ds_options, default=None)
    if selected_ds:
        filters["datasets"] = selected_ds

    # Label
    label_choice = st.selectbox("Label", ["Any", "Malignant (1)", "Benign (0)"])
    if label_choice == "Malignant (1)":
        filters["label"] = 1
    elif label_choice == "Benign (0)":
        filters["label"] = 0

    # Lobe
    lobe_options = af.get("lobes", [])
    selected_lobes = st.multiselect("Lobe", lobe_options, default=None)
    if selected_lobes:
        filters["lobe"] = selected_lobes

    # Zone
    zone_options = af.get("zones", [])
    selected_zones = st.multiselect("Zone", zone_options, default=None)
    if selected_zones:
        filters["lung_zone"] = selected_zones

    # Side
    side = st.selectbox("Side", ["Any", "left", "right"])
    if side != "Any":
        filters["lung_side"] = side

    # Central/peripheral
    cp = st.selectbox("Position", ["Any", "central", "peripheral"])
    if cp != "Any":
        filters["central_peripheral"] = cp

    # Diameter range
    st.subheader("Size (mm)")
    diam_range = st.slider("Diameter", 0.0, 100.0, (0.0, 100.0), step=1.0)
    if diam_range[0] > 0:
        filters["diameter_min"] = diam_range[0]
    if diam_range[1] < 100:
        filters["diameter_max"] = diam_range[1]

    # Pleural distance
    st.subheader("Pleural distance")
    pleural = st.slider("Pleural distance (mm)", 0.0, 80.0, (0.0, 80.0), step=1.0)
    if pleural[0] > 0:
        filters["pleural_distance_min"] = pleural[0]
    if pleural[1] < 80:
        filters["pleural_distance_max"] = pleural[1]

    # Population type
    pop = st.selectbox("Population type", ["Any", "screening", "diagnostic"])
    if pop != "Any":
        filters["population_type"] = pop

    # Label source
    ls = st.selectbox("Label source", ["Any", "histopathology", "radiology"])
    if ls != "Any":
        filters["label_source"] = ls

    # Limit
    filters["limit"] = st.number_input("Max results", 10, 5000, 200, step=50)

    # Sort
    sort_col = st.selectbox(
        "Sort by",
        [
            "None",
            "nodule_mean_diam_mm",
            "pleural_distance_mm",
            "reinsertion_lobe_cc_pct",
            "dataset",
            "lobe_name",
        ],
    )
    if sort_col != "None":
        filters["sort_by"] = sort_col
        filters["sort_ascending"] = st.checkbox("Ascending", value=True)

    return filters


def _build_similarity_sidebar() -> dict:
    """Build the similarity query sidebar."""
    params: dict = {}
    params["annotation_id"] = st.text_input(
        "Reference annotation_id", placeholder="e.g. DLCS24_n0001"
    )
    params["k"] = st.slider("Top-K", 1, 100, 10)
    params["exclude_same_patient"] = st.checkbox("Exclude same patient", value=True)

    label_choice = st.selectbox("Label filter", ["Any", "Malignant (1)", "Benign (0)"])
    if label_choice == "Malignant (1)":
        params["label"] = 1
    elif label_choice == "Benign (0)":
        params["label"] = 0

    return params


def _build_matcher_sidebar(af: dict) -> dict:
    """Build the reinsertion matcher sidebar."""
    params: dict = {}
    lobe_options = af.get("lobes", [])
    params["lobe"] = st.selectbox("Target lobe", lobe_options)
    params["k"] = st.slider("Top-K", 1, 100, 10)

    params["lobe_cc_pct"] = st.slider("Lobe CC %", 0.0, 100.0, 50.0, step=1.0)
    params["pleural_dist_mm"] = st.number_input("Pleural dist (mm)", 0.0, 100.0, 15.0, step=1.0)
    params["diameter_mm"] = st.number_input("Target diameter (mm)", 1.0, 60.0, 10.0, step=0.5)

    label_choice = st.selectbox("Label", ["Any", "Malignant (1)", "Benign (0)"])
    if label_choice == "Malignant (1)":
        params["label"] = 1
    elif label_choice == "Benign (0)":
        params["label"] = 0

    return params


# ---------------------------------------------------------------------------
# Main content renderers
# ---------------------------------------------------------------------------


def _render_search(filters: dict):
    """Render faceted search results."""
    if st.sidebar.button("Search", type="primary"):
        with st.spinner("Searching …"):
            data = _search(filters)
        st.session_state["search_data"] = data

    # Render from session_state so the page survives widget interactions
    data = st.session_state.get("search_data")
    if data is None:
        st.info("Configure filters in the sidebar and click **Search**.")
        return

    total = data["total_matching"]
    results = data["results"]
    facets = data.get("facet_counts", {})

    st.metric("Total matching", f"{total:,}")

    # Facet counts in expander
    if facets:
        with st.expander("Facet counts (full index)"):
            for facet_name, counts in facets.items():
                st.write(f"**{facet_name}**")
                st.json(counts)

    if results:
        df = pd.DataFrame(results)
        display_cols = [
            c
            for c in [
                "annotation_id",
                "dataset",
                "label",
                "lobe_name",
                "lung_side",
                "lung_zone",
                "central_peripheral",
                "nodule_mean_diam_mm",
                "pleural_distance_mm",
                "reinsertion_lobe",
                "reinsertion_nodule_diam_mm",
            ]
            if c in df.columns
        ]
        st.dataframe(df[display_cols], use_container_width=True, height=500)

        # Detail viewer
        st.subheader("Nodule detail")
        selected_id = st.selectbox(
            "Select nodule for detail",
            df["annotation_id"].tolist() if "annotation_id" in df.columns else [],
            key="search_detail_select",
        )
        if selected_id:
            _render_detail(selected_id)

        # Export
        _download_buttons(df, "itrialspace_search")
    else:
        st.info("No results. Adjust filters and try again.")


def _render_similarity(params: dict):
    """Render similarity results with CT viewer for similar nodules."""
    aid = params.get("annotation_id", "").strip()
    if not aid:
        st.info("Enter a reference annotation_id in the sidebar to find similar nodules.")
        return

    if st.sidebar.button("Find similar", type="primary"):
        with st.spinner("Computing similarity …"):
            kwargs = {
                k: v for k, v in params.items() if k not in ("annotation_id", "k") and v is not None
            }
            results = _find_similar(aid, params["k"], **kwargs)

        if results:
            st.session_state["sim_results"] = results
            st.session_state["sim_ref"] = aid
        else:
            st.session_state.pop("sim_results", None)
            st.warning("No similar nodules found.")
            return

    results = st.session_state.get("sim_results")
    aid = st.session_state.get("sim_ref", aid)
    if not results:
        return

    st.subheader(f"Top-{len(results)} similar to {aid}")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    _download_buttons(df, "itrialspace_similar")

    # ── Window preset for all CT views in this section
    sim_window = st.selectbox(
        "CT window (for all views below)",
        ["lung", "mediastinum", "soft_tissue", "bone"],
        key="sim_ct_window",
    )

    # ── Reference nodule
    with st.expander("Reference nodule detail", expanded=True):
        _render_detail(aid)

    # ── Similar nodules — CT comparison
    st.subheader("Compare similar nodules")
    sim_ids = [r["annotation_id"] for r in results if "annotation_id" in r]
    if not sim_ids:
        return

    selected_sim = st.multiselect(
        "Select similar nodules to view",
        sim_ids,
        default=sim_ids[: min(3, len(sim_ids))],
        key="sim_compare_select",
    )

    if selected_sim:
        # Show in a grid: up to 3 per row
        for row_start in range(0, len(selected_sim), 3):
            row_ids = selected_sim[row_start : row_start + 3]
            cols = st.columns(len(row_ids))
            for col, sid in zip(cols, row_ids):
                with col:
                    st.markdown(f"**{sid}**")
                    match = next((r for r in results if r.get("annotation_id") == sid), {})
                    dist = match.get("distance", "?")
                    diam = match.get("nodule_mean_diam_mm", "?")
                    lobe = match.get("lobe_name", "?")
                    label = match.get("label", "?")
                    st.caption(f"dist={dist}  |  diam={diam}mm  |  {lobe}  |  label={label}")

                    png = _get_nodule_view_png(sid, axis="axial", window=sim_window, show_mask=True)
                    if png:
                        st.image(png, use_container_width=True)
                    else:
                        st.warning("CT unavailable")


def _render_matcher(params: dict):
    """Render reinsertion matcher results."""
    if st.sidebar.button("Find matches", type="primary"):
        with st.spinner("Matching …"):
            kwargs = {k: v for k, v in params.items() if k not in ("lobe", "k") and v is not None}
            results = _find_match(params["lobe"], params["k"], **kwargs)
        st.session_state["matcher_results"] = results
        st.session_state["matcher_params"] = {"k": params["k"], "lobe": params["lobe"]}

    results = st.session_state.get("matcher_results")
    mp = st.session_state.get("matcher_params", {})
    if not results:
        st.info("Configure parameters in the sidebar and click **Find matches**.")
        return

    st.subheader(f"Top-{mp.get('k', '?')} matches for {mp.get('lobe', '?')}")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    _download_buttons(df, "itrialspace_matches")


def _render_detail(annotation_id: str):
    """Render full detail panel for a nodule with CT viewer."""
    try:
        detail = _get_nodule_detail(annotation_id)
    except Exception as e:
        st.error(f"Could not load detail: {e}")
        return

    # If detail is a dict with nested 'detail', flatten
    info = detail.get("detail", detail) if isinstance(detail, dict) else detail
    paths = detail.get("_paths", detail.get("paths", {}))

    # ── CT Viewer — Axial slice at nodule centre ─────────────────────────
    ct_abs = paths.get("ct_path", "")

    if ct_abs:
        wcol1, wcol2 = st.columns([1, 1])
        with wcol1:
            view_window = st.selectbox(
                "Window preset",
                ["lung", "mediastinum", "soft_tissue", "bone"],
                key=f"window_{annotation_id}",
            )
        with wcol2:
            show_mask = st.checkbox("Show nodule mask", value=True, key=f"mask_{annotation_id}")

        png = _get_nodule_view_png(
            annotation_id, axis="axial", window=view_window, show_mask=show_mask
        )

        # Volume info badge
        vol_info = _get_volume_info(ct_abs)
        if vol_info:
            s = vol_info["shape"]
            sp = vol_info["spacing_mm"]
            st.caption(
                f"Volume: {s[0]}×{s[1]}×{s[2]}  |  "
                f"Spacing: {sp[0]:.2f}×{sp[1]:.2f}×{sp[2]:.2f} mm  |  "
                f"Centre: ({info.get('coordX','?')}, {info.get('coordY','?')}, {info.get('coordZ','?')})"
            )

        if png:
            st.image(png, caption=f"Axial — {view_window} window", use_container_width=True)
        else:
            st.warning("CT file not accessible from this node. " f"Expected: `{ct_abs}`")
    else:
        st.info("No CT path resolved for this nodule.")

    # ── Metadata tables ──────────────────────────────────────────────────
    st.subheader("Nodule Metadata")
    col1, col2 = st.columns([2, 1])
    with col1:
        core_keys = [
            "annotation_id",
            "dataset",
            "patient_id",
            "label",
            "lobe_name",
            "lung_side",
            "lung_zone",
            "central_peripheral",
            "nodule_mean_diam_mm",
            "nodule_vol_mm3",
            "pleural_distance_mm",
            "airway_distance_mm",
            "cranio_caudal_pct",
            "mediolateral_pct",
            "anteroposterior_pct",
            "coordX",
            "coordY",
            "coordZ",
        ]
        rows = [(k, info.get(k, "")) for k in core_keys if k in info]
        st.table(pd.DataFrame(rows, columns=["Field", "Value"]))

    with col2:
        rkeys = [k for k in info if str(k).startswith("reinsertion_")]
        if rkeys:
            st.write("**Reinsertion coordinates**")
            rrows = [(k, info[k]) for k in rkeys]
            st.table(pd.DataFrame(rrows, columns=["Field", "Value"]))

    # ── File paths ────────────────────────────────────────────────────────
    if paths:
        with st.expander("File paths"):
            for k, v in paths.items():
                import os

                exists = os.path.isfile(str(v)) if v else False
                icon = "✅" if exists else "❌"
                st.text(f"{icon} {k}: {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
