#!/usr/bin/env python3
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
iTrialSpace Retriever CLI — command-line interface.

Usage:
    python -m itrialspace.apps.retriever.cli search --label 1 --lobe right_lung_upper_lobe --limit 20
    python -m itrialspace.apps.retriever.cli similar --id DLCS24_n0001 --k 10
    python -m itrialspace.apps.retriever.cli match  --lobe left_lung_lower_lobe --diameter 12 --k 5
    python -m itrialspace.apps.retriever.cli export --label 1 --format csv -o results.csv
    python -m itrialspace.apps.retriever.cli info
    python -m itrialspace.apps.retriever.cli serve  --port 8421
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Lazy engine loader
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from itrialspace.apps.retriever.engine import RetrieverEngine

        print("Loading NoduleIndex …", file=sys.stderr)
        _engine = RetrieverEngine.from_defaults(verbose=True)
        print(f"Ready — {_engine.n_nodules:,} nodules.\n", file=sys.stderr)
    return _engine


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_search(args):
    engine = _get_engine()
    from itrialspace.apps.retriever.search import SearchFilters

    kwargs = {}
    if args.datasets:
        kwargs["datasets"] = args.datasets
    if args.label is not None:
        kwargs["label"] = args.label
    if args.lobe:
        kwargs["lobe"] = args.lobe
    if args.zone:
        kwargs["lung_zone"] = args.zone
    if args.side:
        kwargs["lung_side"] = args.side
    if args.diam_min is not None:
        kwargs["diameter_min"] = args.diam_min
    if args.diam_max is not None:
        kwargs["diameter_max"] = args.diam_max
    if args.pleural_min is not None:
        kwargs["pleural_distance_min"] = args.pleural_min
    if args.pleural_max is not None:
        kwargs["pleural_distance_max"] = args.pleural_max
    if args.population:
        kwargs["population_type"] = args.population
    if args.label_source:
        kwargs["label_source"] = args.label_source
    kwargs["limit"] = args.limit
    if args.sort:
        kwargs["sort_by"] = args.sort
        kwargs["sort_ascending"] = not args.desc

    sf = SearchFilters(**kwargs)
    result = engine.search(sf)

    print(f"Total matching: {result.total_matching:,}")
    print(f"Showing: {len(result.df)}\n")

    display_cols = [
        "annotation_id",
        "dataset",
        "label",
        "lobe_name",
        "nodule_mean_diam_mm",
        "pleural_distance_mm",
    ]
    cols = [c for c in display_cols if c in result.df.columns]
    print(result.df[cols].to_string(index=False))


def cmd_similar(args):
    engine = _get_engine()

    kwargs = {}
    if args.label is not None:
        kwargs["label"] = args.label
    if args.datasets:
        kwargs["include_datasets"] = args.datasets

    results = engine.find_similar(
        annotation_id=args.id,
        k=args.k,
        exclude_same_patient=not args.allow_same_patient,
        **kwargs,
    )

    if not results:
        print("No similar nodules found.")
        return

    print(f"Top-{args.k} similar to {args.id}:\n")
    rows = []
    for r in results:
        rows.append(
            {
                "rank": r.rank,
                "annotation_id": r.annotation_id,
                "dataset": r.dataset,
                "distance": f"{r.distance:.4f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_match(args):
    engine = _get_engine()

    kwargs = {}
    if args.cc_pct is not None:
        kwargs["lobe_cc_pct"] = args.cc_pct
    if args.pleural is not None:
        kwargs["pleural_dist_mm"] = args.pleural
    if args.diameter is not None:
        kwargs["diameter_mm"] = args.diameter
    if args.label is not None:
        kwargs["label"] = args.label
    if args.datasets:
        kwargs["include_datasets"] = args.datasets

    results = engine.find_reinsertion_match(lobe=args.lobe, k=args.k, **kwargs)

    if not results:
        print("No matches found.")
        return

    print(f"Top-{args.k} reinsertion matches for {args.lobe}:\n")
    rows = []
    for m in results:
        label_str = {0: "benign", 1: "malignant", None: "?"}[m.label]
        rows.append(
            {
                "annotation_id": m.annotation_id,
                "dataset": m.dataset,
                "score": f"{m.score:.4f}",
                "diameter_mm": f"{m.diameter_mm:.1f}",
                "lobe_cc_pct": f"{m.lobe_cc_pct:.1f}",
                "label": label_str,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_export(args):
    engine = _get_engine()
    from itrialspace.apps.retriever.search import SearchFilters

    kwargs = {}
    if args.datasets:
        kwargs["datasets"] = args.datasets
    if args.label is not None:
        kwargs["label"] = args.label
    if args.lobe:
        kwargs["lobe"] = args.lobe
    if args.diam_min is not None:
        kwargs["diameter_min"] = args.diam_min
    if args.diam_max is not None:
        kwargs["diameter_max"] = args.diam_max

    sf = SearchFilters(**kwargs)

    if args.format == "csv":
        path = engine.export_search_csv(sf, args.output, include_paths=args.include_paths)
    else:
        path = engine.export_search_json(sf, args.output, include_paths=args.include_paths)

    print(f"Exported to {path}")


def cmd_info(args):
    engine = _get_engine()
    summary = engine.summary()
    print("iTrialSpace Retriever v0.1.0")
    print(f"Nodules:  {summary['n_nodules']:,}")
    print(f"Datasets: {', '.join(summary['datasets'])}")
    print()
    stats_df = pd.DataFrame(summary["stats"]).T
    print(stats_df.to_string())


def cmd_detail(args):
    engine = _get_engine()
    detail = engine.get_nodule_detail(args.id)

    if args.json:
        # Serialise NaN-safe
        cleaned = {}
        for k, v in detail.items():
            if isinstance(v, float) and pd.isna(v):
                cleaned[k] = None
            elif isinstance(v, pd.Series):
                cleaned[k] = v.to_dict()
            else:
                cleaned[k] = v
        print(json.dumps(cleaned, indent=2, default=str))
    else:
        for k, v in detail.items():
            if not str(k).startswith("_"):
                print(f"  {k:40s}  {v}")


def cmd_serve(args):
    """Launch the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"Starting retriever API on {args.host}:{args.port}")
    uvicorn.run(
        "itrialspace.apps.retriever.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retriever",
        description="iTrialSpace Retriever CLI — search, similarity, matching, export.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- search ---
    p_search = sub.add_parser("search", help="Faceted search")
    p_search.add_argument("--datasets", nargs="+", default=None)
    p_search.add_argument("--label", type=int, choices=[0, 1], default=None)
    p_search.add_argument("--lobe", nargs="+", default=None)
    p_search.add_argument("--zone", nargs="+", default=None)
    p_search.add_argument("--side", default=None)
    p_search.add_argument("--diam-min", type=float, default=None)
    p_search.add_argument("--diam-max", type=float, default=None)
    p_search.add_argument("--pleural-min", type=float, default=None)
    p_search.add_argument("--pleural-max", type=float, default=None)
    p_search.add_argument("--population", choices=["screening", "diagnostic"], default=None)
    p_search.add_argument("--label-source", choices=["histopathology", "radiology"], default=None)
    p_search.add_argument("--limit", type=int, default=50)
    p_search.add_argument("--sort", default=None)
    p_search.add_argument("--desc", action="store_true")
    p_search.set_defaults(func=cmd_search)

    # -- similar ---
    p_sim = sub.add_parser("similar", help="Find similar nodules")
    p_sim.add_argument("--id", required=True, help="Reference annotation_id")
    p_sim.add_argument("--k", type=int, default=10)
    p_sim.add_argument("--label", type=int, choices=[0, 1], default=None)
    p_sim.add_argument("--datasets", nargs="+", default=None)
    p_sim.add_argument("--allow-same-patient", action="store_true")
    p_sim.set_defaults(func=cmd_similar)

    # -- match ---
    p_match = sub.add_parser("match", help="Reinsertion anatomy matching")
    p_match.add_argument("--lobe", required=True)
    p_match.add_argument("--k", type=int, default=10)
    p_match.add_argument("--cc-pct", type=float, default=None)
    p_match.add_argument("--pleural", type=float, default=None)
    p_match.add_argument("--diameter", type=float, default=None)
    p_match.add_argument("--label", type=int, choices=[0, 1], default=None)
    p_match.add_argument("--datasets", nargs="+", default=None)
    p_match.set_defaults(func=cmd_match)

    # -- export ---
    p_export = sub.add_parser("export", help="Export search results")
    p_export.add_argument("-o", "--output", required=True)
    p_export.add_argument("--format", choices=["csv", "json"], default="csv")
    p_export.add_argument("--datasets", nargs="+", default=None)
    p_export.add_argument("--label", type=int, choices=[0, 1], default=None)
    p_export.add_argument("--lobe", nargs="+", default=None)
    p_export.add_argument("--diam-min", type=float, default=None)
    p_export.add_argument("--diam-max", type=float, default=None)
    p_export.add_argument("--include-paths", action="store_true", default=True)
    p_export.add_argument("--no-paths", dest="include_paths", action="store_false")
    p_export.set_defaults(func=cmd_export)

    # -- info ---
    p_info = sub.add_parser("info", help="Index summary statistics")
    p_info.set_defaults(func=cmd_info)

    # -- detail ---
    p_detail = sub.add_parser("detail", help="Full detail for a nodule")
    p_detail.add_argument("--id", required=True, help="annotation_id")
    p_detail.add_argument("--json", action="store_true", help="Output as JSON")
    p_detail.set_defaults(func=cmd_detail)

    # -- serve ---
    p_serve = sub.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8421)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
