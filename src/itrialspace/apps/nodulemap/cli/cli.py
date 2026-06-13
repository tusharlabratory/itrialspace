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
cli.py — itrialspace-nodulemap command-line interface.

Commands:
  build    — Generate embeddings for a model + feature set
  edges    — Compute KNN edge lists
  serve    — Start the FastAPI + frontend server
  info     — Show available models / feature sets / artifacts
  query    — Query neighbors for a node (offline, no server)
  export   — Export neighbors for a node

Usage:
  python -m itrialspace.apps.nodulemap build --model UMAP_2D --feature-set reinsertion_core --outdir ./nodulemap_artifacts
  python -m itrialspace.apps.nodulemap edges --model UMAP_2D --feature-set reinsertion_core --k 5 --mode closest --outdir ./nodulemap_artifacts
  python -m itrialspace.apps.nodulemap serve --artifact-dir ./nodulemap_artifacts --port 8422
  python -m itrialspace.apps.nodulemap info --artifact-dir ./nodulemap_artifacts
  python -m itrialspace.apps.nodulemap query --node DLCS24_DLCS_0001_01 --model UMAP_2D --k 10 --artifact-dir ./nodulemap_artifacts
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="itrialspace-nodulemap",
        description="iTrialSpace NoduleMap — embedding-based nodule similarity explorer",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── build ──────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Generate embeddings")
    p_build.add_argument("--model", default="UMAP_2D", help="Model preset name (default: UMAP_2D)")
    p_build.add_argument(
        "--feature-set", default="reinsertion_core", help="Feature set (default: reinsertion_core)"
    )
    p_build.add_argument(
        "--outdir",
        default=None,
        help="Artifacts output dir (default: $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)",
    )
    p_build.add_argument("--all-2d", action="store_true", help="Build PCA_2D + UMAP_2D + TSNE_2D")

    # ── edges ──────────────────────────────────────────────────────────
    p_edges = sub.add_parser("edges", help="Compute KNN edge lists")
    p_edges.add_argument("--model", default="UMAP_2D", help="Model preset (default: UMAP_2D)")
    p_edges.add_argument("--feature-set", default="reinsertion_core", help="Feature set")
    p_edges.add_argument("--k", type=int, default=5, help="Number of neighbors (default: 5)")
    p_edges.add_argument("--mode", choices=["closest", "farthest"], default="closest")
    p_edges.add_argument(
        "--scope", choices=["cross_dataset", "within_dataset"], default="cross_dataset"
    )
    p_edges.add_argument(
        "--outdir",
        default=None,
        help="Artifacts output dir (default: $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)",
    )
    p_edges.add_argument("--all-k", action="store_true", help="Build edges for k=3,5,10,25")
    p_edges.add_argument(
        "--space",
        choices=["weighted", "feature_l2", "embedding"],
        default="weighted",
        help="Search space for edges (default: weighted = clinical ReinsertionMatcher metric)",
    )

    # ── serve ──────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start NoduleMap web server")
    p_serve.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifacts dir (default: $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)",
    )
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8422)
    p_serve.add_argument("--workers", type=int, default=1)

    # ── info ───────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show available artifacts")
    p_info.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifacts dir (default: $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)",
    )

    # ── query ──────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Query neighbors offline")
    p_query.add_argument("--node", required=True, help="Node ID (e.g. DLCS24_DLCS_0001_01)")
    p_query.add_argument("--model", default="UMAP_2D")
    p_query.add_argument("--feature-set", default="reinsertion_core")
    p_query.add_argument("--k", type=int, default=10)
    p_query.add_argument("--scope", default="cross_dataset")
    p_query.add_argument(
        "--search-backend",
        choices=["weighted", "feature_l2", "embedding"],
        default="weighted",
        help="Similarity backend (default: weighted = clinical ReinsertionMatcher metric)",
    )
    p_query.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifacts dir (default: $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)",
    )
    p_query.add_argument("--json", action="store_true", help="Output as JSON")

    # ── export ─────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export neighbors for a node")
    p_export.add_argument("--node", required=True, help="Node ID")
    p_export.add_argument("--model", default="UMAP_2D")
    p_export.add_argument("--feature-set", default="reinsertion_core")
    p_export.add_argument("--k", type=int, default=10)
    p_export.add_argument("--scope", default="cross_dataset")
    p_export.add_argument(
        "--search-backend",
        choices=["weighted", "feature_l2", "embedding"],
        default="weighted",
        help="Similarity backend (default: weighted = clinical ReinsertionMatcher metric)",
    )
    p_export.add_argument("--format", choices=["csv", "json", "donors"], default="csv")
    p_export.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifacts dir (default: $NODULEMAP_ARTIFACTS or $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts)",
    )
    p_export.add_argument("-o", "--output-dir", default="./nodulemap_exports")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Resolve the artifact directory: explicit flag wins, else
    # $NODULEMAP_ARTIFACTS, else $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts.
    from itrialspace.config import settings

    art_default = str(settings.nodulemap_artifacts_dir())
    for attr in ("outdir", "artifact_dir"):
        if getattr(args, attr, None) is None:
            setattr(args, attr, art_default)

    if args.command == "build":
        _cmd_build(args)
    elif args.command == "edges":
        _cmd_edges(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "query":
        _cmd_query(args)
    elif args.command == "export":
        _cmd_export(args)


# ── Command implementations ──────────────────────────────────────────


def _cmd_build(args):
    from itrialspace.apps.nodulemap.embeddings.build_embeddings import EmbeddingBuilder

    builder = EmbeddingBuilder.from_defaults()

    if args.all_2d:
        results = builder.build_all_2d(feature_set=args.feature_set, outdir=args.outdir)
        for model, res in results.items():
            print(f"\n✓ {model}: {res['embeddings_shape']}")
    else:
        result = builder.build(args.model, args.feature_set, args.outdir)
        print(f"\n✓ {args.model}: {result['embeddings_shape']}")
        print(f"  Artifacts: {args.outdir}")


def _cmd_edges(args):
    from itrialspace.apps.nodulemap.neighbors.build_edges import EdgeBuilder

    eb = EdgeBuilder(args.outdir)

    if args.all_k:
        paths = eb.build_multiple_k(args.model, args.feature_set, mode=args.mode, space=args.space)
        print(f"\n✓ Built edges for k={[3,5,10,25]} (space={args.space})")
    else:
        path = eb.build(
            args.model,
            args.feature_set,
            k=args.k,
            mode=args.mode,
            scope=args.scope,
            space=args.space,
        )
        print(f"\n✓ Edges saved: {path}")


def _cmd_serve(args):
    import uvicorn

    os.environ["NODULEMAP_ARTIFACTS"] = args.artifact_dir
    print(f"Starting NoduleMap server on {args.host}:{args.port}")
    print(f"  Artifacts: {args.artifact_dir}")
    print(f"  Open: http://localhost:{args.port}")
    uvicorn.run(
        "itrialspace.apps.nodulemap.backend.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


def _cmd_info(args):
    from pathlib import Path

    art_dir = Path(args.artifact_dir)
    if not art_dir.exists():
        print(f"Artifact directory not found: {art_dir}")
        sys.exit(1)

    # Find embedding files
    emb_files = sorted(art_dir.glob("embeddings_*.npy"))
    edge_files = sorted(art_dir.glob("edges_*.parquet"))
    meta_files = sorted(art_dir.glob("metadata_*.parquet"))

    print(f"\niTrialSpace NoduleMap — Artifacts in {art_dir}")
    print("=" * 60)

    if emb_files:
        print(f"\nEmbeddings ({len(emb_files)}):")
        for f in emb_files:
            import numpy as np

            arr = np.load(f)
            name = f.stem.replace("embeddings_", "")
            print(f"  {name}: {arr.shape[0]} nodes × {arr.shape[1]}D")
    else:
        print("\nNo embeddings found. Run: itrialspace-nodulemap build")

    if edge_files:
        print(f"\nEdge lists ({len(edge_files)}):")
        for f in edge_files:
            import pandas as pd

            df = pd.read_parquet(f)
            name = f.stem.replace("edges_", "")
            print(f"  {name}: {len(df):,} edges")
    else:
        print("\nNo edge lists found. Run: itrialspace-nodulemap edges")

    if meta_files:
        print(f"\nMetadata ({len(meta_files)}):")
        for f in meta_files:
            import pandas as pd

            df = pd.read_parquet(f)
            name = f.stem.replace("metadata_", "")
            ds_counts = df["dataset"].value_counts()
            print(f"  {name}: {len(df):,} nodes across {len(ds_counts)} datasets")

    print()


def _cmd_query(args):
    import json as jsonlib

    from itrialspace.apps.nodulemap.backend.data_store import DataStore

    ds = DataStore(args.artifact_dir, verbose=False)
    results = ds.query_neighbors(
        node_id=args.node,
        model_name=args.model,
        feature_set=args.feature_set,
        k=args.k,
        scope=args.scope,
        search_backend=args.search_backend,
    )

    if args.json:
        print(jsonlib.dumps(results, indent=2, default=str))
    else:
        print(f"\nNeighbors for {args.node} (model={args.model}, k={args.k}, scope={args.scope})")
        print("-" * 80)
        print(
            f"{'Rank':>4}  {'Node ID':<30}  {'Distance':>10}  {'Similarity':>10}  {'Diam':>6}  {'Lobe'}"
        )
        for r in results:
            lobe = (r.get("reinsertion_lobe") or "—")[:20]
            print(
                f"{r['rank']:>4}  {r['node_id']:<30}  {r['distance']:>10.4f}  {r['similarity']:>10.4f}  {r.get('reinsertion_nodule_diam_mm', 0):>6.1f}  {lobe}"
            )


def _cmd_export(args):
    from itrialspace.apps.nodulemap.backend.data_store import DataStore
    from itrialspace.apps.nodulemap.backend.export import Exporter

    ds = DataStore(args.artifact_dir, verbose=False)
    exporter = Exporter(args.output_dir)

    query_node = ds.get_node(args.node, args.model, args.feature_set)
    neighbors = ds.query_neighbors(
        node_id=args.node,
        model_name=args.model,
        feature_set=args.feature_set,
        k=args.k,
        scope=args.scope,
        search_backend=args.search_backend,
    )

    if args.format == "donors":
        path = exporter.export_candidate_donors(neighbors)
    else:
        path = exporter.export_neighbors(
            query_node=query_node,
            neighbors=neighbors,
            model_name=args.model,
            feature_set=args.feature_set,
            k=args.k,
            scope=args.scope,
            fmt=args.format,
        )
    print(f"✓ Exported: {path}")


if __name__ == "__main__":
    main()
