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
rebuild_artifacts.py -- delete stale NoduleMap artifacts and rebuild from scratch.

This ensures no NLST3D data remains in embeddings, metadata, or edge lists.

Usage:
    python -m itrialspace.apps.nodulemap.tools.rebuild_artifacts --clean
    python -m itrialspace.apps.nodulemap.tools.rebuild_artifacts --clean --outdir ./nodulemap_artifacts
    python -m itrialspace.apps.nodulemap.tools.rebuild_artifacts --clean --models PCA_2D UMAP_2D
    python -m itrialspace.apps.nodulemap.tools.rebuild_artifacts --clean --skip-edges   # faster: only embeddings
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_DIR = str(PROJECT_ROOT / "nodulemap_artifacts")

ARTIFACT_PATTERNS = [
    "embeddings_*.npy",
    "features_*.npy",
    "metadata_*.parquet",
    "edges_*.parquet",
    "preprocessor_*.pkl",
    "info_*.yaml",
    "id_map*",
    "positions_*",
    "*.faiss",
    "*.duckdb",
    "*.duckdb.wal",
]


def clean_artifacts(artifact_dir: str) -> int:
    """Delete all existing artifacts. Returns count of files deleted."""
    deleted = 0
    for pattern in ARTIFACT_PATTERNS:
        for f in glob.glob(os.path.join(artifact_dir, pattern)):
            os.remove(f)
            deleted += 1
    return deleted


def rebuild(
    artifact_dir: str,
    models: list[str] | None = None,
    feature_set: str = "reinsertion_core",
    skip_edges: bool = False,
    space: str = "weighted",
    verbose: bool = True,
) -> dict:
    """Rebuild all artifacts from scratch."""
    from itrialspace.apps.nodulemap.embeddings.build_embeddings import EmbeddingBuilder
    from itrialspace.apps.nodulemap.neighbors.build_edges import EdgeBuilder

    if verbose:
        print(f"\n{'=' * 60}")
        print("  NoduleMap Artifact Rebuild")
        print(f"{'=' * 60}\n")

    # Build NoduleIndex
    builder = EmbeddingBuilder.from_defaults(verbose=verbose)

    # Check NLST3D is absent
    datasets = sorted(builder._df["dataset"].unique().tolist())
    if "NLST3D" in datasets:
        print("FATAL: NLST3D is still present in NoduleIndex!")
        print("Fix datasets.yaml and profile CSVs before rebuilding.")
        sys.exit(1)

    if verbose:
        print(f"\n  Datasets in index: {datasets}")
        print(f"  Total nodules: {len(builder._df):,}")
        print()

    if models is None:
        models = builder.list_models()

    results = {}
    for model in models:
        if verbose:
            print(f"\n{'=' * 60}\n  Building {model} ...\n{'=' * 60}")
        t0 = time.time()
        result = builder.build(model, feature_set, artifact_dir)
        elapsed = time.time() - t0
        results[model] = result
        if verbose:
            print(f"  {model}: {result['embeddings_shape']} ({elapsed:.1f}s)")

    # Build edges
    if not skip_edges:
        eb = EdgeBuilder(artifact_dir, verbose=verbose)
        for model in models:
            if verbose:
                print(f"\n  Building edges for {model} (space={space}) ...")
            eb.build_multiple_k(model, feature_set, space=space)

    # Print dataset inventory from rebuilt metadata
    if verbose:
        print(f"\n{'=' * 60}")
        print("  Post-Rebuild Dataset Inventory")
        print(f"{'=' * 60}\n")

        import pandas as pd

        # Use any metadata file
        meta_path = os.path.join(artifact_dir, f"metadata_{feature_set}__{models[0]}.parquet")
        if os.path.exists(meta_path):
            meta = pd.read_parquet(meta_path)
            print(f"  Total nodes: {len(meta):,}")
            print(f"  Datasets: {sorted(meta['dataset'].unique().tolist())}")
            print()
            print(f"  {'Dataset':<10} {'Count':>8}")
            print("  " + "-" * 20)
            for ds, cnt in meta["dataset"].value_counts().sort_index().items():
                print(f"  {ds:<10} {cnt:>8,}")
            print()
            if "NLST3D" in meta["dataset"].values:
                print("  FAIL: NLST3D found in rebuilt artifacts!")
                sys.exit(1)
            else:
                print("  NLST3D absent: CONFIRMED")

    return results


def main():
    parser = argparse.ArgumentParser(description="Rebuild NoduleMap artifacts")
    parser.add_argument(
        "--clean",
        action="store_true",
        required=True,
        help="Delete existing artifacts before rebuild (required)",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Artifact directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--models", nargs="*", default=None, help="Model names to build (default: all)"
    )
    parser.add_argument(
        "--feature-set", default="reinsertion_core", help="Feature set (default: reinsertion_core)"
    )
    parser.add_argument(
        "--skip-edges", action="store_true", help="Skip edge list computation (faster)"
    )
    parser.add_argument(
        "--space",
        choices=["weighted", "feature_l2", "embedding"],
        default="weighted",
        help="Search space for edges (default: weighted = clinical ReinsertionMatcher metric)",
    )
    args = parser.parse_args()

    artifact_dir = args.outdir
    os.makedirs(artifact_dir, exist_ok=True)

    if args.clean:
        n = clean_artifacts(artifact_dir)
        print(f"Cleaned {n} stale artifact files from {artifact_dir}")

    rebuild(
        artifact_dir=artifact_dir,
        models=args.models,
        feature_set=args.feature_set,
        skip_edges=args.skip_edges,
        space=args.space,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
