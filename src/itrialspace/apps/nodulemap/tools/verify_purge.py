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
verify_purge.py — Post-rebuild verification that NLST3D is fully absent
from every NoduleMap artifact.

Checks:
  1. Metadata parquets:  no rows with dataset == "NLST3D"
  2. Info YAMLs:         n_samples matches metadata row count
  3. Embeddings:         .npy row count matches metadata
  4. Edge parquets:      no src_id / dst_id starting with "NLST3D_"
  5. Preprocessor PKLs:  exist and are loadable
  6. Node count:         consistent across all models

Exit code:
  0 = all checks passed
  1 = at least one check failed

Usage:
    python -m itrialspace.apps.nodulemap.tools.verify_purge --artifact-dir ./nodulemap_artifacts
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def verify(artifact_dir: str) -> bool:
    """Run all checks. Returns True if clean."""
    art = Path(artifact_dir)
    ok = True
    n_nodes_set: set[int] = set()

    print(f"\n{'=' * 64}")
    print("  NoduleMap NLST3D Purge Verification")
    print(f"  Artifact dir: {art}")
    print(f"{'=' * 64}\n")

    # ── 1. Metadata parquets ─────────────────────────────────────────
    meta_files = sorted(art.glob("metadata_*.parquet"))
    if not meta_files:
        _fail("No metadata parquets found")
        return False

    for mf in meta_files:
        df = pd.read_parquet(mf)
        tag = mf.stem.replace("metadata_", "")
        datasets = sorted(df["dataset"].unique())
        n = len(df)
        n_nodes_set.add(n)

        if "NLST3D" in datasets:
            nlst_n = (df["dataset"] == "NLST3D").sum()
            _fail(f"{mf.name}: NLST3D found ({nlst_n} rows)")
            ok = False
        else:
            _pass(f"{mf.name}: {n:,} rows, {len(datasets)} datasets — no NLST3D")

        # Check for NLST3D in node_id
        if "node_id" in df.columns:
            nlst_ids = df["node_id"].str.startswith("NLST3D_").sum()
            if nlst_ids > 0:
                _fail(f"{mf.name}: {nlst_ids} node_ids start with NLST3D_")
                ok = False

    # ── 2. Embeddings .npy ───────────────────────────────────────────
    emb_files = sorted(art.glob("embeddings_*.npy"))
    for ef in emb_files:
        arr = np.load(ef)
        tag = ef.stem.replace("embeddings_", "")
        n_nodes_set.add(arr.shape[0])
        _pass(f"{ef.name}: {arr.shape} float32")

    # ── 3. Info YAMLs ────────────────────────────────────────────────
    info_files = sorted(art.glob("info_*.yaml"))
    for yf in info_files:
        with open(yf) as f:
            info = yaml.safe_load(f)
        n_yaml = info.get("n_samples", -1)
        n_nodes_set.add(n_yaml)
        _pass(f"{yf.name}: n_samples={n_yaml}")

    # ── 4. Edge parquets ─────────────────────────────────────────────
    edge_files = sorted(art.glob("edges_*.parquet"))
    for ef in edge_files:
        edf = pd.read_parquet(ef)
        tag = ef.stem.replace("edges_", "")
        bad_src = edf["src_id"].str.startswith("NLST3D_").sum()
        bad_dst = edf["dst_id"].str.startswith("NLST3D_").sum()
        if bad_src > 0 or bad_dst > 0:
            _fail(f"{ef.name}: NLST3D references ({bad_src} src, {bad_dst} dst)")
            ok = False
        else:
            _pass(f"{ef.name}: {len(edf):,} edges — no NLST3D refs")

    # ── 5. Preprocessor PKLs ────────────────────────────────────────
    pkl_files = sorted(art.glob("preprocessor_*.pkl"))
    for pf in pkl_files:
        try:
            with open(pf, "rb") as f:
                pickle.load(f)
            _pass(f"{pf.name}: loadable")
        except Exception as e:
            _fail(f"{pf.name}: cannot load — {e}")
            ok = False

    # ── 6. Cross-model consistency ───────────────────────────────────
    if len(n_nodes_set) == 1:
        _pass(f"All artifacts have consistent node count: {n_nodes_set.pop():,}")
    else:
        _fail(f"Inconsistent node counts across artifacts: {n_nodes_set}")
        ok = False

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    if ok:
        print("  RESULT: ALL CHECKS PASSED — NLST3D fully purged")
    else:
        print("  RESULT: CHECKS FAILED — NLST3D contamination detected")
    print(f"{'=' * 64}\n")

    return ok


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Verify NLST3D purge from NoduleMap artifacts")
    parser.add_argument("--artifact-dir", required=True, help="Path to artifact directory")
    args = parser.parse_args()

    clean = verify(args.artifact_dir)
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
