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
similarity_report.py — measure NoduleMap search quality across backends.

Quantifies two things so design choices are *measured*, not asserted:

1. **k-NN concordance** — for a random sample of query nodes, the average fraction of the
   top-k neighbours that share the query's lobe / size-bucket / side / (known) malignancy
   label. Higher = more anatomically/clinically coherent neighbourhoods. Compared across
   search backends: ``weighted`` (clinical metric) vs ``feature_l2`` (full feature space)
   vs ``embedding`` (the reduced 2D/16D space — what was used before).
2. **Layout trustworthiness** — sklearn trustworthiness of each 2D embedding w.r.t. the full
   feature space (are *displayed* neighbourhoods faithful?).

Usage:
    python -m itrialspace.apps.nodulemap.tools.similarity_report \\
        --artifact-dir "$NODULEMAP_ARTIFACTS" --k 10 --sample 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from itrialspace.apps.nodulemap.neighbors import search_space

_CONCORDANCE_COLS = {
    "lobe": "reinsertion_lobe",
    "size_bucket": "size_bucket",
    "side": "reinsertion_lung_side",
}


def _list_models(artifact_dir: Path, feature_set: str) -> list[tuple[str, int]]:
    out = []
    for f in sorted(artifact_dir.glob(f"embeddings_{feature_set}__*.npy")):
        model = f.stem.replace(f"embeddings_{feature_set}__", "")
        nd = np.load(f, mmap_mode="r").shape[1]
        out.append((model, nd))
    return out


def _concordance(meta: pd.DataFrame, matrix: np.ndarray, metric: str, sample_idx, k: int) -> dict:
    queries = matrix[sample_idx]
    dists, inds = search_space.knn(matrix, queries, k + 1, metric)
    scores = {key: [] for key in _CONCORDANCE_COLS}
    scores["label"] = []
    for qi, row_inds in zip(sample_idx, inds):
        nbrs = [j for j in row_inds if j != qi][:k]
        if not nbrs:
            continue
        for key, col in _CONCORDANCE_COLS.items():
            if col in meta.columns:
                qv = meta.iloc[qi][col]
                share = np.mean([meta.iloc[j][col] == qv for j in nbrs])
                scores[key].append(float(share))
        # malignancy concordance only over labelled query+neighbours
        if "label" in meta.columns:
            ql = meta.iloc[qi]["label"]
            if pd.notna(ql):
                lab = [meta.iloc[j]["label"] for j in nbrs if pd.notna(meta.iloc[j]["label"])]
                if lab:
                    scores["label"].append(float(np.mean([v == ql for v in lab])))
    return {key: (round(float(np.mean(v)), 4) if v else None) for key, v in scores.items()}


def main():
    ap = argparse.ArgumentParser(description="NoduleMap search-quality report")
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--feature-set", default="reinsertion_core")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=1000, help="query nodes for concordance")
    ap.add_argument("--trust-sample", type=int, default=2000, help="nodes for trustworthiness")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from itrialspace.config import settings

    art = Path(args.artifact_dir or settings.nodulemap_artifacts_dir())
    fs = args.feature_set
    models = _list_models(art, fs)
    if not models:
        raise SystemExit(f"No embeddings found in {art} for feature_set '{fs}'.")

    # Metadata is identical across models; use the first.
    meta = pd.read_parquet(art / f"metadata_{fs}__{models[0][0]}.parquet")
    n = len(meta)
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n, size=min(args.sample, n), replace=False)

    # ── k-NN concordance per backend ───────────────────────────────────
    rows = []
    # model-independent backends
    for backend in ("weighted", "feature_l2"):
        M = search_space.build_search_matrix(backend, art, fs, models[0][0], meta)
        c = _concordance(meta, M, search_space.metric_for(backend), sample_idx, args.k)
        rows.append({"backend": backend, "space": backend, **c})
    # embedding backend — per model (this is the "search in reduced space" being replaced)
    for model, nd in models:
        M = search_space.build_search_matrix("embedding", art, fs, model, meta)
        c = _concordance(meta, M, "l2", sample_idx, args.k)
        rows.append({"backend": "embedding", "space": f"{model}({nd}D)", **c})
    conc = pd.DataFrame(rows)

    # ── Layout trustworthiness (2D models vs full feature space) ────────
    from sklearn.manifold import trustworthiness

    feat = np.load(art / f"features_{fs}.npy").astype(np.float32)
    t_idx = rng.choice(n, size=min(args.trust_sample, n), replace=False)
    trust = []
    for model, nd in models:
        if nd != 2:
            continue
        emb = np.load(art / f"embeddings_{fs}__{model}.npy").astype(np.float32)
        tw = trustworthiness(feat[t_idx], emb[t_idx, :2], n_neighbors=args.k)
        trust.append({"model": model, "trustworthiness": round(float(tw), 4)})

    # ── Emit ────────────────────────────────────────────────────────────
    report = {
        "feature_set": fs,
        "n_nodes": int(n),
        "k": args.k,
        "n_query_sample": int(len(sample_idx)),
        "concordance": conc.to_dict(orient="records"),
        "trustworthiness": trust,
        "notes": [
            "Concordance = mean fraction of top-k neighbours sharing the query's attribute.",
            "weighted = clinical ReinsertionMatcher metric (L1); feature_l2 = full 22-D space; "
            "embedding = reduced layout space (legacy search).",
            "Higher concordance = more coherent donor neighbourhoods for reinsertion.",
        ],
    }
    (art / "nodulemap_similarity_report.json").write_text(json.dumps(report, indent=2))

    md = ["# NoduleMap search-quality report", ""]
    md.append(
        f"- feature_set: `{fs}` · nodes: {n:,} · k={args.k} · " f"query sample={len(sample_idx)}"
    )
    md.append("")
    md.append("## k-NN concordance (fraction of top-k neighbours sharing attribute)")
    md.append("")
    md.append("| backend | space | lobe | size_bucket | side | malignancy |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['backend']} | {r['space']} | {r.get('lobe')} | "
            f"{r.get('size_bucket')} | {r.get('side')} | {r.get('label')} |"
        )
    md.append("")
    md.append("## Layout trustworthiness (2D embedding vs full feature space)")
    md.append("")
    md.append("| layout model | trustworthiness |")
    md.append("|---|---|")
    for t in trust:
        md.append(f"| {t['model']} | {t['trustworthiness']} |")
    md.append("")
    md.append(
        "> weighted/feature_l2 are the recommended **search** backends; 2D embeddings "
        "are **layout only**. Higher concordance for weighted/feature_l2 vs embedding "
        "confirms search should not run in the reduced space."
    )
    (art / "nodulemap_similarity_report.md").write_text("\n".join(md))

    print("\n".join(md))
    print(f"\nWrote {art/'nodulemap_similarity_report.md'} and .json")


if __name__ == "__main__":
    main()
