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
Generate a frozen, checksummed evaluation split from result CSVs.

Computes the canonical-uid intersection across the chosen per-model/condition/task
result CSVs (the reporting-N convention) and writes::

    <out-dir>/<name>.<set>.txt     # sorted canonical uids, one per line
    <out-dir>/splits.json          # manifest: per split -> {n, sha256, source, ...}

The split is reproducible and verifiable: anyone can re-run this tool over the
released results and confirm the ``sha256`` in ``splits.json`` matches.

Examples
--------
# release_v1 (plain scope) for the synthetic set: cases scored by every model on
# the plain condition, all 3 tasks.
python -m itrialspace.evaluation.vlm_eval.make_split \\
    --name release_v1 --set synthetic \\
    --out-dir $ITRIALSPACE_DATA_DIR/vlm_dataset/splits \\
    --results-glob \\
      "$ITRIALSPACE_DATA_DIR/vlm_dataset/synthetic/lung_axial/biomedclip/plain/*_results.csv" \\
      "$ITRIALSPACE_DATA_DIR/vlm_dataset/synthetic/lung_axial/llava_med/plain/*_results.csv" \\
      "$ITRIALSPACE_DATA_DIR/vlm_dataset/synthetic/lung_axial_medgemma/medgemma/plain/*_results.csv" \\
    --description "Cases scored by all 3 models on plain, all 3 tasks."

# release_v1_full: strict intersection across all 4 conditions too (use */*_results.csv).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from itrialspace.evaluation.vlm_eval.splits import canonical_uid, write_split

logger = logging.getLogger(__name__)


def _git_commit(start: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _uids_in_csv(path: str) -> set:
    """Canonical uids present in one result CSV (uses image_path / png_path)."""
    df = pd.read_csv(path)
    col = "image_path" if "image_path" in df.columns else "png_path"
    return set(df[col].astype(str).map(canonical_uid))


def build_split(result_globs: List[str]) -> set:
    """Intersection of canonical uids across every CSV matched by the globs."""
    files: List[str] = []
    for g in result_globs:
        files.extend(glob.glob(g))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no result CSVs matched: {result_globs}")
    logger.info("Intersecting %d result CSVs", len(files))
    common: Optional[set] = None
    for f in files:
        ids = _uids_in_csv(f)
        common = ids if common is None else (common & ids)
        logger.debug("  %-70s n=%d running∩=%d", f, len(ids), len(common))
    return common or set()


def make_split(
    name: str,
    set_name: str,
    out_dir: str,
    result_globs: List[str],
    description: str = "",
) -> dict:
    out = Path(out_dir)
    ids = build_split(result_globs)
    txt = out / f"{name}.{set_name}.txt"
    meta = write_split(ids, txt)

    entry = {
        **meta,
        "set": set_name,
        "file": txt.name,
        "source_globs": result_globs,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(Path(__file__).resolve().parent),
    }

    manifest_path = out / "splits.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault(name, {})[set_name] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[make_split] {name}.{set_name}: N={meta['n']}  sha256={meta['sha256'][:16]}…")
    print(f"             split -> {txt}")
    print(f"             manifest -> {manifest_path}")
    return entry


def main():
    ap = argparse.ArgumentParser(description="Generate a frozen evaluation split.")
    ap.add_argument("--name", required=True, help="Split name, e.g. release_v1.")
    ap.add_argument("--set", dest="set_name", required=True, help="synthetic | real.")
    ap.add_argument("--out-dir", required=True, help="Splits directory (vlm_dataset/splits).")
    ap.add_argument(
        "--results-glob",
        nargs="+",
        required=True,
        help="One or more quoted globs matching the result CSVs to intersect.",
    )
    ap.add_argument("--description", default="", help="Human-readable provenance note.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    make_split(
        name=args.name,
        set_name=args.set_name,
        out_dir=args.out_dir,
        result_globs=args.results_glob,
        description=args.description,
    )


if __name__ == "__main__":
    main()
