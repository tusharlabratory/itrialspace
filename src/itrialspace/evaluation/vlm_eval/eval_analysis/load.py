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
Discover VLM result runs and assemble a single tidy long DataFrame.

The whole analysis suite operates on one normalised table with these columns::

    set, profile, model, condition, task, uid,
    prediction, ground_truth, correct,
    mode, source_dataset, lobe, size_bucket, diam_mm, population

so every table/plot/stat is just a group-by on this frame. The loader
*auto-discovers* whatever is present under the result root(s), so the identical
code path serves the full released dataset and a tiny demo run.

Result-file convention (written by the runners)::

    <model_dir>/<condition>/<task>_results.csv      # condition in CONDITIONS

Per-mode and ``aggregate/`` copies are ignored (only the condition-level file is
read), so nothing is double-counted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

from itrialspace.evaluation.vlm_eval.splits import canonical_uid, load_split

logger = logging.getLogger(__name__)

CONDITIONS = ("plain", "bbox", "contour", "bbox_contour")
TASKS = ("presence", "lobe", "size")

PathLike = Union[str, Path]

# Standard breakdown columns -> candidate source columns in eval_dataset.csv.
_META_MAP = {
    "mode": ["mode"],
    "source_dataset": ["donor_dataset", "dataset", "dataset_name"],
    "lobe": ["ground_truth_lobe", "lobe_name"],
    "size_bucket": ["ground_truth_size_bucket"],
    "diam_mm": ["effective_diam_mm", "nodule_mean_diam_mm"],
    "population": ["population_type"],
}


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def discover_result_files(roots: Iterable[PathLike]) -> List[Dict]:
    """Find condition-level ``<task>_results.csv`` files under the given roots."""
    records: List[Dict] = []
    seen = set()
    for root in roots:
        root = Path(root)
        for task in TASKS:
            for path in root.rglob(f"{task}_results.csv"):
                if path.parent.name not in CONDITIONS:
                    continue  # skip per-mode and aggregate/ copies
                if path in seen:
                    continue
                seen.add(path)
                parts = path.parts
                model_dir = path.parent.parent
                records.append(
                    {
                        "path": path,
                        "task": task,
                        "condition": path.parent.name,
                        "model": model_dir.name,
                        "profile": next(
                            (p for p in parts if p in ("lung_axial", "lung_axial_medgemma")), None
                        ),
                        "set": next((p for p in parts if p in ("synthetic", "real")), "default"),
                        "model_dir": model_dir,
                    }
                )
    logger.info("Discovered %d result files under %s", len(records), list(roots))
    return records


def _find_eval_csv(model_dir: Path) -> Optional[Path]:
    """Locate the eval_dataset.csv that produced a model's results."""
    for cand in (model_dir.parent / "eval_dataset.csv", model_dir / "eval_dataset.csv"):
        if cand.is_file():
            return cand
    return None


def load_metadata(eval_csv: PathLike) -> pd.DataFrame:
    """Per-uid metadata frame with standardised breakdown columns."""
    df = pd.read_csv(eval_csv)
    path_col = "png_path" if "png_path" in df.columns else "image_path"
    out = pd.DataFrame({"uid": df[path_col].astype(str).map(canonical_uid)})
    for std, cands in _META_MAP.items():
        src = next((c for c in cands if c in df.columns), None)
        out[std] = df[src].values if src else pd.NA
    return out.drop_duplicates("uid").set_index("uid")


def load_long(
    roots: Union[PathLike, Iterable[PathLike]],
    split: Optional[Union[PathLike, Dict[str, str]]] = None,
    eval_csvs: Optional[Dict[str, PathLike]] = None,
) -> pd.DataFrame:
    """Build the tidy long results frame.

    Parameters
    ----------
    roots : path or iterable of paths
        Result root(s) to scan (e.g. ``vlm_dataset`` or a demo output dir).
    split : path or {set: path}, optional
        Frozen split file(s) to restrict to. A single path applies to every set;
        a dict maps set name -> split file.
    eval_csvs : {set: path}, optional
        Explicit eval_dataset.csv per set (overrides auto-discovery).

    Returns
    -------
    pd.DataFrame
        One row per (model, condition, task, uid).
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    records = discover_result_files(roots)
    if not records:
        raise FileNotFoundError(f"no *_results.csv found under {list(roots)}")

    # Resolve split files per set.
    split_ids: Dict[str, set] = {}
    if split is not None:
        if isinstance(split, dict):
            split_ids = {k: load_split(v) for k, v in split.items()}
        else:
            ids = load_split(split)
            split_ids = {r["set"]: ids for r in records}

    # Metadata cache per (set, eval_csv).
    meta_cache: Dict[Path, pd.DataFrame] = {}
    frames: List[pd.DataFrame] = []
    for rec in records:
        df = pd.read_csv(rec["path"])
        col = "image_path" if "image_path" in df.columns else "png_path"
        df = df.assign(
            uid=df[col].astype(str).map(canonical_uid),
            set=rec["set"],
            profile=rec["profile"],
            model=rec["model"],
            condition=rec["condition"],
            task=rec["task"],
        )
        df["correct"] = (
            df["correct"].map(_truthy)
            if "correct" in df.columns
            else (df["prediction"].astype(str) == df["ground_truth"].astype(str))
        )
        keep = [
            "uid",
            "set",
            "profile",
            "model",
            "condition",
            "task",
            "prediction",
            "ground_truth",
            "correct",
        ]
        df = df[[c for c in keep if c in df.columns]].drop_duplicates("uid")

        if rec["set"] in split_ids:
            df = df[df["uid"].isin(split_ids[rec["set"]])]

        # Attach metadata.
        eval_csv = None
        if eval_csvs and rec["set"] in eval_csvs:
            eval_csv = Path(eval_csvs[rec["set"]])
        else:
            eval_csv = _find_eval_csv(rec["model_dir"])
        if eval_csv is not None:
            if eval_csv not in meta_cache:
                meta_cache[eval_csv] = load_metadata(eval_csv)
            df = df.join(meta_cache[eval_csv], on="uid")

        frames.append(df)

    long = pd.concat(frames, ignore_index=True)
    logger.info(
        "Long frame: %d rows | sets=%s models=%s conditions=%s tasks=%s",
        len(long),
        sorted(long["set"].unique()),
        sorted(long["model"].unique()),
        sorted(long["condition"].unique()),
        sorted(long["task"].unique()),
    )
    return long
