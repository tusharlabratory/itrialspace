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
Frozen evaluation splits for the VLM benchmark.

A *split* is an immutable, checksummed list of canonical case ids (``uid``\\ s)
that a result is reported on, so that reruns and re-reports score exactly the
same cases. See ``docs/vlm_eval.md`` (section "Reporting convention") and
``docs/dataset_card.md``.

Canonical uid
-------------
The stable identity of a case is **not** the ``case_id`` column -- that is a
non-unique positional index in both ``eval_dataset.csv`` and the per-model
result CSVs (it restarts per shard, so unrelated cases collide). The stable key
is the slice image path with the condition-variant sub-dir and the file
extension stripped, e.g.::

    .../slices/mode1_controlled_prevalence/NLST_cohort__C0000.png       -> mode1_controlled_prevalence/NLST_cohort__C0000
    .../slices/mode1_controlled_prevalence/bbox/NLST_cohort__C0000.png  -> mode1_controlled_prevalence/NLST_cohort__C0000

so the same case maps to one uid across all four image conditions, and results
join cleanly to ``eval_dataset.csv``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Set, Union

import pandas as pd

# Overlay sub-directory names inserted between the mode dir and the file name.
CONDITION_DIRS = ("bbox", "contour", "bbox_contour")

# Path columns that may carry a slice path, in preference order.
_PATH_COLS = (
    "image_path",
    "png_path",
    "png_bbox_path",
    "png_contour_path",
    "png_bbox_contour_path",
)

PathLike = Union[str, os.PathLike]


def canonical_uid(image_path: str) -> str:
    """Canonical case uid from a slice image path (condition + extension stripped)."""
    tail = str(image_path).split("/slices/")[-1]
    parts = [p for p in tail.split("/") if p not in CONDITION_DIRS]
    parts[-1] = os.path.splitext(parts[-1])[0]
    return "/".join(parts)


def uid_series(df: pd.DataFrame) -> pd.Series:
    """Series of canonical uids for a results / eval DataFrame.

    Uses the first available path column (``image_path`` for result CSVs,
    ``png_path`` for eval datasets).
    """
    col = next((c for c in _PATH_COLS if c in df.columns), None)
    if col is None:
        raise KeyError(
            f"DataFrame has none of the slice-path columns {_PATH_COLS}; "
            "cannot derive canonical uids."
        )
    return df[col].astype(str).map(canonical_uid)


def load_split(path: PathLike) -> Set[str]:
    """Load a split file (one uid per line; blank lines and ``#`` comments ignored)."""
    ids: Set[str] = set()
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.add(s)
    return ids


def filter_to_split(df: pd.DataFrame, case_ids: Union[PathLike, Iterable[str]]) -> pd.DataFrame:
    """Restrict a DataFrame to rows whose canonical uid is in ``case_ids``.

    ``case_ids`` may be a path to a split file or an iterable of uids.
    """
    if isinstance(case_ids, (str, os.PathLike)):
        case_ids = load_split(case_ids)
    wanted = set(case_ids)
    return df[uid_series(df).isin(wanted)].copy()


def sha256_of_ids(ids: Iterable[str]) -> str:
    """Stable content hash of a uid set (sorted, newline-joined)."""
    h = hashlib.sha256()
    for u in sorted(set(ids)):
        h.update(u.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_split(ids: Iterable[str], out_path: PathLike) -> dict:
    """Write a sorted split file; return ``{"n", "sha256"}`` metadata."""
    ids = sorted(set(ids))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(ids) + "\n")
    return {"n": len(ids), "sha256": sha256_of_ids(ids)}
