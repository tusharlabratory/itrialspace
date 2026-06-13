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
Dataset adapters — one class per dataset.

Each adapter knows:
  - which column is the patient_id
  - which column is the annotation_id
  - which columns are dataset-specific metadata (go into meta{})
  - any special pre-processing needed before loading
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class _BaseAdapter:
    dataset: str = ""
    patient_id_col: str = "PatientID"
    annotation_id_col: Optional[str] = "AnnotationID"

    # Columns that are dataset-specific (go to meta dict, not core fields)
    # Computed as: all columns NOT in CORE_COLS and NOT identity cols
    _extra_col_cache: Optional[list[str]] = None

    def extra_cols(self, df_columns: list[str]) -> list[str]:
        from itrialspace.core.schema import CORE_COLS

        core_set = set(CORE_COLS) | {
            "coordX",
            "coordY",
            "coordZ",
            self.patient_id_col,
            self.annotation_id_col or "",
            "ct_path",
        }
        return [c for c in df_columns if c not in core_set]

    def get_patient_id(self, row: pd.Series) -> str:
        v = row.get(self.patient_id_col)
        return str(v) if pd.notna(v) else "unknown"

    def get_annotation_id(self, row: pd.Series) -> str:
        if self.annotation_id_col and self.annotation_id_col in row.index:
            v = row.get(self.annotation_id_col)
            if pd.notna(v):
                return str(v)
        # fallback: dataset + index
        return f"{self.dataset}_unknown"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook for dataset-specific pre-processing. Override as needed."""
        return df


class DLCS24Adapter(_BaseAdapter):
    dataset = "DLCS24"
    patient_id_col = "patient-id"
    annotation_id_col = "AnnotationID"


class LUNA25Adapter(_BaseAdapter):
    dataset = "LUNA25"
    patient_id_col = "PatientID"
    annotation_id_col = "AnnotationID"


class LUNA16Adapter(_BaseAdapter):
    dataset = "LUNA16"
    patient_id_col = "seriesuid"
    annotation_id_col = "AnnotationID"


class LUNGxAdapter(_BaseAdapter):
    dataset = "LUNGx"
    patient_id_col = "PatientID"
    annotation_id_col = "AnnotationID"


class LNDbv4Adapter(_BaseAdapter):
    dataset = "LNDbv4"
    patient_id_col = "PatientID"
    annotation_id_col = "AnnotationID"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Aggregate multi-radiologist rows: group by unique_NoduleID,
        # take mean of numeric ratings, keep first for string fields
        if "unique_NoduleID" not in df.columns:
            return df
        return df  # loaded as-is; aggregation is optional and caller-driven


class NSCLCRAdapter(_BaseAdapter):
    dataset = "NSCLCR"
    patient_id_col = "PatientID"
    annotation_id_col = "AnnotationID"


class IMDCTAdapter(_BaseAdapter):
    dataset = "IMDCT"
    patient_id_col = "PatientID"
    annotation_id_col = None  # IMDCT has no AnnotationID column

    def get_annotation_id(self, row: pd.Series) -> str:
        # Synthesise from PatientID + row index (set by loader)
        pid = self.get_patient_id(row)
        idx = row.get("_row_index", "0")
        return f"IMDCT_{pid}_{idx}"


# ── Registry ──────────────────────────────────────────────────────────────────

ADAPTERS: dict[str, _BaseAdapter] = {
    "DLCS24": DLCS24Adapter(),
    "LUNA25": LUNA25Adapter(),
    "LUNA16": LUNA16Adapter(),
    "LUNGx": LUNGxAdapter(),
    "LNDbv4": LNDbv4Adapter(),
    "NSCLCR": NSCLCRAdapter(),
    "IMDCT": IMDCTAdapter(),
}


def get_adapter(dataset: str) -> _BaseAdapter:
    if dataset not in ADAPTERS:
        raise ValueError(f"No adapter for dataset '{dataset}'. Known: {list(ADAPTERS)}")
    return ADAPTERS[dataset]
