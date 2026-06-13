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
CohortManifest — the final output of iTrialSpace.

A structured manifest describing every case in a synthetic trial
cohort. Wraps a DataFrame with export, audit, and path verification.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from itrialspace.site.spec import TrialSpec

# ── Manifest column schema ────────────────────────────────────────────────────

MANIFEST_COLS = [
    # Case identity
    "case_id",
    "nodule_idx",
    "is_primary_nodule",
    "n_nodules_in_case",
    "companion_group_id",
    "trial_name",
    "trial_template",
    "bootstrap_id",
    # Host CT
    "host_patient_id",
    "host_dataset",
    "host_ct_path",
    "host_organ_seg_path",
    # Donor nodule
    "donor_patient_id",
    "donor_annotation_id",
    "donor_dataset",
    "donor_nodule_mask_path",
    "donor_ct_path",
    "donor_refined_seg_path",
    # Insertion parameters
    "insertion_coord_x",
    "insertion_coord_y",
    "insertion_coord_z",
    "insertion_lobe",
    "insertion_lobe_cc_pct",
    "insertion_lobe_ml_pct",
    "insertion_lobe_ap_pct",
    "insertion_mode",
    # Nodule characteristics
    "nodule_diam_mm",
    "effective_diam_mm",
    "scale_factor",
    "warp_applied",
    "label",
    # Anatomy
    "nodule_lobe_name",
    "nodule_lung_side",
    "nodule_lung_zone",
    "nodule_central_peripheral",
    "pleural_distance_mm",
    # Demographics
    "patient_age",
    "patient_sex",
    "smoking_status",
    "pack_years",
    # Cohort metadata
    "cohort_mode",
    "size_bucket",
    "population_type",
    "label_source",
]

# Additional columns for digital twin isolation manifests.
# These are appended to MANIFEST_COLS when the mode is digital_twin_isolation.
ISOLATION_MANIFEST_COLS = MANIFEST_COLS + [
    "mode",
    "target_annotation_id",
    "target_nodule_mask_path",
    "target_diameter_mm",
    "target_label",
    "target_lobe",
    "target_side",
    "target_zone",
    "host_n_nodules",
    "isolation_case_index",
]

# Additional columns for digital twin complete manifests.
# One row per nodule, all nodules for the same patient share a case_id.
COMPLETE_MANIFEST_COLS = MANIFEST_COLS + [
    "mode",
    "host_n_nodules",
    "annotation_ids",
    "diameters_mm",
    "labels",
    "lobes",
    "sides",
    "zones",
]

# Additional columns for digital twin cross manifests.
# Explicit host/donor provenance with placement and pairing metadata.
CROSS_MANIFEST_COLS = MANIFEST_COLS + [
    "mode",
    "donor_transfer_mode",
    "pairing_policy",
    "placement_strategy",
    "cross_case_group_id",
]


class CohortManifest:
    """Structured manifest for a synthetic imaging trial cohort."""

    def __init__(self, df: pd.DataFrame, spec: TrialSpec):
        self._df = df
        self._spec = spec

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def spec(self) -> TrialSpec:
        return self._spec

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        n_mal = (self._df["label"] == 1).sum()
        n_ben = (self._df["label"] == 0).sum()
        # Show total rows vs unique cases when companions present
        if "is_primary_nodule" in self._df.columns:
            n_primary = int(self._df["is_primary_nodule"].sum())
            if n_primary < len(self):
                return (
                    f"CohortManifest('{self._spec.trial_name}' | "
                    f"{n_primary} cases ({len(self)} total rows) | "
                    f"{n_mal} malignant, {n_ben} benign)"
                )
        return (
            f"CohortManifest('{self._spec.trial_name}' | "
            f"{len(self)} cases | {n_mal} malignant, {n_ben} benign)"
        )

    # ── Export ─────────────────────────────────────────────────────────────────

    def to_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._df.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        records = self._df.to_dict(orient="records")
        with open(path, "w") as f:
            json.dump(
                {
                    "trial_name": self._spec.trial_name,
                    "trial_template": self._spec.trial_template,
                    "n_cases": len(self),
                    "cohort_mode": self._spec.cohort_mode,
                    "seed": self._spec.seed,
                    "cases": records,
                },
                f,
                indent=2,
                default=str,
            )

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Summary statistics of the manifest."""
        df = self._df
        s: dict = {
            "trial_name": self._spec.trial_name,
            "n_cases": len(df),
            "cohort_mode": self._spec.cohort_mode,
        }

        if "label" in df.columns:
            labels = df["label"].dropna()
            s["n_malignant"] = int((labels == 1).sum())
            s["n_benign"] = int((labels == 0).sum())
            s["n_unlabelled"] = int(df["label"].isna().sum())
            if len(labels) > 0:
                s["malignancy_rate"] = round(float((labels == 1).mean()), 4)

        if "size_bucket" in df.columns:
            s["size_distribution"] = df["size_bucket"].value_counts().to_dict()

        if "insertion_lobe" in df.columns:
            s["lobe_distribution"] = df["insertion_lobe"].value_counts().to_dict()

        if "donor_dataset" in df.columns:
            s["donor_datasets"] = df["donor_dataset"].value_counts().to_dict()

        if "host_dataset" in df.columns:
            s["host_datasets"] = df["host_dataset"].value_counts().to_dict()

        if "warp_applied" in df.columns:
            s["warp_distribution"] = df["warp_applied"].value_counts().to_dict()

        # Multi-nodule statistics
        if "is_primary_nodule" in df.columns:
            n_primary = int(df["is_primary_nodule"].sum())
            n_companion = len(df) - n_primary
            if n_companion > 0:
                s["n_primary_cases"] = n_primary
                s["n_companion_insertions"] = n_companion
                s["total_manifest_rows"] = len(df)
                # Count cases with >1 nodule
                if "n_nodules_in_case" in df.columns:
                    primary_rows = df[df["is_primary_nodule"] == True]
                    s["multi_nodule_cases"] = int((primary_rows["n_nodules_in_case"] > 1).sum())

        return s

    def audit(self) -> pd.DataFrame:
        """Compare actual manifest statistics to TrialSpec targets.

        Returns:
            DataFrame with columns: metric, target, actual, deviation.
        """
        rows = []
        spec = self._spec
        df = self._df

        # Prevalence
        if spec.malignancy_prevalence is not None and "label" in df.columns:
            labels = df["label"].dropna()
            actual_prev = float((labels == 1).mean()) if len(labels) > 0 else 0.0
            rows.append(
                {
                    "metric": "malignancy_prevalence",
                    "target": spec.malignancy_prevalence,
                    "actual": round(actual_prev, 4),
                    "deviation": round(actual_prev - spec.malignancy_prevalence, 4),
                }
            )

        # N cases
        rows.append(
            {
                "metric": "n_cases",
                "target": spec.n_cases,
                "actual": len(df),
                "deviation": len(df) - spec.n_cases,
            }
        )

        # Size distribution
        if (
            spec.nodule_spec
            and spec.nodule_spec.size_distribution
            and spec.nodule_spec.size_distribution.bucket_weights
            and "size_bucket" in df.columns
        ):
            actual_dist = df["size_bucket"].value_counts(normalize=True)
            for bucket, target_w in spec.nodule_spec.size_distribution.bucket_weights.items():
                actual_w = actual_dist.get(bucket, 0.0)
                rows.append(
                    {
                        "metric": f"size_{bucket}",
                        "target": round(target_w, 3),
                        "actual": round(float(actual_w), 3),
                        "deviation": round(float(actual_w) - target_w, 3),
                    }
                )

        # Companion nodule audit
        if "is_primary_nodule" in df.columns:
            n_companion = int((df["is_primary_nodule"] == False).sum())
            if n_companion > 0:
                rows.append(
                    {
                        "metric": "companion_nodules_added",
                        "target": "N/A",
                        "actual": n_companion,
                        "deviation": 0,
                    }
                )

        return pd.DataFrame(rows)

    # ── Path verification ─────────────────────────────────────────────────────

    def verify_paths(self) -> pd.DataFrame:
        """Check which files actually exist on disk.

        Returns:
            DataFrame of missing files with columns: case_id, path_type, path.
        """
        path_cols = [
            "host_ct_path",
            "host_organ_seg_path",
            "donor_nodule_mask_path",
            "donor_ct_path",
        ]
        missing = []
        for _, row in self._df.iterrows():
            for col in path_cols:
                if col in row and pd.notna(row[col]):
                    path = str(row[col])
                    if not os.path.exists(path):
                        missing.append(
                            {
                                "case_id": row.get("case_id", ""),
                                "path_type": col,
                                "path": path,
                            }
                        )
        return pd.DataFrame(missing)

    # ── Subsetting ────────────────────────────────────────────────────────────

    def malignant_cases(self) -> CohortManifest:
        return CohortManifest(
            self._df[self._df["label"] == 1].reset_index(drop=True),
            self._spec,
        )

    def benign_cases(self) -> CohortManifest:
        return CohortManifest(
            self._df[self._df["label"] == 0].reset_index(drop=True),
            self._spec,
        )

    def filter_lobe(self, lobe: str) -> CohortManifest:
        return CohortManifest(
            self._df[self._df["insertion_lobe"] == lobe].reset_index(drop=True),
            self._spec,
        )

    def filter_size(self, min_mm: float, max_mm: float) -> CohortManifest:
        mask = (self._df["effective_diam_mm"] >= min_mm) & (self._df["effective_diam_mm"] <= max_mm)
        return CohortManifest(
            self._df[mask].reset_index(drop=True),
            self._spec,
        )
