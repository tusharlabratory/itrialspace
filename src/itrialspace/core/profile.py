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
NoduleProfile — the unified representation of a single lung nodule
across all 7 iTrialSpace datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoduleProfile:
    """
    A single nodule record, normalised from any iTrialSpace dataset CSV.

    The 53 core pipeline fields are typed attributes.
    Dataset-specific columns (Lung-RADS, staging, pack-years, etc.)
    live in `meta` and are also retained in the NoduleIndex DataFrame.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    annotation_id: str  # AnnotationID (universal across datasets)
    patient_id: str  # normalised from 7 different source col names
    dataset: str  # 'DLCS24' | 'LUNA25' | 'LUNA16' | ...
    ct_path: str

    # ── Label ─────────────────────────────────────────────────────────────────
    label: Optional[int] = None  # 0=benign, 1=malignant, None=unknown

    # ── Bounding box ──────────────────────────────────────────────────────────
    coord_x: float = 0.0
    coord_y: float = 0.0
    coord_z: float = 0.0
    w: float = 0.0
    h: float = 0.0
    d: float = 0.0

    # ── Morphology ────────────────────────────────────────────────────────────
    nodule_mean_diam_mm: float = 0.0
    nodule_vol_mm3: Optional[float] = None

    # ── Anatomy ───────────────────────────────────────────────────────────────
    lobe_name: str = "unknown"
    lung_side: str = "unknown"
    lung_zone: str = "unknown"
    organ_label_id: int = 0
    organ_label_name: str = ""
    central_peripheral: str = "unknown"
    nearby_organs_10mm: Optional[str] = None

    # ── Position — whole-lung % ───────────────────────────────────────────────
    cranio_caudal_pct: Optional[float] = None
    mediolateral_pct: Optional[float] = None
    anteroposterior_pct: Optional[float] = None

    # ── Position — lobe % ────────────────────────────────────────────────────
    # NOTE: lobe_cc_pct is clean (use freely).
    #       lobe_ml_pct and lobe_ap_pct have known outliers in DLCS24
    #       due to a normalisation bug — check dataset_flags before using.
    lobe_cc_pct: Optional[float] = None
    lobe_ml_pct: Optional[float] = None
    lobe_ap_pct: Optional[float] = None

    # ── Distances (mm) ────────────────────────────────────────────────────────
    dist_to_trachea_mm: Optional[float] = None
    dist_to_aorta_mm: Optional[float] = None
    dist_to_heart_mm: Optional[float] = None
    dist_to_esophagus_mm: Optional[float] = None
    dist_to_pulmonary_vein_mm: Optional[float] = None
    dist_to_superior_vena_cava_mm: Optional[float] = None
    pleural_distance_mm: Optional[float] = None
    airway_distance_mm: Optional[float] = None

    # ── Inter-nodule ──────────────────────────────────────────────────────────
    n_nodules_in_patient: int = 1
    nearest_nodule_id: Optional[str] = None
    nearest_nodule_dist_mm: Optional[float] = None
    nearest_dx_mm: Optional[float] = None
    nearest_dy_mm: Optional[float] = None
    nearest_dz_mm: Optional[float] = None
    all_nodule_ids: Optional[str] = None
    all_nodule_dists_mm: Optional[str] = None
    ipsilateral: Optional[bool] = None
    nearest_same_lobe: Optional[bool] = None
    bilateral_distribution: bool = False

    # ── Reinsertion — 100% complete, primary query / match index ─────────────
    reinsertion_lobe: str = "unknown"
    reinsertion_lung_side: str = "unknown"
    reinsertion_lung_zone: str = "unknown"
    reinsertion_lobe_cc_pct: float = 0.0
    reinsertion_lobe_ml_pct: float = 0.0
    reinsertion_lobe_ap_pct: float = 0.0
    reinsertion_lung_cc_pct: float = 0.0
    reinsertion_lung_ml_pct: float = 0.0
    reinsertion_lung_ap_pct: float = 0.0
    reinsertion_pleural_dist_mm: Optional[float] = None
    reinsertion_airway_dist_mm: Optional[float] = None
    reinsertion_nodule_diam_mm: float = 0.0

    # ── Dataset-specific metadata ─────────────────────────────────────────────
    # All extra columns from the source CSV land here.
    # Also retained in NoduleIndex DataFrame under original column names.
    #
    # Examples by dataset:
    #   DLCS24:  'Lung-RADS score', 'Smoking Status', 'Age', 'Sex', 'annotation_group'
    #   LUNA25:  'LesionID', 'NoduleID', 'StudyDate', 'Split', 'age', 'gender'
    #   LNDbv4:  'Malignancy' (raw float), 'Texture', 'Spiculation', 'NumRadsFound'
    #   NSCLCR:  'overall.stage', 'histology', 'survival.time', 'deadstatus.event'
    #   IMDCT:   'nodules_types', 'Spiculation', 'Emphysema', 'PET', 'uptake_class'
    #   LUNGx:   'Final Diagnosis', 'ANod_score', 'meta_prob'
    meta: dict = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def is_malignant(self) -> Optional[bool]:
        if self.label is None:
            return None
        return bool(self.label)

    @property
    def is_upper_lobe(self) -> bool:
        return "upper" in self.lobe_name

    @property
    def is_peripheral(self) -> bool:
        return self.central_peripheral == "peripheral"

    @property
    def size_bucket(self) -> str:
        d = self.nodule_mean_diam_mm
        if d < 5:
            return "<5mm"
        if d < 10:
            return "5-10mm"
        if d < 15:
            return "10-15mm"
        if d < 20:
            return "15-20mm"
        if d < 30:
            return "20-30mm"
        return ">30mm"

    def __repr__(self) -> str:
        lbl = {0: "benign", 1: "malignant", None: "unlabelled"}[self.label]
        return (
            f"NoduleProfile({self.dataset} | {self.annotation_id} | "
            f"{self.nodule_mean_diam_mm:.1f}mm | {self.lobe_name} | {lbl})"
        )
