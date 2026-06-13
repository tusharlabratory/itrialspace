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
Schema constants — the 53 columns present in every nodule profile CSV,
plus enums for categorical values.
"""

from enum import Enum

# ── Core columns present in ALL 7 datasets ────────────────────────────────────
CORE_COLS = [
    # identity
    "ct_path",
    # bounding box
    "coordX",
    "coordY",
    "coordZ",
    "w",
    "h",
    "d",
    # morphology
    "nodule_mean_diam_mm",
    "nodule_vol_mm3",
    # anatomy
    "organ_label_id",
    "organ_label_name",
    "nearby_organs_10mm",
    "lung_side",
    "lobe_name",
    "lung_zone",
    "central_peripheral",
    # position – whole lung %
    "cranio_caudal_pct",
    "mediolateral_pct",
    "anteroposterior_pct",
    # position – lobe %
    "lobe_cc_pct",
    "lobe_ml_pct",
    "lobe_ap_pct",
    # distances mm
    "dist_to_trachea_mm",
    "dist_to_aorta_mm",
    "dist_to_heart_mm",
    "dist_to_esophagus_mm",
    "dist_to_pulmonary_vein_mm",
    "dist_to_superior_vena_cava_mm",
    "pleural_distance_mm",
    "airway_distance_mm",
    # inter-nodule
    "n_nodules_in_patient",
    "nearest_nodule_id",
    "nearest_nodule_dist_mm",
    "nearest_dx_mm",
    "nearest_dy_mm",
    "nearest_dz_mm",
    "all_nodule_ids",
    "all_nodule_dists_mm",
    "ipsilateral",
    "nearest_same_lobe",
    "bilateral_distribution",
    # reinsertion (100% complete across all datasets)
    "reinsertion_lobe",
    "reinsertion_lung_side",
    "reinsertion_lung_zone",
    "reinsertion_lobe_cc_pct",
    "reinsertion_lobe_ml_pct",
    "reinsertion_lobe_ap_pct",
    "reinsertion_lung_cc_pct",
    "reinsertion_lung_ml_pct",
    "reinsertion_lung_ap_pct",
    "reinsertion_pleural_dist_mm",
    "reinsertion_airway_dist_mm",
    "reinsertion_nodule_diam_mm",
]

# Pipeline-computed columns that map to NoduleProfile typed fields
PROFILING_COLS = [c for c in CORE_COLS if c != "ct_path"]

# Reinsertion subset — 100% complete, primary query index
REINSERTION_COLS = [c for c in CORE_COLS if c.startswith("reinsertion_")]

# ── Categorical values ─────────────────────────────────────────────────────────
LOBE_NAMES = [
    "right_lung_upper_lobe",
    "right_lung_middle_lobe",
    "right_lung_lower_lobe",
    "left_lung_upper_lobe",
    "left_lung_lower_lobe",
]
LOBE_NAMES_WITH_BG = LOBE_NAMES + ["background", "unknown"]

ZONE_NAMES = ["upper_zone", "middle_zone", "lower_zone"]
SIDE_NAMES = ["right", "left", "unknown"]
CP_VALUES = ["central", "peripheral", "unknown"]

DATASET_NAMES = ["DLCS24", "LUNA25", "LUNA16", "LUNGx", "LNDbv4", "NSCLCR", "IMDCT"]


class Lobe(str, Enum):
    RIGHT_UPPER = "right_lung_upper_lobe"
    RIGHT_MIDDLE = "right_lung_middle_lobe"
    RIGHT_LOWER = "right_lung_lower_lobe"
    LEFT_UPPER = "left_lung_upper_lobe"
    LEFT_LOWER = "left_lung_lower_lobe"
    BACKGROUND = "background"
    UNKNOWN = "unknown"


class Zone(str, Enum):
    UPPER = "upper_zone"
    MIDDLE = "middle_zone"
    LOWER = "lower_zone"


class Side(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    UNKNOWN = "unknown"


class CentralPeripheral(str, Enum):
    CENTRAL = "central"
    PERIPHERAL = "peripheral"
    UNKNOWN = "unknown"


# ── Known data quality flags per dataset ──────────────────────────────────────
DATASET_FLAGS: dict[str, dict] = {
    "DLCS24": {
        "lobe_ml_pct_unreliable": True,  # 93% outside [0,100] — pipeline normalisation bug
        "lobe_ap_pct_unreliable": True,  # 50.7% outside [0,100]
        "lobe_cc_pct_unreliable": False,
        "has_label": True,
        "all_malignant": False,
        "all_unlabelled": False,
        "label_is_soft": False,
        "notes": "Binary label. Includes Lung-RADS scores, smoking, age/sex.",
    },
    "LUNA25": {
        "lobe_ml_pct_unreliable": False,
        "lobe_ap_pct_unreliable": False,
        "lobe_cc_pct_unreliable": False,
        "has_label": True,
        "all_malignant": False,
        "all_unlabelled": False,
        "label_is_soft": False,
        "notes": "Binary label. Largest dataset (6,163 nodules). Multi-nodule screening.",
    },
    "LUNA16": {
        "lobe_ml_pct_unreliable": False,
        "lobe_ap_pct_unreliable": False,
        "lobe_cc_pct_unreliable": False,
        "has_label": False,
        "all_malignant": False,
        "all_unlabelled": True,
        "label_is_soft": False,
        "notes": "No malignancy label. Use for reinsertion/anatomy only.",
    },
    "LUNGx": {
        "lobe_ml_pct_unreliable": False,
        "lobe_ap_pct_unreliable": False,
        "lobe_cc_pct_unreliable": False,
        "has_label": True,
        "all_malignant": False,
        "all_unlabelled": False,
        "label_is_soft": False,
        "notes": "Benchmark challenge dataset. CADx_label binary. 49.4% malignant.",
    },
    "LNDbv4": {
        "lobe_ml_pct_unreliable": False,
        "lobe_ap_pct_unreliable": False,
        "lobe_cc_pct_unreliable": False,
        "has_label": True,
        "all_malignant": False,
        "all_unlabelled": False,
        "label_is_soft": True,  # continuous 1–5 radiologist average
        "label_threshold": 3.0,
        "notes": "Malignancy is float avg of radiologist ratings (1–5). Threshold >=3 used.",
    },
    "NSCLCR": {
        "lobe_ml_pct_unreliable": False,
        "lobe_ap_pct_unreliable": False,
        "lobe_cc_pct_unreliable": False,
        "has_label": True,
        "all_malignant": True,  # 100% cancer cohort
        "all_unlabelled": False,
        "label_is_soft": False,
        "notes": "All malignant (lung cancer patients). Large tumours (median 52mm). Has staging/survival.",
    },
    "IMDCT": {
        "lobe_ml_pct_unreliable": False,
        "lobe_ap_pct_unreliable": False,
        "lobe_cc_pct_unreliable": False,
        "has_label": True,
        "all_malignant": False,
        "all_unlabelled": False,
        "label_is_soft": False,
        "notes": "Label is string 'benign'/'cancer'. Has nodule type, PET, spiculation.",
    },
}
