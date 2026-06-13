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
DatasetLoader — reads a nodule profile CSV and returns a list of
NoduleProfile objects with unified schema.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from itrialspace.core.label import LabelNormaliser
from itrialspace.core.profile import NoduleProfile
from itrialspace.io.adapters.adapters import get_adapter

# Map CSV column names → NoduleProfile attribute names
_COL_MAP = {
    "coordX": "coord_x",
    "coordY": "coord_y",
    "coordZ": "coord_z",
}


def _f(v, default=None) -> Optional[float]:
    """Safe float conversion."""
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _i(v, default=None) -> Optional[int]:
    """Safe int conversion."""
    try:
        f = float(v)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return default


def _b(v) -> Optional[bool]:
    """Safe bool conversion."""
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _s(v, default: str = "") -> str:
    """Safe string conversion."""
    if pd.isna(v):
        return default
    return str(v).strip()


class DatasetLoader:
    """
    Loads a nodule profile CSV for a given dataset and returns
    a list of NoduleProfile objects.

    Usage:
        profiles = DatasetLoader.load("DLCS24", "/path/to/DLCS24_nodule_profiles.csv")
    """

    @staticmethod
    def load(dataset: str, csv_path: str) -> list[NoduleProfile]:
        adapter = get_adapter(dataset)
        normaliser = LabelNormaliser(dataset)

        df = pd.read_csv(csv_path, low_memory=False)
        df = adapter.preprocess(df)

        # Add row index for adapters that need it (IMDCT)
        df["_row_index"] = df.index.astype(str)

        extra_cols = adapter.extra_cols(list(df.columns))
        profiles = []

        for _, row in df.iterrows():
            pid = adapter.get_patient_id(row)
            aid = adapter.get_annotation_id(row)
            label = normaliser.normalise(row)

            meta = {
                c: (None if pd.isna(row.get(c)) else row.get(c))
                for c in extra_cols
                if c != "_row_index"
            }

            p = NoduleProfile(
                # identity
                annotation_id=aid,
                patient_id=pid,
                dataset=dataset,
                ct_path=_s(row.get("ct_path")),
                label=label,
                # bounding box
                coord_x=_f(row.get("coordX"), 0.0),
                coord_y=_f(row.get("coordY"), 0.0),
                coord_z=_f(row.get("coordZ"), 0.0),
                w=_f(row.get("w"), 0.0),
                h=_f(row.get("h"), 0.0),
                d=_f(row.get("d"), 0.0),
                # morphology
                nodule_mean_diam_mm=_f(row.get("nodule_mean_diam_mm"), 0.0),
                nodule_vol_mm3=_f(row.get("nodule_vol_mm3")),
                # anatomy
                lobe_name=_s(row.get("lobe_name"), "unknown"),
                lung_side=_s(row.get("lung_side"), "unknown"),
                lung_zone=_s(row.get("lung_zone"), "unknown"),
                organ_label_id=_i(row.get("organ_label_id"), 0),
                organ_label_name=_s(row.get("organ_label_name")),
                central_peripheral=_s(row.get("central_peripheral"), "unknown"),
                nearby_organs_10mm=_s(row.get("nearby_organs_10mm")) or None,
                # position – lung %
                cranio_caudal_pct=_f(row.get("cranio_caudal_pct")),
                mediolateral_pct=_f(row.get("mediolateral_pct")),
                anteroposterior_pct=_f(row.get("anteroposterior_pct")),
                # position – lobe %
                lobe_cc_pct=_f(row.get("lobe_cc_pct")),
                lobe_ml_pct=_f(row.get("lobe_ml_pct")),
                lobe_ap_pct=_f(row.get("lobe_ap_pct")),
                # distances
                dist_to_trachea_mm=_f(row.get("dist_to_trachea_mm")),
                dist_to_aorta_mm=_f(row.get("dist_to_aorta_mm")),
                dist_to_heart_mm=_f(row.get("dist_to_heart_mm")),
                dist_to_esophagus_mm=_f(row.get("dist_to_esophagus_mm")),
                dist_to_pulmonary_vein_mm=_f(row.get("dist_to_pulmonary_vein_mm")),
                dist_to_superior_vena_cava_mm=_f(row.get("dist_to_superior_vena_cava_mm")),
                pleural_distance_mm=_f(row.get("pleural_distance_mm")),
                airway_distance_mm=_f(row.get("airway_distance_mm")),
                # inter-nodule
                n_nodules_in_patient=_i(row.get("n_nodules_in_patient"), 1),
                nearest_nodule_id=_s(row.get("nearest_nodule_id")) or None,
                nearest_nodule_dist_mm=_f(row.get("nearest_nodule_dist_mm")),
                nearest_dx_mm=_f(row.get("nearest_dx_mm")),
                nearest_dy_mm=_f(row.get("nearest_dy_mm")),
                nearest_dz_mm=_f(row.get("nearest_dz_mm")),
                all_nodule_ids=_s(row.get("all_nodule_ids")) or None,
                all_nodule_dists_mm=_s(row.get("all_nodule_dists_mm")) or None,
                ipsilateral=_b(row.get("ipsilateral")),
                nearest_same_lobe=_b(row.get("nearest_same_lobe")),
                bilateral_distribution=_b(row.get("bilateral_distribution")) or False,
                # reinsertion (100% complete)
                reinsertion_lobe=_s(row.get("reinsertion_lobe"), "unknown"),
                reinsertion_lung_side=_s(row.get("reinsertion_lung_side"), "unknown"),
                reinsertion_lung_zone=_s(row.get("reinsertion_lung_zone"), "unknown"),
                reinsertion_lobe_cc_pct=_f(row.get("reinsertion_lobe_cc_pct"), 0.0),
                reinsertion_lobe_ml_pct=_f(row.get("reinsertion_lobe_ml_pct"), 0.0),
                reinsertion_lobe_ap_pct=_f(row.get("reinsertion_lobe_ap_pct"), 0.0),
                reinsertion_lung_cc_pct=_f(row.get("reinsertion_lung_cc_pct"), 0.0),
                reinsertion_lung_ml_pct=_f(row.get("reinsertion_lung_ml_pct"), 0.0),
                reinsertion_lung_ap_pct=_f(row.get("reinsertion_lung_ap_pct"), 0.0),
                reinsertion_pleural_dist_mm=_f(row.get("reinsertion_pleural_dist_mm")),
                reinsertion_airway_dist_mm=_f(row.get("reinsertion_airway_dist_mm")),
                reinsertion_nodule_diam_mm=_f(row.get("reinsertion_nodule_diam_mm"), 0.0),
                meta=meta,
            )
            profiles.append(p)

        return profiles

    @staticmethod
    def load_to_dataframe(dataset: str, csv_path: str) -> pd.DataFrame:
        """
        Load CSV, add normalised `label`, `patient_id`, `dataset` columns,
        and return the full DataFrame (core + extra columns).
        Faster than load() for index-building.
        """
        adapter = get_adapter(dataset)
        normaliser = LabelNormaliser(dataset)

        df = pd.read_csv(csv_path, low_memory=False)
        df = adapter.preprocess(df)
        df["_row_index"] = df.index.astype(str)

        # Normalised identity columns
        df["dataset"] = dataset
        df["patient_id"] = df.apply(adapter.get_patient_id, axis=1)
        df["annotation_id"] = df.apply(adapter.get_annotation_id, axis=1)
        df["label"] = normaliser.normalise_series(df)

        # Drop internal helper
        df = df.drop(columns=["_row_index"])

        return df
