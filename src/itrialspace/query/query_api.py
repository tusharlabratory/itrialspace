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
NoduleQuery — fluent builder API for querying the NoduleIndex.

Usage:
    results = (
        idx.query()
        .datasets(['DLCS24', 'LUNA25'])
        .label(1)
        .lobe('right_lung_upper_lobe')
        .diameter(min=8.0, max=20.0)
        .pleural_distance(max=10.0)
        .sample(n=100, seed=42)
        .fetch()
    )
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from itrialspace.core.profile import NoduleProfile
from itrialspace.site.spec import validate_dataset_names


class NoduleQuery:
    """
    Fluent query builder. All filter methods return `self` for chaining.
    Call `.fetch()` or `.count()` to execute.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._mask = pd.Series(True, index=df.index)
        self._sample_n: Optional[int] = None
        self._sample_seed: Optional[int] = None

    # ── Dataset filters ────────────────────────────────────────────────────────

    def datasets(self, names: Union[str, list[str]]) -> "NoduleQuery":
        """Include only these datasets."""
        if isinstance(names, str):
            names = [names]
        validate_dataset_names(names)
        self._mask &= self._df["dataset"].isin(names)
        return self

    def exclude_datasets(self, names: Union[str, list[str]]) -> "NoduleQuery":
        """Exclude these datasets."""
        if isinstance(names, str):
            names = [names]
        self._mask &= ~self._df["dataset"].isin(names)
        return self

    # ── Label filters ──────────────────────────────────────────────────────────

    def label(self, val: Optional[int]) -> "NoduleQuery":
        """Filter by label: 0=benign, 1=malignant, None=unlabelled."""
        if val is None:
            self._mask &= self._df["label"].isna()
        else:
            self._mask &= self._df["label"] == val
        return self

    def labelled(self) -> "NoduleQuery":
        """Only rows with a non-null label."""
        self._mask &= self._df["label"].notna()
        return self

    # ── Anatomy filters ────────────────────────────────────────────────────────

    def lobe(self, names: Union[str, list[str]]) -> "NoduleQuery":
        """Filter by lobe_name."""
        if isinstance(names, str):
            names = [names]
        self._mask &= self._df["lobe_name"].isin(names)
        return self

    def lung_side(self, side: str) -> "NoduleQuery":
        """'left' or 'right'."""
        self._mask &= self._df["lung_side"] == side
        return self

    def lung_zone(self, zone: Union[str, list[str]]) -> "NoduleQuery":
        """'upper_zone', 'middle_zone', or 'lower_zone'."""
        if isinstance(zone, str):
            zone = [zone]
        self._mask &= self._df["lung_zone"].isin(zone)
        return self

    def central_peripheral(self, val: str) -> "NoduleQuery":
        """'central' or 'peripheral'."""
        self._mask &= self._df["central_peripheral"] == val
        return self

    # ── Size filters ───────────────────────────────────────────────────────────

    def diameter(self, min: Optional[float] = None, max: Optional[float] = None) -> "NoduleQuery":
        """Filter by nodule_mean_diam_mm."""
        if min is not None:
            self._mask &= self._df["nodule_mean_diam_mm"] >= min
        if max is not None:
            self._mask &= self._df["nodule_mean_diam_mm"] <= max
        return self

    def volume(self, min: Optional[float] = None, max: Optional[float] = None) -> "NoduleQuery":
        """Filter by nodule_vol_mm3."""
        if min is not None:
            self._mask &= self._df["nodule_vol_mm3"].fillna(0) >= min
        if max is not None:
            self._mask &= self._df["nodule_vol_mm3"].fillna(0) <= max
        return self

    # ── Distance filters ───────────────────────────────────────────────────────

    def pleural_distance(
        self, min: Optional[float] = None, max: Optional[float] = None
    ) -> "NoduleQuery":
        """Filter by pleural_distance_mm."""
        col = self._df["pleural_distance_mm"]
        if min is not None:
            self._mask &= col.fillna(999) >= min
        if max is not None:
            self._mask &= col.fillna(999) <= max
        return self

    def airway_distance(
        self, min: Optional[float] = None, max: Optional[float] = None
    ) -> "NoduleQuery":
        """Filter by airway_distance_mm."""
        col = self._df["airway_distance_mm"]
        if min is not None:
            self._mask &= col.fillna(999) >= min
        if max is not None:
            self._mask &= col.fillna(999) <= max
        return self

    # ── Reinsertion filters (primary index — 100% complete) ────────────────────

    def reinsertion_lobe(self, name: Union[str, list[str]]) -> "NoduleQuery":
        """Filter by reinsertion_lobe."""
        if isinstance(name, str):
            name = [name]
        self._mask &= self._df["reinsertion_lobe"].isin(name)
        return self

    def reinsertion_zone(self, zone: Union[str, list[str]]) -> "NoduleQuery":
        """Filter by reinsertion_lung_zone."""
        if isinstance(zone, str):
            zone = [zone]
        self._mask &= self._df["reinsertion_lung_zone"].isin(zone)
        return self

    def reinsertion_side(self, side: str) -> "NoduleQuery":
        """Filter by reinsertion_lung_side."""
        self._mask &= self._df["reinsertion_lung_side"] == side
        return self

    def reinsertion_diameter(
        self, min: Optional[float] = None, max: Optional[float] = None
    ) -> "NoduleQuery":
        """Filter by reinsertion_nodule_diam_mm."""
        col = self._df["reinsertion_nodule_diam_mm"]
        if min is not None:
            self._mask &= col >= min
        if max is not None:
            self._mask &= col <= max
        return self

    def reinsertion_pleural(
        self, min: Optional[float] = None, max: Optional[float] = None
    ) -> "NoduleQuery":
        """Filter by reinsertion_pleural_dist_mm."""
        col = self._df["reinsertion_pleural_dist_mm"]
        if min is not None:
            self._mask &= col.fillna(999) >= min
        if max is not None:
            self._mask &= col.fillna(999) <= max
        return self

    def reinsertion_cc_pct(
        self, min: Optional[float] = None, max: Optional[float] = None
    ) -> "NoduleQuery":
        """Filter by reinsertion_lobe_cc_pct (craniocaudal position within lobe)."""
        col = self._df["reinsertion_lobe_cc_pct"]
        if min is not None:
            self._mask &= col >= min
        if max is not None:
            self._mask &= col <= max
        return self

    # ── Multi-nodule filters ─────────────────────────────────────────────────

    def n_nodules_in_patient(
        self, min: Optional[int] = None, max: Optional[int] = None
    ) -> "NoduleQuery":
        """Filter by number of nodules in the source patient."""
        col = self._df["n_nodules_in_patient"]
        if min is not None:
            self._mask &= col >= min
        if max is not None:
            self._mask &= col <= max
        return self

    # ── Sampling ───────────────────────────────────────────────────────────────

    def sample(self, n: int, seed: Optional[int] = None) -> "NoduleQuery":
        """Random sample of n results (applied at fetch time)."""
        self._sample_n = n
        self._sample_seed = seed
        return self

    # ── Execute ────────────────────────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        """Execute query and return matching rows as a DataFrame."""
        result = self._df[self._mask].copy()
        if self._sample_n is not None:
            n = min(self._sample_n, len(result))
            result = result.sample(n=n, random_state=self._sample_seed)
        return result.reset_index(drop=True)

    def count(self) -> int:
        """Return count without fetching data."""
        return int(self._mask.sum())

    def exists(self) -> bool:
        """Return True if any rows match."""
        return self.count() > 0

    def fetch_profiles(self) -> list[NoduleProfile]:
        """
        Execute query and return matching rows as NoduleProfile objects.
        Note: slower than fetch() — use for small result sets.
        """
        from itrialspace.io.loader import _b, _f, _i, _s

        profiles = []
        for _, row in self.fetch().iterrows():
            ds = row.get("dataset", "unknown")
            from itrialspace.core.schema import CORE_COLS

            core_set = set(CORE_COLS) | {
                "ct_path",
                "dataset",
                "patient_id",
                "annotation_id",
                "label",
            }
            meta = {k: v for k, v in row.items() if k not in core_set and not k.startswith("_")}
            p = NoduleProfile(
                annotation_id=str(row.get("annotation_id", "")),
                patient_id=str(row.get("patient_id", "")),
                dataset=ds,
                ct_path=_s(row.get("ct_path")),
                label=_i(row.get("label")),
                coord_x=_f(row.get("coordX"), 0.0),
                coord_y=_f(row.get("coordY"), 0.0),
                coord_z=_f(row.get("coordZ"), 0.0),
                w=_f(row.get("w"), 0.0),
                h=_f(row.get("h"), 0.0),
                d=_f(row.get("d"), 0.0),
                nodule_mean_diam_mm=_f(row.get("nodule_mean_diam_mm"), 0.0),
                nodule_vol_mm3=_f(row.get("nodule_vol_mm3")),
                lobe_name=_s(row.get("lobe_name"), "unknown"),
                lung_side=_s(row.get("lung_side"), "unknown"),
                lung_zone=_s(row.get("lung_zone"), "unknown"),
                organ_label_id=_i(row.get("organ_label_id"), 0),
                organ_label_name=_s(row.get("organ_label_name")),
                central_peripheral=_s(row.get("central_peripheral"), "unknown"),
                nearby_organs_10mm=_s(row.get("nearby_organs_10mm")) or None,
                cranio_caudal_pct=_f(row.get("cranio_caudal_pct")),
                mediolateral_pct=_f(row.get("mediolateral_pct")),
                anteroposterior_pct=_f(row.get("anteroposterior_pct")),
                lobe_cc_pct=_f(row.get("lobe_cc_pct")),
                lobe_ml_pct=_f(row.get("lobe_ml_pct")),
                lobe_ap_pct=_f(row.get("lobe_ap_pct")),
                dist_to_trachea_mm=_f(row.get("dist_to_trachea_mm")),
                dist_to_aorta_mm=_f(row.get("dist_to_aorta_mm")),
                dist_to_heart_mm=_f(row.get("dist_to_heart_mm")),
                dist_to_esophagus_mm=_f(row.get("dist_to_esophagus_mm")),
                dist_to_pulmonary_vein_mm=_f(row.get("dist_to_pulmonary_vein_mm")),
                dist_to_superior_vena_cava_mm=_f(row.get("dist_to_superior_vena_cava_mm")),
                pleural_distance_mm=_f(row.get("pleural_distance_mm")),
                airway_distance_mm=_f(row.get("airway_distance_mm")),
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

    def __repr__(self) -> str:
        return f"NoduleQuery(matching={self.count():,} rows)"
