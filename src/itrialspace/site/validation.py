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
Spec validation — checks a TrialSpec against the available NoduleIndex
for feasibility and correctness.
"""

from __future__ import annotations

import pandas as pd

from itrialspace.core.schema import CP_VALUES, DATASET_NAMES, LOBE_NAMES, SIDE_NAMES, ZONE_NAMES
from itrialspace.site.spec import (
    COMPANION_STRATEGY_VALUES,
    HISTOPATH_DATASETS,
    LABEL_SOURCE_MAP,
    LABEL_SOURCE_VALUES,
    N_NODULES_FILTER_VALUES,
    POPULATION_TYPE_MAP,
    POPULATION_TYPE_VALUES,
    TrialSpec,
)


class SpecValidator:
    """Validates a TrialSpec against the available NoduleIndex."""

    def __init__(self, index_df: pd.DataFrame):
        self._df = index_df

    def validate(self, spec: TrialSpec) -> list[str]:
        """Returns list of error messages. Empty = valid."""
        errors: list[str] = []

        # Cohort mode
        if spec.cohort_mode not in ("native", "matched", "synthetic"):
            errors.append(f"Invalid cohort_mode: '{spec.cohort_mode}'")

        # Insertion mode
        if spec.insertion_mode not in ("profile_faithful", "prescribed", "randomised"):
            errors.append(f"Invalid insertion_mode: '{spec.insertion_mode}'")

        # Prevalence bounds
        if spec.malignancy_prevalence is not None:
            if not 0.0 <= spec.malignancy_prevalence <= 1.0:
                errors.append(
                    f"malignancy_prevalence must be in [0,1], got {spec.malignancy_prevalence}"
                )

        if not 0.0 <= spec.no_nodule_fraction <= 1.0:
            errors.append(f"no_nodule_fraction must be in [0,1], got {spec.no_nodule_fraction}")

        # Dataset names
        for ds_list_name in (
            "source_datasets",
            "exclude_datasets",
            "host_datasets",
            "donor_datasets",
            "exclude_training_datasets",
        ):
            ds_list = getattr(spec, ds_list_name)
            if ds_list:
                for ds in ds_list:
                    if ds not in DATASET_NAMES:
                        errors.append(f"Unknown dataset '{ds}' in {ds_list_name}")

        # N cases
        if spec.n_cases < 1:
            errors.append(f"n_cases must be >= 1, got {spec.n_cases}")

        # Nodule spec validation
        ns = spec.nodule_spec
        if ns:
            if ns.label is not None and ns.label not in (0, 1):
                errors.append(f"NoduleSpec.label must be 0, 1, or None, got {ns.label}")

            if ns.lobe:
                for l in ns.lobe:
                    if l not in LOBE_NAMES:
                        errors.append(f"Unknown lobe '{l}' in NoduleSpec")

            if ns.zone:
                for z in ns.zone:
                    if z not in ZONE_NAMES:
                        errors.append(f"Unknown zone '{z}' in NoduleSpec")

            if ns.side and ns.side not in SIDE_NAMES:
                errors.append(f"Unknown side '{ns.side}' in NoduleSpec")

            if ns.central_peripheral and ns.central_peripheral not in CP_VALUES:
                errors.append(f"Unknown central_peripheral '{ns.central_peripheral}'")

            if ns.diameter_range:
                lo, hi = ns.diameter_range
                if lo >= hi:
                    errors.append(f"diameter_range min ({lo}) >= max ({hi})")

            if ns.size_distribution and ns.size_distribution.bucket_weights:
                total = sum(ns.size_distribution.bucket_weights.values())
                if abs(total - 1.0) > 0.05:
                    errors.append(f"Size distribution weights sum to {total:.3f}, expected ~1.0")

            if ns.label_source and ns.label_source not in LABEL_SOURCE_VALUES:
                errors.append(
                    f"NoduleSpec.label_source must be one of {LABEL_SOURCE_VALUES}, "
                    f"got '{ns.label_source}'"
                )

            if ns.population_type and ns.population_type not in POPULATION_TYPE_VALUES:
                errors.append(
                    f"NoduleSpec.population_type must be one of {POPULATION_TYPE_VALUES}, "
                    f"got '{ns.population_type}'"
                )

            if ns.n_nodules_filter and ns.n_nodules_filter not in N_NODULES_FILTER_VALUES:
                errors.append(
                    f"NoduleSpec.n_nodules_filter must be one of {N_NODULES_FILTER_VALUES}, "
                    f"got '{ns.n_nodules_filter}'"
                )

        # Companion strategy validation
        if spec.companion_strategy not in COMPANION_STRATEGY_VALUES:
            errors.append(
                f"companion_strategy must be one of {COMPANION_STRATEGY_VALUES}, "
                f"got '{spec.companion_strategy}'"
            )

        # Insertion spec validation
        ispec = spec.insertion_spec
        if ispec:
            if ispec.target_lobe and ispec.target_lobe not in LOBE_NAMES:
                errors.append(f"Unknown target_lobe '{ispec.target_lobe}'")
            if ispec.target_zone and ispec.target_zone not in ZONE_NAMES:
                errors.append(f"Unknown target_zone '{ispec.target_zone}'")
            if ispec.max_scale_factor < 1.0:
                errors.append(f"max_scale_factor must be >= 1.0, got {ispec.max_scale_factor}")

        return errors

    def feasibility_report(self, spec: TrialSpec) -> dict:
        """Detailed feasibility analysis against available data.

        Returns dict with:
            pool_size: total matching nodules
            malignant_pool: count of malignant nodules in pool
            benign_pool: count of benign nodules in pool
            n_requested_malignant / n_requested_benign: derived from spec
            feasible: bool
            warnings: list of strings
        """
        df = self._df.copy()
        warnings: list[str] = []

        # Apply dataset filters
        if spec.source_datasets:
            df = df[df["dataset"].isin(spec.source_datasets)]
        if spec.exclude_datasets:
            df = df[~df["dataset"].isin(spec.exclude_datasets)]
        if spec.donor_datasets:
            df = df[df["dataset"].isin(spec.donor_datasets)]
        if spec.exclude_training_datasets:
            df = df[~df["dataset"].isin(spec.exclude_training_datasets)]

        # Apply nodule spec filters
        ns = spec.nodule_spec
        if ns:
            if ns.diameter_range:
                lo, hi = ns.diameter_range
                df = df[
                    (df["reinsertion_nodule_diam_mm"] >= lo)
                    & (df["reinsertion_nodule_diam_mm"] <= hi)
                ]
            if ns.lobe:
                df = df[df["reinsertion_lobe"].isin(ns.lobe)]
            if ns.zone:
                df = df[df["reinsertion_lung_zone"].isin(ns.zone)]
            if ns.side:
                df = df[df["reinsertion_lung_side"] == ns.side]
            if ns.exclude_all_malignant_datasets:
                df = df[~df["dataset"].isin(["NSCLCR"])]

            # Label source filter
            if ns.label_source and ns.label_source != "all":
                allowed = LABEL_SOURCE_MAP.get(ns.label_source)
                if allowed:
                    df = df[df["dataset"].isin(allowed)]
            elif ns.require_histopath_label:
                df = df[df["dataset"].isin(HISTOPATH_DATASETS)]

            # Population type filter
            if ns.population_type and ns.population_type != "all":
                allowed = POPULATION_TYPE_MAP.get(ns.population_type)
                if allowed:
                    df = df[df["dataset"].isin(allowed)]

            # Multi-nodule filter
            if ns.n_nodules_filter == "single_only":
                df = df[df["n_nodules_in_patient"] == 1]
            elif ns.n_nodules_filter == "multi_only":
                df = df[df["n_nodules_in_patient"] >= 2]

        pool_size = len(df)
        mal_pool = int((df["label"] == 1).sum()) if "label" in df.columns else 0
        ben_pool = int((df["label"] == 0).sum()) if "label" in df.columns else 0

        # Compute requested counts
        n_nodule_cases = int(spec.n_cases * (1.0 - spec.no_nodule_fraction))
        n_mal = 0
        n_ben = 0
        if spec.malignancy_prevalence is not None:
            n_mal = int(n_nodule_cases * spec.malignancy_prevalence)
            n_ben = n_nodule_cases - n_mal
        else:
            n_ben = n_nodule_cases  # no prevalence constraint

        # Multi-nodule pool statistics
        n_multi = (
            int((df["n_nodules_in_patient"] >= 2).sum())
            if "n_nodules_in_patient" in df.columns
            else 0
        )
        n_single = (
            int((df["n_nodules_in_patient"] == 1).sum())
            if "n_nodules_in_patient" in df.columns
            else pool_size
        )

        feasible = True
        if pool_size < n_nodule_cases:
            warnings.append(
                f"Pool size ({pool_size}) < requested cases ({n_nodule_cases}). "
                f"Sampling with replacement may be needed."
            )
        if spec.malignancy_prevalence is not None:
            if mal_pool < n_mal:
                warnings.append(f"Malignant pool ({mal_pool}) < requested malignant ({n_mal})")
                feasible = False
            if ben_pool < n_ben:
                warnings.append(f"Benign pool ({ben_pool}) < requested benign ({n_ben})")

        # Companion strategy feasibility
        if spec.companion_strategy != "none":
            multi_patients = (
                df[df["n_nodules_in_patient"] >= 2]["patient_id"].nunique()
                if "patient_id" in df.columns
                else 0
            )
            if multi_patients < n_nodule_cases * 0.1:
                warnings.append(
                    f"companion_strategy='{spec.companion_strategy}' requested but only "
                    f"{multi_patients} multi-nodule patients in pool"
                )

        return {
            "pool_size": pool_size,
            "malignant_pool": mal_pool,
            "benign_pool": ben_pool,
            "n_requested_malignant": n_mal,
            "n_requested_benign": n_ben,
            "n_nodule_cases": n_nodule_cases,
            "datasets_in_pool": df["dataset"].unique().tolist() if len(df) > 0 else [],
            "multi_nodule_pool": n_multi,
            "single_nodule_pool": n_single,
            "feasible": feasible,
            "warnings": warnings,
        }
