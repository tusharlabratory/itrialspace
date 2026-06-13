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
CohortBuilder — the main orchestrator.

Resolves a TrialSpec into a CohortManifest by:
1. Validating the spec
2. Selecting nodule and host pools via NoduleQuery
3. Assigning labels and sizes
4. Pairing hosts with donors (native/matched/synthetic)
5. Planning insertions
6. Resolving file paths
7. Assembling the manifest
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from itrialspace.index.nodule_index import NoduleIndex
from itrialspace.query.matcher import ReinsertionMatcher, ReinsertionTarget
from itrialspace.site.insertion_planner import InsertionPlanner
from itrialspace.site.manifest import CohortManifest
from itrialspace.site.path_resolver import PathResolver
from itrialspace.site.sampling import DistributionSampler, _diameter_to_bucket
from itrialspace.site.spec import (
    HISTOPATH_DATASETS,
    LABEL_SOURCE_MAP,
    POPULATION_TYPE_MAP,
    SCREENING_DATASETS,
    DigitalTwinCompleteSpec,
    DigitalTwinCrossSpec,
    DigitalTwinIsolationSpec,
    TrialSpec,
)
from itrialspace.site.validation import SpecValidator

# Demographics column mapping per dataset (meta column names → manifest columns)
_DEMO_COL_MAP: dict[str, dict[str, str]] = {
    "DLCS24": {
        "Age": "patient_age",
        "Sex": "patient_sex",
        "Smoking Status (Current/Former/Never/Unknown)": "smoking_status",
    },
    "LUNA25": {"age": "patient_age", "gender": "patient_sex"},
    "IMDCT": {},
    "NSCLCR": {},
    "LUNGx": {},
    "LNDbv4": {},
    "LUNA16": {},
}


class CohortBuilder:
    """Resolves a TrialSpec into a CohortManifest."""

    def __init__(self, index: NoduleIndex, path_resolver: PathResolver):
        self._index = index
        self._resolver = path_resolver
        self._matcher = ReinsertionMatcher(index)
        self._planner = InsertionPlanner(index.df)

    def build(self, spec: TrialSpec, verbose: bool = True) -> CohortManifest:
        """Main entry point: TrialSpec → CohortManifest.

        If spec.n_bootstrap > 1, returns the first bootstrap replicate.
        Use build_all() for multiple replicates.
        """
        # Validate
        validator = SpecValidator(self._index.df)
        errors = validator.validate(spec)
        if errors:
            raise ValueError("Invalid TrialSpec:\n" + "\n".join(f"  - {e}" for e in errors))

        if verbose:
            report = validator.feasibility_report(spec)
            if report["warnings"]:
                for w in report["warnings"]:
                    warnings.warn(f"iTrialSpace feasibility: {w}")
            print(
                f"Pool: {report['pool_size']} nodules | "
                f"Malignant: {report['malignant_pool']} | "
                f"Benign: {report['benign_pool']}"
            )

        rng = np.random.default_rng(spec.seed)

        # Build based on mode
        if spec.cohort_mode == "native":
            rows = self._build_native(spec, rng, verbose)
        elif spec.cohort_mode == "matched":
            rows = self._build_matched(spec, rng, verbose)
        elif spec.cohort_mode == "synthetic":
            rows = self._build_synthetic(spec, rng, verbose)
        else:
            raise ValueError(f"Unknown cohort_mode: {spec.cohort_mode}")

        df = pd.DataFrame(rows)
        return CohortManifest(df, spec)

    def build_all(self, spec: TrialSpec, verbose: bool = False) -> list[CohortManifest]:
        """Build multiple bootstrap replicates."""
        manifests = []
        for i in range(spec.n_bootstrap):
            boot_spec = TrialSpec(
                **{
                    **spec.__dict__,
                    "seed": spec.seed + i,
                }
            )
            manifest = self.build(boot_spec, verbose=verbose)
            # Tag bootstrap_id
            manifest.df["bootstrap_id"] = i
            manifests.append(manifest)
        return manifests

    # ── Mode: Native ──────────────────────────────────────────────────────────

    def _build_native(self, spec: TrialSpec, rng: np.random.Generator, verbose: bool) -> list[dict]:
        """Patient + their own nodule. No cross-patient mixing."""
        pool = self._select_nodule_pool(spec)

        if spec.malignancy_prevalence is not None:
            pool = self._sample_by_prevalence(pool, spec, rng)
        else:
            pool = self._sample_pool(pool, spec.n_cases, rng)

        rows = []
        for case_id, (_, nod) in enumerate(pool.iterrows()):
            case_rows = self._build_case_rows(
                case_id=case_id,
                spec=spec,
                donor_row=nod,
                host_row=nod,  # same patient
                target_diam_mm=None,
                rng=rng,
            )
            rows.extend(case_rows)

        return rows

    # ── Mode: Matched ─────────────────────────────────────────────────────────

    def _build_matched(
        self, spec: TrialSpec, rng: np.random.Generator, verbose: bool
    ) -> list[dict]:
        """Host CT + anatomically-matched donor nodule via ReinsertionMatcher."""
        host_pool = self._select_host_pool(spec)
        donor_pool = self._select_nodule_pool(spec)

        # Sample target sizes
        target_sizes = self._sample_target_sizes(spec, rng)

        # Sample labels
        labels = self._sample_label_assignments(spec, rng)

        # Sample hosts
        n = spec.n_cases
        if len(host_pool) < n:
            host_indices = rng.choice(len(host_pool), size=n, replace=True)
        else:
            host_indices = rng.choice(len(host_pool), size=n, replace=False)

        rows = []
        for case_id in range(n):
            host_row = host_pool.iloc[host_indices[case_id]]
            target_diam = target_sizes[case_id] if target_sizes is not None else None
            label = labels[case_id] if labels is not None else None

            # Build reinsertion target for matching
            target = ReinsertionTarget(
                lobe=str(host_row.get("reinsertion_lobe", "right_lung_upper_lobe")),
                lobe_cc_pct=float(host_row.get("reinsertion_lobe_cc_pct", 50)),
                diameter_mm=target_diam,
                label=label,
                include_datasets=spec.donor_datasets or [],
                exclude_datasets=spec.exclude_datasets or [],
            )

            match = self._matcher.find_best(target)
            if match is None:
                # Fallback: relax label constraint
                target.label = None
                match = self._matcher.find_best(target)

            if match is None:
                if verbose:
                    warnings.warn(f"Case {case_id}: no donor found, skipping")
                continue

            case_rows = self._build_case_rows(
                case_id=case_id,
                spec=spec,
                donor_row=match.row,
                host_row=host_row,
                target_diam_mm=target_diam,
                rng=rng,
                label_override=label,
            )
            rows.extend(case_rows)

        return rows

    # ── Mode: Synthetic ───────────────────────────────────────────────────────

    def _build_synthetic(
        self, spec: TrialSpec, rng: np.random.Generator, verbose: bool
    ) -> list[dict]:
        """Any nodule + any host CT, paired independently."""
        donor_pool = self._select_nodule_pool(spec)
        host_pool = self._select_host_pool(spec)

        target_sizes = self._sample_target_sizes(spec, rng)
        labels = self._sample_label_assignments(spec, rng)

        n = spec.n_cases

        rows = []
        for case_id in range(n):
            label = labels[case_id] if labels is not None else None
            target_diam = target_sizes[case_id] if target_sizes is not None else None

            # Select donor matching label and size
            donor_row = self._find_donor(donor_pool, label, target_diam, rng)
            if donor_row is None:
                if verbose:
                    warnings.warn(f"Case {case_id}: no suitable donor, skipping")
                continue

            # Select host randomly
            host_idx = rng.integers(0, len(host_pool))
            host_row = host_pool.iloc[host_idx]

            case_rows = self._build_case_rows(
                case_id=case_id,
                spec=spec,
                donor_row=donor_row,
                host_row=host_row,
                target_diam_mm=target_diam,
                rng=rng,
                label_override=label,
            )
            rows.extend(case_rows)

        return rows

    # ── Pool selection ────────────────────────────────────────────────────────

    def _filter_missing_masks(
        self,
        pool: pd.DataFrame,
        dataset: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Drop rows whose nodule mask file does not exist on disk.

        This prevents downstream insertion failures caused by annotations
        in the NoduleIndex that lack a corresponding segmentation mask
        (upstream data gap).  Skipped when the mask directory doesn't exist
        (e.g. in unit-test environments with synthetic data).
        """
        mask_dir = os.path.join(self._resolver._base_dir, "masks", dataset, "nodule_seg")
        if not os.path.isdir(mask_dir):
            return pool

        mask_exists = pool.apply(
            lambda row: os.path.isfile(
                self._resolver.resolve_nodule_mask_path(
                    dataset,
                    str(row["annotation_id"]),
                    ct_id=PathResolver.extract_ct_id(str(row.get("ct_path", ""))),
                )
            ),
            axis=1,
        )
        n_missing = (~mask_exists).sum()
        if n_missing > 0 and verbose:
            dropped = pool.loc[~mask_exists, "annotation_id"].tolist()
            print(f"  ⚠ Dropped {n_missing} annotation(s) with missing nodule masks: " f"{dropped}")
        return pool[mask_exists].reset_index(drop=True)

    def _select_nodule_pool(self, spec: TrialSpec) -> pd.DataFrame:
        """Use NoduleQuery to build the donor nodule pool."""
        q = self._index.query()

        # Dataset constraints
        ds = spec.donor_datasets or spec.source_datasets
        if ds:
            q = q.datasets(ds)
        if spec.exclude_datasets:
            q = q.exclude_datasets(spec.exclude_datasets)

        # Training exclusion — remove datasets the model was trained on
        if spec.exclude_training_datasets:
            q = q.exclude_datasets(spec.exclude_training_datasets)

        # Nodule spec constraints
        ns = spec.nodule_spec
        if ns:
            if ns.diameter_range:
                q = q.reinsertion_diameter(min=ns.diameter_range[0], max=ns.diameter_range[1])
            if ns.lobe:
                q = q.reinsertion_lobe(ns.lobe)
            if ns.zone:
                q = q.reinsertion_zone(ns.zone)
            if ns.side:
                q = q.reinsertion_side(ns.side)
            if ns.pleural_distance_range:
                q = q.reinsertion_pleural(
                    min=ns.pleural_distance_range[0],
                    max=ns.pleural_distance_range[1],
                )
            if ns.exclude_all_malignant_datasets:
                q = q.exclude_datasets(["NSCLCR"])

            # Label source filter: histopathology, radiology, or all
            if ns.label_source and ns.label_source != "all":
                allowed = LABEL_SOURCE_MAP.get(ns.label_source)
                if allowed:
                    q = q.datasets(allowed)
            elif ns.require_histopath_label:  # deprecated fallback
                q = q.datasets(HISTOPATH_DATASETS)

            # Population type filter: screening, diagnostic, or all
            if ns.population_type and ns.population_type != "all":
                allowed = POPULATION_TYPE_MAP.get(ns.population_type)
                if allowed:
                    q = q.datasets(allowed)

            # Multi-nodule filter
            if ns.n_nodules_filter == "single_only":
                q = q.n_nodules_in_patient(min=1, max=1)
            elif ns.n_nodules_filter == "multi_only":
                q = q.n_nodules_in_patient(min=2)

        pool = q.fetch()

        # Filter out annotations whose mask files are missing on disk.
        # This prevents deterministic Stage-2 failures for all build paths
        # (modes 1-10 synthetic/matched/native AND digital-twin modes).
        if hasattr(self, "_resolver") and hasattr(self._resolver, "_base_dir"):
            datasets = pool["dataset"].unique() if "dataset" in pool.columns else []
            parts = []
            for ds in datasets:
                ds_mask = pool["dataset"] == ds
                ds_pool = pool[ds_mask]
                filtered = self._filter_missing_masks(ds_pool, ds, verbose=True)
                parts.append(filtered)
            if parts:
                pool = pd.concat(parts, ignore_index=True)

        return pool

    def _select_host_pool(self, spec: TrialSpec) -> pd.DataFrame:
        """Select unique patient CTs for the host pool."""
        q = self._index.query()

        ds = spec.host_datasets or spec.source_datasets
        if ds:
            q = q.datasets(ds)
        if spec.exclude_datasets:
            q = q.exclude_datasets(spec.exclude_datasets)

        # Training exclusion — remove datasets the model was trained on
        if spec.exclude_training_datasets:
            q = q.exclude_datasets(spec.exclude_training_datasets)

        # Population type filter applies to hosts too
        ns = spec.nodule_spec
        if ns and ns.population_type and ns.population_type != "all":
            allowed = POPULATION_TYPE_MAP.get(ns.population_type)
            if allowed:
                q = q.datasets(allowed)

        pool = q.fetch()
        # Deduplicate to one row per patient (first nodule row)
        pool = pool.drop_duplicates(subset=["patient_id", "ct_path"], keep="first")

        # Filter out hosts whose organ segmentation file is missing on disk.
        # This prevents deterministic Stage-2 "Failed to load organ seg" errors.
        if hasattr(self, "_resolver") and hasattr(self._resolver, "_base_dir"):

            def _has_organ_seg(row):
                seg_path = self._resolver.resolve_organ_seg_path(
                    str(row.get("dataset", "")),
                    PathResolver.extract_ct_id(str(row.get("ct_path", ""))),
                )
                return os.path.isfile(seg_path)

            seg_exists = pool.apply(_has_organ_seg, axis=1)
            n_missing = (~seg_exists).sum()
            if n_missing > 0:
                dropped_ids = pool.loc[~seg_exists, "patient_id"].tolist()
                print(
                    f"  \u26a0 Dropped {n_missing} host(s) with missing organ "
                    f"segmentations: {dropped_ids}"
                )
                pool = pool[seg_exists].reset_index(drop=True)

        return pool.reset_index(drop=True)

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def _sample_by_prevalence(
        self, pool: pd.DataFrame, spec: TrialSpec, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Sample from pool respecting malignancy prevalence."""
        prev = spec.malignancy_prevalence
        if prev is None:
            return self._sample_pool(pool, spec.n_cases, rng)

        n = spec.n_cases
        n_mal = int(round(n * prev))
        n_ben = n - n_mal

        mal_pool = pool[pool["label"] == 1]
        ben_pool = pool[pool["label"] == 0]

        mal_n = min(n_mal, len(mal_pool))
        ben_n = min(n_ben, len(ben_pool))

        mal_sample = (
            mal_pool.sample(n=mal_n, random_state=rng.integers(2**31))
            if mal_n > 0
            else mal_pool.iloc[:0]
        )
        ben_sample = (
            ben_pool.sample(n=ben_n, random_state=rng.integers(2**31))
            if ben_n > 0
            else ben_pool.iloc[:0]
        )

        result = pd.concat([mal_sample, ben_sample], ignore_index=True)
        return result.sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)

    def _sample_pool(self, pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Simple random sample from pool."""
        replace = len(pool) < n
        return pool.sample(n=n, replace=replace, random_state=rng.integers(2**31)).reset_index(
            drop=True
        )

    def _sample_target_sizes(
        self, spec: TrialSpec, rng: np.random.Generator
    ) -> Optional[np.ndarray]:
        """Sample target diameters from spec if size distribution is defined."""
        ns = spec.nodule_spec
        if ns and ns.size_distribution:
            sd = ns.size_distribution
            if sd.bucket_weights:
                return DistributionSampler.sample_size_from_buckets(
                    sd.bucket_weights, spec.n_cases, rng
                )
            elif sd.mean_log_mm is not None and sd.std_log_mm is not None:
                return DistributionSampler.sample_size_lognormal(
                    spec.n_cases, sd.mean_log_mm, sd.std_log_mm, sd.min_mm, sd.max_mm, rng
                )
        return None

    def _sample_label_assignments(
        self, spec: TrialSpec, rng: np.random.Generator
    ) -> Optional[np.ndarray]:
        """Sample label assignments from prevalence."""
        if spec.malignancy_prevalence is not None:
            return DistributionSampler.sample_labels(spec.n_cases, spec.malignancy_prevalence, rng)
        return None

    def _find_donor(
        self,
        pool: pd.DataFrame,
        label: Optional[int],
        target_diam: Optional[float],
        rng: np.random.Generator,
    ) -> Optional[pd.Series]:
        """Find a suitable donor nodule from the pool."""
        candidates = pool

        # Filter by label
        if label is not None:
            label_match = candidates[candidates["label"] == label]
            if len(label_match) > 0:
                candidates = label_match

        # Filter by size (within 50% tolerance for initial search)
        if target_diam is not None and len(candidates) > 5:
            lo = target_diam * 0.5
            hi = target_diam * 2.0
            size_match = candidates[
                (candidates["reinsertion_nodule_diam_mm"] >= lo)
                & (candidates["reinsertion_nodule_diam_mm"] <= hi)
            ]
            if len(size_match) > 0:
                candidates = size_match

        if len(candidates) == 0:
            return None

        idx = rng.integers(0, len(candidates))
        return candidates.iloc[idx]

    # ── Companion resolution ─────────────────────────────────────────────────

    def _resolve_companions(
        self,
        primary_row: pd.Series,
        spec: TrialSpec,
    ) -> list[pd.Series]:
        """Find companion nodules from the same donor patient.

        Uses the all_nodule_ids field (comma-delimited annotation IDs)
        to look up sibling nodules in the index DataFrame.

        Returns list of pd.Series rows (excluding the primary itself).
        Skipped in native mode (nodules already exist in the CT).
        """
        if spec.companion_strategy == "none":
            return []
        if spec.cohort_mode == "native":
            return []

        all_ids_str = primary_row.get("all_nodule_ids")
        if pd.isna(all_ids_str) or not all_ids_str:
            return []

        all_ids = [aid.strip() for aid in str(all_ids_str).split(",")]
        primary_id = str(primary_row.get("annotation_id", ""))
        companion_ids = [aid for aid in all_ids if aid and aid != primary_id]

        if not companion_ids:
            return []

        # Look up companions in the full index
        idx_df = self._index.df
        companions = idx_df[idx_df["annotation_id"].isin(companion_ids)]

        # Apply strategy filter
        if spec.companion_strategy == "ipsilateral":
            primary_side = primary_row.get("reinsertion_lung_side", "")
            companions = companions[companions["reinsertion_lung_side"] == primary_side]
        elif spec.companion_strategy == "same_lobe":
            primary_lobe = primary_row.get("reinsertion_lobe", "")
            companions = companions[companions["reinsertion_lobe"] == primary_lobe]
        # "all_companions" → no further filter

        return [row for _, row in companions.iterrows()]

    # ── Case row assembly ─────────────────────────────────────────────────────

    def _build_case_rows(
        self,
        case_id: int,
        spec: TrialSpec,
        donor_row: pd.Series,
        host_row: pd.Series,
        target_diam_mm: Optional[float],
        rng: np.random.Generator,
        label_override: Optional[int] = None,
    ) -> list[dict]:
        """Assemble manifest rows for a case, including any companions."""
        primary = self._build_single_row(
            case_id=case_id,
            spec=spec,
            donor_row=donor_row,
            host_row=host_row,
            target_diam_mm=target_diam_mm,
            rng=rng,
            label_override=label_override,
            nodule_idx=0,
            is_primary=True,
        )
        rows = [primary]

        # Resolve companions
        companions = self._resolve_companions(donor_row, spec)
        for i, comp_row in enumerate(companions):
            comp = self._build_single_row(
                case_id=case_id,
                spec=spec,
                donor_row=comp_row,
                host_row=host_row,
                target_diam_mm=None,  # companions keep original size
                rng=rng,
                label_override=None,  # companions keep their own label
                nodule_idx=i + 1,
                is_primary=False,
            )
            rows.append(comp)

        # Set n_nodules_in_case on all rows
        n = len(rows)
        for r in rows:
            r["n_nodules_in_case"] = n
            r["companion_group_id"] = case_id

        return rows

    def _build_single_row(
        self,
        case_id: int,
        spec: TrialSpec,
        donor_row: pd.Series,
        host_row: pd.Series,
        target_diam_mm: Optional[float],
        rng: np.random.Generator,
        label_override: Optional[int] = None,
        nodule_idx: int = 0,
        is_primary: bool = True,
    ) -> dict:
        """Assemble a single manifest row."""
        # Insertion planning
        plan = self._planner.plan(
            nodule_row=donor_row,
            mode=spec.insertion_mode,
            target_diam_mm=target_diam_mm,
            insertion_spec=spec.insertion_spec,
            rng=rng,
        )

        # Path resolution
        host_ds = str(host_row.get("dataset", ""))
        donor_ds = str(donor_row.get("dataset", ""))
        host_ct_rel = str(host_row.get("ct_path", ""))
        donor_ct_rel = str(donor_row.get("ct_path", ""))
        donor_ann_id = str(donor_row.get("annotation_id", ""))
        host_ct_id = PathResolver.extract_ct_id(host_ct_rel)
        donor_ct_id = PathResolver.extract_ct_id(donor_ct_rel)

        # Label
        if label_override is not None:
            label = label_override
        else:
            raw_label = donor_row.get("label")
            label = int(raw_label) if pd.notna(raw_label) else None

        # Diameter
        donor_diam = float(donor_row.get("reinsertion_nodule_diam_mm", 0))
        effective_diam = plan.effective_diam_mm if plan.effective_diam_mm > 0 else donor_diam

        # Size bucket
        size_bucket = _diameter_to_bucket(effective_diam) if effective_diam > 0 else ""

        # Population type and label source
        population_type = "screening" if donor_ds in SCREENING_DATASETS else "diagnostic"
        label_source = "histopathology" if donor_ds in HISTOPATH_DATASETS else "radiology"

        # Demographics extraction
        patient_age, patient_sex, smoking_status, pack_years = self._extract_demographics(
            host_row, host_ds
        )

        return {
            # Case identity
            "case_id": case_id,
            "nodule_idx": nodule_idx,
            "is_primary_nodule": is_primary,
            "n_nodules_in_case": 1,  # updated by _build_case_rows
            "companion_group_id": case_id,  # updated by _build_case_rows
            "trial_name": spec.trial_name,
            "trial_template": spec.trial_template or "",
            "bootstrap_id": 0,
            # Host CT
            "host_patient_id": str(host_row.get("patient_id", "")),
            "host_dataset": host_ds,
            "host_ct_path": self._resolver.resolve_ct_path(host_ds, host_ct_rel),
            "host_organ_seg_path": self._resolver.resolve_organ_seg_path(host_ds, host_ct_id),
            # Donor nodule
            "donor_patient_id": str(donor_row.get("patient_id", "")),
            "donor_annotation_id": donor_ann_id,
            "donor_dataset": donor_ds,
            "donor_nodule_mask_path": self._resolver.resolve_nodule_mask_path(
                donor_ds, donor_ann_id, ct_id=donor_ct_id
            ),
            "donor_ct_path": self._resolver.resolve_ct_path(donor_ds, donor_ct_rel),
            "donor_refined_seg_path": self._resolver.resolve_refined_seg_path(
                donor_ds, donor_ct_id
            ),
            # Insertion
            "insertion_coord_x": plan.insertion_coord_x,
            "insertion_coord_y": plan.insertion_coord_y,
            "insertion_coord_z": plan.insertion_coord_z,
            "insertion_lobe": plan.insertion_lobe,
            "insertion_lobe_cc_pct": plan.insertion_lobe_cc_pct,
            "insertion_lobe_ml_pct": plan.insertion_lobe_ml_pct,
            "insertion_lobe_ap_pct": plan.insertion_lobe_ap_pct,
            "insertion_mode": plan.insertion_mode,
            # Nodule characteristics
            "nodule_diam_mm": donor_diam,
            "effective_diam_mm": effective_diam,
            "scale_factor": plan.scale_factor,
            "warp_applied": plan.warp_applied,
            "label": label,
            # Anatomy
            "nodule_lobe_name": str(donor_row.get("reinsertion_lobe", "")),
            "nodule_lung_side": str(donor_row.get("reinsertion_lung_side", "")),
            "nodule_lung_zone": str(donor_row.get("reinsertion_lung_zone", "")),
            "nodule_central_peripheral": str(donor_row.get("central_peripheral", "")),
            "pleural_distance_mm": donor_row.get("reinsertion_pleural_dist_mm"),
            # Demographics
            "patient_age": patient_age,
            "patient_sex": patient_sex,
            "smoking_status": smoking_status,
            "pack_years": pack_years,
            # Cohort metadata
            "cohort_mode": spec.cohort_mode,
            "size_bucket": size_bucket,
            "population_type": population_type,
            "label_source": label_source,
        }

    @staticmethod
    def _extract_demographics(
        row: pd.Series, dataset: str
    ) -> tuple[Optional[float], Optional[str], Optional[str], Optional[float]]:
        """Extract demographics from meta columns (best-effort)."""
        col_map = _DEMO_COL_MAP.get(dataset, {})

        age = None
        sex = None
        smoking = None
        pack_years = None

        for src_col, tgt in col_map.items():
            val = row.get(src_col)
            if pd.isna(val) if isinstance(val, float) else (val is None):
                continue
            if tgt == "patient_age":
                try:
                    age = float(val)
                except (ValueError, TypeError):
                    pass
            elif tgt == "patient_sex":
                sex = str(val)
            elif tgt == "smoking_status":
                smoking = str(val)
            elif tgt == "pack_years":
                try:
                    pack_years = float(val)
                except (ValueError, TypeError):
                    pass

        return age, sex, smoking, pack_years

    # ── Digital twin isolation ────────────────────────────────────────────────

    def build_digital_twin_isolation(
        self,
        spec: DigitalTwinIsolationSpec,
        verbose: bool = True,
    ) -> CohortManifest:
        """Build a digital twin isolation manifest.

        For each CT scan in the dataset, generates one isolation case
        per native nodule. Each case pairs the host's clean anatomy mask
        with exactly one native nodule mask.

        Matching is at the CT-scan level: a nodule is only paired with
        the CT it was originally annotated from.  Patients with multiple
        CT scans produce separate cases per CT (not grouped by patient).

        Returns a CohortManifest whose DataFrame uses the extended
        ISOLATION_MANIFEST_COLS schema.
        """
        from itrialspace.site.sampling import _diameter_to_bucket

        rng = np.random.default_rng(spec.seed)
        dataset = spec.dataset

        # 1. Select nodule pool from the target dataset
        q = self._index.query().datasets([dataset])

        # Apply nodule-level filters
        ns = spec.nodule_spec
        if ns:
            if ns.label is not None:
                q = q.label(ns.label)
            if ns.diameter_range:
                lo, hi = ns.diameter_range
                if lo is not None:
                    q = q.reinsertion_diameter(min=lo)
                if hi is not None:
                    q = q.reinsertion_diameter(max=hi)

        pool = q.fetch()

        # Filter out annotations whose mask files are missing on disk
        pool = self._filter_missing_masks(pool, dataset, verbose=verbose)

        if len(pool) == 0:
            raise ValueError(
                f"No eligible nodules found in dataset '{dataset}' with the given filters."
            )

        # 2. Group by patient (for user-facing selection), then by CT scan
        grouped_by_patient = pool.groupby("patient_id")
        all_patient_ids = sorted(grouped_by_patient.groups.keys())

        total_eligible_patients = len(all_patient_ids)
        total_eligible_nodules = len(pool)

        # 3. Select patients
        if spec.all_patients:
            selected_ids = all_patient_ids
        else:
            n = min(spec.max_patients, total_eligible_patients)
            indices = rng.choice(total_eligible_patients, size=n, replace=False)
            indices.sort()
            selected_ids = [all_patient_ids[i] for i in indices]

        total_selected_patients = len(selected_ids)

        # Gather nodules for selected patients, then group by ct_path
        selected_pool = pool[pool["patient_id"].isin(selected_ids)]
        grouped_by_ct = selected_pool.groupby("ct_path")
        total_eligible_cts = len(grouped_by_ct)

        if verbose:
            print(f"Dataset: {dataset}")
            print(f"  Eligible patients: {total_eligible_patients}")
            print(f"  Selected patients: {total_selected_patients}")
            print(f"  CT scans (selected): {total_eligible_cts}")
            print(f"  Eligible nodules (total pool): {total_eligible_nodules}")

        # 4. Build isolation cases — group by CT scan, not patient
        rows = []
        case_id = 0

        for ct_path_key in sorted(grouped_by_ct.groups.keys()):
            ct_nodules = grouped_by_ct.get_group(ct_path_key).reset_index(drop=True)
            host_n_nodules = len(ct_nodules)
            patient_id = ct_nodules.iloc[0]["patient_id"]

            # Optionally cap nodules per patient
            if spec.max_nodules_per_patient is not None:
                ct_nodules = ct_nodules.head(spec.max_nodules_per_patient)

            for iso_idx, (_, nod) in enumerate(ct_nodules.iterrows()):
                ct_rel = str(nod.get("ct_path", ""))
                ct_id = PathResolver.extract_ct_id(ct_rel)
                ann_id = str(nod.get("annotation_id", ""))
                donor_diam = float(nod.get("reinsertion_nodule_diam_mm", 0))
                size_bucket = _diameter_to_bucket(donor_diam) if donor_diam > 0 else ""

                raw_label = nod.get("label")
                label_val = int(raw_label) if pd.notna(raw_label) else None

                population_type = "screening" if dataset in SCREENING_DATASETS else "diagnostic"
                label_source = "histopathology" if dataset in HISTOPATH_DATASETS else "radiology"

                patient_age, patient_sex, smoking_status, pack_years = self._extract_demographics(
                    nod, dataset
                )

                host_ct_abs = self._resolver.resolve_ct_path(dataset, ct_rel)
                donor_ct_abs = self._resolver.resolve_ct_path(dataset, ct_rel)

                # Safety: host and donor must reference the same CT scan
                assert host_ct_abs == donor_ct_abs, (
                    f"CT mismatch: host={host_ct_abs} != donor={donor_ct_abs} "
                    f"for annotation {ann_id}"
                )

                row = {
                    # Case identity
                    "case_id": case_id,
                    "nodule_idx": 0,
                    "is_primary_nodule": True,
                    "n_nodules_in_case": 1,
                    "companion_group_id": case_id,
                    "trial_name": spec.trial_name,
                    "trial_template": "",
                    "bootstrap_id": 0,
                    # Host CT (same CT scan as the nodule)
                    "host_patient_id": str(patient_id),
                    "host_dataset": dataset,
                    "host_ct_path": host_ct_abs,
                    "host_organ_seg_path": self._resolver.resolve_organ_seg_path(dataset, ct_id),
                    # Donor = same CT scan's nodule (native)
                    "donor_patient_id": str(patient_id),
                    "donor_annotation_id": ann_id,
                    "donor_dataset": dataset,
                    "donor_nodule_mask_path": self._resolver.resolve_nodule_mask_path(
                        dataset, ann_id, ct_id=ct_id
                    ),
                    "donor_ct_path": donor_ct_abs,
                    "donor_refined_seg_path": self._resolver.resolve_refined_seg_path(
                        dataset, ct_id
                    ),
                    # Insertion — use nodule's own reinsertion coords (profile_faithful)
                    "insertion_coord_x": float(nod.get("coordX", 0)),
                    "insertion_coord_y": float(nod.get("coordY", 0)),
                    "insertion_coord_z": float(nod.get("coordZ", 0)),
                    "insertion_lobe": str(nod.get("reinsertion_lobe", "")),
                    "insertion_lobe_cc_pct": float(nod.get("reinsertion_lobe_cc_pct", 0)),
                    "insertion_lobe_ml_pct": float(nod.get("reinsertion_lobe_ml_pct", 0)),
                    "insertion_lobe_ap_pct": float(nod.get("reinsertion_lobe_ap_pct", 0)),
                    "insertion_mode": "profile_faithful",
                    # Nodule characteristics
                    "nodule_diam_mm": donor_diam,
                    "effective_diam_mm": donor_diam,
                    "scale_factor": 1.0,
                    "warp_applied": "none",
                    "label": label_val,
                    # Anatomy
                    "nodule_lobe_name": str(nod.get("reinsertion_lobe", "")),
                    "nodule_lung_side": str(nod.get("reinsertion_lung_side", "")),
                    "nodule_lung_zone": str(nod.get("reinsertion_lung_zone", "")),
                    "nodule_central_peripheral": str(nod.get("central_peripheral", "")),
                    "pleural_distance_mm": nod.get("reinsertion_pleural_dist_mm"),
                    # Demographics
                    "patient_age": patient_age,
                    "patient_sex": patient_sex,
                    "smoking_status": smoking_status,
                    "pack_years": pack_years,
                    # Cohort metadata
                    "cohort_mode": "native",
                    "size_bucket": size_bucket,
                    "population_type": population_type,
                    "label_source": label_source,
                    # Digital twin isolation columns
                    "mode": "digital_twin_isolation",
                    "target_annotation_id": ann_id,
                    "target_nodule_mask_path": self._resolver.resolve_nodule_mask_path(
                        dataset, ann_id, ct_id=ct_id
                    ),
                    "target_diameter_mm": donor_diam,
                    "target_label": label_val,
                    "target_lobe": str(nod.get("reinsertion_lobe", "")),
                    "target_side": str(nod.get("reinsertion_lung_side", "")),
                    "target_zone": str(nod.get("reinsertion_lung_zone", "")),
                    "host_n_nodules": host_n_nodules,
                    "isolation_case_index": iso_idx,
                }
                rows.append(row)
                case_id += 1

        total_generated = len(rows)

        if verbose:
            print(f"  Generated isolation cases: {total_generated}")
            print("\nSummary:")
            print(f"  Total eligible patients:     {total_eligible_patients}")
            print(f"  Total selected patients:     {total_selected_patients}")
            print(f"  Total CT scans:              {total_eligible_cts}")
            print(f"  Total eligible nodules:      {total_eligible_nodules}")
            print(f"  Total generated cases:       {total_generated}")

        df = pd.DataFrame(rows)

        # Create a lightweight TrialSpec wrapper for CohortManifest compatibility
        wrapper_spec = TrialSpec(
            trial_name=spec.trial_name,
            n_cases=total_generated,
            cohort_mode="native",
            seed=spec.seed,
            source_datasets=[dataset],
            base_dir=spec.base_dir,
        )

        return CohortManifest(df, wrapper_spec)

    # ── Digital twin complete ─────────────────────────────────────────────────

    def build_digital_twin_complete(
        self,
        spec: DigitalTwinCompleteSpec,
        verbose: bool = True,
    ) -> CohortManifest:
        """Build a digital twin complete manifest.

        For each CT scan in the dataset, generates ONE case containing
        ALL native nodules from that CT.  Rows sharing a case_id are processed
        sequentially by Stage 2 (insert_manifest), accumulating into one
        combined conditioning mask per CT scan.

        Matching is at the CT-scan level: a nodule is only paired with
        the CT it was originally annotated from.  Patients with multiple
        CT scans produce one case per CT (not one case per patient).

        Returns a CohortManifest whose DataFrame uses the extended
        COMPLETE_MANIFEST_COLS schema.
        """
        from itrialspace.site.sampling import _diameter_to_bucket

        rng = np.random.default_rng(spec.seed)
        dataset = spec.dataset

        # 1. Select nodule pool from the target dataset
        q = self._index.query().datasets([dataset])

        # Apply nodule-level filters
        ns = spec.nodule_spec
        if ns:
            if ns.label is not None:
                q = q.label(ns.label)
            if ns.diameter_range:
                lo, hi = ns.diameter_range
                if lo is not None:
                    q = q.reinsertion_diameter(min=lo)
                if hi is not None:
                    q = q.reinsertion_diameter(max=hi)

        pool = q.fetch()

        # Filter out annotations whose mask files are missing on disk
        pool = self._filter_missing_masks(pool, dataset, verbose=verbose)

        if len(pool) == 0:
            raise ValueError(
                f"No eligible nodules found in dataset '{dataset}' with the given filters."
            )

        # 2. Group by patient (for user-facing selection), then by CT scan
        grouped_by_patient = pool.groupby("patient_id")
        all_patient_ids = sorted(grouped_by_patient.groups.keys())

        total_eligible_patients = len(all_patient_ids)
        total_eligible_nodules = len(pool)

        # 3. Select patients
        if spec.all_patients:
            selected_ids = all_patient_ids
        else:
            n = min(spec.max_patients, total_eligible_patients)
            indices = rng.choice(total_eligible_patients, size=n, replace=False)
            indices.sort()
            selected_ids = [all_patient_ids[i] for i in indices]

        total_selected_patients = len(selected_ids)

        # Gather nodules for selected patients, then group by ct_path
        selected_pool = pool[pool["patient_id"].isin(selected_ids)]
        grouped_by_ct = selected_pool.groupby("ct_path")
        total_eligible_cts = len(grouped_by_ct)

        if verbose:
            print(f"Dataset: {dataset}")
            print(f"  Eligible patients: {total_eligible_patients}")
            print(f"  Selected patients: {total_selected_patients}")
            print(f"  CT scans (selected): {total_eligible_cts}")
            print(f"  Eligible nodules (total pool): {total_eligible_nodules}")

        # 4. Build complete cases — one case_id per CT scan, one row per nodule
        rows = []
        case_id = 0
        total_nodules = 0

        for ct_path_key in sorted(grouped_by_ct.groups.keys()):
            ct_nodules = grouped_by_ct.get_group(ct_path_key).reset_index(drop=True)
            n_nodules = len(ct_nodules)
            total_nodules += n_nodules
            patient_id = ct_nodules.iloc[0]["patient_id"]

            # Collect per-CT summary lists for the case-level columns
            ann_ids_list = []
            diams_list = []
            labels_list = []
            lobes_list = []
            sides_list = []
            zones_list = []

            for _, nod in ct_nodules.iterrows():
                ann_ids_list.append(str(nod.get("annotation_id", "")))
                diams_list.append(str(round(float(nod.get("reinsertion_nodule_diam_mm", 0)), 2)))
                raw_label = nod.get("label")
                labels_list.append(str(int(raw_label)) if pd.notna(raw_label) else "")
                lobes_list.append(str(nod.get("reinsertion_lobe", "")))
                sides_list.append(str(nod.get("reinsertion_lung_side", "")))
                zones_list.append(str(nod.get("reinsertion_lung_zone", "")))

            # Semicolon-delimited summary strings
            ann_ids_str = ";".join(ann_ids_list)
            diams_str = ";".join(diams_list)
            labels_str = ";".join(labels_list)
            lobes_str = ";".join(lobes_list)
            sides_str = ";".join(sides_list)
            zones_str = ";".join(zones_list)

            for nod_idx, (_, nod) in enumerate(ct_nodules.iterrows()):
                ct_rel = str(nod.get("ct_path", ""))
                ct_id = PathResolver.extract_ct_id(ct_rel)
                ann_id = str(nod.get("annotation_id", ""))
                donor_diam = float(nod.get("reinsertion_nodule_diam_mm", 0))
                size_bucket = _diameter_to_bucket(donor_diam) if donor_diam > 0 else ""

                raw_label = nod.get("label")
                label_val = int(raw_label) if pd.notna(raw_label) else None

                population_type = "screening" if dataset in SCREENING_DATASETS else "diagnostic"
                label_source = "histopathology" if dataset in HISTOPATH_DATASETS else "radiology"

                patient_age, patient_sex, smoking_status, pack_years = self._extract_demographics(
                    nod, dataset
                )

                host_ct_abs = self._resolver.resolve_ct_path(dataset, ct_rel)
                donor_ct_abs = self._resolver.resolve_ct_path(dataset, ct_rel)

                # Safety: host and donor must reference the same CT scan
                assert host_ct_abs == donor_ct_abs, (
                    f"CT mismatch: host={host_ct_abs} != donor={donor_ct_abs} "
                    f"for annotation {ann_id}"
                )

                row = {
                    # Case identity — same case_id for all nodules in this CT
                    "case_id": case_id,
                    "nodule_idx": nod_idx,
                    "is_primary_nodule": nod_idx == 0,
                    "n_nodules_in_case": n_nodules,
                    "companion_group_id": case_id,
                    "trial_name": spec.trial_name,
                    "trial_template": "",
                    "bootstrap_id": 0,
                    # Host CT (same CT scan as the nodule)
                    "host_patient_id": str(patient_id),
                    "host_dataset": dataset,
                    "host_ct_path": host_ct_abs,
                    "host_organ_seg_path": self._resolver.resolve_organ_seg_path(dataset, ct_id),
                    # Donor = same CT scan's nodule (native)
                    "donor_patient_id": str(patient_id),
                    "donor_annotation_id": ann_id,
                    "donor_dataset": dataset,
                    "donor_nodule_mask_path": self._resolver.resolve_nodule_mask_path(
                        dataset, ann_id, ct_id=ct_id
                    ),
                    "donor_ct_path": donor_ct_abs,
                    "donor_refined_seg_path": self._resolver.resolve_refined_seg_path(
                        dataset, ct_id
                    ),
                    # Insertion — profile_faithful at native coordinates
                    "insertion_coord_x": float(nod.get("coordX", 0)),
                    "insertion_coord_y": float(nod.get("coordY", 0)),
                    "insertion_coord_z": float(nod.get("coordZ", 0)),
                    "insertion_lobe": str(nod.get("reinsertion_lobe", "")),
                    "insertion_lobe_cc_pct": float(nod.get("reinsertion_lobe_cc_pct", 0)),
                    "insertion_lobe_ml_pct": float(nod.get("reinsertion_lobe_ml_pct", 0)),
                    "insertion_lobe_ap_pct": float(nod.get("reinsertion_lobe_ap_pct", 0)),
                    "insertion_mode": "profile_faithful",
                    # Nodule characteristics
                    "nodule_diam_mm": donor_diam,
                    "effective_diam_mm": donor_diam,
                    "scale_factor": 1.0,
                    "warp_applied": "none",
                    "label": label_val,
                    # Anatomy
                    "nodule_lobe_name": str(nod.get("reinsertion_lobe", "")),
                    "nodule_lung_side": str(nod.get("reinsertion_lung_side", "")),
                    "nodule_lung_zone": str(nod.get("reinsertion_lung_zone", "")),
                    "nodule_central_peripheral": str(nod.get("central_peripheral", "")),
                    "pleural_distance_mm": nod.get("reinsertion_pleural_dist_mm"),
                    # Demographics
                    "patient_age": patient_age,
                    "patient_sex": patient_sex,
                    "smoking_status": smoking_status,
                    "pack_years": pack_years,
                    # Cohort metadata
                    "cohort_mode": "native",
                    "size_bucket": size_bucket,
                    "population_type": population_type,
                    "label_source": label_source,
                    # Digital twin complete columns
                    "mode": "digital_twin_complete",
                    "host_n_nodules": n_nodules,
                    "annotation_ids": ann_ids_str,
                    "diameters_mm": diams_str,
                    "labels": labels_str,
                    "lobes": lobes_str,
                    "sides": sides_str,
                    "zones": zones_str,
                }
                rows.append(row)

            case_id += 1

        total_generated_cases = case_id
        total_rows = len(rows)

        if verbose:
            print(f"  Generated cases: {total_generated_cases}")
            print(f"  Total manifest rows: {total_rows}")
            print("\nSummary:")
            print(f"  Total eligible patients:     {total_eligible_patients}")
            print(f"  Total selected patients:     {total_selected_patients}")
            print(f"  Total CT scans:              {total_eligible_cts}")
            print(f"  Total nodules:               {total_nodules}")
            print(f"  Total generated cases:       {total_generated_cases}")

        df = pd.DataFrame(rows)

        # Create a lightweight TrialSpec wrapper for CohortManifest compatibility
        wrapper_spec = TrialSpec(
            trial_name=spec.trial_name,
            n_cases=total_generated_cases,
            cohort_mode="native",
            seed=spec.seed,
            source_datasets=[dataset],
            base_dir=spec.base_dir,
        )

        return CohortManifest(df, wrapper_spec)

    # ── Digital twin cross ────────────────────────────────────────────────────

    def build_digital_twin_cross(
        self,
        spec: DigitalTwinCrossSpec,
        verbose: bool = True,
    ) -> CohortManifest:
        """Build a digital twin cross manifest.

        Pairs host anatomy from one patient/dataset with donor nodule(s)
        from a *different* patient/dataset.  Supports three pairing
        policies and two donor transfer modes.

        Returns a CohortManifest whose DataFrame uses the extended
        CROSS_MANIFEST_COLS schema.
        """

        rng = np.random.default_rng(spec.seed)

        # ── 1. Build host pool ───────────────────────────────────────────
        host_q = self._index.query().datasets([spec.host_dataset])
        if spec.exclude_training_datasets:
            host_q = host_q.exclude_datasets(spec.exclude_training_datasets)
        host_pool = host_q.fetch()
        # Deduplicate to one row per host patient
        host_pool = host_pool.drop_duplicates(
            subset=["patient_id", "ct_path"], keep="first"
        ).reset_index(drop=True)

        total_eligible_hosts = len(host_pool)
        if total_eligible_hosts == 0:
            raise ValueError(f"No eligible host patients in dataset '{spec.host_dataset}'.")

        # Select hosts
        if spec.all_host_patients:
            selected_hosts = host_pool
        else:
            n = min(spec.max_host_patients, total_eligible_hosts)
            indices = rng.choice(total_eligible_hosts, size=n, replace=False)
            indices.sort()
            selected_hosts = host_pool.iloc[indices].reset_index(drop=True)

        # ── 2. Build donor pool ──────────────────────────────────────────
        donor_q = self._index.query().datasets([spec.donor_dataset])
        if spec.exclude_training_datasets:
            donor_q = donor_q.exclude_datasets(spec.exclude_training_datasets)

        ns = spec.nodule_spec
        if ns:
            if ns.label is not None:
                donor_q = donor_q.label(ns.label)
            if ns.diameter_range:
                lo, hi = ns.diameter_range
                if lo is not None:
                    donor_q = donor_q.reinsertion_diameter(min=lo)
                if hi is not None:
                    donor_q = donor_q.reinsertion_diameter(max=hi)

        donor_pool = donor_q.fetch()

        if len(donor_pool) == 0:
            raise ValueError(
                f"No eligible donor nodules in dataset '{spec.donor_dataset}' "
                f"with the given filters."
            )

        # Group donors by patient
        donor_grouped = donor_pool.groupby("patient_id")
        all_donor_patient_ids = sorted(donor_grouped.groups.keys())
        total_eligible_donor_patients = len(all_donor_patient_ids)
        total_eligible_donor_nodules = len(donor_pool)

        # Select donor patients
        if spec.all_donor_patients:
            selected_donor_ids = all_donor_patient_ids
        elif spec.max_donor_patients is not None:
            n = min(spec.max_donor_patients, total_eligible_donor_patients)
            indices = rng.choice(total_eligible_donor_patients, size=n, replace=False)
            indices.sort()
            selected_donor_ids = [all_donor_patient_ids[i] for i in indices]
        else:
            # max_donor_nodules: take donors that cover that many nodules
            selected_donor_ids = all_donor_patient_ids  # will limit nodules below

        # Collect selected donor nodules
        donor_nodules = []
        for pid in selected_donor_ids:
            patient_nods = donor_grouped.get_group(pid).reset_index(drop=True)
            for _, nod in patient_nods.iterrows():
                donor_nodules.append(nod)

        # Apply max_donor_nodules cap
        if spec.max_donor_nodules is not None and len(donor_nodules) > spec.max_donor_nodules:
            rng.shuffle(donor_nodules)
            donor_nodules = donor_nodules[: spec.max_donor_nodules]

        total_selected_hosts = len(selected_hosts)
        total_selected_donor_nodules = len(donor_nodules)

        if verbose:
            print(f"Host dataset:  {spec.host_dataset}")
            print(f"Donor dataset: {spec.donor_dataset}")
            print(f"  Eligible host patients:       {total_eligible_hosts}")
            print(f"  Selected host patients:       {total_selected_hosts}")
            print(f"  Eligible donor patients:      {total_eligible_donor_patients}")
            print(f"  Eligible donor nodules:       {total_eligible_donor_nodules}")
            print(f"  Selected donor nodules:       {total_selected_donor_nodules}")

        # ── 3. Build cross cases ─────────────────────────────────────────
        rows = []
        case_id = 0

        if spec.pairing_policy == "one_to_one":
            # Each donor nodule paired with one host (round-robin across hosts)
            for i, donor_nod in enumerate(donor_nodules):
                host_idx = i % total_selected_hosts
                host_row = selected_hosts.iloc[host_idx]
                new_rows = self._build_cross_case_rows(
                    case_id=case_id,
                    spec=spec,
                    host_row=host_row,
                    donor_nodules_for_case=[donor_nod],
                    rng=rng,
                )
                rows.extend(new_rows)
                case_id += 1

        elif spec.pairing_policy == "one_to_many_hosts":
            # Same donor nodule paired with N hosts
            n_hosts = min(spec.n_hosts_per_donor, total_selected_hosts)
            for donor_nod in donor_nodules:
                # Select N distinct hosts for this donor
                host_indices = rng.choice(total_selected_hosts, size=n_hosts, replace=False)
                for h_idx in host_indices:
                    host_row = selected_hosts.iloc[h_idx]
                    new_rows = self._build_cross_case_rows(
                        case_id=case_id,
                        spec=spec,
                        host_row=host_row,
                        donor_nodules_for_case=[donor_nod],
                        rng=rng,
                    )
                    rows.extend(new_rows)
                    case_id += 1

        elif spec.pairing_policy == "donor_patient_complete":
            # All donor nodules from one donor patient → one host
            # Group donor_nodules back by patient
            from collections import defaultdict

            by_patient: dict[str, list] = defaultdict(list)
            for nod in donor_nodules:
                by_patient[str(nod.get("patient_id", ""))].append(nod)

            for i, (donor_pid, nods) in enumerate(by_patient.items()):
                host_idx = i % total_selected_hosts
                host_row = selected_hosts.iloc[host_idx]
                new_rows = self._build_cross_case_rows(
                    case_id=case_id,
                    spec=spec,
                    host_row=host_row,
                    donor_nodules_for_case=nods,
                    rng=rng,
                )
                rows.extend(new_rows)
                case_id += 1

        total_generated_cases = case_id
        total_rows = len(rows)

        if verbose:
            print(f"\n  Generated cross cases:  {total_generated_cases}")
            print(f"  Total manifest rows:    {total_rows}")
            print("\nSummary:")
            print(f"  Total eligible host patients:     {total_eligible_hosts}")
            print(f"  Total selected host patients:     {total_selected_hosts}")
            print(f"  Total eligible donor patients:    {total_eligible_donor_patients}")
            print(f"  Total eligible donor nodules:     {total_eligible_donor_nodules}")
            print(f"  Total selected donor nodules:     {total_selected_donor_nodules}")
            print(f"  Total generated cross cases:      {total_generated_cases}")

        df = pd.DataFrame(rows)

        wrapper_spec = TrialSpec(
            trial_name=spec.trial_name,
            n_cases=total_generated_cases,
            cohort_mode="cross",
            seed=spec.seed,
            host_datasets=[spec.host_dataset],
            donor_datasets=[spec.donor_dataset],
            base_dir=spec.base_dir,
        )

        return CohortManifest(df, wrapper_spec)

    def _build_cross_case_rows(
        self,
        case_id: int,
        spec: DigitalTwinCrossSpec,
        host_row: pd.Series,
        donor_nodules_for_case: list[pd.Series],
        rng: np.random.Generator,
    ) -> list[dict]:
        """Assemble manifest rows for a single cross case.

        For donor_transfer_mode="single", donor_nodules_for_case has 1 element.
        For "complete" or "donor_patient_complete" pairing, it may have many.
        """
        from itrialspace.site.sampling import _diameter_to_bucket

        host_ds = spec.host_dataset
        host_pid = str(host_row.get("patient_id", ""))
        host_ct_rel = str(host_row.get("ct_path", ""))
        host_ct_id = PathResolver.extract_ct_id(host_ct_rel)

        host_age, host_sex, host_smoking, host_pack_years = self._extract_demographics(
            host_row, host_ds
        )

        rows = []
        n_nodules = len(donor_nodules_for_case)

        for nod_idx, donor_nod in enumerate(donor_nodules_for_case):
            donor_ds = spec.donor_dataset
            donor_pid = str(donor_nod.get("patient_id", ""))
            donor_ann_id = str(donor_nod.get("annotation_id", ""))
            donor_ct_rel = str(donor_nod.get("ct_path", ""))
            donor_ct_id = PathResolver.extract_ct_id(donor_ct_rel)
            donor_diam = float(donor_nod.get("reinsertion_nodule_diam_mm", 0))
            size_bucket = _diameter_to_bucket(donor_diam) if donor_diam > 0 else ""

            raw_label = donor_nod.get("label")
            label_val = int(raw_label) if pd.notna(raw_label) else None

            population_type = "screening" if donor_ds in SCREENING_DATASETS else "diagnostic"
            label_source = "histopathology" if donor_ds in HISTOPATH_DATASETS else "radiology"

            # Insertion planning — use donor's reinsertion profile
            # profile_faithful_transfer: preserve donor's anatomical coordinates
            # host_constrained_transfer: use prescribed mode to resample in host
            if spec.placement_strategy == "profile_faithful_transfer":
                plan = self._planner.plan(
                    nodule_row=donor_nod,
                    mode="profile_faithful",
                    target_diam_mm=None,
                    insertion_spec=None,
                    rng=rng,
                )
            else:
                # host_constrained_transfer: use prescribed mode with donor's lobe info
                from itrialspace.site.spec import InsertionSpec

                insertion_spec = InsertionSpec(
                    target_lobe=str(donor_nod.get("reinsertion_lobe", "")),
                )
                plan = self._planner.plan(
                    nodule_row=donor_nod,
                    mode="prescribed",
                    target_diam_mm=None,
                    insertion_spec=insertion_spec,
                    rng=rng,
                )

            row = {
                # Case identity
                "case_id": case_id,
                "nodule_idx": nod_idx,
                "is_primary_nodule": nod_idx == 0,
                "n_nodules_in_case": n_nodules,
                "companion_group_id": case_id,
                "trial_name": spec.trial_name,
                "trial_template": "",
                "bootstrap_id": 0,
                # Host CT — from host patient
                "host_patient_id": host_pid,
                "host_dataset": host_ds,
                "host_ct_path": self._resolver.resolve_ct_path(host_ds, host_ct_rel),
                "host_organ_seg_path": self._resolver.resolve_organ_seg_path(host_ds, host_ct_id),
                # Donor — from different patient/dataset
                "donor_patient_id": donor_pid,
                "donor_annotation_id": donor_ann_id,
                "donor_dataset": donor_ds,
                "donor_nodule_mask_path": self._resolver.resolve_nodule_mask_path(
                    donor_ds, donor_ann_id, ct_id=donor_ct_id
                ),
                "donor_ct_path": self._resolver.resolve_ct_path(donor_ds, donor_ct_rel),
                "donor_refined_seg_path": self._resolver.resolve_refined_seg_path(
                    donor_ds, donor_ct_id
                ),
                # Insertion — from planner
                "insertion_coord_x": plan.insertion_coord_x,
                "insertion_coord_y": plan.insertion_coord_y,
                "insertion_coord_z": plan.insertion_coord_z,
                "insertion_lobe": plan.insertion_lobe,
                "insertion_lobe_cc_pct": plan.insertion_lobe_cc_pct,
                "insertion_lobe_ml_pct": plan.insertion_lobe_ml_pct,
                "insertion_lobe_ap_pct": plan.insertion_lobe_ap_pct,
                "insertion_mode": plan.insertion_mode,
                # Nodule characteristics
                "nodule_diam_mm": donor_diam,
                "effective_diam_mm": (
                    plan.effective_diam_mm if plan.effective_diam_mm > 0 else donor_diam
                ),
                "scale_factor": plan.scale_factor,
                "warp_applied": plan.warp_applied,
                "label": label_val,
                # Anatomy (donor's native anatomy)
                "nodule_lobe_name": str(donor_nod.get("reinsertion_lobe", "")),
                "nodule_lung_side": str(donor_nod.get("reinsertion_lung_side", "")),
                "nodule_lung_zone": str(donor_nod.get("reinsertion_lung_zone", "")),
                "nodule_central_peripheral": str(donor_nod.get("central_peripheral", "")),
                "pleural_distance_mm": donor_nod.get("reinsertion_pleural_dist_mm"),
                # Demographics (from host)
                "patient_age": host_age,
                "patient_sex": host_sex,
                "smoking_status": host_smoking,
                "pack_years": host_pack_years,
                # Cohort metadata
                "cohort_mode": "cross",
                "size_bucket": size_bucket,
                "population_type": population_type,
                "label_source": label_source,
                # Digital twin cross columns
                "mode": "digital_twin_cross",
                "donor_transfer_mode": spec.donor_transfer_mode,
                "pairing_policy": spec.pairing_policy,
                "placement_strategy": spec.placement_strategy,
                "cross_case_group_id": case_id,
            }
            rows.append(row)

        return rows
