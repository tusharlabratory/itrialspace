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
Trial specification dataclasses — pure data, no logic.

These define what a synthetic imaging trial looks like:
demographics, nodule characteristics, prevalence, insertion mode, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from itrialspace.config import settings


def _default_base_dir() -> str:
    """Resolve the data root at instantiation time (env-portable)."""
    return str(settings.data_dir())


# Datasets that have been permanently removed and must be rejected at all entry points.
_REMOVED_DATASETS = {"NLST3D"}


def validate_dataset_names(names: list[str] | None) -> None:
    """Raise ValueError if any removed dataset is specified."""
    if names is None:
        return
    bad = _REMOVED_DATASETS.intersection(names)
    if bad:
        raise ValueError(
            f"Dataset(s) {bad} removed due to quality issues and cannot be used. "
            f"Remove them from your dataset list."
        )


# ── Size distribution ────────────────────────────────────────────────────────

SIZE_BUCKETS = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-30mm", ">30mm"]

SIZE_BUCKET_RANGES: dict[str, tuple[float, float]] = {
    "<5mm": (1.0, 5.0),
    "5-10mm": (5.0, 10.0),
    "10-15mm": (10.0, 15.0),
    "15-20mm": (15.0, 20.0),
    "20-30mm": (20.0, 30.0),
    ">30mm": (30.0, 60.0),
}

# Open-ended aggregate buckets used by some published trial templates (UKLS, DANTE,
# NELSON, MILD) in their reported size distributions. These are NOT part of the
# canonical SIZE_BUCKETS; the sampler resolves them via SIZE_BUCKET_RANGES first,
# then these aliases.
SIZE_BUCKET_ALIASES: dict[str, tuple[float, float]] = {
    ">20mm": (20.0, 60.0),  # i.e. >=20mm, up to the max nodule diameter
}


@dataclass
class SizeDistribution:
    """How to sample nodule diameters.

    Either discrete bucket weights or continuous log-normal parameters.
    Exactly one mode should be specified.
    """

    # Discrete: keys from SIZE_BUCKETS, weights summing to ~1.0
    bucket_weights: Optional[dict[str, float]] = None

    # Continuous: truncated log-normal
    mean_log_mm: Optional[float] = None
    std_log_mm: Optional[float] = None
    min_mm: float = 3.0
    max_mm: float = 60.0


# ── Demographics ─────────────────────────────────────────────────────────────


@dataclass
class DemographicSpec:
    """Demographics constraints (best-effort; only DLCS24/LUNA25 have these)."""

    age_range: Optional[tuple[int, int]] = None
    age_mean: Optional[float] = None
    age_std: Optional[float] = None
    sex_ratio_male: Optional[float] = None  # 0.0–1.0
    smoking_status: Optional[list[str]] = None  # ["Current", "Former", ...]
    pack_years_min: Optional[float] = None


# ── Nodule specification ─────────────────────────────────────────────────────

HISTOPATH_DATASETS = ["DLCS24", "LUNA25", "NSCLCR", "IMDCT", "LUNGx"]
RADIOLOGY_DATASETS = ["LUNA16", "LNDbv4"]
SCREENING_DATASETS = ["DLCS24", "LUNA16", "LUNA25", "LNDbv4"]
DIAGNOSTIC_DATASETS = ["NSCLCR", "IMDCT", "LUNGx"]

# Valid values for label_source and population_type filters
LABEL_SOURCE_VALUES = ["histopathology", "radiology", "all"]
POPULATION_TYPE_VALUES = ["screening", "diagnostic", "all"]

# Mapping from label_source / population_type → dataset lists
LABEL_SOURCE_MAP: dict[str, list[str]] = {
    "histopathology": HISTOPATH_DATASETS,
    "radiology": RADIOLOGY_DATASETS,
}
POPULATION_TYPE_MAP: dict[str, list[str]] = {
    "screening": SCREENING_DATASETS,
    "diagnostic": DIAGNOSTIC_DATASETS,
}

# Valid values for multi-nodule controls
N_NODULES_FILTER_VALUES = ["any", "single_only", "multi_only"]
COMPANION_STRATEGY_VALUES = ["none", "all_companions", "ipsilateral", "same_lobe"]


@dataclass
class NoduleSpec:
    """Constraints on which nodules to include in the cohort."""

    label: Optional[int] = None  # 0, 1, or None=any
    diameter_range: Optional[tuple[float, float]] = None  # (min_mm, max_mm)
    size_distribution: Optional[SizeDistribution] = None
    lobe: Optional[list[str]] = None
    zone: Optional[list[str]] = None
    side: Optional[str] = None
    central_peripheral: Optional[str] = None
    pleural_distance_range: Optional[tuple[float, float]] = None
    exclude_all_malignant_datasets: bool = False  # skip NSCLCR

    # ── Label source filter ────────────────────────────────────
    # Controls which datasets are eligible based on how labels were obtained.
    #   "histopathology" → DLCS24, LUNA25, NSCLCR, IMDCT, LUNGx
    #   "radiology"      → LUNA16, LNDbv4 (radiologist suspicion level / RSL)
    #   "all"            → no restriction (default)
    #   None             → same as "all"
    label_source: Optional[str] = None

    # ── Population type filter ─────────────────────────────────
    # Controls whether to draw from screening or diagnostic populations.
    #   "screening"  → DLCS24, LUNA16, LUNA25, LNDbv4
    #   "diagnostic" → NSCLCR, IMDCT, LUNGx
    #   "all"        → no restriction (default)
    #   None         → same as "all"
    population_type: Optional[str] = None

    # Deprecated: use label_source="histopathology" instead
    require_histopath_label: bool = False

    # ── Multi-nodule donor pool filter ─────────────────────────
    # Controls which nodules are eligible based on how many nodules
    # the donor patient has.
    #   "any"          → no filter (default, current behavior)
    #   "single_only"  → only n_nodules_in_patient == 1
    #   "multi_only"   → only n_nodules_in_patient >= 2
    #   None           → same as "any"
    n_nodules_filter: Optional[str] = None


# ── Insertion specification ──────────────────────────────────────────────────


@dataclass
class InsertionSpec:
    """Controls for the insertion planner."""

    target_lobe: Optional[str] = None
    target_zone: Optional[str] = None
    target_side: Optional[str] = None
    scale_tolerance: float = 0.20  # max fractional mismatch before scaling
    allow_isotropic_scale: bool = True
    allow_anisotropic_warp: bool = False
    max_scale_factor: float = 1.5  # never scale more than 50%
    min_pleural_dist_mm: float = 2.0  # safety margin from pleura


# ── Top-level trial specification ────────────────────────────────────────────


@dataclass
class TrialSpec:
    """Complete specification of a synthetic imaging trial."""

    # ── Identity ──────────────────────────────────────────────
    trial_name: str = "untitled_trial"
    trial_template: Optional[str] = None  # e.g. "NLST" (from knowledge base)
    description: str = ""
    seed: int = 42

    # ── Cohort size ───────────────────────────────────────────
    n_cases: int = 500
    cohort_mode: str = "synthetic"  # "native" | "matched" | "synthetic"

    # ── Demographics ──────────────────────────────────────────
    demographics: Optional[DemographicSpec] = None

    # ── Nodule specification ──────────────────────────────────
    nodule_spec: Optional[NoduleSpec] = None

    # ── Prevalence ────────────────────────────────────────────
    malignancy_prevalence: Optional[float] = None  # e.g. 0.04 for 4%
    no_nodule_fraction: float = 0.0  # fraction of clean cases

    # ── Dataset constraints ───────────────────────────────────
    source_datasets: Optional[list[str]] = None
    exclude_datasets: Optional[list[str]] = None
    host_datasets: Optional[list[str]] = None  # CTs for matched/synthetic
    donor_datasets: Optional[list[str]] = None  # nodules for matched/synthetic

    # ── Training exclusion ─────────────────────────────────────
    # Datasets the model was trained on — excluded from BOTH donor and host pools.
    # Use this to ensure test cohorts have zero data leakage from training.
    # e.g. exclude_training_datasets=["LUNA16", "LNDbv4"]
    exclude_training_datasets: Optional[list[str]] = None

    # ── Multi-nodule companion inclusion ──────────────────────
    # When a donor nodule comes from a multi-nodule patient, controls
    # whether sibling nodules are also added to the manifest for that case.
    #   "none"            → one nodule per case (default, current behavior)
    #   "all_companions"  → include all sibling nodules from donor patient
    #   "ipsilateral"     → include only same-side companions
    #   "same_lobe"       → include only same-lobe companions
    companion_strategy: str = "none"

    # ── Insertion ─────────────────────────────────────────────
    insertion_mode: str = "profile_faithful"
    insertion_spec: Optional[InsertionSpec] = None

    # ── Stratification ────────────────────────────────────────
    stratify_by: Optional[list[str]] = None  # e.g. ["lobe_name", "label"]

    # ── Bootstrapping ─────────────────────────────────────────
    n_bootstrap: int = 1
    bootstrap_replace: bool = True

    # ── Path config ───────────────────────────────────────────
    base_dir: str = field(default_factory=_default_base_dir)

    def __post_init__(self):
        for attr in (
            "source_datasets",
            "exclude_datasets",
            "host_datasets",
            "donor_datasets",
            "exclude_training_datasets",
        ):
            validate_dataset_names(getattr(self, attr, None))


# ── Digital twin isolation specification ─────────────────────────────────────


@dataclass
class DigitalTwinIsolationSpec:
    """Specification for a digital twin isolation study.

    This spec is fundamentally different from TrialSpec: instead of
    specifying n_cases with donor-host pairing, it selects host patients
    from a single dataset and generates one isolation case per native nodule.

    The host body/anatomy segmentation masks are already nodule-free.
    Each native nodule has its own individual mask file.
    """

    trial_name: str = "digital_twin_isolation"
    dataset: str = ""  # Single source dataset
    max_patients: Optional[int] = None  # Max host patients
    all_patients: bool = False  # Use all eligible patients
    max_nodules_per_patient: Optional[int] = (
        None  # Cap isolation cases per CT scan (applied per-CT, not per-patient)
    )
    nodule_spec: Optional[NoduleSpec] = None  # Nodule filters
    seed: int = 42
    exclude_training_datasets: Optional[list[str]] = None
    base_dir: str = field(default_factory=_default_base_dir)

    def __post_init__(self):
        validate_dataset_names(self.exclude_training_datasets)


# ── Digital twin complete specification ──────────────────────────────────────


@dataclass
class DigitalTwinCompleteSpec:
    """Specification for a digital twin complete study.

    Complementary to DigitalTwinIsolationSpec: instead of one case per
    nodule, this mode produces one case per patient containing ALL native
    nodules.  The host body/anatomy segmentation masks are already
    nodule-free; each native nodule has its own individual mask file.

    Stage 2 will insert all nodule masks into the host anatomy with
    label value 23, producing a single combined conditioning mask per case.
    """

    trial_name: str = "digital_twin_complete"
    dataset: str = ""  # Single source dataset
    max_patients: Optional[int] = None  # Max host patients
    all_patients: bool = False  # Use all eligible patients
    nodule_spec: Optional[NoduleSpec] = None  # Nodule filters
    seed: int = 42
    exclude_training_datasets: Optional[list[str]] = None
    base_dir: str = field(default_factory=_default_base_dir)

    def __post_init__(self):
        validate_dataset_names(self.exclude_training_datasets)


# ── Digital twin cross specification ─────────────────────────────────────────

# Valid values for donor_transfer_mode and pairing_policy
DONOR_TRANSFER_MODES = ["single", "complete"]
PAIRING_POLICIES = ["one_to_one", "one_to_many_hosts", "donor_patient_complete"]
PLACEMENT_STRATEGIES = ["profile_faithful_transfer", "host_constrained_transfer"]


@dataclass
class DigitalTwinCrossSpec:
    """Specification for a digital twin cross study.

    Pairs host anatomy from one patient with donor nodule(s) from a
    *different* patient — potentially from a different dataset.

    This enables controlled counterfactual experiments:
      - Same donor nodule, different host anatomies
      - Same host anatomy, different donor nodules
      - Cross-dataset anatomy/nodule mixing

    The host body/anatomy segmentation masks are already nodule-free.
    Each donor nodule has its own individual mask file.
    """

    trial_name: str = "digital_twin_cross"

    # Host pool configuration
    host_dataset: str = ""  # Dataset for host anatomies
    max_host_patients: Optional[int] = None  # Max host patients to select
    all_host_patients: bool = False  # Use all eligible host patients

    # Donor pool configuration
    donor_dataset: str = ""  # Dataset for donor nodules
    max_donor_patients: Optional[int] = None  # Max donor patients to select
    all_donor_patients: bool = False  # Use all eligible donor patients
    max_donor_nodules: Optional[int] = None  # Max donor nodules to use

    # Cross-patient pairing configuration
    donor_transfer_mode: str = "single"  # "single" | "complete"
    pairing_policy: str = (
        "one_to_one"  # "one_to_one" | "one_to_many_hosts" | "donor_patient_complete"
    )
    n_hosts_per_donor: int = 3  # For one_to_many_hosts policy

    # Placement strategy
    placement_strategy: str = (
        "profile_faithful_transfer"  # "profile_faithful_transfer" | "host_constrained_transfer"
    )

    # Nodule filters
    nodule_spec: Optional[NoduleSpec] = None

    seed: int = 42
    exclude_training_datasets: Optional[list[str]] = None
    base_dir: str = field(default_factory=_default_base_dir)

    def __post_init__(self):
        validate_dataset_names(self.exclude_training_datasets)
        if self.donor_transfer_mode not in DONOR_TRANSFER_MODES:
            raise ValueError(
                f"donor_transfer_mode must be one of {DONOR_TRANSFER_MODES}, "
                f"got '{self.donor_transfer_mode}'"
            )
        if self.pairing_policy not in PAIRING_POLICIES:
            raise ValueError(
                f"pairing_policy must be one of {PAIRING_POLICIES}, " f"got '{self.pairing_policy}'"
            )
        if self.placement_strategy not in PLACEMENT_STRATEGIES:
            raise ValueError(
                f"placement_strategy must be one of {PLACEMENT_STRATEGIES}, "
                f"got '{self.placement_strategy}'"
            )
