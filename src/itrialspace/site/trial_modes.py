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
Pre-wired trial mode factories — convenience functions that produce
TrialSpec objects for the 13 scientific trial modes.

Each function returns either a single TrialSpec or a list of TrialSpecs,
ready to be passed to CohortBuilder.build() (or, for digital twin modes,
to CohortBuilder.build_digital_twin_isolation() / build_digital_twin_complete()
/ build_digital_twin_cross()).
"""

from __future__ import annotations

from typing import Optional

from itrialspace.core.schema import LOBE_NAMES
from itrialspace.site.spec import (
    SIZE_BUCKETS,
    DemographicSpec,
    DigitalTwinCompleteSpec,
    DigitalTwinCrossSpec,
    DigitalTwinIsolationSpec,
    InsertionSpec,
    NoduleSpec,
    TrialSpec,
)


def _apply_nodule_filters(
    spec: TrialSpec,
    label_source: Optional[str],
    population_type: Optional[str],
    n_nodules_filter: Optional[str] = None,
) -> None:
    """Apply label_source, population_type, and n_nodules_filter to a spec's NoduleSpec (in-place)."""
    if label_source is None and population_type is None and n_nodules_filter is None:
        return
    if spec.nodule_spec is None:
        spec.nodule_spec = NoduleSpec()
    if label_source:
        spec.nodule_spec.label_source = label_source
    if population_type:
        spec.nodule_spec.population_type = population_type
    if n_nodules_filter:
        spec.nodule_spec.n_nodules_filter = n_nodules_filter


def controlled_prevalence_study(
    n_cases: int = 1000,
    prevalence: float = 0.05,
    template: Optional[str] = "NLST",
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    companion_strategy: str = "none",
    n_nodules_filter: Optional[str] = None,
    **overrides,
) -> TrialSpec:
    """Mode 1: Fixed malignancy prevalence.

    Produces n_cases with exactly `prevalence` fraction malignant.
    Uses a template trial for demographics and size distribution if provided.

    Args:
        exclude_training_datasets: Datasets to exclude (e.g. model was trained on them).
        label_source: "histopathology", "radiology", or "all".
        population_type: "screening", "diagnostic", or "all".
        companion_strategy: "none", "all_companions", "ipsilateral", or "same_lobe".
        n_nodules_filter: "any", "single_only", or "multi_only".
    """
    if template:
        from itrialspace.site.knowledge_base import TrialKnowledgeBase

        kb = TrialKnowledgeBase()
        spec = kb.to_trial_spec(
            template,
            n_cases=n_cases,
            cohort_mode="synthetic",
            seed=seed,
            overrides={"malignancy_prevalence": prevalence, **overrides},
        )
    else:
        spec = TrialSpec(
            trial_name=f"prevalence_{prevalence:.0%}",
            n_cases=n_cases,
            malignancy_prevalence=prevalence,
            cohort_mode="synthetic",
            seed=seed,
        )
        for k, v in overrides.items():
            if hasattr(spec, k):
                setattr(spec, k, v)

    # Apply filters
    if exclude_training_datasets:
        spec.exclude_training_datasets = exclude_training_datasets
    spec.companion_strategy = companion_strategy
    _apply_nodule_filters(spec, label_source, population_type, n_nodules_filter)
    return spec


def size_detection_curve(
    n_per_bucket: int = 100,
    size_buckets: Optional[list[str]] = None,
    label: int = 1,
    template: Optional[str] = "NLST",
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    **overrides,
) -> list[TrialSpec]:
    """Mode 2: One TrialSpec per size bucket.

    Returns list of specs (one per bucket), each with n_per_bucket cases
    of a fixed label. Use for FROC size-detection curves.
    """
    if size_buckets is None:
        size_buckets = SIZE_BUCKETS

    from itrialspace.site.spec import SIZE_BUCKET_RANGES

    specs = []
    for i, bucket in enumerate(size_buckets):
        lo, hi = SIZE_BUCKET_RANGES[bucket]
        spec = TrialSpec(
            trial_name=f"size_curve_{bucket}",
            trial_template=template,
            n_cases=n_per_bucket,
            cohort_mode="synthetic",
            malignancy_prevalence=1.0 if label == 1 else 0.0,
            nodule_spec=NoduleSpec(
                label=label,
                diameter_range=(lo, hi),
            ),
            seed=seed + i,
        )
        for k, v in overrides.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        if exclude_training_datasets:
            spec.exclude_training_datasets = exclude_training_datasets
        _apply_nodule_filters(spec, label_source, population_type)
        specs.append(spec)
    return specs


def location_sensitivity(
    n_per_lobe: int = 100,
    label: int = 1,
    diameter_range: tuple[float, float] = (6.0, 15.0),
    template: Optional[str] = "NLST",
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    **overrides,
) -> list[TrialSpec]:
    """Mode 3: One TrialSpec per lobe.

    Same nodule size range, varies only location.
    Measures detection sensitivity by anatomical position.
    """
    specs = []
    for i, lobe in enumerate(LOBE_NAMES):
        spec = TrialSpec(
            trial_name=f"lobe_{lobe}",
            trial_template=template,
            n_cases=n_per_lobe,
            cohort_mode="synthetic",
            malignancy_prevalence=1.0 if label == 1 else 0.0,
            nodule_spec=NoduleSpec(
                label=label,
                diameter_range=diameter_range,
                lobe=[lobe],
            ),
            insertion_mode="prescribed",
            insertion_spec=InsertionSpec(target_lobe=lobe),
            seed=seed + i,
        )
        for k, v in overrides.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        if exclude_training_datasets:
            spec.exclude_training_datasets = exclude_training_datasets
        _apply_nodule_filters(spec, label_source, population_type)
        specs.append(spec)
    return specs


def demographic_stratification(
    n_per_stratum: int = 200,
    strata: Optional[list[dict]] = None,
    template: str = "NLST",
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    **overrides,
) -> list[TrialSpec]:
    """Mode 4: Separate cohorts by demographic strata.

    Each stratum dict can contain:
        sex (str), age_range (tuple), smoking_status (list), pack_years_min (float)
    """
    if strata is None:
        strata = [
            {"sex_ratio_male": 1.0, "age_range": (55, 65)},
            {"sex_ratio_male": 1.0, "age_range": (65, 75)},
            {"sex_ratio_male": 0.0, "age_range": (55, 65)},
            {"sex_ratio_male": 0.0, "age_range": (65, 75)},
        ]

    from itrialspace.site.knowledge_base import TrialKnowledgeBase

    kb = TrialKnowledgeBase()

    specs = []
    for i, stratum in enumerate(strata):
        demo = DemographicSpec(
            age_range=stratum.get("age_range"),
            sex_ratio_male=stratum.get("sex_ratio_male"),
            smoking_status=stratum.get("smoking_status"),
            pack_years_min=stratum.get("pack_years_min"),
        )
        spec = kb.to_trial_spec(
            template,
            n_cases=n_per_stratum,
            cohort_mode="synthetic",
            seed=seed + i,
            overrides={"demographics": demo, **overrides},
        )
        spec.trial_name = f"demo_stratum_{i}"
        if exclude_training_datasets:
            spec.exclude_training_datasets = exclude_training_datasets
        _apply_nodule_filters(spec, label_source, population_type)
        specs.append(spec)
    return specs


def counterfactual_cohort(
    n_cases: int = 500,
    template: str = "NLST",
    vary_param: str = "malignancy_prevalence",
    values: Optional[list] = None,
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
) -> list[TrialSpec]:
    """Mode 5: Same base cohort with one parameter varied.

    Useful for measuring the effect of a single variable on algorithm performance.
    """
    if values is None:
        values = [0.01, 0.02, 0.05, 0.10, 0.20]

    from itrialspace.site.knowledge_base import TrialKnowledgeBase

    kb = TrialKnowledgeBase()

    specs = []
    for i, val in enumerate(values):
        spec = kb.to_trial_spec(
            template,
            n_cases=n_cases,
            cohort_mode="synthetic",
            seed=seed,  # same seed for same base cohort
            overrides={vary_param: val},
        )
        spec.trial_name = f"counterfactual_{vary_param}={val}"
        if exclude_training_datasets:
            spec.exclude_training_datasets = exclude_training_datasets
        _apply_nodule_filters(spec, label_source, population_type)
        specs.append(spec)
    return specs


def cross_dataset_generalization(
    n_cases: int = 300,
    datasets: Optional[list[str]] = None,
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    **overrides,
) -> list[TrialSpec]:
    """Mode 6: One spec per dataset as sole donor.

    Tests whether algorithm performance varies by data source.
    """
    if datasets is None:
        datasets = ["DLCS24", "LUNA25", "LNDbv4", "IMDCT", "LUNGx"]

    specs = []
    for i, ds in enumerate(datasets):
        spec = TrialSpec(
            trial_name=f"cross_ds_{ds}",
            n_cases=n_cases,
            cohort_mode="synthetic",
            donor_datasets=[ds],
            seed=seed + i,
        )
        for k, v in overrides.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        if exclude_training_datasets:
            spec.exclude_training_datasets = exclude_training_datasets
        _apply_nodule_filters(spec, label_source, population_type)
        specs.append(spec)
    return specs


def bootstrap_confidence(
    base_spec: TrialSpec,
    n_bootstrap: int = 100,
) -> TrialSpec:
    """Mode 7: Set n_bootstrap on a spec for confidence interval estimation.

    CohortBuilder.build_all() will produce n_bootstrap independent manifests.
    """
    base_spec.n_bootstrap = n_bootstrap
    return base_spec


def algorithm_comparison(
    n_cases: int = 500,
    template: str = "NLST",
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    companion_strategy: str = "none",
    n_nodules_filter: Optional[str] = None,
    **overrides,
) -> TrialSpec:
    """Mode 8: Single standardised cohort for comparing algorithms.

    Fixed seed ensures identical manifest across runs.
    Use the same manifest to evaluate multiple models.

    Args:
        companion_strategy: "none", "all_companions", "ipsilateral", or "same_lobe".
        n_nodules_filter: "any", "single_only", or "multi_only".
    """
    from itrialspace.site.knowledge_base import TrialKnowledgeBase

    kb = TrialKnowledgeBase()
    spec = kb.to_trial_spec(
        template,
        n_cases=n_cases,
        cohort_mode="synthetic",
        seed=seed,
        overrides=overrides,
    )
    spec.trial_name = f"algo_comparison_{template}_{n_cases}"
    if exclude_training_datasets:
        spec.exclude_training_datasets = exclude_training_datasets
    spec.companion_strategy = companion_strategy
    _apply_nodule_filters(spec, label_source, population_type, n_nodules_filter)
    return spec


def screening_protocol_simulation(
    template: str = "NLST",
    n_rounds: int = 3,
    n_cases_per_round: int = 500,
    prevalence_decay: float = 0.7,
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    **overrides,
) -> list[TrialSpec]:
    """Mode 9: Multi-round screening protocol simulation.

    Each round has decreasing malignancy prevalence (simulating
    that cancers caught in round 1 are removed from later rounds).
    """
    from itrialspace.site.knowledge_base import TrialKnowledgeBase

    kb = TrialKnowledgeBase()

    base_spec = kb.to_trial_spec(
        template,
        n_cases=n_cases_per_round,
        cohort_mode="synthetic",
        seed=seed,
    )
    base_prev = base_spec.malignancy_prevalence or 0.04

    specs = []
    for r in range(n_rounds):
        round_prev = base_prev * (prevalence_decay**r)
        spec = kb.to_trial_spec(
            template,
            n_cases=n_cases_per_round,
            cohort_mode="synthetic",
            seed=seed + r,
            overrides={"malignancy_prevalence": round_prev, **overrides},
        )
        spec.trial_name = f"screening_round_{r+1}_prev={round_prev:.3f}"
        if exclude_training_datasets:
            spec.exclude_training_datasets = exclude_training_datasets
        _apply_nodule_filters(spec, label_source, population_type)
        specs.append(spec)
    return specs


def multi_nodule_realism_study(
    n_cases: int = 500,
    multi_nodule_fraction: float = 0.25,
    companion_strategy: str = "all_companions",
    template: Optional[str] = "NLST",
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
    label_source: Optional[str] = None,
    population_type: Optional[str] = None,
    **overrides,
) -> list[TrialSpec]:
    """Mode 10: Multi-nodule realism study.

    Creates two sub-cohorts:
    1. Single-nodule cases (1 - multi_nodule_fraction)
    2. Multi-nodule cases with companions (multi_nodule_fraction)

    Allows measuring the impact of multi-nodule context on detection
    performance. In NLST, ~25% of screened patients have multiple nodules.

    Args:
        multi_nodule_fraction: Fraction of cases drawn from multi-nodule patients.
        companion_strategy: How to include companion nodules ("all_companions",
            "ipsilateral", "same_lobe").
    """
    n_multi = int(n_cases * multi_nodule_fraction)
    n_single = n_cases - n_multi

    # Single-nodule sub-cohort
    single_spec = TrialSpec(
        trial_name="multi_nodule_study_single",
        trial_template=template,
        n_cases=n_single,
        cohort_mode="synthetic",
        nodule_spec=NoduleSpec(n_nodules_filter="single_only"),
        companion_strategy="none",
        seed=seed,
    )
    for k, v in overrides.items():
        if hasattr(single_spec, k):
            setattr(single_spec, k, v)
    if exclude_training_datasets:
        single_spec.exclude_training_datasets = exclude_training_datasets
    _apply_nodule_filters(single_spec, label_source, population_type)

    # Multi-nodule sub-cohort
    multi_spec = TrialSpec(
        trial_name="multi_nodule_study_multi",
        trial_template=template,
        n_cases=n_multi,
        cohort_mode="synthetic",
        nodule_spec=NoduleSpec(n_nodules_filter="multi_only"),
        companion_strategy=companion_strategy,
        seed=seed + 1,
    )
    for k, v in overrides.items():
        if hasattr(multi_spec, k):
            setattr(multi_spec, k, v)
    if exclude_training_datasets:
        multi_spec.exclude_training_datasets = exclude_training_datasets
    _apply_nodule_filters(multi_spec, label_source, population_type)

    return [single_spec, multi_spec]


def digital_twin_isolation(
    dataset: str,
    max_patients: Optional[int] = None,
    all_patients: bool = False,
    max_nodules_per_patient: Optional[int] = None,
    label: Optional[int] = None,
    diameter_min: Optional[float] = None,
    diameter_max: Optional[float] = None,
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
) -> DigitalTwinIsolationSpec:
    """Mode 11: Digital twin isolation study.

    For each host patient in the selected dataset, generate one trial case
    per native nodule.  Each case uses the host's own clean anatomy mask
    paired with exactly one of its native nodule masks.

    Scientific purpose:
        • Isolate lesion behaviour within fixed host anatomy
        • Remove multi-nodule interference
        • Enable lesion-level evaluation
        • Allow later comparison with complete multi-nodule digital twin trials

    Args:
        dataset: Source dataset name (e.g. "DLCS24", "LUNA16").
        max_patients: Process at most N eligible host patients.
            Mutually exclusive with all_patients.
        all_patients: Process every eligible host patient.
            Mutually exclusive with max_patients.
        max_nodules_per_patient: Cap the number of isolation cases per patient.
        label: Filter nodules by label (0=benign, 1=malignant, None=any).
        diameter_min: Minimum nodule diameter in mm (inclusive).
        diameter_max: Maximum nodule diameter in mm (inclusive).
        seed: Random seed for patient sampling when max_patients is set.
        exclude_training_datasets: Datasets to exclude entirely.

    Returns:
        DigitalTwinIsolationSpec carrying the full configuration.

    Raises:
        ValueError: If both max_patients and all_patients are specified, or neither.
    """
    if max_patients is not None and all_patients:
        raise ValueError("--max-patients and --all-patients are mutually exclusive")
    if max_patients is None and not all_patients:
        raise ValueError("Specify either --max-patients N or --all-patients")

    nodule_spec = NoduleSpec(
        label=label,
        diameter_range=(
            (diameter_min, diameter_max)
            if (diameter_min is not None or diameter_max is not None)
            else None
        ),
    )

    return DigitalTwinIsolationSpec(
        trial_name=f"digital_twin_isolation_{dataset}",
        dataset=dataset,
        max_patients=max_patients,
        all_patients=all_patients,
        max_nodules_per_patient=max_nodules_per_patient,
        nodule_spec=nodule_spec,
        seed=seed,
        exclude_training_datasets=exclude_training_datasets,
    )


def digital_twin_complete(
    dataset: str,
    max_patients: Optional[int] = None,
    all_patients: bool = False,
    label: Optional[int] = None,
    diameter_min: Optional[float] = None,
    diameter_max: Optional[float] = None,
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
) -> DigitalTwinCompleteSpec:
    """Mode 12: Digital twin complete study.

    For each host patient in the selected dataset, generate one trial case
    containing ALL native nodules.  The host's clean anatomy mask is paired
    with every native nodule mask belonging to that patient.

    This is the complement to digital_twin_isolation (Mode 11):
        - Isolation  → one nodule per case  (N nodules → N cases)
        - Complete   → all nodules per case (N nodules → 1 case)

    Scientific purpose:
        • Reconstruct the complete native nodule burden
        • Compare with isolation trials to measure multi-nodule detection suppression
        • Identify context-dependent model behaviour and lesion-specific failures

    Args:
        dataset: Source dataset name (e.g. "DLCS24", "LUNA16").
        max_patients: Process at most N eligible host patients.
            Mutually exclusive with all_patients.
        all_patients: Process every eligible host patient.
            Mutually exclusive with max_patients.
        label: Filter nodules by label (0=benign, 1=malignant, None=any).
        diameter_min: Minimum nodule diameter in mm (inclusive).
        diameter_max: Maximum nodule diameter in mm (inclusive).
        seed: Random seed for patient sampling when max_patients is set.
        exclude_training_datasets: Datasets to exclude entirely.

    Returns:
        DigitalTwinCompleteSpec carrying the full configuration.

    Raises:
        ValueError: If both max_patients and all_patients are specified, or neither.
    """
    if max_patients is not None and all_patients:
        raise ValueError("--max-patients and --all-patients are mutually exclusive")
    if max_patients is None and not all_patients:
        raise ValueError("Specify either --max-patients N or --all-patients")

    nodule_spec = NoduleSpec(
        label=label,
        diameter_range=(
            (diameter_min, diameter_max)
            if (diameter_min is not None or diameter_max is not None)
            else None
        ),
    )

    return DigitalTwinCompleteSpec(
        trial_name=f"digital_twin_complete_{dataset}",
        dataset=dataset,
        max_patients=max_patients,
        all_patients=all_patients,
        nodule_spec=nodule_spec,
        seed=seed,
        exclude_training_datasets=exclude_training_datasets,
    )


def digital_twin_cross(
    host_dataset: str,
    donor_dataset: str,
    max_host_patients: Optional[int] = None,
    all_host_patients: bool = False,
    max_donor_patients: Optional[int] = None,
    all_donor_patients: bool = False,
    max_donor_nodules: Optional[int] = None,
    donor_transfer_mode: str = "single",
    pairing_policy: str = "one_to_one",
    n_hosts_per_donor: int = 3,
    placement_strategy: str = "profile_faithful_transfer",
    label: Optional[int] = None,
    diameter_min: Optional[float] = None,
    diameter_max: Optional[float] = None,
    seed: int = 42,
    exclude_training_datasets: Optional[list[str]] = None,
) -> DigitalTwinCrossSpec:
    """Mode 13: Digital twin cross study.

    Pairs host anatomy from one patient with donor nodule(s) from a
    *different* patient, potentially from a different dataset.

    This enables controlled counterfactual experiments:
        - Same donor nodule across multiple host anatomies
        - Same host anatomy with different donor nodules
        - Cross-dataset anatomy/nodule mixing
        - Full donor burden transfer to new anatomy

    Supported pairing policies:
        - one_to_one: each donor nodule paired with one host
        - one_to_many_hosts: same donor paired with N hosts (crucial for anatomy-dependent analysis)
        - donor_patient_complete: all nodules from one donor patient paired to one host

    Supported donor transfer modes:
        - single: one donor nodule per case
        - complete: all donor nodules from the donor patient per case

    Args:
        host_dataset: Dataset for host anatomies (e.g. "DLCS24").
        donor_dataset: Dataset for donor nodules (e.g. "LUNA25").
        max_host_patients: Process at most N host patients.
            Mutually exclusive with all_host_patients.
        all_host_patients: Use all eligible host patients.
            Mutually exclusive with max_host_patients.
        max_donor_patients: Process at most N donor patients.
            Mutually exclusive with all_donor_patients.
        all_donor_patients: Use all eligible donor patients.
            Mutually exclusive with max_donor_patients.
        max_donor_nodules: Max total donor nodules used.
        donor_transfer_mode: "single" or "complete".
        pairing_policy: "one_to_one", "one_to_many_hosts", or "donor_patient_complete".
        n_hosts_per_donor: Number of hosts per donor (one_to_many_hosts policy).
        placement_strategy: "profile_faithful_transfer" or "host_constrained_transfer".
        label: Filter donor nodules by label (0=benign, 1=malignant, None=any).
        diameter_min: Min donor nodule diameter in mm (inclusive).
        diameter_max: Max donor nodule diameter in mm (inclusive).
        seed: Random seed for patient sampling.
        exclude_training_datasets: Datasets to exclude entirely.

    Returns:
        DigitalTwinCrossSpec carrying the full configuration.

    Raises:
        ValueError: If mutual exclusion constraints are violated.
    """
    if max_host_patients is not None and all_host_patients:
        raise ValueError("--max-host-patients and --all-host-patients are mutually exclusive")
    if max_host_patients is None and not all_host_patients:
        raise ValueError("Specify either --max-host-patients N or --all-host-patients")
    if max_donor_patients is not None and all_donor_patients:
        raise ValueError("--max-donor-patients and --all-donor-patients are mutually exclusive")
    if max_donor_patients is None and not all_donor_patients and max_donor_nodules is None:
        raise ValueError(
            "Specify either --max-donor-patients N, --all-donor-patients, or --max-donor-nodules N"
        )

    nodule_spec = NoduleSpec(
        label=label,
        diameter_range=(
            (diameter_min, diameter_max)
            if (diameter_min is not None or diameter_max is not None)
            else None
        ),
    )

    trial_name = f"digital_twin_cross_{host_dataset}_x_{donor_dataset}"

    return DigitalTwinCrossSpec(
        trial_name=trial_name,
        host_dataset=host_dataset,
        donor_dataset=donor_dataset,
        max_host_patients=max_host_patients,
        all_host_patients=all_host_patients,
        max_donor_patients=max_donor_patients,
        all_donor_patients=all_donor_patients,
        max_donor_nodules=max_donor_nodules,
        donor_transfer_mode=donor_transfer_mode,
        pairing_policy=pairing_policy,
        n_hosts_per_donor=n_hosts_per_donor,
        placement_strategy=placement_strategy,
        nodule_spec=nodule_spec,
        seed=seed,
        exclude_training_datasets=exclude_training_datasets,
    )
