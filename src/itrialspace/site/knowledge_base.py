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
Clinical trial knowledge base — loads published trial parameters
from trials.yaml and converts them into TrialSpec objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from itrialspace.site.spec import (
    DemographicSpec,
    NoduleSpec,
    SizeDistribution,
    TrialSpec,
)

_DEFAULT_YAML = Path(__file__).parent / "config" / "trials.yaml"


class TrialKnowledgeBase:
    """Loads published trial parameters and produces TrialSpec objects."""

    def __init__(self, yaml_path: Optional[str] = None):
        if yaml is None:
            raise ImportError("pyyaml is required: pip install pyyaml")
        path = Path(yaml_path) if yaml_path else _DEFAULT_YAML
        with open(path) as f:
            data = yaml.safe_load(f)
        self._trials: dict[str, dict] = data.get("trials", {})

    @property
    def available_trials(self) -> list[str]:
        return list(self._trials.keys())

    def get_template(self, trial_name: str) -> dict:
        if trial_name not in self._trials:
            raise KeyError(f"Unknown trial '{trial_name}'. " f"Available: {self.available_trials}")
        return self._trials[trial_name]

    def to_trial_spec(
        self,
        trial_name: str,
        n_cases: int = 500,
        cohort_mode: str = "synthetic",
        insertion_mode: str = "profile_faithful",
        overrides: Optional[dict] = None,
        seed: int = 42,
    ) -> TrialSpec:
        """Convert a published trial template into a TrialSpec.

        Args:
            trial_name: Template name (e.g. "NLST", "NELSON").
            n_cases: Number of cases in the cohort.
            cohort_mode: "native", "matched", or "synthetic".
            insertion_mode: "profile_faithful", "prescribed", or "randomised".
            overrides: Dict to patch any TrialSpec field after construction.
            seed: Random seed for reproducibility.

        Returns:
            Fully populated TrialSpec.
        """
        tmpl = self.get_template(trial_name)

        # Demographics
        demo = None
        if "demographics" in tmpl:
            d = tmpl["demographics"]
            age_range = tuple(d["age_range"]) if "age_range" in d else None
            demo = DemographicSpec(
                age_range=age_range,
                age_mean=d.get("age_mean"),
                age_std=d.get("age_std"),
                sex_ratio_male=d.get("sex_ratio_male"),
                smoking_status=d.get("smoking_status"),
                pack_years_min=d.get("pack_years_min"),
            )

        # Size distribution
        size_dist = None
        if "size_distribution" in tmpl:
            sd = tmpl["size_distribution"]
            size_dist = SizeDistribution(
                bucket_weights=sd.get("bucket_weights"),
            )

        # Nodule spec
        nodule_spec = NoduleSpec(size_distribution=size_dist)

        # Prevalence
        mal_prev = None
        if "nodule_prevalence" in tmpl:
            mal_prev = tmpl["nodule_prevalence"].get("malignancy_rate")

        # Source datasets
        source_ds = tmpl.get("applicable_source_datasets")

        spec = TrialSpec(
            trial_name=f"{trial_name}_cohort",
            trial_template=trial_name,
            description=f"Cohort based on {tmpl.get('full_name', trial_name)} "
            f"({tmpl.get('publication', 'N/A')})",
            seed=seed,
            n_cases=n_cases,
            cohort_mode=cohort_mode,
            demographics=demo,
            nodule_spec=nodule_spec,
            malignancy_prevalence=mal_prev,
            source_datasets=source_ds,
            insertion_mode=insertion_mode,
        )

        # Apply overrides
        if overrides:
            for k, v in overrides.items():
                if hasattr(spec, k):
                    setattr(spec, k, v)

        return spec

    def describe(self, trial_name: str) -> str:
        """Human-readable summary of a trial template."""
        tmpl = self.get_template(trial_name)
        lines = [
            f"=== {tmpl.get('full_name', trial_name)} ===",
            f"Publication: {tmpl.get('publication', 'N/A')}",
            f"Population: {tmpl.get('population', 'N/A')}",
            f"N participants: {tmpl.get('n_participants', 'N/A')}",
        ]
        if "demographics" in tmpl:
            d = tmpl["demographics"]
            lines.append(f"Age range: {d.get('age_range', 'N/A')}")
            lines.append(f"Male ratio: {d.get('sex_ratio_male', 'N/A')}")
            lines.append(f"Min pack-years: {d.get('pack_years_min', 'N/A')}")
        if "nodule_prevalence" in tmpl:
            np_ = tmpl["nodule_prevalence"]
            lines.append(f"Any nodule prevalence: {np_.get('any_nodule', 'N/A')}")
            lines.append(f"Malignancy rate: {np_.get('malignancy_rate', 'N/A')}")
        if "size_distribution" in tmpl:
            sd = tmpl["size_distribution"]
            if "bucket_weights" in sd:
                lines.append("Size distribution:")
                for bucket, w in sd["bucket_weights"].items():
                    lines.append(f"  {bucket}: {w:.0%}")
        lines.append(f"Source datasets: {tmpl.get('applicable_source_datasets', 'N/A')}")
        return "\n".join(lines)

    def compare(self, trial_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Side-by-side comparison table of trial templates."""
        if trial_names is None:
            trial_names = self.available_trials

        rows = []
        for name in trial_names:
            tmpl = self.get_template(name)
            demo = tmpl.get("demographics", {})
            prev = tmpl.get("nodule_prevalence", {})
            row = {
                "trial": name,
                "population": tmpl.get("population"),
                "n_participants": tmpl.get("n_participants"),
                "age_range": str(demo.get("age_range", "")),
                "male_ratio": demo.get("sex_ratio_male"),
                "pack_years_min": demo.get("pack_years_min"),
                "nodule_prevalence": prev.get("any_nodule"),
                "malignancy_rate": prev.get("malignancy_rate"),
            }
            rows.append(row)

        return pd.DataFrame(rows).set_index("trial")
