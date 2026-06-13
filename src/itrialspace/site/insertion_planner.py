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
Insertion planner — determines insertion coordinates and scaling
for each case in the synthetic cohort.

Three modes:
- profile_faithful: use reinsertion_* columns directly from the nodule profile
- prescribed: user specifies target anatomy; planner samples from observed distributions
- randomised: uniform sample within target lobe's observed percentile ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from itrialspace.site.spec import InsertionSpec


@dataclass
class InsertionPlan:
    """Result of planning a single nodule insertion."""

    insertion_coord_x: float = 0.0
    insertion_coord_y: float = 0.0
    insertion_coord_z: float = 0.0
    insertion_lobe: str = "unknown"
    insertion_lobe_cc_pct: float = 0.0
    insertion_lobe_ml_pct: float = 0.0
    insertion_lobe_ap_pct: float = 0.0
    insertion_mode: str = "profile_faithful"
    scale_factor: float = 1.0
    scale_factor_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    warp_applied: str = "none"  # "none" | "isotropic" | "anisotropic"
    effective_diam_mm: float = 0.0
    feasible: bool = True
    infeasibility_reason: str = ""


class InsertionPlanner:
    """Plans insertion coordinates and scaling for synthetic nodule placement."""

    def __init__(self, index_df: pd.DataFrame):
        """
        Args:
            index_df: The NoduleIndex DataFrame (used to compute lobe
                      statistics for prescribed/randomised modes).
        """
        self._df = index_df
        self._lobe_stats = self._compute_lobe_ranges()

    def plan(
        self,
        nodule_row: pd.Series,
        mode: str,
        target_diam_mm: Optional[float] = None,
        insertion_spec: Optional["InsertionSpec"] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> InsertionPlan:
        """Dispatch to mode-specific planner.

        Args:
            nodule_row: Row from NoduleIndex DataFrame (the donor nodule).
            mode: "profile_faithful", "prescribed", or "randomised".
            target_diam_mm: Desired diameter (triggers scaling if different from donor).
            insertion_spec: InsertionSpec for prescribed/randomised modes.
            rng: Random generator (required for prescribed/randomised).

        Returns:
            InsertionPlan with coordinates, scaling, and feasibility.
        """
        if mode == "profile_faithful":
            plan = self._plan_profile_faithful(nodule_row)
        elif mode == "prescribed":
            if rng is None:
                raise ValueError("rng required for prescribed mode")
            plan = self._plan_prescribed(nodule_row, insertion_spec, rng)
        elif mode == "randomised":
            if rng is None:
                raise ValueError("rng required for randomised mode")
            plan = self._plan_randomised(nodule_row, insertion_spec, rng)
        else:
            raise ValueError(f"Unknown insertion mode: {mode}")

        # Apply scaling if target diameter specified
        if target_diam_mm is not None:
            donor_diam = float(nodule_row.get("reinsertion_nodule_diam_mm", 0))
            if donor_diam > 0:
                sf, warp = self._compute_scale(donor_diam, target_diam_mm, insertion_spec)
                plan.scale_factor = sf
                plan.warp_applied = warp
                plan.effective_diam_mm = donor_diam * sf
            else:
                plan.effective_diam_mm = target_diam_mm
        else:
            plan.effective_diam_mm = float(nodule_row.get("reinsertion_nodule_diam_mm", 0))

        plan.insertion_mode = mode
        return plan

    # ── Mode implementations ──────────────────────────────────────────────────

    def _plan_profile_faithful(self, row: pd.Series) -> InsertionPlan:
        """Use reinsertion_* columns directly."""
        return InsertionPlan(
            insertion_coord_x=float(row.get("coordX", 0)),
            insertion_coord_y=float(row.get("coordY", 0)),
            insertion_coord_z=float(row.get("coordZ", 0)),
            insertion_lobe=str(row.get("reinsertion_lobe", "unknown")),
            insertion_lobe_cc_pct=float(row.get("reinsertion_lobe_cc_pct", 0)),
            insertion_lobe_ml_pct=float(row.get("reinsertion_lobe_ml_pct", 0)),
            insertion_lobe_ap_pct=float(row.get("reinsertion_lobe_ap_pct", 0)),
        )

    def _plan_prescribed(
        self,
        row: pd.Series,
        spec: Optional["InsertionSpec"],
        rng: np.random.Generator,
    ) -> InsertionPlan:
        """Sample position within prescribed lobe/zone from observed distributions."""
        from itrialspace.site.spec import InsertionSpec

        spec = spec or InsertionSpec()
        target_lobe = spec.target_lobe or str(row.get("reinsertion_lobe", "unknown"))

        stats = self._lobe_stats.get(target_lobe)
        if stats is None:
            return InsertionPlan(
                insertion_lobe=target_lobe,
                feasible=False,
                infeasibility_reason=f"No statistics for lobe '{target_lobe}'",
            )

        cc = rng.uniform(stats["cc_p10"], stats["cc_p90"])
        ml = rng.uniform(stats["ml_p10"], stats["ml_p90"])
        ap = rng.uniform(stats["ap_p10"], stats["ap_p90"])

        return InsertionPlan(
            insertion_lobe=target_lobe,
            insertion_lobe_cc_pct=cc,
            insertion_lobe_ml_pct=ml,
            insertion_lobe_ap_pct=ap,
        )

    def _plan_randomised(
        self,
        row: pd.Series,
        spec: Optional["InsertionSpec"],
        rng: np.random.Generator,
    ) -> InsertionPlan:
        """Uniform sample within target lobe's observed ranges."""
        from itrialspace.site.spec import InsertionSpec

        spec = spec or InsertionSpec()
        target_lobe = spec.target_lobe or str(row.get("reinsertion_lobe", "unknown"))

        stats = self._lobe_stats.get(target_lobe)
        if stats is None:
            return InsertionPlan(
                insertion_lobe=target_lobe,
                feasible=False,
                infeasibility_reason=f"No statistics for lobe '{target_lobe}'",
            )

        cc = rng.uniform(stats["cc_min"], stats["cc_max"])
        ml = rng.uniform(stats["ml_min"], stats["ml_max"])
        ap = rng.uniform(stats["ap_min"], stats["ap_max"])

        return InsertionPlan(
            insertion_lobe=target_lobe,
            insertion_lobe_cc_pct=cc,
            insertion_lobe_ml_pct=ml,
            insertion_lobe_ap_pct=ap,
        )

    # ── Scaling ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_scale(
        donor_diam: float,
        target_diam: float,
        spec: Optional["InsertionSpec"],
    ) -> tuple[float, str]:
        """Compute scale factor and warp type.

        Priority:
        1. Constrained match (within tolerance) → no warp
        2. Isotropic scale → uniform resize
        3. Anisotropic warp → per-axis stretch (opt-in)

        Returns:
            (scale_factor, warp_type) where warp_type is "none"|"isotropic"|"anisotropic"
        """
        from itrialspace.site.spec import InsertionSpec

        spec = spec or InsertionSpec()
        if donor_diam <= 0:
            return (1.0, "none")

        ratio = target_diam / donor_diam

        # Within tolerance → no scaling needed
        if abs(ratio - 1.0) <= spec.scale_tolerance:
            return (1.0, "none")

        # Isotropic scale
        if spec.allow_isotropic_scale and ratio <= spec.max_scale_factor:
            return (ratio, "isotropic")

        # Anisotropic warp
        if spec.allow_anisotropic_warp and ratio <= spec.max_scale_factor:
            return (ratio, "anisotropic")

        # Scale factor exceeds maximum — clamp
        clamped = min(ratio, spec.max_scale_factor)
        warp = "isotropic" if spec.allow_isotropic_scale else "anisotropic"
        return (clamped, warp)

    # ── Lobe statistics ───────────────────────────────────────────────────────

    def _compute_lobe_ranges(self) -> dict[str, dict]:
        """Compute per-lobe position statistics from the index.

        Uses reinsertion_* columns (100% complete) to build
        min/max/percentile ranges for each lobe.
        """
        from itrialspace.core.schema import LOBE_NAMES

        stats: dict[str, dict] = {}
        for lobe in LOBE_NAMES:
            mask = self._df["reinsertion_lobe"] == lobe
            sub = self._df.loc[mask]
            if len(sub) < 5:
                continue

            cc = sub["reinsertion_lobe_cc_pct"].dropna()
            ml = sub["reinsertion_lobe_ml_pct"].dropna()
            ap = sub["reinsertion_lobe_ap_pct"].dropna()

            if len(cc) < 5:
                continue

            stats[lobe] = {
                "n": len(sub),
                "cc_min": float(cc.min()),
                "cc_max": float(cc.max()),
                "cc_p10": float(cc.quantile(0.10)),
                "cc_p90": float(cc.quantile(0.90)),
                "cc_median": float(cc.median()),
                "ml_min": float(ml.min()),
                "ml_max": float(ml.max()),
                "ml_p10": float(ml.quantile(0.10)),
                "ml_p90": float(ml.quantile(0.90)),
                "ap_min": float(ap.min()),
                "ap_max": float(ap.max()),
                "ap_p10": float(ap.quantile(0.10)),
                "ap_p90": float(ap.quantile(0.90)),
            }

        return stats
