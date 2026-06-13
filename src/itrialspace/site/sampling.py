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
Distribution sampling utilities — stateless helper functions for
sampling nodule sizes, labels, and lobe assignments from trial
specifications.
"""

from __future__ import annotations

import numpy as np

from itrialspace.site.spec import SIZE_BUCKET_ALIASES, SIZE_BUCKET_RANGES


class DistributionSampler:
    """Stateless sampling helpers for trial cohort generation."""

    @staticmethod
    def sample_size_from_buckets(
        bucket_weights: dict[str, float],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample n diameters from weighted size buckets.

        Within each bucket, diameters are uniformly distributed
        over the bucket's range.

        Returns:
            Array of float diameters in mm, shape (n,).
        """
        buckets = list(bucket_weights.keys())
        weights = np.array([bucket_weights[b] for b in buckets], dtype=float)
        weights /= weights.sum()

        bucket_indices = rng.choice(len(buckets), size=n, p=weights)
        diameters = np.empty(n, dtype=float)

        for i, idx in enumerate(bucket_indices):
            b = buckets[idx]
            rng_lohi = SIZE_BUCKET_RANGES.get(b) or SIZE_BUCKET_ALIASES.get(b)
            if rng_lohi is None:
                raise KeyError(
                    f"Unknown size bucket {b!r}; valid: "
                    f"{list(SIZE_BUCKET_RANGES) + list(SIZE_BUCKET_ALIASES)}"
                )
            lo, hi = rng_lohi
            diameters[i] = rng.uniform(lo, hi)

        return diameters

    @staticmethod
    def sample_size_lognormal(
        n: int,
        mean_log_mm: float,
        std_log_mm: float,
        min_mm: float,
        max_mm: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample n diameters from a truncated log-normal distribution."""
        samples = []
        while len(samples) < n:
            batch = rng.lognormal(mean_log_mm, std_log_mm, size=n * 2)
            valid = batch[(batch >= min_mm) & (batch <= max_mm)]
            samples.extend(valid.tolist())
        return np.array(samples[:n])

    @staticmethod
    def sample_labels(
        n: int,
        malignancy_rate: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample n binary labels at given malignancy rate.

        Returns:
            Array of int (0 or 1), shape (n,).
        """
        return rng.binomial(1, malignancy_rate, size=n).astype(int)

    @staticmethod
    def sample_labels_by_size(
        sizes: np.ndarray,
        rate_by_bucket: dict[str, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Assign labels conditioned on nodule size.

        Uses per-bucket malignancy rates from clinical trial data.
        More realistic than a flat prevalence.
        """
        labels = np.zeros(len(sizes), dtype=int)
        for i, d in enumerate(sizes):
            bucket = _diameter_to_bucket(d)
            rate = rate_by_bucket.get(bucket, 0.0)
            labels[i] = rng.binomial(1, rate)
        return labels

    @staticmethod
    def sample_lobe_distribution(
        lobe_weights: dict[str, float],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample n lobe assignments from weighted distribution.

        Returns:
            Array of str lobe names, shape (n,).
        """
        lobes = list(lobe_weights.keys())
        weights = np.array([lobe_weights[l] for l in lobes], dtype=float)
        weights /= weights.sum()
        indices = rng.choice(len(lobes), size=n, p=weights)
        return np.array([lobes[i] for i in indices])

    @staticmethod
    def assign_no_nodule_cases(
        n: int,
        no_nodule_fraction: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return boolean mask where True = no nodule (clean case).

        Returns:
            Array of bool, shape (n,).
        """
        return rng.random(n) < no_nodule_fraction


def _diameter_to_bucket(d: float) -> str:
    """Map a diameter in mm to its size bucket string."""
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
