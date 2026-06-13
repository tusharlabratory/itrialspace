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
Preprocessing profiles for VLM evaluation dataset extraction.

Each profile defines how 2D slices are extracted from 3D CT volumes.
Models declare which profile they need; the dataset build step applies it.

Usage in SLURM:
    python -m itrialspace.evaluation.vlm_eval.build_dataset --profile lung_axial ...

To add a new profile, define it in PROFILES below. The SLURM sub file
just passes --profile <name> and the extraction pipeline adapts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PreprocessProfile:
    """Configuration for CT slice extraction and preprocessing."""

    name: str

    # CT windowing
    window_center: float = -600.0
    window_width: float = 1500.0

    # Slice orientation: "axial", "coronal", "sagittal"
    slice_plane: str = "axial"

    # Output format: "png_gray", "png_rgb", "png_medgemma_rgb", "npy"
    output_format: str = "png_gray"

    # How many extra slices on each side of the centre (0 = centre only)
    num_context_slices: int = 1

    # Whether to save QC overlay PNGs alongside slices
    save_overlay: bool = False


# ── Named profiles ───────────────────────────────────────────────────────────
# Each entry is usable via --profile <key> in the CLI.

PROFILES: Dict[str, PreprocessProfile] = {
    "lung_axial": PreprocessProfile(
        name="lung_axial",
        window_center=-600.0,
        window_width=1500.0,
        slice_plane="axial",
        output_format="png_gray",
        num_context_slices=1,
    ),
    "mediastinal_axial": PreprocessProfile(
        name="mediastinal_axial",
        window_center=40.0,
        window_width=400.0,
        slice_plane="axial",
        output_format="png_gray",
        num_context_slices=1,
    ),
    "bone_axial": PreprocessProfile(
        name="bone_axial",
        window_center=400.0,
        window_width=1800.0,
        slice_plane="axial",
        output_format="png_gray",
        num_context_slices=1,
    ),
    "lung_coronal": PreprocessProfile(
        name="lung_coronal",
        window_center=-600.0,
        window_width=1500.0,
        slice_plane="coronal",
        output_format="png_gray",
        num_context_slices=1,
    ),
    "lung_sagittal": PreprocessProfile(
        name="lung_sagittal",
        window_center=-600.0,
        window_width=1500.0,
        slice_plane="sagittal",
        output_format="png_gray",
        num_context_slices=1,
    ),
    "lung_axial_npy": PreprocessProfile(
        name="lung_axial_npy",
        window_center=-600.0,
        window_width=1500.0,
        slice_plane="axial",
        output_format="npy",
        num_context_slices=1,
    ),
    "lung_axial_medgemma": PreprocessProfile(
        name="lung_axial_medgemma",
        # window_center/width are ignored for png_medgemma_rgb; the format
        # uses fixed 3-channel encoding: R=wide(-1024,1024), G=soft(-135,215),
        # B=brain(0,80) per the MedGemma 1.5 CT training protocol.
        window_center=-600.0,
        window_width=1500.0,
        slice_plane="axial",
        output_format="png_medgemma_rgb",
        num_context_slices=1,
    ),
}

DEFAULT_PROFILE = "lung_axial"


def get_profile(name: str) -> PreprocessProfile:
    """Look up a profile by name. Raises ValueError if not found."""
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"Unknown preprocessing profile '{name}'. " f"Available profiles: {available}"
        )
    return PROFILES[name]


def list_profiles() -> list[str]:
    """Return sorted list of available profile names."""
    return sorted(PROFILES)
