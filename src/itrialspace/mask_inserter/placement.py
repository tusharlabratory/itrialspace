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

# -*- coding: utf-8 -*-
"""
placement.py — Percentile-to-voxel conversion, snapping, pleural-margin
enforcement, and overlap/collision avoidance for nodule insertion.

Coordinate convention: all index operations are performed **after**
canonicalising both the host CT and organ-segmentation to RAS+ orientation
using ``nibabel.as_closest_canonical``.  In RAS+:

  axis 0 → Right→Left        (X / mediolateral)
  axis 1 → Anterior→Posterior (Y / anteroposterior)
  axis 2 → Inferior→Superior  (Z / craniocaudal)

Percentile mapping:

  voxel_i = i_min + (ml_pct / 100) × (i_max − i_min)
  voxel_j = j_min + (ap_pct / 100) × (j_max − j_min)
  voxel_k = k_min + (cc_pct / 100) × (k_max − k_min)

where (i_min…i_max, j_min…j_max, k_min…k_max) is the bounding box of
the target structure mask in canonical index space.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class PlacementResult:
    """Output of the placement algorithm."""

    center_ijk: Optional[Tuple[int, int, int]] = None
    center_world_mm: Optional[Tuple[float, float, float]] = None
    is_feasible: bool = True
    snapped: bool = False
    snap_distance_vox: float = 0.0
    pleural_dist_mm: float = float("inf")
    overlap_fraction: float = 0.0
    warnings: list = field(default_factory=list)
    reason: str = ""


@dataclass
class PlacementConfig:
    """Parameters governing placement decisions."""

    max_snap_radius_vox: int = 25
    min_pleural_dist_mm: float = 2.0
    check_overlap: bool = True
    max_overlap_fraction: float = 0.05
    max_collision_shift_vox: int = 15


# ── Lobe label resolution ────────────────────────────────────────────────

# Canonical VISTA3D lobe labels.
DEFAULT_LOBE_LABELS = {
    "left_lung_upper_lobe": 28,
    "left_lung_lower_lobe": 29,
    "right_lung_upper_lobe": 30,
    "right_lung_middle_lobe": 31,
    "right_lung_lower_lobe": 32,
}

DEFAULT_LOBE_ALIASES = {
    "lul": "left_lung_upper_lobe",
    "lll": "left_lung_lower_lobe",
    "rul": "right_lung_upper_lobe",
    "rml": "right_lung_middle_lobe",
    "rll": "right_lung_lower_lobe",
    "left_upper": "left_lung_upper_lobe",
    "left_lower": "left_lung_lower_lobe",
    "right_upper": "right_lung_upper_lobe",
    "right_middle": "right_lung_middle_lobe",
    "right_lower": "right_lung_lower_lobe",
}


def resolve_lobe_label(
    lobe_name: str,
    lobe_labels: dict | None = None,
    lobe_aliases: dict | None = None,
) -> int:
    """Resolve a human-readable lobe name to its integer segmentation label.

    Parameters
    ----------
    lobe_name : str
        Raw value from the manifest ``insertion_lobe`` column.
    lobe_labels : dict, optional
        Canonical name → int mapping.  Defaults to VISTA3D labels.
    lobe_aliases : dict, optional
        Alias → canonical name mapping.

    Returns
    -------
    int   Label ID in the organ segmentation.

    Raises
    ------
    KeyError  If the lobe name cannot be resolved.
    """
    lobe_labels = lobe_labels or DEFAULT_LOBE_LABELS
    lobe_aliases = lobe_aliases or DEFAULT_LOBE_ALIASES

    key = lobe_name.strip().lower().replace(" ", "_")

    # Direct hit
    if key in lobe_labels:
        return lobe_labels[key]

    # Alias
    canonical = lobe_aliases.get(key)
    if canonical and canonical in lobe_labels:
        return lobe_labels[canonical]

    raise KeyError(
        f"Cannot resolve lobe '{lobe_name}' (normalised='{key}'). "
        f"Known lobes: {list(lobe_labels.keys())}; aliases: {list(lobe_aliases.keys())}"
    )


# ── Target mask extraction ────────────────────────────────────────────────


def extract_target_mask(
    seg_data: np.ndarray,
    insertion_lobe: str,
    lobe_labels: dict | None = None,
    lobe_aliases: dict | None = None,
    lung_labels: list[int] | None = None,
) -> Tuple[np.ndarray, str]:
    """Extract a binary mask for the target structure.

    Strategy:
    1. If the target lobe label exists in ``seg_data``, use it.
    2. Else if the target *side* (left/right) is identifiable, use the union
       of that side's lobes as fallback.
    3. Else use the whole-lung mask (union of all lobe labels).
    4. If nothing found, return an empty mask.

    Parameters
    ----------
    seg_data : ndarray
        3-D integer organ segmentation (canonical RAS+ space).
    insertion_lobe : str
        Target lobe from the manifest.
    lobe_labels, lobe_aliases : dict, optional
    lung_labels : list[int], optional
        Label IDs that are considered "lung parenchyma".

    Returns
    -------
    (mask, fallback_level) where *fallback_level* is one of
    ``"lobe"``, ``"side"``, ``"lung"``, or ``"empty"``.
    """
    lobe_labels = lobe_labels or DEFAULT_LOBE_LABELS
    lobe_aliases = lobe_aliases or DEFAULT_LOBE_ALIASES
    lung_labels = lung_labels or list(lobe_labels.values())

    # Attempt exact lobe
    try:
        label_id = resolve_lobe_label(insertion_lobe, lobe_labels, lobe_aliases)
    except KeyError:
        label_id = None

    if label_id is not None and np.any(seg_data == label_id):
        return (seg_data == label_id).astype(np.uint8), "lobe"

    # Fallback: side
    key = insertion_lobe.strip().lower().replace(" ", "_")
    canonical = lobe_aliases.get(key, key)
    if "left" in canonical:
        side_ids = [v for k, v in lobe_labels.items() if "left" in k]
    elif "right" in canonical:
        side_ids = [v for k, v in lobe_labels.items() if "right" in k]
    else:
        side_ids = []

    if side_ids:
        side_mask = np.isin(seg_data, side_ids).astype(np.uint8)
        if side_mask.any():
            return side_mask, "side"

    # Fallback: whole lung
    lung_mask = np.isin(seg_data, lung_labels).astype(np.uint8)
    if lung_mask.any():
        return lung_mask, "lung"

    return np.zeros_like(seg_data, dtype=np.uint8), "empty"


# ── Bounding-box helpers ──────────────────────────────────────────────────


def mask_bounding_box(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of nonzero voxels.

    Returns (mins, maxs) each of shape (ndim,).  Raises ValueError on empty.
    """
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        raise ValueError("Empty mask — no bounding box.")
    return nz.min(axis=0), nz.max(axis=0)


# ── Core percentile → voxel ──────────────────────────────────────────────


def percentile_to_voxel(
    target_mask: np.ndarray,
    cc_pct: float,
    ml_pct: float,
    ap_pct: float,
) -> Tuple[int, int, int]:
    """Convert lobe-relative percentile coordinates to an initial voxel index.

    In RAS+ canonical space:
      axis 0 = mediolateral  (R→L)   ← ml_pct
      axis 1 = anteroposterior (A→P) ← ap_pct
      axis 2 = craniocaudal  (I→S)   ← cc_pct

    **Coordinate reliability notes (iTrialSpace schema)**:
      - cc_pct: clean 0-100 percentile; used directly for the K axis.
      - ml_pct / ap_pct: known pipeline-normalisation bug — 93 %+ of values
        fall outside [0, 100] in real datasets.  When either is outside the
        valid range the function falls back to the lobe centroid at the target
        CC slab for the I/J axes, ensuring the initial candidate always lands
        inside (or very close to) the lobe.

    Returns (i, j, k) in canonical index space.  The initial candidate is
    guaranteed to land inside (or within a few voxels of) the target mask.
    """
    mins, maxs = mask_bounding_box(target_mask)

    # ── K axis: map cc_pct through *occupied* slices only ────────────────
    # The bounding-box range [k_min, k_max] can span large internal gaps
    # (e.g. post-surgical anatomy in NSCLCR where lobes are fragmented).
    # Interpolating within the bounding box would map cc_pct into a void.
    # Instead, collect the k-slices that actually contain lobe voxels and
    # use cc_pct to index into that sorted array.  This guarantees k0
    # always lands on a slice with lobe tissue.
    cc_pct_clamped = max(0.0, min(100.0, cc_pct))
    k_occupied = np.unique(np.nonzero(target_mask)[2])
    if len(k_occupied) == 1:
        k0 = int(k_occupied[0])
    else:
        idx = cc_pct_clamped / 100.0 * (len(k_occupied) - 1)
        k0 = int(k_occupied[int(round(idx))])

    # ── I/J axes: use percentile only when both values are in [0, 100] ───
    # Outside that range the values are mm-scale artefacts from the upstream
    # normalisation bug; in that case use the centroid of lobe voxels in the
    # CC slab around k0 (robust fallback).
    #
    # Even when ml/ap are in-range, the bounding-box interpolation can land
    # outside a non-convex lobe cross-section.  We start from the
    # interpolated position but snap (i0, j0) to the nearest actual lobe
    # voxel in the CC slab so the candidate is always inside the mask.
    k_lo = max(0, k0 - 5)
    k_hi = min(target_mask.shape[2], k0 + 6)
    slab = target_mask[:, :, k_lo:k_hi]
    nz = np.argwhere(slab)
    if len(nz) == 0:
        # Should not happen now that k0 is on an occupied slice, but
        # keep the whole-lobe fallback for safety.
        nz = np.argwhere(target_mask)

    if 0.0 <= ml_pct <= 100.0 and 0.0 <= ap_pct <= 100.0:
        # Bounding-box interpolation → snap to nearest slab voxel
        i_raw = mins[0] + (ml_pct / 100.0) * (maxs[0] - mins[0])
        j_raw = mins[1] + (ap_pct / 100.0) * (maxs[1] - mins[1])
        dists2d = ((nz[:, 0] - i_raw) ** 2 + (nz[:, 1] - j_raw) ** 2) ** 0.5
        nearest = nz[int(dists2d.argmin())]
        i0, j0 = int(nearest[0]), int(nearest[1])
    else:
        # Pick the lobe voxel nearest to the centroid within the CC slab.
        ci = float(nz[:, 0].mean())
        cj = float(nz[:, 1].mean())
        dists2d = ((nz[:, 0] - ci) ** 2 + (nz[:, 1] - cj) ** 2) ** 0.5
        nearest = nz[int(dists2d.argmin())]
        i0, j0 = int(nearest[0]), int(nearest[1])

    return (i0, j0, k0)


# ── Snap to nearest valid voxel ──────────────────────────────────────────


def snap_to_mask(
    target_mask: np.ndarray,
    ijk: Tuple[int, int, int],
    max_radius: int = 25,
) -> Tuple[Tuple[int, int, int], float]:
    """Snap *ijk* to the nearest voxel inside *target_mask*.

    Uses a distance-transform approach for efficiency: compute the Euclidean
    distance from the candidate point to all True voxels and pick the closest.

    Parameters
    ----------
    target_mask : ndarray  (bool / uint8)
    ijk : (i, j, k)
    max_radius : int   Search radius in voxels.

    Returns
    -------
    ((i, j, k), dist)  –  snapped index and distance in voxels.
                          Returns (ijk, 0.0) if already inside.
                          Returns (None, inf) if no valid voxel within radius.
    """
    i, j, k = ijk
    # Clamp to array bounds
    shape = target_mask.shape
    ic = max(0, min(i, shape[0] - 1))
    jc = max(0, min(j, shape[1] - 1))
    kc = max(0, min(k, shape[2] - 1))

    if target_mask[ic, jc, kc]:
        return (ic, jc, kc), 0.0

    # Extract a local cube around the candidate
    lo = np.array([ic, jc, kc]) - max_radius
    hi = np.array([ic, jc, kc]) + max_radius + 1
    lo_clamped = np.maximum(lo, 0)
    hi_clamped = np.minimum(hi, shape)

    local_mask = target_mask[
        lo_clamped[0] : hi_clamped[0],
        lo_clamped[1] : hi_clamped[1],
        lo_clamped[2] : hi_clamped[2],
    ]

    nz = np.argwhere(local_mask > 0)
    if nz.size == 0:
        return None, float("inf")  # type: ignore[return-value]

    # Shift to global coords, compute distance
    nz_global = nz + lo_clamped[None, :]
    dists = np.linalg.norm(nz_global - np.array([ic, jc, kc])[None, :], axis=1)
    best = int(np.argmin(dists))
    best_dist = float(dists[best])

    if best_dist > max_radius:
        return None, float("inf")  # type: ignore[return-value]

    best_ijk = tuple(int(x) for x in nz_global[best])
    return best_ijk, best_dist  # type: ignore[return-value]


# ── Pleural distance enforcement ─────────────────────────────────────────


def compute_pleural_distance_map(
    lung_mask: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
) -> np.ndarray:
    """Euclidean distance transform from the lung boundary into the interior.

    Returns an array of same shape as *lung_mask* where each voxel stores
    its distance (mm) to the nearest lung-boundary voxel.
    Voxels outside the lung have distance 0.
    """
    # Interior voxels = lung_mask > 0
    dist = ndimage.distance_transform_edt(
        lung_mask.astype(bool),
        sampling=voxel_spacing_mm,
    )
    return dist.astype(np.float32)


def enforce_pleural_margin(
    target_mask: np.ndarray,
    pleural_map: np.ndarray,
    ijk: Tuple[int, int, int],
    min_dist_mm: float,
    max_shift: int = 15,
) -> Tuple[Tuple[int, int, int], float]:
    """Move *ijk* inward if it violates the pleural safety margin.

    Searches among target-mask voxels within *max_shift* for the closest
    one that satisfies the margin.

    Returns (new_ijk, pleural_dist_mm).
    """
    i, j, k = ijk
    shape = target_mask.shape
    ic = max(0, min(i, shape[0] - 1))
    jc = max(0, min(j, shape[1] - 1))
    kc = max(0, min(k, shape[2] - 1))

    current_dist = float(pleural_map[ic, jc, kc])
    if current_dist >= min_dist_mm:
        return (ic, jc, kc), current_dist

    # Search nearby for a valid voxel
    lo = np.maximum(np.array([ic, jc, kc]) - max_shift, 0)
    hi = np.minimum(np.array([ic, jc, kc]) + max_shift + 1, shape)

    local_mask = target_mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    local_pleu = pleural_map[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

    # Candidate voxels: inside target mask AND satisfying margin
    candidates = np.argwhere((local_mask > 0) & (local_pleu >= min_dist_mm))
    if candidates.size == 0:
        # Relax: just pick deepest voxel inside target mask within region
        in_mask = np.argwhere(local_mask > 0)
        if in_mask.size == 0:
            return (ic, jc, kc), current_dist
        pleu_vals = local_pleu[in_mask[:, 0], in_mask[:, 1], in_mask[:, 2]]
        best = int(np.argmax(pleu_vals))
        g = tuple(int(x) for x in (in_mask[best] + lo))
        return g, float(pleu_vals[best])  # type: ignore[return-value]

    # Pick closest candidate to original point
    cand_global = candidates + lo[None, :]
    dists = np.linalg.norm(
        cand_global - np.array([ic, jc, kc])[None, :],
        axis=1,
    )
    best = int(np.argmin(dists))
    g = tuple(int(x) for x in cand_global[best])
    pdist = float(pleural_map[g[0], g[1], g[2]])
    return g, pdist  # type: ignore[return-value]


# ── Overlap / collision check ─────────────────────────────────────────────


def check_overlap(
    existing_mask: np.ndarray | None,
    donor_mask_host: np.ndarray,
    max_fraction: float = 0.05,
) -> Tuple[float, bool]:
    """Compute overlap fraction between a new donor mask and existing nodules.

    Returns (fraction, is_acceptable).
    """
    if existing_mask is None:
        return 0.0, True
    donor_vox = donor_mask_host > 0
    n_donor = int(donor_vox.sum())
    if n_donor == 0:
        return 0.0, True
    # Defensive: skip overlap check if shapes don't match (different CT series)
    if existing_mask.shape != donor_mask_host.shape:
        return 0.0, True
    overlap = int((donor_vox & (existing_mask > 0)).sum())
    frac = overlap / n_donor
    return frac, frac <= max_fraction


def resolve_collision(
    target_mask: np.ndarray,
    existing_mask: np.ndarray,
    ijk: Tuple[int, int, int],
    max_shift: int = 15,
    rng: np.random.Generator | None = None,
) -> Tuple[int, int, int] | None:
    """Attempt to shift *ijk* to avoid collision with existing nodules.

    Returns a new center or None if no collision-free spot found.
    """
    rng = rng or np.random.default_rng(42)
    shape = target_mask.shape
    i, j, k = ijk

    lo = np.maximum(np.array([i, j, k]) - max_shift, 0)
    hi = np.minimum(np.array([i, j, k]) + max_shift + 1, shape)

    local_target = target_mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    local_exist = existing_mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

    candidates = np.argwhere((local_target > 0) & (local_exist == 0))
    if candidates.size == 0:
        return None

    cand_global = candidates + lo[None, :]
    dists = np.linalg.norm(
        cand_global - np.array([i, j, k])[None, :],
        axis=1,
    )
    # Sort by distance, pick from top-5 with slight randomness for diversity
    order = np.argsort(dists)[:5]
    pick = rng.choice(order)
    return tuple(int(x) for x in cand_global[pick])  # type: ignore[return-value]


# ── Deterministic RNG seed ────────────────────────────────────────────────


def make_seed(
    trial_name: str,
    case_id: str,
    nodule_idx: int = 0,
    global_seed: int = 42,
) -> int:
    """Derive a deterministic seed from case identifiers.

    Uses SHA-256 truncated to 32 bits, mixed with the global seed.
    """
    h = hashlib.sha256(f"{trial_name}|{case_id}|{nodule_idx}|{global_seed}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# ── Top-level placement orchestrator ──────────────────────────────────────


def compute_placement(
    seg_data: np.ndarray,
    affine: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
    insertion_lobe: str,
    cc_pct: float,
    ml_pct: float,
    ap_pct: float,
    config: PlacementConfig | None = None,
    existing_combined_mask: np.ndarray | None = None,
    lobe_labels: dict | None = None,
    lobe_aliases: dict | None = None,
    lung_labels: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> PlacementResult:
    """Full placement pipeline: percentile → voxel → snap → pleural → collision.

    Parameters
    ----------
    seg_data : ndarray
        Host organ segmentation in canonical RAS+ space.
    affine : ndarray (4×4)
        Canonical RAS+ affine (voxel → world mm).
    voxel_spacing_mm : (sx, sy, sz)
        Voxel dimensions in mm along each RAS+ axis.
    insertion_lobe, cc_pct, ml_pct, ap_pct : float
        Target from the manifest.
    config : PlacementConfig
    existing_combined_mask : ndarray, optional
        Existing nodule mask in host space (for overlap checking).
    lobe_labels, lobe_aliases, lung_labels : dict / list, optional
    rng : numpy Generator, optional

    Returns
    -------
    PlacementResult
    """
    cfg = config or PlacementConfig()
    result = PlacementResult()

    # 1. Extract target mask
    target_mask, fallback = extract_target_mask(
        seg_data,
        insertion_lobe,
        lobe_labels,
        lobe_aliases,
        lung_labels,
    )
    if fallback == "empty":
        result.is_feasible = False
        result.reason = "No lung voxels found in organ segmentation."
        return result
    if fallback != "lobe":
        result.warnings.append(f"Lobe not found; using {fallback} fallback.")

    # 2. Percentile → initial voxel
    try:
        ijk0 = percentile_to_voxel(target_mask, cc_pct, ml_pct, ap_pct)
    except ValueError as e:
        result.is_feasible = False
        result.reason = f"Percentile mapping failed: {e}"
        return result

    # 3. Snap to target mask
    ijk, snap_dist = snap_to_mask(target_mask, ijk0, cfg.max_snap_radius_vox)
    if ijk is None:
        result.is_feasible = False
        result.reason = (
            f"Cannot snap to target mask within {cfg.max_snap_radius_vox} vox "
            f"(initial candidate {ijk0})."
        )
        return result
    if snap_dist > 0:
        result.snapped = True
        result.snap_distance_vox = snap_dist
        result.warnings.append(
            f"Snapped {snap_dist:.1f} vox from initial candidate {ijk0} to {ijk}."
        )

    # 4. Pleural margin
    if cfg.min_pleural_dist_mm > 0:
        # Build lung mask for pleural distance
        all_lung_labels = lung_labels or list((lobe_labels or DEFAULT_LOBE_LABELS).values())
        lung_mask = np.isin(seg_data, all_lung_labels).astype(np.uint8)
        if lung_mask.any():
            pleural_map = compute_pleural_distance_map(lung_mask, voxel_spacing_mm)
            ijk, pdist = enforce_pleural_margin(
                target_mask,
                pleural_map,
                ijk,
                cfg.min_pleural_dist_mm,
                max_shift=cfg.max_collision_shift_vox,
            )
            result.pleural_dist_mm = pdist
            if pdist < cfg.min_pleural_dist_mm:
                result.warnings.append(
                    f"Pleural margin {pdist:.1f} mm < {cfg.min_pleural_dist_mm} mm; "
                    "could not fully satisfy margin."
                )

    # 5. Collision check (deferred until donor mask is placed)
    # We record the center; collision is re-checked in the inserter after
    # the donor mask is pasted.

    result.center_ijk = ijk

    # Convert to world coords
    ijk_h = np.array([ijk[0], ijk[1], ijk[2], 1.0])
    world = affine @ ijk_h
    result.center_world_mm = tuple(float(x) for x in world[:3])  # type: ignore[assignment]

    return result
