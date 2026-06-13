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
resample.py — Donor mask extraction, scaling/resampling into host
geometry, and binary cleanup.

The resampling pipeline:

1. Load donor nodule mask and canonicalise to RAS+.
2. Extract the largest connected component (in case of multi-label artifacts).
3. Build a SimpleITK similarity or affine transform that:
   a. Scales the donor by the manifest scale_factor (isotropic or per-axis).
   b. Translates the donor centre to the target insertion point in host space.
4. Resample the donor mask onto the host CT grid using nearest-neighbour.
5. Apply binary cleanup: threshold, remove small components, fill holes.
6. Return the mask as a numpy array matching host shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


@dataclass
class ResampleConfig:
    """Parameters governing resampling behaviour."""

    interpolation: str = "nearest"  # nearest | linear
    binary_threshold: float = 0.5
    min_component_size_vox: int = 3
    fill_holes: bool = True


# ── NIfTI helpers ─────────────────────────────────────────────────────────


def load_nifti_canonical(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Load a NIfTI file, canonicalise to RAS+.

    Returns (data, affine_4x4, voxel_spacing).
    """
    img = nib.load(path)
    img_can = nib.as_closest_canonical(img)
    data = np.asarray(img_can.dataobj)
    aff = img_can.affine.copy()
    spacing = tuple(float(s) for s in img_can.header.get_zooms()[:3])
    return data, aff, spacing


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: str,
    *,
    dtype: np.dtype | type = np.uint8,
) -> None:
    """Save a numpy array as a NIfTI file (optionally compressed)."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = nib.Nifti1Image(data.astype(dtype), affine)
    nib.save(img, path)


# ── Connected-component extraction ───────────────────────────────────────


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Extract the largest connected component from a binary mask.

    Returns a uint8 mask with only the largest component set to 1.
    """
    binary = (mask > 0).astype(np.uint8)
    labelled, n_labels = ndimage.label(binary)
    if n_labels <= 1:
        return binary

    sizes = ndimage.sum(binary, labelled, range(1, n_labels + 1))
    biggest = int(np.argmax(sizes)) + 1
    return (labelled == biggest).astype(np.uint8)


# ── Centre of mass ────────────────────────────────────────────────────────


def mask_centre_of_mass(mask: np.ndarray) -> Tuple[float, float, float]:
    """Centre of mass of nonzero voxels in voxel coordinates."""
    com = ndimage.center_of_mass(mask.astype(float))
    return (float(com[0]), float(com[1]), float(com[2]))


# ── Donor bounding-box crop ──────────────────────────────────────────────


def crop_to_bbox(
    mask: np.ndarray,
    padding: int = 2,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Crop a 3-D mask to its tight bounding box plus padding.

    Returns (cropped_mask, origin_offset_ijk).
    """
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return mask, (0, 0, 0)
    lo = np.maximum(nz.min(axis=0) - padding, 0)
    hi = np.minimum(nz.max(axis=0) + padding + 1, mask.shape)
    cropped = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    return cropped, tuple(int(x) for x in lo)  # type: ignore[return-value]


# ── Scipy-based resampling ───────────────────────────────────────────────


def resample_donor_to_host(
    donor_mask: np.ndarray,
    donor_affine: np.ndarray,
    donor_spacing: Tuple[float, float, float],
    host_shape: Tuple[int, int, int],
    host_affine: np.ndarray,
    host_spacing: Tuple[float, float, float],
    target_center_ijk: Tuple[int, int, int],
    scale_factor: float = 1.0,
    config: ResampleConfig | None = None,
) -> np.ndarray:
    """Resample a donor mask into host space.

    Algorithm:
    1. Extract largest connected component from donor mask.
    2. Crop to tight bounding box.
    3. Scale the cropped mask using scipy.ndimage.zoom (if scale_factor != 1).
    4. Adjust voxel sizes: zoom each axis by donor_spacing/host_spacing so
       the physical size is preserved when pasted into host voxel grid.
    5. Paste the (scaled) donor into host space centred at target_center_ijk.
    6. Binary cleanup.

    Returns a uint8 array of shape ``host_shape``.
    """
    cfg = config or ResampleConfig()

    # ── Extract largest component and crop ────────────────────────────────
    donor_lcc = largest_connected_component(donor_mask)
    cropped, _ = crop_to_bbox(donor_lcc, padding=1)

    # ── Compute effective zoom factors ────────────────────────────────────
    # Two reasons to zoom:
    #   a) scale_factor from the manifest (e.g. grow/shrink the nodule)
    #   b) spacing mismatch between donor and host voxel grids
    zoom_factors = tuple(scale_factor * (ds / hs) for ds, hs in zip(donor_spacing, host_spacing))

    needs_zoom = any(abs(z - 1.0) > 0.01 for z in zoom_factors)

    if needs_zoom:
        zoomed = ndimage.zoom(
            cropped.astype(np.float32),
            zoom_factors,
            order=1,
        )
        zoomed = (zoomed >= 0.5).astype(np.uint8)
    else:
        zoomed = cropped

    # ── Paste into host space ─────────────────────────────────────────────
    result = _paste_cropped(zoomed, host_shape, target_center_ijk)

    # ── Binary cleanup ────────────────────────────────────────────────────
    result = binary_cleanup(result, cfg)

    return result


def _paste_cropped(
    cropped: np.ndarray,
    host_shape: Tuple[int, int, int],
    center_ijk: Tuple[int, int, int],
) -> np.ndarray:
    """Paste a cropped donor mask into host space centred at center_ijk."""
    result = np.zeros(host_shape, dtype=np.uint8)
    d_shape = np.array(cropped.shape)
    center = np.array(center_ijk)

    # Compute destination region (donor centre-of-mass → target centre)
    donor_com = np.array(ndimage.center_of_mass((cropped > 0).astype(float)))
    offset = np.round(center - donor_com).astype(int)

    dst_lo = offset
    dst_hi = offset + d_shape

    # Clip to host bounds
    src_lo = np.maximum(-dst_lo, 0)
    src_hi = d_shape - np.maximum(dst_hi - np.array(host_shape), 0)
    dst_lo_c = np.maximum(dst_lo, 0)
    dst_hi_c = np.minimum(dst_hi, np.array(host_shape))

    if np.all(src_lo < src_hi) and np.all(dst_lo_c < dst_hi_c):
        result[
            dst_lo_c[0] : dst_hi_c[0],
            dst_lo_c[1] : dst_hi_c[1],
            dst_lo_c[2] : dst_hi_c[2],
        ] = cropped[
            src_lo[0] : src_hi[0],
            src_lo[1] : src_hi[1],
            src_lo[2] : src_hi[2],
        ]

    return result


# ── Binary mask cleanup ──────────────────────────────────────────────────


def binary_cleanup(mask: np.ndarray, config: ResampleConfig | None = None) -> np.ndarray:
    """Threshold, remove small components, optionally fill holes."""
    cfg = config or ResampleConfig()

    # Threshold
    binary = (mask >= cfg.binary_threshold).astype(np.uint8)

    # Remove small connected components
    if cfg.min_component_size_vox > 0:
        labelled, n = ndimage.label(binary)
        if n > 0:
            sizes = ndimage.sum(binary, labelled, range(1, n + 1))
            for idx, size in enumerate(sizes, start=1):
                if size < cfg.min_component_size_vox:
                    binary[labelled == idx] = 0

    # Fill holes
    if cfg.fill_holes:
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    return binary


# ── Direct paste (fallback / no-scale path) ──────────────────────────────


def paste_donor_direct(
    donor_mask: np.ndarray,
    host_shape: Tuple[int, int, int],
    target_center_ijk: Tuple[int, int, int],
) -> np.ndarray:
    """Paste a donor mask directly into host space at the given centre.

    Used when scale_factor == 1.0 and donor/host have identical spacing,
    so no resampling is needed — just a simple array copy with bounds
    clipping.

    Returns a uint8 array of shape ``host_shape``.
    """
    donor_lcc = largest_connected_component(donor_mask)
    donor_com = mask_centre_of_mass(donor_lcc)

    # Compute the offset to align donor COM with target centre
    offset = np.array(target_center_ijk) - np.array(donor_com)
    offset = np.round(offset).astype(int)

    result = np.zeros(host_shape, dtype=np.uint8)

    # Compute source and destination slicing
    nz = np.argwhere(donor_lcc > 0)
    if nz.size == 0:
        return result

    for idx in range(len(nz)):
        i, j, k = nz[idx] + offset
        if 0 <= i < host_shape[0] and 0 <= j < host_shape[1] and 0 <= k < host_shape[2]:
            result[i, j, k] = 1

    return result


# ── Efficient paste using array slicing ──────────────────────────────────


def paste_donor_array(
    donor_mask: np.ndarray,
    host_shape: Tuple[int, int, int],
    target_center_ijk: Tuple[int, int, int],
) -> np.ndarray:
    """Faster paste using array slicing rather than per-voxel iteration.

    Centres the donor mask at target_center_ijk in host space.
    Returns a uint8 array of shape ``host_shape``.
    """
    donor_lcc = largest_connected_component(donor_mask)
    donor_com = np.array(mask_centre_of_mass(donor_lcc))
    target = np.array(target_center_ijk, dtype=float)

    # Shift = how much to move donor indices to align COM with target
    shift = np.round(target - donor_com).astype(int)

    # Compute valid overlap region
    d_shape = np.array(donor_lcc.shape)
    h_shape = np.array(host_shape)

    # Donor index ranges in host space
    dst_lo = shift
    dst_hi = shift + d_shape

    # Clip to host bounds
    src_lo = np.maximum(-dst_lo, 0)
    src_hi = d_shape - np.maximum(dst_hi - h_shape, 0)
    dst_lo_clipped = np.maximum(dst_lo, 0)
    dst_hi_clipped = np.minimum(dst_hi, h_shape)

    result = np.zeros(host_shape, dtype=np.uint8)

    # Check for valid overlap
    if np.all(src_lo < src_hi) and np.all(dst_lo_clipped < dst_hi_clipped):
        donor_crop = donor_lcc[src_lo[0] : src_hi[0], src_lo[1] : src_hi[1], src_lo[2] : src_hi[2]]
        result[
            dst_lo_clipped[0] : dst_hi_clipped[0],
            dst_lo_clipped[1] : dst_hi_clipped[1],
            dst_lo_clipped[2] : dst_hi_clipped[2],
        ] = donor_crop

    return result
