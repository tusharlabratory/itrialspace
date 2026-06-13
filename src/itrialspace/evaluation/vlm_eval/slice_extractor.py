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
Extract 2D slices from 3D CT volumes for VLM evaluation.

Slice selection priority:
1. Co-located input_mask.nii.gz label-23 centroid (same geometry as synthetic CT)
2. Inserted mask (_mask.nii.gz) label-23 centroid (needs resampling check)
3. Skip case (no arbitrary middle-slice fallback)

Supports multiple slice planes (axial, coronal, sagittal), CT windows, and
output formats via PreprocessProfile.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from itrialspace.evaluation.vlm_eval.preprocess_profiles import (
    PreprocessProfile,
)

logger = logging.getLogger(__name__)

# Nodule label in the iTrialSpace combined organ+nodule masks
NODULE_LABEL = 23

# Lung CT window: centre -600 HU, width 1500 HU → [-1350, 150]
_DEFAULT_WINDOW_CENTER = -600.0
_DEFAULT_WINDOW_WIDTH = 1500.0


def _to_radiological(arr_2d: np.ndarray) -> np.ndarray:
    """Orient a raw axial slice (X, Y) → radiological display (rows=A→P, cols=R→L).

    Assumes the NIfTI data is in RAS+ orientation (axis-0→R, axis-1→A).
    After .T the image rows follow axis-1 (A) and cols follow axis-0 (R).
    flipud reverses rows so top = Anterior; fliplr reverses cols so left = Right.
    """
    return np.fliplr(np.flipud(arr_2d.T))


# Number of slices on each side of the centre for the mini-stack
_STACK_MARGIN = 1


def _extract_plane_slice(
    volume: np.ndarray,
    plane: str,
    index: int,
) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along the given plane.

    Parameters
    ----------
    volume : ndarray (X, Y, Z) in RAS+ NIfTI convention
    plane : "axial" | "coronal" | "sagittal"
    index : slice index along that axis

    Returns
    -------
    2D ndarray (already oriented for radiological display)
    """
    if plane == "axial":
        return _to_radiological(volume[:, :, index])
    elif plane == "coronal":
        # coronal = anterior–posterior axis (axis 1 in RAS)
        raw = volume[:, index, :]  # (X, Z)
        return np.flipud(raw.T)  # rows=Z (sup→inf), cols=X (R→L)
    elif plane == "sagittal":
        # sagittal = right–left axis (axis 0 in RAS)
        raw = volume[index, :, :]  # (Y, Z)
        return np.flipud(raw.T)  # rows=Z (sup→inf), cols=Y (A→P)
    else:
        raise ValueError(f"Unknown slice plane: '{plane}'. Use axial/coronal/sagittal.")


@dataclass
class SliceInfo:
    """Metadata about how a slice was selected."""

    center_z: int
    method: str  # "input_mask" | "inserted_mask" | "skipped"
    mask_path: str = ""
    n_nodule_voxels: int = 0


def _apply_ct_window(
    data: np.ndarray,
    center: float = _DEFAULT_WINDOW_CENTER,
    width: float = _DEFAULT_WINDOW_WIDTH,
) -> np.ndarray:
    """Apply CT windowing and normalise to [0, 255] uint8."""
    lower = center - width / 2.0
    upper = center + width / 2.0
    data = np.clip(data, lower, upper)
    data = ((data - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return data


# MedGemma 1.5 3-channel CT windowing protocol
# Source: https://github.com/Google-Health/medgemma/blob/main/notebooks/high_dimensional_ct_hugging_face.ipynb
_MEDGEMMA_WINDOW_CLIPS = [
    (-1024, 1024),  # R: wide
    (-135, 215),  # G: soft tissue
    (0, 80),  # B: brain
]


def _apply_medgemma_ct_window(data: np.ndarray) -> np.ndarray:
    """Apply MedGemma 1.5 3-channel CT windowing.

    Each RGB channel encodes a different HU window:
      R = wide (-1024, 1024)
      G = soft tissue (-135, 215)
      B = brain (0, 80)

    Returns (H, W, 3) uint8 array.
    """
    channels = []
    for lo, hi in _MEDGEMMA_WINDOW_CLIPS:
        ch = np.clip(data, lo, hi).astype(np.float32)
        ch = (ch - lo) / (hi - lo) * 255.0
        channels.append(np.round(ch).astype(np.uint8))
    return np.stack(channels, axis=-1)


def _clean_artifact_slices(
    mask_data: np.ndarray,
    threshold: float = 0.5,
    nodule_label: Optional[int] = None,
) -> np.ndarray:
    """Zero out artifact slices where the mask fills > *threshold* of the plane.

    Some real CT nodule masks (e.g. NSCLCR, IMDCT) have a last-slice artifact
    where the entire z-plane is filled with non-zero values.  This corrupts
    centroid calculations and overlay generation.

    When *nodule_label* is ``None`` (binary-mask mode), any axial slice whose
    total non-zero fraction exceeds *threshold* is zeroed out entirely.

    When *nodule_label* is set (multi-label segmentation mask, e.g. 23 for
    nodule), only that label's fill-fraction is checked, and only that label
    is zeroed on artifact slices.  This prevents dense multi-organ masks from
    accidentally wiping small nodule labels.

    Parameters
    ----------
    mask_data : ndarray (X, Y, Z)
        3-D mask volume (modified **in-place** and returned).
    threshold : float
        Fraction of the plane area above which a slice is considered artifact.
    nodule_label : int or None
        If set, restrict artifact detection and cleaning to this label only.

    Returns
    -------
    ndarray
        The cleaned mask volume (same object, modified in-place).
    """
    plane_area = mask_data.shape[0] * mask_data.shape[1]
    for z in range(mask_data.shape[2]):
        if nodule_label is None:
            # Binary mask: check all non-zero voxels, zero entire slice
            if np.count_nonzero(mask_data[:, :, z]) > plane_area * threshold:
                logger.debug("Zeroing artifact slice z=%d (>%.0f%% filled)", z, threshold * 100)
                mask_data[:, :, z] = 0
        else:
            # Multi-label mask: check only the target label
            label_count = np.count_nonzero(mask_data[:, :, z] == nodule_label)
            if label_count > plane_area * threshold:
                logger.debug(
                    "Zeroing label %d on artifact slice z=%d (>%.0f%% filled)",
                    nodule_label,
                    z,
                    threshold * 100,
                )
                mask_data[:, :, z][mask_data[:, :, z] == nodule_label] = 0
    return mask_data


def _nodule_centroid_from_mask(
    mask_path: str,
    nodule_label: Optional[int] = None,
) -> Optional[Tuple[int, int, int]]:
    """Return (x, y, z) voxel centroid of the nodule label in a mask, or None.

    Parameters
    ----------
    mask_path : str
        Path to the mask NIfTI file.
    nodule_label : int or None
        Label value to threshold.  ``None`` means *any non-zero* voxel
        (useful for per-nodule binary masks from real CT datasets).
        An explicit integer (e.g. 23) selects only that label.
    """
    if not mask_path or not os.path.isfile(mask_path):
        return None
    try:
        mask_nii = nib.load(mask_path)
        mask_data = nib.as_closest_canonical(mask_nii).get_fdata(dtype=np.float32)
        # Remove artifact slices (e.g. full-plane fills) before centroid calc
        _clean_artifact_slices(mask_data, nodule_label=nodule_label)
        if nodule_label is None:
            voxels = np.argwhere(mask_data > 0)
        else:
            voxels = np.argwhere(mask_data == nodule_label)
        if len(voxels) == 0:
            return None
        cx, cy, cz = voxels.mean(axis=0)
        return int(round(cx)), int(round(cy)), int(round(cz))
    except Exception as e:
        logger.warning("Failed loading mask %s: %s", mask_path, e)
        return None


def _resolve_input_mask(ct_path: str) -> Optional[str]:
    """Find the input_mask.nii.gz co-located with the synthetic CT.

    This mask is in the same resampled geometry as the synthetic CT
    (typically 512x512x256) and contains label 23 for the nodule.
    """
    ct_dir = os.path.dirname(ct_path)
    input_mask = os.path.join(ct_dir, "input_mask.nii.gz")
    if os.path.isfile(input_mask):
        return input_mask
    return None


def extract_slice(
    ct_path: str,
    output_png: str,
    input_mask_path: Optional[str] = None,
    inserted_mask_path: Optional[str] = None,
    window_center: float = _DEFAULT_WINDOW_CENTER,
    window_width: float = _DEFAULT_WINDOW_WIDTH,
    save_overlay: bool = False,
    profile: Optional[PreprocessProfile] = None,
    nodule_label: Optional[int] = NODULE_LABEL,
) -> Optional[SliceInfo]:
    """Extract a lesion-centred slice from a CT volume and save.

    Parameters
    ----------
    ct_path : str
        Path to the 3D synthetic CT NIfTI file.
    output_png : str
        Where to save the output (PNG or .npy depending on profile).
    input_mask_path : str, optional
        Path to input_mask.nii.gz (co-located with synthetic CT, same geometry).
        This is the preferred source for nodule localisation.
    inserted_mask_path : str, optional
        Path to the inserted mask (_mask.nii.gz). Used as fallback only if
        it has the same shape as the CT.
    window_center, window_width : float
        CT windowing parameters. Overridden by *profile* if provided.
    save_overlay : bool
        If True, save a second PNG with the nodule mask overlaid.
        Overridden by *profile* if provided.
    profile : PreprocessProfile, optional
        Full preprocessing profile. When given, its ``window_center``,
        ``window_width``, ``save_overlay``, ``slice_plane``,
        ``num_context_slices``, and ``output_format`` take precedence over
        the individual keyword arguments.

    Returns
    -------
    SliceInfo or None
        Metadata about the extracted slice, or None if the case was skipped.
    """
    # Resolve effective parameters from profile (if any)
    if profile is not None:
        window_center = profile.window_center
        window_width = profile.window_width
        save_overlay = profile.save_overlay
        slice_plane = profile.slice_plane
        num_context = profile.num_context_slices
        output_format = profile.output_format
    else:
        slice_plane = "axial"
        num_context = _STACK_MARGIN
        output_format = "png_gray"
    nii = nib.load(ct_path)
    # Reorient to RAS+ canonical so _to_radiological() / _extract_plane_slice()
    # produce correct radiological display regardless of native NIfTI orientation.
    # Matches retriever/slicer.py convention.
    nii = nib.as_closest_canonical(nii)
    ct_data = nii.get_fdata(dtype=np.float32)
    ct_shape = ct_data.shape

    # Determine which axis corresponds to the requested slice plane
    _axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    slice_axis = _axis_map[slice_plane]
    max_idx = ct_shape[slice_axis] - 1

    # ── Strategy 1: input_mask.nii.gz (co-located, same geometry) ────────
    if input_mask_path is None:
        input_mask_path = _resolve_input_mask(ct_path)

    centroid = _nodule_centroid_from_mask(input_mask_path, nodule_label=nodule_label)
    if centroid is not None:
        center_idx = int(np.clip(centroid[slice_axis], 0, max_idx))
        info = SliceInfo(
            center_z=center_idx,
            method="input_mask",
            mask_path=input_mask_path,
            n_nodule_voxels=_count_nodule_voxels(input_mask_path, nodule_label=nodule_label),
        )
        logger.info(
            "Case %s: %s slice idx=%d via input_mask centroid (%d nodule voxels)",
            os.path.basename(output_png),
            slice_plane,
            center_idx,
            info.n_nodule_voxels,
        )
        _save_slice_stack(
            ct_data,
            center_idx,
            output_png,
            window_center,
            window_width,
            slice_plane,
            num_context,
            output_format,
        )

        if save_overlay and input_mask_path:
            _save_overlay(
                ct_data,
                input_mask_path,
                center_idx,
                output_png,
                window_center,
                window_width,
                slice_plane,
                nodule_label=nodule_label,
            )

        return info

    # ── Strategy 2: inserted mask (only if same shape as CT) ─────────────
    if inserted_mask_path and os.path.isfile(inserted_mask_path):
        try:
            imask_nii = nib.load(inserted_mask_path)
            if imask_nii.shape == ct_shape:
                centroid = _nodule_centroid_from_mask(inserted_mask_path, nodule_label=nodule_label)
                if centroid is not None:
                    center_idx = int(np.clip(centroid[slice_axis], 0, max_idx))
                    info = SliceInfo(
                        center_z=center_idx,
                        method="inserted_mask",
                        mask_path=inserted_mask_path,
                        n_nodule_voxels=_count_nodule_voxels(
                            inserted_mask_path, nodule_label=nodule_label
                        ),
                    )
                    logger.info(
                        "Case %s: %s slice idx=%d via inserted_mask (same shape)",
                        os.path.basename(output_png),
                        slice_plane,
                        center_idx,
                    )
                    _save_slice_stack(
                        ct_data,
                        center_idx,
                        output_png,
                        window_center,
                        window_width,
                        slice_plane,
                        num_context,
                        output_format,
                    )
                    if save_overlay:
                        _save_overlay(
                            ct_data,
                            inserted_mask_path,
                            center_idx,
                            output_png,
                            window_center,
                            window_width,
                            slice_plane,
                            nodule_label=nodule_label,
                        )
                    return info
            else:
                logger.debug(
                    "Case %s: inserted mask shape %s != CT shape %s, skipping",
                    os.path.basename(output_png),
                    imask_nii.shape,
                    ct_shape,
                )
        except Exception as e:
            logger.warning("Failed loading inserted mask %s: %s", inserted_mask_path, e)

    # ── No reliable localisation → skip ──────────────────────────────────
    logger.warning(
        "SKIP %s: no nodule localisation source available (no input_mask, "
        "no compatible inserted_mask)",
        os.path.basename(output_png),
    )
    return None


def _count_nodule_voxels(
    mask_path: str,
    nodule_label: Optional[int] = None,
) -> int:
    """Count nodule voxels without keeping full array in memory."""
    try:
        data = nib.load(mask_path).get_fdata(dtype=np.float32)
        if nodule_label is None:
            return int(np.sum(data > 0))
        return int(np.sum(data == nodule_label))
    except Exception:
        return 0


def _save_slice_stack(
    ct_data: np.ndarray,
    center_idx: int,
    output_png: str,
    window_center: float,
    window_width: float,
    slice_plane: str = "axial",
    num_context: int = _STACK_MARGIN,
    output_format: str = "png_gray",
) -> None:
    """Save the centre slice (+ optional neighbours) using the requested format."""
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    _axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    max_idx = ct_data.shape[_axis_map[slice_plane]] - 1

    # Centre slice
    oriented = _extract_plane_slice(ct_data, slice_plane, center_idx)

    _save_single_slice(oriented, output_png, window_center, window_width, output_format)

    # Context slices (saved alongside for optional multi-slice analysis)
    if num_context > 0:
        base, ext = os.path.splitext(output_png)
        for offset in range(-num_context, num_context + 1):
            if offset == 0:
                continue
            idx = center_idx + offset
            if 0 <= idx <= max_idx:
                nb_oriented = _extract_plane_slice(ct_data, slice_plane, idx)
                suffix_ext = ".npy" if output_format == "npy" else ext
                nb_path = f"{base}_z{offset:+d}{suffix_ext}"
                _save_single_slice(nb_oriented, nb_path, window_center, window_width, output_format)


def _save_single_slice(
    oriented: np.ndarray,
    out_path: str,
    window_center: float,
    window_width: float,
    output_format: str,
) -> None:
    """Window and save one 2D slice in the requested format."""
    if output_format == "png_medgemma_rgb":
        rgb = _apply_medgemma_ct_window(oriented)
        Image.fromarray(rgb, mode="RGB").save(out_path)
    elif output_format == "npy":
        img_arr = _apply_ct_window(oriented, window_center, window_width)
        npy_path = os.path.splitext(out_path)[0] + ".npy"
        np.save(npy_path, img_arr)
    elif output_format == "png_rgb":
        img_arr = _apply_ct_window(oriented, window_center, window_width)
        rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(out_path)
    else:  # png_gray (default)
        img_arr = _apply_ct_window(oriented, window_center, window_width)
        Image.fromarray(img_arr, mode="L").save(out_path)


def _save_overlay(
    ct_data: np.ndarray,
    mask_path: str,
    center_idx: int,
    output_png: str,
    window_center: float,
    window_width: float,
    slice_plane: str = "axial",
    nodule_label: Optional[int] = NODULE_LABEL,
) -> None:
    """Save an overlay PNG with the nodule mask highlighted for QC."""
    try:
        mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)

        # Extract mask slice for the same plane
        _axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
        axis = _axis_map[slice_plane]
        slicing = [slice(None)] * 3
        slicing[axis] = center_idx
        raw_slice = mask_data[tuple(slicing)]
        if nodule_label is None:
            mask_slice = (raw_slice > 0).astype(np.uint8)
        else:
            mask_slice = (raw_slice == nodule_label).astype(np.uint8)

        ct_slice_raw = ct_data[tuple(slicing)]
        ct_arr = _apply_ct_window(ct_slice_raw, window_center, window_width)
        rgb = np.stack([ct_arr, ct_arr, ct_arr], axis=-1)

        # Red overlay where mask == 1, 50% opacity
        overlay_mask = mask_slice > 0
        rgb[overlay_mask, 0] = np.clip(rgb[overlay_mask, 0].astype(int) + 100, 0, 255).astype(
            np.uint8
        )
        rgb[overlay_mask, 1] = (rgb[overlay_mask, 1] * 0.5).astype(np.uint8)
        rgb[overlay_mask, 2] = (rgb[overlay_mask, 2] * 0.5).astype(np.uint8)

        base, ext = os.path.splitext(output_png)
        overlay_path = f"{base}_overlay{ext}"

        # Apply same radiological orientation to RGB overlay
        if slice_plane == "axial":
            oriented = np.fliplr(np.flipud(rgb.transpose(1, 0, 2)))
        elif slice_plane == "coronal":
            oriented = np.flipud(rgb.transpose(1, 0, 2))
        else:  # sagittal
            oriented = np.flipud(rgb.transpose(1, 0, 2))

        img = Image.fromarray(oriented, mode="RGB")
        img.save(overlay_path)
    except Exception as e:
        logger.warning("Failed to save overlay for %s: %s", output_png, e)


# ── Public helper: extract a slice at an arbitrary z-index ───────────────────


def extract_slice_at_z(
    ct_path: str,
    z_index: int,
    output_png: str,
    profile: Optional["PreprocessProfile"] = None,
    window_center: float = _DEFAULT_WINDOW_CENTER,
    window_width: float = _DEFAULT_WINDOW_WIDTH,
) -> None:
    """Extract and save a 2D slice at a specific z-index (no mask needed).

    Uses the same windowing, orientation, and output format as
    :func:`extract_slice`, but requires no mask—caller supplies the z-index
    directly.  Intended for generating nodule-free negative slices.

    Parameters
    ----------
    ct_path : str
        Path to the 3D CT NIfTI file.
    z_index : int
        Axial slice index (after RAS+ reorientation).
    output_png : str
        Where to save the output image.
    profile : PreprocessProfile, optional
        Preprocessing profile (controls windowing, output format, etc.).
    window_center, window_width : float
        CT windowing. Overridden by *profile* if provided.
    """
    if profile is not None:
        window_center = profile.window_center
        window_width = profile.window_width
        slice_plane = profile.slice_plane
        num_context = profile.num_context_slices
        output_format = profile.output_format
    else:
        slice_plane = "axial"
        num_context = _STACK_MARGIN
        output_format = "png_gray"

    nii = nib.load(ct_path)
    nii = nib.as_closest_canonical(nii)
    ct_data = nii.get_fdata(dtype=np.float32)

    _axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    max_idx = ct_data.shape[_axis_map[slice_plane]] - 1
    z_index = int(np.clip(z_index, 0, max_idx))

    _save_slice_stack(
        ct_data,
        z_index,
        output_png,
        window_center,
        window_width,
        slice_plane,
        num_context,
        output_format,
    )
