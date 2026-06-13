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
Generate bounding-box and contour overlay images from plain slices + masks.

Given a plain CT slice PNG and the corresponding lesion mask, produces:
- bbox overlay: plain slice with a bright bounding-box rectangle around the lesion
- contour overlay: plain slice with the lesion boundary drawn as a contour
- bbox_contour overlay: both combined

All overlays keep the same spatial dimensions and orientation as the input
plain slice so model preprocessors (resize, crop, normalize) handle them
identically.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from itrialspace.evaluation.vlm_eval.slice_extractor import (
    _clean_artifact_slices,
    _to_radiological,
)

logger = logging.getLogger(__name__)

# Valid image conditions
VALID_CONDITIONS = ("plain", "bbox", "contour", "bbox_contour")

# Overlay colours (RGB)
BBOX_COLOR = (0, 255, 0)  # bright green
CONTOUR_COLOR = (255, 50, 50)  # red
BBOX_WIDTH = 2  # pixels
CONTOUR_WIDTH = 2  # pixels
# Padding (pixels) around bounding box
BBOX_PAD = 3


def _get_mask_slice_2d(
    mask_path: str,
    slice_index: int,
    slice_plane: str = "axial",
    nodule_label: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Extract a 2D binary lesion mask at the given slice, radiologically oriented.

    Returns a boolean 2D array matching the orientation of
    ``_extract_plane_slice`` output, or None if mask cannot be loaded
    or has no lesion voxels on that slice.

    Parameters
    ----------
    nodule_label : int or None
        Label value to threshold.  ``None`` (default) means *any non-zero*
        voxel — suitable for per-nodule binary masks.  An explicit integer
        (e.g. 23) selects only that label from a multi-label mask.
    """
    if not mask_path or not os.path.isfile(mask_path):
        return None
    try:
        mask_nii = nib.load(mask_path)
        mask_data = nib.as_closest_canonical(mask_nii).get_fdata(dtype=np.float32)
        _clean_artifact_slices(mask_data, nodule_label=nodule_label)
    except Exception as e:
        logger.warning("Cannot load mask %s: %s", mask_path, e)
        return None

    _axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = _axis_map.get(slice_plane, 2)
    max_idx = mask_data.shape[axis] - 1
    if slice_index < 0 or slice_index > max_idx:
        return None

    # Extract 2D mask at that index
    slicing: List = [slice(None)] * 3
    slicing[axis] = slice_index
    raw_slice = mask_data[tuple(slicing)]
    if nodule_label is None:
        binary = (raw_slice > 0).astype(np.uint8)
    else:
        binary = (raw_slice == nodule_label).astype(np.uint8)

    if binary.sum() == 0:
        return None

    # Orient to match the plain slice PNG
    if slice_plane == "axial":
        oriented = _to_radiological(binary)
    elif slice_plane == "coronal":
        oriented = np.flipud(binary.T)
    else:  # sagittal
        oriented = np.flipud(binary.T)

    return oriented.astype(bool)


def _mask_to_bbox(binary_mask: np.ndarray, pad: int = BBOX_PAD) -> Tuple[int, int, int, int]:
    """Compute (x0, y0, x1, y1) bounding box from a 2D boolean mask.

    Returns coordinates in (col_min, row_min, col_max, row_max) order
    suitable for ``ImageDraw.rectangle``.
    """
    rows, cols = np.nonzero(binary_mask)
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())
    h, w = binary_mask.shape
    x0 = max(c_min - pad, 0)
    y0 = max(r_min - pad, 0)
    x1 = min(c_max + pad, w - 1)
    y1 = min(r_max + pad, h - 1)
    return x0, y0, x1, y1


def _mask_to_contour(binary_mask: np.ndarray) -> np.ndarray:
    """Extract single-pixel boundary contour from a 2D boolean mask.

    Uses morphological erosion to find boundary pixels.
    Returns a boolean 2D array where True = contour pixel.
    """
    from scipy.ndimage import binary_erosion

    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    eroded = binary_erosion(binary_mask, iterations=1)
    contour = binary_mask & ~eroded
    return contour


def _plain_to_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is RGB (convert grayscale if needed)."""
    if img.mode == "L":
        return img.convert("RGB")
    if img.mode == "RGB":
        return img
    return img.convert("RGB")


def draw_bbox_overlay(
    plain_img: Image.Image,
    binary_mask: np.ndarray,
    color: Tuple[int, int, int] = BBOX_COLOR,
    width: int = BBOX_WIDTH,
) -> Image.Image:
    """Draw a bounding-box rectangle on a copy of the plain image."""
    rgb = _plain_to_rgb(plain_img).copy()
    x0, y0, x1, y1 = _mask_to_bbox(binary_mask)
    draw = ImageDraw.Draw(rgb)
    for i in range(width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)
    return rgb


def draw_contour_overlay(
    plain_img: Image.Image,
    binary_mask: np.ndarray,
    color: Tuple[int, int, int] = CONTOUR_COLOR,
    width: int = CONTOUR_WIDTH,
) -> Image.Image:
    """Draw the lesion contour on a copy of the plain image."""
    from scipy.ndimage import binary_dilation

    rgb = _plain_to_rgb(plain_img).copy()
    contour = _mask_to_contour(binary_mask)

    # Thicken contour if width > 1
    if width > 1:
        contour = binary_dilation(contour, iterations=width - 1)

    arr = np.array(rgb)
    arr[contour, 0] = color[0]
    arr[contour, 1] = color[1]
    arr[contour, 2] = color[2]
    return Image.fromarray(arr)


def draw_bbox_contour_overlay(
    plain_img: Image.Image,
    binary_mask: np.ndarray,
    bbox_color: Tuple[int, int, int] = BBOX_COLOR,
    contour_color: Tuple[int, int, int] = CONTOUR_COLOR,
) -> Image.Image:
    """Draw both bounding box and contour on a copy of the plain image."""
    img = draw_bbox_overlay(plain_img, binary_mask, color=bbox_color)
    img = draw_contour_overlay(img, binary_mask, color=contour_color)
    return img


def generate_overlays(
    plain_png_path: str,
    mask_path: str,
    slice_index: int,
    slice_plane: str = "axial",
    conditions: Tuple[str, ...] = ("bbox", "contour", "bbox_contour"),
    nodule_label: Optional[int] = None,
) -> dict:
    """Generate overlay PNGs for a single case.

    Parameters
    ----------
    plain_png_path : str
        Path to the existing plain slice PNG.
    mask_path : str
        Path to the lesion mask NIfTI (input_mask.nii.gz or per-nodule mask).
    slice_index : int
        Slice index along the extraction axis.
    slice_plane : str
        "axial", "coronal", or "sagittal".
    conditions : tuple of str
        Which overlays to generate ("bbox", "contour", "bbox_contour").
    nodule_label : int or None
        Label value to threshold in the mask.  ``None`` (default) means
        *any non-zero* voxel.  Use 23 for synthetic iTrialSpace masks.

    Returns
    -------
    dict
        Mapping condition name -> output path (or None if generation failed).
    """
    result = {}
    for cond in conditions:
        if cond not in ("bbox", "contour", "bbox_contour"):
            continue
        result[cond] = None

    if not os.path.isfile(plain_png_path):
        logger.warning("Plain PNG missing: %s", plain_png_path)
        return result

    binary_mask = _get_mask_slice_2d(mask_path, slice_index, slice_plane, nodule_label=nodule_label)
    if binary_mask is None:
        logger.warning(
            "No lesion mask on slice %d for %s — skipping overlays",
            slice_index,
            plain_png_path,
        )
        return result

    plain_img = Image.open(plain_png_path)
    parent_dir = os.path.dirname(plain_png_path) or "."
    filename = os.path.basename(plain_png_path)
    base_name, ext = os.path.splitext(filename)

    # Discover plain context slices (e.g. _z-1, _z+1) for multi-slice models.
    # We draw overlays on context slices too (if the lesion is visible there)
    # so MedGemma sees consistent visual guidance across all input slices.
    import re as _re

    ctx_pattern = _re.compile(_re.escape(base_name) + r"_z([+-]\d+)" + _re.escape(ext) + "$")
    context_siblings: List[tuple] = []  # (z_offset, full_path)
    for fname in os.listdir(parent_dir):
        m = ctx_pattern.match(fname)
        if m:
            z_offset = int(m.group(1))
            context_siblings.append((z_offset, os.path.join(parent_dir, fname)))

    for cond in conditions:
        if cond == "bbox":
            overlay_img = draw_bbox_overlay(plain_img, binary_mask)
        elif cond == "contour":
            overlay_img = draw_contour_overlay(plain_img, binary_mask)
        elif cond == "bbox_contour":
            overlay_img = draw_bbox_contour_overlay(plain_img, binary_mask)
        else:
            continue

        # Save into a condition sub-folder: {parent}/{condition}/{filename}
        cond_dir = os.path.join(parent_dir, cond)
        os.makedirs(cond_dir, exist_ok=True)
        out_path = os.path.join(cond_dir, filename)
        overlay_img.save(out_path)
        result[cond] = out_path

        # Generate overlays for context slices too.  If the lesion mask
        # has data on the neighbouring slice we draw the overlay; otherwise
        # copy the plain slice so the file still exists for discovery.
        for z_offset, ctx_path in context_siblings:
            ctx_filename = os.path.basename(ctx_path)
            ctx_out = os.path.join(cond_dir, ctx_filename)
            if os.path.exists(ctx_out):
                continue

            ctx_slice_idx = slice_index + z_offset
            ctx_mask = _get_mask_slice_2d(
                mask_path, ctx_slice_idx, slice_plane, nodule_label=nodule_label
            )

            if ctx_mask is not None:
                # Lesion visible on this slice — draw overlay
                ctx_plain = Image.open(ctx_path)
                if cond == "bbox":
                    ctx_overlay = draw_bbox_overlay(ctx_plain, ctx_mask)
                elif cond == "contour":
                    ctx_overlay = draw_contour_overlay(ctx_plain, ctx_mask)
                else:  # bbox_contour
                    ctx_overlay = draw_bbox_contour_overlay(ctx_plain, ctx_mask)
                ctx_overlay.save(ctx_out)
            else:
                # No lesion on this slice — copy the plain version
                import shutil

                shutil.copy2(ctx_path, ctx_out)

    return result


def generate_qc_grid(
    plain_path: str,
    bbox_path: Optional[str],
    contour_path: Optional[str],
    bbox_contour_path: Optional[str],
    output_path: str,
) -> Optional[str]:
    """Save a side-by-side QC grid of all conditions for one case.

    Returns the output path or None on failure.
    """
    paths = [
        ("plain", plain_path),
        ("bbox", bbox_path),
        ("contour", contour_path),
        ("bbox+contour", bbox_contour_path),
    ]
    images = []
    labels = []
    for label, p in paths:
        if p and os.path.isfile(p):
            images.append(Image.open(p).convert("RGB"))
            labels.append(label)

    if len(images) < 2:
        return None

    # Create horizontal grid
    widths = [img.width for img in images]
    max_h = max(img.height for img in images)
    label_h = 20
    total_w = sum(widths)
    grid = Image.new("RGB", (total_w, max_h + label_h), color=(0, 0, 0))

    draw = ImageDraw.Draw(grid)
    x_offset = 0
    for img, label in zip(images, labels):
        grid.paste(img, (x_offset, label_h))
        draw.text((x_offset + 4, 2), label, fill=(255, 255, 255))
        x_offset += img.width

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid.save(output_path)
    return output_path
