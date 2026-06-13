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
NIfTISlicer — extract 2-D slices from NIfTI CT/mask volumes.

Supports axial, coronal, sagittal views with optional overlay of
nodule masks or organ segmentations.  Returns numpy arrays that
can be converted to PNG by the API layer or rendered directly in
matplotlib / Streamlit.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


class SliceAxis(str, Enum):
    AXIAL = "axial"  # Z-axis
    CORONAL = "coronal"  # Y-axis
    SAGITTAL = "sagittal"  # X-axis


@dataclass
class SliceResult:
    """A rendered 2-D slice with metadata."""

    image: np.ndarray  # 2-D float array (HU or normalised)
    mask: Optional[np.ndarray]  # 2-D binary mask (or None)
    axis: SliceAxis
    slice_index: int
    shape_3d: tuple[int, int, int]
    window_center: float
    window_width: float
    spacing_mm: tuple[float, float, float]

    @property
    def windowed(self) -> np.ndarray:
        """Apply CT window and return uint8 [0, 255]."""
        lo = self.window_center - self.window_width / 2
        hi = self.window_center + self.window_width / 2
        img = np.clip(self.image, lo, hi)
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        return img

    def to_png_bytes(self, overlay_alpha: float = 0.35, labels: bool = True) -> bytes:
        """Render as PNG bytes with optional mask overlay and orientation labels."""
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")

        base = self.windowed
        rgb = np.stack([base, base, base], axis=-1)  # grayscale → RGB

        if self.mask is not None and self.mask.any():
            overlay = np.zeros_like(rgb)
            overlay[self.mask > 0] = [255, 50, 50]  # red overlay
            alpha = overlay_alpha
            blended = (rgb.astype(float) * (1 - alpha) + overlay.astype(float) * alpha).astype(
                np.uint8
            )
            # Only blend where mask is active
            rgb[self.mask > 0] = blended[self.mask > 0]

        img = Image.fromarray(rgb)

        # ── Orientation labels (radiology convention) ──────────────────
        if labels:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
                )
            except (IOError, OSError):
                font = ImageFont.load_default()

            h_img, w_img = rgb.shape[0], rgb.shape[1]
            label_color = (255, 255, 0)  # yellow
            shadow = (0, 0, 0)
            m = 4  # margin

            def _lbl(x, y, text, anchor=None):
                draw.text((x + 1, y + 1), text, fill=shadow, font=font, anchor=anchor)
                draw.text((x, y), text, fill=label_color, font=font, anchor=anchor)

            if self.axis == SliceAxis.AXIAL:
                _lbl(m, h_img // 2, "R", "lm")  # left  = patient-Right
                _lbl(w_img - m, h_img // 2, "L", "rm")  # right = patient-Left
                _lbl(w_img // 2, m, "A", "mt")  # top   = Anterior
                _lbl(w_img // 2, h_img - m, "P", "mb")  # bot   = Posterior
            elif self.axis == SliceAxis.CORONAL:
                _lbl(m, h_img // 2, "R", "lm")
                _lbl(w_img - m, h_img // 2, "L", "rm")
                _lbl(w_img // 2, m, "S", "mt")  # top   = Superior
                _lbl(w_img // 2, h_img - m, "I", "mb")  # bot   = Inferior
            elif self.axis == SliceAxis.SAGITTAL:
                _lbl(m, h_img // 2, "A", "lm")  # left  = Anterior
                _lbl(w_img - m, h_img // 2, "P", "rm")  # right = Posterior
                _lbl(w_img // 2, m, "S", "mt")
                _lbl(w_img // 2, h_img - m, "I", "mb")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


# Default CT window presets
WINDOW_PRESETS = {
    "lung": {"center": -600, "width": 1500},
    "mediastinum": {"center": 40, "width": 400},
    "bone": {"center": 400, "width": 1800},
    "soft_tissue": {"center": 50, "width": 350},
}


class NIfTISlicer:
    """
    Load NIfTI volumes and extract 2-D slices for the viewer.

    Uses a small LRU cache to avoid re-loading the same volume
    on sequential slice requests (e.g. scrolling through axial slices).
    """

    def __init__(self, cache_size: int = 4):
        if nib is None:
            raise ImportError("nibabel is required: pip install nibabel")
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size
        self._affine_cache: dict[str, np.ndarray] = {}

    # ── Volume loading ────────────────────────────────────────────────────────

    def _load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load a NIfTI volume, reorient to RAS+ canonical, return (data, affine)."""
        if path in self._cache:
            return self._cache[path], self._affine_cache[path]

        if not os.path.isfile(path):
            raise FileNotFoundError(f"NIfTI file not found: {path}")

        img = nib.load(path)
        # Reorient to RAS+ canonical so axes are always:
        #   axis-0 = R→L (X),  axis-1 = A→P (Y),  axis-2 = I→S (Z)
        canon = nib.as_closest_canonical(img)
        data = np.asarray(canon.dataobj, dtype=np.float32)

        # Squeeze extra dims (some files are 5-D)
        while data.ndim > 3:
            data = data[..., 0]

        affine = canon.affine

        # Cache management (LRU eviction)
        if len(self._cache) >= self._cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
            self._affine_cache.pop(evict, None)
        self._cache[path] = data
        self._affine_cache[path] = affine
        self._cache_order.append(path)

        return data, affine

    def volume_shape(self, path: str) -> tuple[int, int, int]:
        """Return the 3-D shape of a NIfTI volume."""
        data, _ = self._load(path)
        return data.shape[:3]  # type: ignore[return-value]

    def voxel_spacing(self, path: str) -> tuple[float, float, float]:
        """Return voxel spacing in mm from the affine."""
        _, affine = self._load(path)
        spacing = np.abs(np.diag(affine)[:3])
        return tuple(float(s) for s in spacing)  # type: ignore[return-value]

    # ── Slice extraction ──────────────────────────────────────────────────────

    def get_slice(
        self,
        ct_path: str,
        axis: SliceAxis = SliceAxis.AXIAL,
        index: Optional[int] = None,
        mask_path: Optional[str] = None,
        window: str = "lung",
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
    ) -> SliceResult:
        """Extract a 2-D slice from a CT volume.

        Args:
            ct_path: Path to CT NIfTI.
            axis: Slice orientation.
            index: Slice index. None = middle slice.
            mask_path: Optional path to mask NIfTI for overlay.
            window: CT window preset name.
            window_center: Override window center.
            window_width: Override window width.
        """
        data, affine = self._load(ct_path)
        shape = data.shape[:3]
        spacing = tuple(float(s) for s in np.abs(np.diag(affine)[:3]))

        # Determine axis dimension
        axis_dim = {"axial": 2, "coronal": 1, "sagittal": 0}[axis.value]
        max_idx = shape[axis_dim] - 1

        if index is None:
            index = max_idx // 2
        index = max(0, min(index, max_idx))

        # Extract slice and apply radiology display transform.
        # After RAS+ canonical load:
        #   axis-0 = R→L (X),  axis-1 = A→P (Y),  axis-2 = I→S (Z)
        if axis_dim == 2:  # Axial — fixed Z, shows (X, Y) plane
            img_slice = data[:, :, index]
        elif axis_dim == 1:  # Coronal — fixed Y, shows (X, Z) plane
            img_slice = data[:, index, :]
        else:  # Sagittal — fixed X, shows (Y, Z) plane
            img_slice = data[index, :, :]

        # Apply radiology display transform:
        #   Axial:    .T → (Y,X) then fliplr (R on left) then flipud (A at top)
        #   Coronal:  .T → (Z,X) then fliplr (R on left) then flipud (S at top)
        #   Sagittal: .T → (Z,Y) then fliplr (A on left) then flipud (S at top)
        img_slice = np.flipud(np.fliplr(img_slice.T))

        # Mask overlay
        mask_slice = None
        if mask_path and os.path.isfile(mask_path):
            mask_data, _ = self._load(mask_path)
            # Ensure compatible shape
            if mask_data.shape[:3] == shape:
                if axis_dim == 2:
                    mask_slice = mask_data[:, :, index]
                elif axis_dim == 1:
                    mask_slice = mask_data[:, index, :]
                else:
                    mask_slice = mask_data[index, :, :]
                # Same radiology display transform as CT
                mask_slice = np.flipud(np.fliplr(mask_slice.T))
                mask_slice = (mask_slice > 0).astype(np.uint8)

        # Window
        wc = (
            window_center
            if window_center is not None
            else WINDOW_PRESETS.get(window, WINDOW_PRESETS["lung"])["center"]
        )
        ww = (
            window_width
            if window_width is not None
            else WINDOW_PRESETS.get(window, WINDOW_PRESETS["lung"])["width"]
        )

        return SliceResult(
            image=img_slice,
            mask=mask_slice,
            axis=SliceAxis(axis),
            slice_index=index,
            shape_3d=shape,  # type: ignore[arg-type]
            window_center=wc,
            window_width=ww,
            spacing_mm=spacing,  # type: ignore[arg-type]
        )

    def get_nodule_slice(
        self,
        ct_path: str,
        mask_path: str,
        coord_x: float,
        coord_y: float,
        coord_z: float,
        axis: SliceAxis = SliceAxis.AXIAL,
        window: str = "lung",
    ) -> SliceResult:
        """Extract the slice passing through the nodule centre.

        Coordinates are in world/physical space (mm) as stored in the
        nodule profile CSVs.  We convert to voxel indices using the
        inverse affine.
        """
        data, affine = self._load(ct_path)
        inv = np.linalg.inv(affine)
        world = np.array([coord_x, coord_y, coord_z, 1.0])
        voxel = inv @ world

        # Map axis → voxel dimension
        axis_to_dim = {"axial": 2, "coronal": 1, "sagittal": 0}
        dim = axis_to_dim[axis.value]
        idx = int(round(voxel[dim]))

        return self.get_slice(
            ct_path,
            axis=axis,
            index=idx,
            mask_path=mask_path,
            window=window,
        )
