#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
run_itrialspace_to_ct.py — Integration pipeline: iTrialSpace masks -> NodMAISI -> synthetic CTs.

Reads an iTrialSpace CohortManifest (CSV/JSON) or a directory of inserted masks,
prepares NodMAISI-compatible inputs, runs ControlNet-conditioned CT synthesis,
and produces synthetic CTs with QC outputs.

Usage:
    # From manifest (reads host_ct_path, host_organ_seg_path, inserted mask from audit)
    python tools/run_itrialspace_to_ct.py \\
        --manifest /path/to/manifest.csv \\
        --audit /path/to/audit.json \\
        --config tools/integration_config.yaml \\
        --outdir $ITRIALSPACE_DATA_DIR/.../generated_cts/mode1 \\
        --jobs 1

    # From a directory of inserted masks
    python tools/run_itrialspace_to_ct.py \\
        --mask-root $ITRIALSPACE_DATA_DIR/.../inserted_masks/mode1_controlled_prevalence \\
        --config tools/integration_config.yaml \\
        --outdir $ITRIALSPACE_DATA_DIR/.../generated_cts/mode1

    # Dry-run (validate inputs, skip inference)
    python tools/run_itrialspace_to_ct.py \\
        --audit /path/to/audit.json \\
        --config tools/integration_config.yaml \\
        --outdir /tmp/test --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import nibabel as nib
import numpy as np

logger = logging.getLogger("itrialspace_to_ct")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CaseSpec:
    """Specification for a single case to synthesize."""

    case_id: str
    mask_path: str  # path to combined inserted mask (label map)
    host_ct_path: str = ""  # optional: for QC overlay / linking
    dim: tuple = (512, 512, 256)  # (X, Y, Z) volume dimensions
    spacing: tuple = (0.7, 0.7, 1.25)  # voxel spacing in mm
    subdir: str = ""  # subdirectory bucket (e.g., size_curve_10-15mm)
    host_patient_id: str = ""
    host_dataset: str = ""
    donor_annotation_id: str = ""
    trial_name: str = ""


@dataclass
class CaseResult:
    """Result record for a single case."""

    case_id: str = ""
    status: str = "pending"  # success | failed | skipped | dry_run
    reason: str = ""
    # Paths
    input_mask_path: str = ""
    host_ct_path: str = ""
    synthetic_ct_path: str = ""
    qc_png_paths: list = field(default_factory=list)
    # Geometry
    original_dim: tuple = ()
    original_spacing: tuple = ()
    nodmaisi_dim: tuple = ()
    nodmaisi_spacing: tuple = ()
    resized: bool = False
    # Timing
    prep_sec: float = 0.0
    inference_sec: float = 0.0
    qc_sec: float = 0.0
    total_sec: float = 0.0
    # Sanity checks
    shape_match: bool = False
    has_nodule_label: bool = False
    ct_min_hu: float = 0.0
    ct_max_hu: float = 0.0
    has_nan: bool = False


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML integration config, expanding ${VAR} placeholders portably."""
    from itrialspace.config import settings

    return settings.load_yaml(config_path)


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


def _mask_stem(path: str) -> str:
    """Derive a case ID from a mask filename.

    Strips ``_mask.nii.gz`` or ``.nii.gz`` suffix to produce a
    human-readable, globally unique identifier.
    """
    name = os.path.basename(path)
    for suffix in ("_mask.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _build_host_ct_index(manifest_root: str) -> tuple[dict[str, str], dict[str, str]]:
    """Build host CT lookup indices from manifest JSON/CSVs.

    Returns two dicts:
      - **pid_index**: ``host_patient_id → host_ct_path``
      - **case_index**: ``"<trial_name>/<case_id_int>" → host_ct_path``

    The *pid_index* handles most datasets.  The *case_index* is a fallback
    for LUNA16 whose mask filenames use truncated hashes that don't match
    the full DICOM UID stored as ``host_patient_id`` in the manifest.
    """
    import glob as _glob

    pid_index: dict[str, str] = {}
    case_index: dict[str, str] = {}

    def _ingest(records: list, trial_name: str):
        for rec in records:
            pid = str(rec.get("host_patient_id", ""))
            hcp = str(rec.get("host_ct_path", ""))
            cid = rec.get("case_id")
            if not hcp:
                continue
            if pid:
                pid_index[pid] = hcp
            if trial_name and cid is not None:
                case_index[f"{trial_name}/{int(cid)}"] = hcp

    # JSON manifests (structured, fast)
    for jp in sorted(_glob.glob(os.path.join(manifest_root, "**", "*.json"), recursive=True)):
        try:
            with open(jp) as f:
                data = json.load(f)
            cases = data.get("cases") or data.get("records") or []
            tname = data.get("trial_name", "")
            # If no trial_name field, infer from filename stem
            if not tname:
                tname = os.path.basename(jp).replace(".json", "")
            _ingest(cases, tname)
        except Exception:
            continue

    # CSV manifests (supplement)
    for cp in sorted(_glob.glob(os.path.join(manifest_root, "**", "*.csv"), recursive=True)):
        try:
            import pandas as pd

            df = pd.read_csv(cp)
            tname = ""
            if "trial_name" in df.columns:
                tname = str(df["trial_name"].iloc[0])
            if not tname:
                tname = os.path.basename(cp).replace(".csv", "")
            for _, row in df.iterrows():
                rec = row.to_dict()
                rec = {k: (str(v) if pd.notna(v) else "") for k, v in rec.items()}
                pid = rec.get("host_patient_id", "")
                hcp = rec.get("host_ct_path", "")
                cid = rec.get("case_id")
                if not hcp:
                    continue
                if pid and pid not in pid_index:
                    pid_index[pid] = hcp
                if tname and cid:
                    key = f"{tname}/{int(float(cid))}"
                    if key not in case_index:
                        case_index[key] = hcp
        except Exception:
            continue

    logger.info(
        "Host-CT index: %d by patient-id, %d by trial/case from %s",
        len(pid_index),
        len(case_index),
        manifest_root,
    )
    return pid_index, case_index


def _parse_host_id(mask_stem: str) -> str:
    """Extract the host patient ID from an iTrialSpace mask filename stem.

    Expects the ``--host-<ID>--`` convention, e.g.
    ``iTS--size_curve_10-15mm--C0000--host-DLCS_1092--src-DLCS24--nod-…``
    returns ``"DLCS_1092"``.  Returns ``""`` if the pattern isn't found.
    """
    import re

    m = re.search(r"--host-(.+?)--", mask_stem)
    return m.group(1) if m else ""


def _parse_trial_case(mask_stem: str) -> tuple[str, str]:
    """Extract (trial_name, integer_case_id) from an iTrialSpace mask filename.

    ``iTS--size_curve_10-15mm--C0036--host-…`` → ``("size_curve_10-15mm", "36")``.
    Returns ``("", "")`` if either field isn't found.
    """
    import re

    parts = mask_stem.split("--")
    # parts: ["iTS", "size_curve_10-15mm", "C0036", "host-…", …]
    trial = parts[1] if len(parts) > 1 else ""
    cid_match = re.search(r"--C(\d+)--", mask_stem)
    cid = str(int(cid_match.group(1))) if cid_match else ""
    return trial, cid


def cases_from_audit(audit_path: str, cfg: dict) -> list[CaseSpec]:
    """Build CaseSpecs from an iTrialSpace audit.json.

    If *audit_path* is a file, reads it directly.
    If *audit_path* does not exist but its parent directory contains
    sub-directory audit.json files, merge them all.

    Each case_id is the mask filename stem (without ``_mask.nii.gz``),
    which is globally unique and self-documenting.
    """
    audit_files: list[str] = []
    root_dir: str = ""

    if os.path.isfile(audit_path):
        audit_files.append(audit_path)
        root_dir = os.path.dirname(audit_path)
    else:
        # Try to discover audit.json files in subdirectories
        parent = os.path.dirname(audit_path)
        if os.path.isdir(parent):
            import glob as _glob

            audit_files = sorted(
                _glob.glob(os.path.join(parent, "**", "audit.json"), recursive=True)
            )
            root_dir = parent

    if not audit_files:
        logger.error("No audit.json found at %s or in subdirectories", audit_path)
        return []

    seen: dict[str, CaseSpec] = {}
    for af in audit_files:
        with open(af) as f:
            audit = json.load(f)

        # Derive subdirectory name from the audit file's parent
        af_dir = os.path.dirname(af)
        subdir = os.path.basename(af_dir) if af_dir != root_dir else ""

        records = audit.get("records", [])
        for rec in records:
            if rec.get("status") != "success":
                continue

            combined_path = rec.get("output_combined_path", "")
            if not combined_path or not os.path.isfile(combined_path):
                logger.warning(
                    "Case %s: combined mask not found at %s", rec.get("case_id"), combined_path
                )
                continue

            cid = _mask_stem(combined_path)
            if cid in seen:
                continue

            try:
                img = nib.load(combined_path)
                img_can = nib.as_closest_canonical(img)
                dim = tuple(int(d) for d in img_can.shape[:3])
                spacing = tuple(float(s) for s in img_can.header.get_zooms()[:3])
            except Exception as e:
                logger.warning("Case %s: cannot read mask header: %s", cid, e)
                continue

            seen[cid] = CaseSpec(
                case_id=cid,
                mask_path=combined_path,
                host_ct_path=rec.get("host_ct_path", ""),
                dim=dim,
                spacing=spacing,
                subdir=subdir,
                host_patient_id=rec.get("host_patient_id", ""),
                host_dataset=rec.get("host_dataset", ""),
                donor_annotation_id=rec.get("donor_annotation_id", ""),
                trial_name=audit.get("trial_name", ""),
            )

    cases = list(seen.values())
    src = (
        audit_path if len(audit_files) == 1 else f"{len(audit_files)} audit files under {root_dir}"
    )
    logger.info("Discovered %d cases from %s", len(cases), src)
    return cases


def cases_from_mask_root(mask_root: str, cfg: dict) -> list[CaseSpec]:
    """Build CaseSpecs by scanning a directory of inserted mask NIfTIs.

    Searches recursively so subdirectory layouts (per-bucket, per-lobe, etc.)
    are handled automatically.  Each case_id is the mask filename stem
    (without ``_mask.nii.gz``), which is globally unique.

    Host CT paths are resolved from the iTrialSpace manifest JSON / CSV
    files (``itrialspace.manifest_root`` in config).  These manifests already
    record ``host_ct_path`` for every case:

    1. Look up by ``host_patient_id`` (parsed from ``--host-<ID>--``).
    2. Fallback: look up by ``trial_name/case_id_int`` (parsed from
       ``--<trial>--C<nnnn>--``).  Handles LUNA16 whose mask filenames
       use truncated hashes instead of the full DICOM UID.
    """
    import glob

    # Build host-CT lookup from manifests (JSON + CSV)
    manifest_root = cfg.get("itrialspace", {}).get("manifest_root", "")
    pid_index: dict[str, str] = {}
    case_index: dict[str, str] = {}
    if manifest_root and os.path.isdir(manifest_root):
        pid_index, case_index = _build_host_ct_index(manifest_root)

    # Recursive search: finds masks in root *and* subdirectories
    masks = sorted(glob.glob(os.path.join(mask_root, "**", "*.nii.gz"), recursive=True))

    cases = []
    n_host_found = 0
    for mp in masks:
        case_id = _mask_stem(mp)
        # Derive subdir from relative path (empty if mask is at root)
        mp_dir = os.path.dirname(mp)
        subdir = os.path.basename(mp_dir) if mp_dir != mask_root else ""

        # Resolve host CT via manifest lookup
        host_ct = ""
        # Strategy 1: by host_patient_id
        host_id = _parse_host_id(case_id)
        if host_id and host_id in pid_index:
            host_ct = pid_index[host_id]
        # Strategy 2: by trial_name / integer case_id (handles LUNA16 hash IDs)
        if not host_ct:
            trial, cid_int = _parse_trial_case(case_id)
            key = f"{trial}/{cid_int}"
            if key in case_index:
                host_ct = case_index[key]
        if host_ct:
            n_host_found += 1

        try:
            img = nib.load(mp)
            img_can = nib.as_closest_canonical(img)
            dim = tuple(int(d) for d in img_can.shape[:3])
            spacing = tuple(float(s) for s in img_can.header.get_zooms()[:3])
        except Exception as e:
            logger.warning("Skipping %s: %s", mp, e)
            continue

        cases.append(
            CaseSpec(
                case_id=case_id,
                mask_path=mp,
                host_ct_path=host_ct,
                dim=dim,
                spacing=spacing,
                subdir=subdir,
            )
        )

    logger.info(
        "Discovered %d masks from %s (host CT resolved: %d/%d)",
        len(cases),
        mask_root,
        n_host_found,
        len(cases),
    )
    return cases


def cases_from_manifest(manifest_path: str, mask_root: str, cfg: dict) -> list[CaseSpec]:
    """Build CaseSpecs from an iTrialSpace CSV manifest + mask directory."""
    import pandas as pd

    df = pd.read_csv(manifest_path)
    if "case_id" not in df.columns:
        raise ValueError("Manifest must have a 'case_id' column")

    # Group by case_id, take first row per case
    grouped = df.groupby("case_id", sort=False).first().reset_index()
    cases = []
    for _, row in grouped.iterrows():
        cid = str(row["case_id"])
        # Try to find the combined mask in mask_root by pattern
        import glob

        pattern = os.path.join(mask_root, f"*C{cid.zfill(4)}*_mask.nii.gz")
        matches = glob.glob(pattern)
        if not matches:
            pattern2 = os.path.join(mask_root, f"*case{cid.zfill(4)}*")
            matches = glob.glob(pattern2)
        if not matches:
            logger.warning("Case %s: no mask found in %s", cid, mask_root)
            continue
        mp = matches[0]

        try:
            img = nib.load(mp)
            img_can = nib.as_closest_canonical(img)
            dim = tuple(int(d) for d in img_can.shape[:3])
            spacing = tuple(float(s) for s in img_can.header.get_zooms()[:3])
        except Exception as e:
            logger.warning("Case %s: cannot read %s: %s", cid, mp, e)
            continue

        cases.append(
            CaseSpec(
                case_id=cid,
                mask_path=mp,
                host_ct_path=str(row.get("host_ct_path", "")),
                dim=dim,
                spacing=spacing,
                host_patient_id=str(row.get("host_patient_id", "")),
                host_dataset=str(row.get("host_dataset", "")),
                donor_annotation_id=str(row.get("donor_annotation_id", "")),
                trial_name=str(row.get("trial_name", "")),
            )
        )
    logger.info("Discovered %d cases from manifest %s", len(cases), manifest_path)
    return cases


# ---------------------------------------------------------------------------
# Geometry validation / adjustment
# ---------------------------------------------------------------------------


def snap_to_valid_size(dim: tuple, geom_cfg: dict, spacing: tuple | None = None) -> tuple:
    """Snap volume dimensions to nearest valid NodMAISI size.

    If *spacing* is provided and the physical FOV in XY
    (spacing[0] * dim[0]) is below the NodMAISI minimum (256 mm),
    the smallest valid XY size whose grid alone meets the threshold
    is chosen.  This allows ``prepare_mask_for_nodmaisi`` to clamp
    the spacing to the minimum FOV / snapped_dim, satisfying the
    ``check_input`` guard.
    """
    valid_xy = sorted(geom_cfg.get("valid_xy_sizes", [256, 384, 512]))
    valid_z = sorted(geom_cfg.get("valid_z_sizes", [128, 256, 384, 512, 640, 768]))
    min_fov_xy = geom_cfg.get("min_fov_xy_mm", 256)

    def _nearest(val, choices):
        return min(choices, key=lambda c: abs(c - val))

    xy = _nearest(dim[0], valid_xy)  # NodMAISI requires dim[0] == dim[1]

    # If the original physical FOV is below the minimum, pick the
    # smallest valid XY size so floor-spacing = min_fov / xy stays
    # reasonable (closer to original spacing).
    if spacing is not None and spacing[0] * dim[0] < min_fov_xy:
        # Choose largest valid xy to keep adjusted spacing close to original
        xy = valid_xy[-1]  # 512 — safest bet for borderline CTs

    z = _nearest(dim[2], valid_z)
    return (xy, xy, z)


def prepare_mask_for_nodmaisi(
    case: CaseSpec, output_dir: str, geom_cfg: dict, nodule_label: int = 23
) -> dict:
    """Load mask, ensure RAS+ orientation, resize if needed, write dataset JSON.

    Returns dict with keys: label_path, dim, spacing, resized, original_dim.
    """
    img = nib.load(case.mask_path)
    img_can = nib.as_closest_canonical(img)
    data = np.asarray(img_can.dataobj)
    spacing = tuple(float(s) for s in img_can.header.get_zooms()[:3])
    original_dim = data.shape[:3]

    target_dim = snap_to_valid_size(original_dim, geom_cfg, spacing=spacing)
    resized = target_dim != original_dim

    if resized:
        from scipy.ndimage import zoom as ndimage_zoom

        # NodMAISI requires isotropic XY spacing (spacing[0] == spacing[1]).
        # When the original volume has non-square XY (dim_x != dim_y) and
        # target_dim forces square XY, a naive resize creates anisotropic
        # spacing.  Fix: pad the shorter XY axis to match the longer one
        # *before* resizing, so the zoom factor is identical for X and Y.
        dx, dy, dz = data.shape[:3]
        if dx != dy and target_dim[0] == target_dim[1]:
            pad_dim = max(dx, dy)
            pad_x = pad_dim - dx
            pad_y = pad_dim - dy
            # Symmetric padding (centered content)
            px_before, px_after = pad_x // 2, pad_x - pad_x // 2
            py_before, py_after = pad_y // 2, pad_y - pad_y // 2
            data = np.pad(
                data,
                ((px_before, px_after), (py_before, py_after), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            logger.info(
                "Padded non-square XY (%d,%d) -> (%d,%d) before resize "
                "to preserve isotropic spacing.",
                dx,
                dy,
                pad_dim,
                pad_dim,
            )
            original_dim = data.shape[:3]  # update for zoom calculation

        zoom_factors = tuple(t / o for t, o in zip(target_dim, original_dim))
        # Use nearest-neighbor for label maps to preserve integer values
        data_resized = ndimage_zoom(data.astype(np.float32), zoom_factors, order=0)
        data = np.round(data_resized).astype(np.uint8)
        # Adjust spacing proportionally
        spacing = tuple(s * o / t for s, o, t in zip(spacing, original_dim, target_dim))

    # NodMAISI requires spacing[0] == spacing[1] (exact equality).
    # Source NIfTI headers sometimes store pixdim with float32 rounding,
    # producing sp_x ≈ sp_y but not exactly equal.  Force equality when
    # the target grid is square.
    if target_dim[0] == target_dim[1] and spacing[0] != spacing[1]:
        avg_sp = (spacing[0] + spacing[1]) / 2.0
        spacing = (avg_sp, avg_sp, spacing[2])

    # Ensure FOV meets NodMAISI's minimum (256 mm in XY).
    # If the physical extent is still below the threshold after snapping
    # (because the original CT was physically small), clamp spacing up so
    # that   target_dim[0] * spacing[0] >= min_fov.
    min_fov_xy = geom_cfg.get("min_fov_xy_mm", 256)
    fov_xy = target_dim[0] * spacing[0]
    if fov_xy < min_fov_xy:
        floor_sp = min_fov_xy / target_dim[0]
        logger.warning(
            "FOV too small (%.1f mm < %d mm). Clamping XY spacing "
            "from %.4f to %.4f mm to satisfy NodMAISI constraint.",
            fov_xy,
            min_fov_xy,
            spacing[0],
            floor_sp,
        )
        spacing = (floor_sp, floor_sp, spacing[2])

    # Clamp Z spacing to NodMAISI's valid range [0.5, 5.0] mm.
    # Ultra-thin-slice CTs (e.g. 0.27 mm) can produce z-spacing below
    # the minimum after resizing to 256 slices.
    sp_z = spacing[2]
    if sp_z < 0.5:
        logger.warning(
            "Z spacing %.4f mm below minimum 0.5 mm; clamping to 0.5 mm.",
            sp_z,
        )
        spacing = (spacing[0], spacing[1], 0.5)
    elif sp_z > 5.0:
        logger.warning(
            "Z spacing %.4f mm above maximum 5.0 mm; clamping to 5.0 mm.",
            sp_z,
        )
        spacing = (spacing[0], spacing[1], 5.0)

    # Save the prepared mask to the case output directory
    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, "input_mask.nii.gz")
    # Build a new affine with the (possibly adjusted) spacing
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), label_path)

    return {
        "label_path": label_path,
        "dim": list(target_dim),
        "spacing": list(spacing),
        "resized": resized,
        "original_dim": list(original_dim),
    }


def write_dataset_json(label_path: str, dim: list, spacing: list, output_dir: str) -> str:
    """Write a NodMAISI-compatible dataset JSON for a single case."""
    # NodMAISI expects label path relative to data_base_dir
    dataset = {
        "name": "itrialspace_case",
        "numTest": 1,
        "testing": [
            {
                "label": os.path.basename(label_path),
                "fold": 0,
                "dim": dim,
                "spacing": spacing,
            }
        ],
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    return json_path


# ---------------------------------------------------------------------------
# NodMAISI inference wrapper
# ---------------------------------------------------------------------------

# Default and fallback sliding-window sizes for autoencoder inference.
# If the first attempt OOMs or produces no output, retry with the smaller size.
_DEFAULT_AE_WINDOW = [80, 80, 64]
_FALLBACK_AE_WINDOW = [64, 64, 48]


def validate_condition_mask(label_path: str, cfg: dict) -> dict:
    """Pre-NodMAISI QC gate — validate the prepared condition mask.

    Checks:
      1. File exists and is readable.
      2. Mask has the expected nodule label (default 23).
      3. Dimensions match a valid NodMAISI size.
      4. No NaN/Inf values.
      5. Nodule label occupies a physically plausible volume.

    Returns a dict: ``{"ok": bool, "reason": str, "details": dict}``.
    """
    nodule_label = cfg.get("itrialspace", {}).get("nodule_label_value", 23)
    geom_cfg = cfg.get("geometry", {})
    valid_xy = set(geom_cfg.get("valid_xy_sizes", [256, 384, 512]))
    valid_z = set(geom_cfg.get("valid_z_sizes", [128, 256, 384, 512, 640, 768]))

    details: dict = {}
    try:
        img = nib.load(label_path)
        data = np.asarray(img.dataobj)
        details["shape"] = list(data.shape[:3])
        details["dtype"] = str(data.dtype)
    except Exception as e:
        return {"ok": False, "reason": f"Cannot read mask: {e}", "details": details}

    # 1) NaN / Inf check
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return {"ok": False, "reason": "Mask contains NaN or Inf values", "details": details}

    # 2) Dimension validity
    dim = data.shape[:3]
    if dim[0] != dim[1]:
        return {
            "ok": False,
            "reason": f"X/Y dimensions differ ({dim[0]} vs {dim[1]}); NodMAISI requires square XY",
            "details": details,
        }
    if dim[0] not in valid_xy:
        return {
            "ok": False,
            "reason": f"XY size {dim[0]} not in valid set {sorted(valid_xy)}",
            "details": details,
        }
    if dim[2] not in valid_z:
        return {
            "ok": False,
            "reason": f"Z size {dim[2]} not in valid set {sorted(valid_z)}",
            "details": details,
        }

    # 3) Nodule label presence
    n_nodule_vox = int(np.sum(data == nodule_label))
    details["n_nodule_voxels"] = n_nodule_vox
    if n_nodule_vox == 0:
        return {
            "ok": False,
            "reason": f"Mask contains no voxels with nodule label {nodule_label}",
            "details": details,
        }

    # 4) Plausibility: nodule should be < 10% of total volume
    total_voxels = int(np.prod(dim))
    ratio = n_nodule_vox / total_voxels if total_voxels > 0 else 1.0
    details["nodule_volume_ratio"] = round(ratio, 6)
    if ratio > 0.10:
        return {
            "ok": False,
            "reason": f"Nodule label occupies {ratio:.1%} of volume (>10%); likely corrupt mask",
            "details": details,
        }

    return {"ok": True, "reason": "", "details": details}


def run_nodmaisi_inference(
    case_dir: str,
    dataset_json: str,
    cfg: dict,
    dry_run: bool = False,
    ae_window: list | None = None,
) -> str | None:
    """Run NodMAISI inference for a single case via subprocess.

    Parameters
    ----------
    ae_window : list, optional
        Override autoencoder_sliding_window_infer_size for this run.
        Used by the retry path to reduce memory pressure.

    Returns the path to the generated synthetic CT, or None on failure.
    """
    nodmaisi_cfg = cfg["nodmaisi"]
    infer_cfg = cfg["inference"]
    project_dir = nodmaisi_cfg["project_dir"]

    # Build a temporary environment config that points to this case's data
    env_override = {
        "model_dir": os.path.join(project_dir, "models") + "/",
        "output_dir": case_dir,
        "tfevent_path": os.path.join(case_dir, "tfevent"),
        "trained_autoencoder_path": _resolve_path(nodmaisi_cfg["autoencoder_path"], project_dir),
        "trained_diffusion_path": _resolve_path(nodmaisi_cfg["diffusion_unet_path"], project_dir),
        "trained_controlnet_path": _resolve_path(nodmaisi_cfg["controlnet_path"], project_dir),
        "exp_name": "iTS_case",
        "data_base_dir": [case_dir],
        "json_data_list": [dataset_json],
    }

    # Apply sliding-window override for retry / reduced-memory runs
    effective_window = ae_window or infer_cfg.get(
        "autoencoder_sliding_window_infer_size", _DEFAULT_AE_WINDOW
    )
    env_override["autoencoder_sliding_window_infer_size"] = effective_window

    # Write temporary env config
    tmp_env_path = os.path.join(case_dir, "_tmp_env.json")
    with open(tmp_env_path, "w") as f:
        json.dump(env_override, f, indent=2)

    model_config = _resolve_path(nodmaisi_cfg["model_config"], project_dir)
    controlnet_config = _resolve_path(nodmaisi_cfg["controlnet_config"], project_dir)

    if dry_run:
        logger.info("[DRY RUN] Would run NodMAISI with env=%s", tmp_env_path)
        return None

    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "scripts.infer_testV2_controlnet",
        "-c",
        model_config,
        "-e",
        tmp_env_path,
        "-t",
        controlnet_config,
    ]

    env = os.environ.copy()
    env["MONAI_DATA_DIRECTORY"] = project_dir

    logger.info("Running NodMAISI: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=project_dir,
        env=env,
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error(
            "NodMAISI failed (rc=%d):\nSTDOUT: %s\nSTDERR: %s",
            result.returncode,
            result.stdout[-2000:],
            result.stderr[-2000:],
        )
        # Persist stderr for structured failure diagnostics
        stderr_path = os.path.join(case_dir, "_last_stderr.txt")
        with open(stderr_path, "w") as f:
            f.write(result.stderr[-4000:] if result.stderr else "")
        return None

    # Find the generated image in case_dir (MONAI SaveImage appends _image suffix)
    import glob

    candidates = glob.glob(os.path.join(case_dir, "*_image.nii.gz"))
    if not candidates:
        candidates = glob.glob(os.path.join(case_dir, "*.nii.gz"))
        candidates = [
            c
            for c in candidates
            if "input_mask" not in c and "dataset" not in c and "_label" not in c
        ]

    if candidates:
        # Rename to standard name
        out_cfg = cfg.get("output", {})
        final_name = out_cfg.get("synthetic_ct_name", "synthetic_ct.nii.gz")
        final_path = os.path.join(case_dir, final_name)
        os.rename(candidates[0], final_path)
        return final_path

    logger.error("No synthetic CT found in %s after inference", case_dir)
    # Persist stderr even on rc=0 for diagnostic purposes
    stderr_path = os.path.join(case_dir, "_last_stderr.txt")
    with open(stderr_path, "w") as f:
        f.write(result.stderr[-4000:] if result.stderr else "")
    return None


def _resolve_path(p: str, base: str) -> str:
    """Resolve a possibly relative path against a base directory."""
    if os.path.isabs(p):
        return p
    return os.path.join(base, p)


# ---------------------------------------------------------------------------
# QC: PNG generation
# ---------------------------------------------------------------------------


def generate_qc(
    case_dir: str,
    synthetic_ct_path: str,
    mask_path: str,
    host_ct_path: str,
    cfg: dict,
    nodule_label: int = 23,
) -> list[str]:
    """Generate QC PNG montages showing host CT, mask overlay, and synthetic CT."""
    qc_cfg = cfg.get("qc", {})
    if not qc_cfg.get("enabled", True):
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    except ImportError:
        logger.warning("matplotlib not available; skipping QC PNGs")
        return []

    num_slices = qc_cfg.get("num_slices", 3)
    wc = qc_cfg.get("window_center", -600)
    ww = qc_cfg.get("window_width", 1500)
    dpi = qc_cfg.get("dpi", 150)
    vmin, vmax = wc - ww / 2, wc + ww / 2

    qc_dir = os.path.join(case_dir, cfg.get("output", {}).get("qc_dir_name", "qc"))
    os.makedirs(qc_dir, exist_ok=True)

    # Load volumes
    synth_data = np.asarray(nib.as_closest_canonical(nib.load(synthetic_ct_path)).dataobj)
    mask_data = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)

    has_host = host_ct_path and os.path.isfile(host_ct_path)
    if has_host:
        try:
            host_data = np.asarray(nib.as_closest_canonical(nib.load(host_ct_path)).dataobj)
        except Exception:
            has_host = False
            host_data = None
    else:
        host_data = None

    # Find axial slices that contain nodule label
    nodule_slices = np.where(np.any(mask_data == nodule_label, axis=(0, 1)))[0]
    if len(nodule_slices) == 0:
        # Fallback: evenly spaced slices through the volume
        nodule_slices = np.linspace(0, synth_data.shape[2] - 1, num_slices + 2, dtype=int)[1:-1]
    else:
        # Pick evenly spaced slices through the nodule Z-range
        indices = np.linspace(0, len(nodule_slices) - 1, num_slices, dtype=int)
        nodule_slices = nodule_slices[indices]

    png_paths = []
    for si, k in enumerate(nodule_slices):
        ncols = 3 if has_host else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        col = 0
        if has_host:
            axes[col].imshow(
                host_data[:, :, k].T,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="equal",
            )
            axes[col].set_title(f"Host CT (z={k})")
            axes[col].axis("off")
            col += 1

        # Mask overlay on synthetic CT
        axes[col].imshow(
            synth_data[:, :, k].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", aspect="equal"
        )
        nodule_overlay = np.ma.masked_where(mask_data[:, :, k] != nodule_label, mask_data[:, :, k])
        axes[col].imshow(nodule_overlay.T, cmap="autumn", alpha=0.5, origin="lower", aspect="equal")
        axes[col].set_title(f"Synth CT + Mask (z={k})")
        axes[col].axis("off")
        col += 1

        # Synthetic CT alone
        axes[col].imshow(
            synth_data[:, :, k].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", aspect="equal"
        )
        axes[col].set_title(f"Synthetic CT (z={k})")
        axes[col].axis("off")

        png_name = f"qc_slice_{si:02d}_z{k:04d}.png"
        png_path = os.path.join(qc_dir, png_name)
        fig.tight_layout()
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(png_path)

    return png_paths


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_check(synthetic_ct_path: str, mask_path: str, nodule_label: int = 23) -> dict:
    """Run sanity checks on the synthetic CT."""
    checks = {
        "shape_match": False,
        "has_nodule_label": False,
        "ct_min_hu": 0.0,
        "ct_max_hu": 0.0,
        "has_nan": False,
    }
    try:
        ct_img = nib.load(synthetic_ct_path)
        ct_data = np.asarray(ct_img.dataobj)
        mask_img = nib.load(mask_path)
        mask_data = np.asarray(mask_img.dataobj)

        checks["shape_match"] = ct_data.shape[:3] == mask_data.shape[:3]
        checks["has_nodule_label"] = bool(np.any(mask_data == nodule_label))
        checks["ct_min_hu"] = float(np.nanmin(ct_data))
        checks["ct_max_hu"] = float(np.nanmax(ct_data))
        checks["has_nan"] = bool(np.any(np.isnan(ct_data)))
    except Exception as e:
        logger.error("Sanity check failed: %s", e)
    return checks


# ---------------------------------------------------------------------------
# Per-case orchestrator
# ---------------------------------------------------------------------------


def process_one_case(case: CaseSpec, cfg: dict, outdir: str, dry_run: bool = False) -> CaseResult:
    """Full pipeline for a single case: prepare -> infer -> QC -> audit."""
    t0 = time.time()
    result = CaseResult(
        case_id=case.case_id, input_mask_path=case.mask_path, host_ct_path=case.host_ct_path
    )

    out_cfg = cfg.get("output", {})
    # Mirror input subdirectory structure: subdir/case_id
    if case.subdir:
        case_dir = os.path.join(outdir, case.subdir, case.case_id)
    else:
        case_dir = os.path.join(outdir, case.case_id)
    os.makedirs(case_dir, exist_ok=True)

    geom_cfg = cfg.get("geometry", {})
    nodule_label = cfg.get("itrialspace", {}).get("nodule_label_value", 23)

    # --- Step 1: Prepare mask ---
    t_prep = time.time()
    try:
        prep = prepare_mask_for_nodmaisi(case, case_dir, geom_cfg, nodule_label)
    except Exception as e:
        result.status = "failed"
        result.reason = f"Mask preparation failed: {e}"
        result.total_sec = time.time() - t0
        return result

    result.original_dim = tuple(prep["original_dim"])
    result.original_spacing = tuple(case.spacing)
    result.nodmaisi_dim = tuple(prep["dim"])
    result.nodmaisi_spacing = tuple(prep["spacing"])
    result.resized = prep["resized"]

    dataset_json = write_dataset_json(prep["label_path"], prep["dim"], prep["spacing"], case_dir)
    result.prep_sec = time.time() - t_prep

    # --- Step 1b: Pre-NodMAISI QC gate (validate condition mask) ---
    qc_gate = validate_condition_mask(prep["label_path"], cfg)
    if not qc_gate["ok"]:
        result.status = "failed"
        result.reason = f"Pre-NodMAISI QC gate rejected mask: {qc_gate['reason']}"
        result.total_sec = time.time() - t0
        return result

    # --- Step 2: Symlink host CT (optional) ---
    if case.host_ct_path and os.path.isfile(case.host_ct_path):
        link_name = out_cfg.get("host_ct_link_name", "host_ct.nii.gz")
        link_path = os.path.join(case_dir, link_name)
        if not os.path.exists(link_path):
            try:
                os.symlink(case.host_ct_path, link_path)
            except OSError:
                pass  # skip if symlinking not supported

    # --- Step 3: Run NodMAISI inference ---
    if dry_run:
        result.status = "dry_run"
        result.reason = "Dry run — inference skipped"
        result.total_sec = time.time() - t0
        return result

    t_infer = time.time()
    synthetic_ct_path = run_nodmaisi_inference(case_dir, dataset_json, cfg)
    result.inference_sec = time.time() - t_infer

    # Retry with smaller sliding window if first attempt produced no output
    if synthetic_ct_path is None:
        infer_cfg = cfg.get("inference", {})
        current_window = infer_cfg.get("autoencoder_sliding_window_infer_size", _DEFAULT_AE_WINDOW)
        if current_window != _FALLBACK_AE_WINDOW:
            logger.warning(
                "Case %s: first attempt produced no output, retrying with "
                "reduced sliding window %s → %s",
                case.case_id,
                current_window,
                _FALLBACK_AE_WINDOW,
            )
            t_retry = time.time()
            synthetic_ct_path = run_nodmaisi_inference(
                case_dir,
                dataset_json,
                cfg,
                ae_window=_FALLBACK_AE_WINDOW,
            )
            retry_sec = time.time() - t_retry
            result.inference_sec += retry_sec
            if synthetic_ct_path is not None:
                logger.info(
                    "Case %s: retry succeeded with reduced window (%.1fs)",
                    case.case_id,
                    retry_sec,
                )

    if synthetic_ct_path is None:
        # Collect structured diagnostics for the failure
        diag: dict = {"case_id": case.case_id}
        try:
            diag["mask_dim"] = list(result.nodmaisi_dim)
            diag["mask_spacing"] = list(result.nodmaisi_spacing)
            diag["mask_resized"] = result.resized
            # Check for GPU memory info
            try:
                import torch

                if torch.cuda.is_available():
                    diag["gpu_mem_allocated_MB"] = round(torch.cuda.memory_allocated() / 1e6, 1)
                    diag["gpu_mem_reserved_MB"] = round(torch.cuda.memory_reserved() / 1e6, 1)
                    diag["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
            # Log stderr tail from last subprocess run
            last_stderr_path = os.path.join(case_dir, "_last_stderr.txt")
            if os.path.isfile(last_stderr_path):
                with open(last_stderr_path) as f:
                    diag["stderr_tail"] = f.read()[-2000:]
        except Exception:
            pass

        # Write structured failure log
        fail_log_path = os.path.join(case_dir, "nodmaisi_failure.json")
        with open(fail_log_path, "w") as f:
            json.dump(diag, f, indent=2, default=str)

        result.status = "failed"
        result.reason = (
            f"NodMAISI inference produced no output "
            f"(dim={list(result.nodmaisi_dim)}, retried={current_window != _FALLBACK_AE_WINDOW})"
        )
        result.total_sec = time.time() - t0
        return result

    result.synthetic_ct_path = synthetic_ct_path

    # --- Step 4: Sanity checks ---
    checks = sanity_check(synthetic_ct_path, prep["label_path"], nodule_label)
    result.shape_match = checks["shape_match"]
    result.has_nodule_label = checks["has_nodule_label"]
    result.ct_min_hu = checks["ct_min_hu"]
    result.ct_max_hu = checks["ct_max_hu"]
    result.has_nan = checks["has_nan"]

    # --- Step 5: QC PNGs ---
    t_qc = time.time()
    try:
        result.qc_png_paths = generate_qc(
            case_dir, synthetic_ct_path, prep["label_path"], case.host_ct_path, cfg, nodule_label
        )
    except Exception as e:
        logger.warning("QC generation failed for case %s: %s", case.case_id, e)
    result.qc_sec = time.time() - t_qc

    result.status = "success"
    result.total_sec = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Write per-case audit JSON
# ---------------------------------------------------------------------------


def write_case_audit(result: CaseResult, case_dir: str, cfg: dict):
    """Write a per-case NodMAISI audit JSON."""
    audit = {
        "case_id": result.case_id,
        "status": result.status,
        "reason": result.reason,
        "timestamp": datetime.now().isoformat(),
        "input_mask_path": result.input_mask_path,
        "host_ct_path": result.host_ct_path,
        "synthetic_ct_path": result.synthetic_ct_path,
        "geometry": {
            "original_dim": result.original_dim,
            "original_spacing": result.original_spacing,
            "nodmaisi_dim": result.nodmaisi_dim,
            "nodmaisi_spacing": result.nodmaisi_spacing,
            "resized": result.resized,
        },
        "nodmaisi_params": {
            "num_inference_steps": cfg.get("inference", {}).get("num_inference_steps", 30),
            "noise_factor": cfg.get("inference", {}).get("noise_factor", 1.0),
            "modality": cfg.get("inference", {}).get("modality", 1),
            "seed": cfg.get("inference", {}).get("seed", 42),
            "autoencoder_sliding_window_infer_size": cfg.get("inference", {}).get(
                "autoencoder_sliding_window_infer_size", [80, 80, 64]
            ),
        },
        "timing": {
            "prep_sec": round(result.prep_sec, 2),
            "inference_sec": round(result.inference_sec, 2),
            "qc_sec": round(result.qc_sec, 2),
            "total_sec": round(result.total_sec, 2),
        },
        "sanity_checks": {
            "shape_match": result.shape_match,
            "has_nodule_label": result.has_nodule_label,
            "ct_min_hu": round(result.ct_min_hu, 1),
            "ct_max_hu": round(result.ct_max_hu, 1),
            "has_nan": result.has_nan,
        },
        "qc_pngs": result.qc_png_paths,
    }
    audit_name = cfg.get("output", {}).get("audit_name", "nodmaisi_audit.json")
    audit_path = os.path.join(case_dir, audit_name)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_itrialspace_to_ct",
        description="iTrialSpace masks -> NodMAISI -> synthetic CTs",
    )
    parser.add_argument("--manifest", "-m", help="iTrialSpace manifest CSV")
    parser.add_argument("--audit", "-a", help="iTrialSpace audit.json (preferred)")
    parser.add_argument("--mask-root", help="Directory of inserted mask NIfTIs")
    parser.add_argument("--config", "-c", required=True, help="Integration config YAML")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Parallelism (future)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no inference")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--case-ids", nargs="*", help="Process only these case IDs")

    args = parser.parse_args(argv)

    # Logging
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    # --- Discover cases ---
    if args.audit:
        cases = cases_from_audit(args.audit, cfg)
    elif args.mask_root:
        cases = cases_from_mask_root(args.mask_root, cfg)
    elif args.manifest and args.mask_root:
        cases = cases_from_manifest(args.manifest, args.mask_root, cfg)
    elif args.manifest:
        # Need mask_root too if not using audit
        parser.error("--manifest requires --mask-root to locate combined masks")
        return 1
    else:
        parser.error("Provide --audit, --mask-root, or --manifest + --mask-root")
        return 1

    # Filter by case IDs if specified
    if args.case_ids:
        id_set = set(args.case_ids)
        cases = [c for c in cases if c.case_id in id_set]

    if not cases:
        logger.error("No cases found. Check input paths.")
        return 1

    os.makedirs(args.outdir, exist_ok=True)

    # --- Process cases ---
    results = []
    t_total = time.time()
    for i, case in enumerate(cases):
        logger.info("=== Case %d/%d: %s ===", i + 1, len(cases), case.case_id)
        result = process_one_case(case, cfg, args.outdir, dry_run=args.dry_run)
        results.append(result)

        # Write per-case audit
        if case.subdir:
            case_dir = os.path.join(args.outdir, case.subdir, case.case_id)
        else:
            case_dir = os.path.join(args.outdir, case.case_id)
        write_case_audit(result, case_dir, cfg)

        logger.info("  Status: %s  Time: %.1fs", result.status, result.total_sec)

        # Free GPU memory between cases
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # --- Summary report ---
    elapsed = time.time() - t_total
    n_ok = sum(1 for r in results if r.status == "success")
    n_fail = sum(1 for r in results if r.status == "failed")
    n_dry = sum(1 for r in results if r.status == "dry_run")

    print(f"\n{'=' * 60}")
    print(f" iTrialSpace -> NodMAISI Pipeline {'DRY RUN' if args.dry_run else 'COMPLETE'}")
    print(f"{'=' * 60}")
    print(f"  Total cases:   {len(results)}")
    print(f"  Success:       {n_ok}")
    print(f"  Failed:        {n_fail}")
    if n_dry:
        print(f"  Dry-run:       {n_dry}")
    print(f"  Elapsed:       {elapsed:.1f}s")
    print(f"  Output dir:    {args.outdir}")

    if n_fail > 0:
        print("\nFailed cases:")
        for r in results:
            if r.status == "failed":
                print(f"  case {r.case_id}: {r.reason}")

    # Write summary — use per-case path for single-case (array job) runs
    # to avoid race conditions when many tasks write to the same outdir.
    summary = {
        "pipeline": "itrialspace_to_nodmaisi",
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "n_success": n_ok,
        "n_failed": n_fail,
        "n_dry_run": n_dry,
        "elapsed_sec": round(elapsed, 1),
        "config": args.config,
        "cases": [
            {
                "case_id": r.case_id,
                "status": r.status,
                "reason": r.reason,
                "synthetic_ct": r.synthetic_ct_path,
                "total_sec": round(r.total_sec, 2),
            }
            for r in results
        ],
    }
    if len(results) == 1:
        # Array-job mode: write per-case summary inside the case output dir
        # Check flat layout first, then sub-bin layout
        case_dir = os.path.join(args.outdir, results[0].case_id)
        if not os.path.isdir(case_dir):
            # Search sub-directories for the case (sub-bin layout)
            for entry in os.listdir(args.outdir) if os.path.isdir(args.outdir) else []:
                candidate = os.path.join(args.outdir, entry, results[0].case_id)
                if os.path.isdir(candidate):
                    case_dir = candidate
                    break
        if os.path.isdir(case_dir):
            summary_path = os.path.join(case_dir, "pipeline_summary.json")
        else:
            summary_path = os.path.join(args.outdir, "pipeline_summary.json")
    else:
        summary_path = os.path.join(args.outdir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return 1 if n_fail > 0 and n_ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
