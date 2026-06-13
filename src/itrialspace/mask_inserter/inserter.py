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
inserter.py — Main orchestration for the mask insertion pipeline.

Public API:
  insert_case(row, ...)       – process a single manifest row
  insert_manifest(path, ...)  – process an entire manifest CSV/JSON

Workflow per row:
  1. Resolve paths (resolver_bridge)
  2. Load host organ segmentation → canonical RAS+
  3. Compute placement (placement.py)
  4. Load donor nodule mask → canonical RAS+
  5. Resample donor into host geometry (resample.py)
  6. Paste into combined mask; check collision
  7. Save per-nodule mask and update combined mask
  8. Return audit record
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from itrialspace.mask_inserter.placement import (
    PlacementConfig,
    check_overlap,
    compute_placement,
    make_seed,
    resolve_collision,
)
from itrialspace.mask_inserter.resample import (
    ResampleConfig,
    load_nifti_canonical,
    resample_donor_to_host,
    save_nifti,
)
from itrialspace.mask_inserter.resolver_bridge import ResolverBridge

logger = logging.getLogger("itrialspace.mask_inserter")


def _compact_id(raw_id: str, max_len: int = 20) -> str:
    """Shorten IDs longer than *max_len* to a deterministic 8-char hex hash."""
    s = str(raw_id)
    if len(s) <= max_len:
        return s
    return hashlib.sha256(s.encode()).hexdigest()[:8]


# ── Configuration loading ─────────────────────────────────────────────────

_DEFAULTS_YAML = Path(__file__).parent / "config" / "defaults.yaml"


def load_config(config_path: str | None = None) -> dict:
    """Load YAML config, merging with defaults."""
    if yaml is None:
        raise ImportError("pyyaml required: pip install pyyaml")

    with open(_DEFAULTS_YAML) as f:
        defaults = yaml.safe_load(f) or {}

    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        defaults = _deep_merge(defaults, overrides)

    return defaults


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _placement_config_from_dict(d: dict) -> PlacementConfig:
    p = d.get("placement", {})
    return PlacementConfig(
        max_snap_radius_vox=p.get("max_snap_radius_vox", 25),
        min_pleural_dist_mm=p.get("min_pleural_dist_mm", 2.0),
        check_overlap=p.get("check_overlap", True),
        max_overlap_fraction=p.get("max_overlap_fraction", 0.05),
        max_collision_shift_vox=p.get("max_collision_shift_vox", 15),
    )


def _resample_config_from_dict(d: dict) -> ResampleConfig:
    r = d.get("resample", {})
    return ResampleConfig(
        interpolation=r.get("interpolation", "nearest"),
        binary_threshold=r.get("binary_threshold", 0.5),
        min_component_size_vox=r.get("min_component_size_vox", 3),
        fill_holes=r.get("fill_holes", True),
    )


# ── Audit record ──────────────────────────────────────────────────────────


@dataclass
class InsertionAuditRecord:
    """Audit trail for a single nodule insertion."""

    case_id: str = ""
    nodule_idx: int = 0
    is_primary: bool = True
    companion_group_id: str = ""
    status: str = "success"  # success | skipped | failed
    reason: str = ""
    # Paths used
    host_ct_path: str = ""
    host_organ_seg_path: str = ""
    donor_nodule_mask_path: str = ""
    # Placement
    insertion_lobe: str = ""
    cc_pct: float = 0.0
    ml_pct: float = 0.0
    ap_pct: float = 0.0
    target_ijk: Tuple[int, int, int] | None = None
    target_world_mm: Tuple[float, float, float] | None = None
    snapped: bool = False
    snap_distance_vox: float = 0.0
    pleural_dist_mm: float = 0.0
    overlap_fraction: float = 0.0
    placement_warnings: list = field(default_factory=list)
    # Resample
    scale_factor: float = 1.0
    donor_mask_voxels: int = 0
    resampled_mask_voxels: int = 0
    # Output
    output_mask_path: str = ""
    output_combined_path: str = ""
    host_id_compact: str = ""
    nodule_label_value: int = 23
    elapsed_sec: float = 0.0


# ── Per-row insertion ─────────────────────────────────────────────────────


def insert_case(
    row: pd.Series | dict,
    *,
    output_dir: str,
    config: dict | None = None,
    resolver: ResolverBridge | None = None,
    existing_combined: np.ndarray | None = None,
    existing_affine: np.ndarray | None = None,
    trial_name: str = "unnamed",
    seed: int = 42,
    dry_run: bool = False,
) -> Tuple[InsertionAuditRecord, np.ndarray | None, np.ndarray | None]:
    """Insert a single donor nodule into host space.

    Parameters
    ----------
    row : Series or dict
        One row from the CohortManifest.
    output_dir : str
        Root output directory.
    config : dict
        Merged configuration (see ``load_config``).
    resolver : ResolverBridge
        For path resolution.
    existing_combined : ndarray, optional
        Current combined mask for this host (updated in-place for multi-nodule).
    existing_affine : ndarray, optional
        Affine of the existing combined mask.
    trial_name : str
    seed : int
    dry_run : bool
        If True, compute placement but don't write files.

    Returns
    -------
    (audit_record, updated_combined_mask, combined_affine)
    """
    t0 = time.time()
    if isinstance(row, dict):
        row = pd.Series(row)

    cfg = config or load_config()
    resolver = resolver or ResolverBridge()
    placement_cfg = _placement_config_from_dict(cfg)
    resample_cfg = _resample_config_from_dict(cfg)

    case_id = str(row.get("case_id", "unknown"))
    nodule_idx = int(row.get("nodule_idx", 0))

    audit = InsertionAuditRecord(
        case_id=case_id,
        nodule_idx=nodule_idx,
        is_primary=bool(row.get("is_primary_nodule", True)),
        companion_group_id=str(row.get("companion_group_id", "")),
        insertion_lobe=str(row.get("insertion_lobe", "")),
        cc_pct=float(row.get("insertion_lobe_cc_pct", 0)),
        ml_pct=float(row.get("insertion_lobe_ml_pct", 0)),
        ap_pct=float(row.get("insertion_lobe_ap_pct", 0)),
        scale_factor=float(row.get("scale_factor", 1.0)),
    )

    # ── 1. Resolve paths ─────────────────────────────────────────────────
    paths = resolver.resolve(row)
    missing = paths.validate()
    if missing:
        audit.status = "failed"
        audit.reason = f"Missing files: {missing}"
        audit.elapsed_sec = time.time() - t0
        logger.warning("Row %s/%d: %s", case_id, nodule_idx, audit.reason)
        return audit, existing_combined, existing_affine

    audit.host_ct_path = paths.host_ct
    audit.host_organ_seg_path = paths.host_organ_seg
    audit.donor_nodule_mask_path = paths.donor_nodule_mask

    # ── 2. Load host organ segmentation ──────────────────────────────────
    try:
        seg_data, seg_aff, seg_spacing = load_nifti_canonical(paths.host_organ_seg)
    except Exception as e:
        audit.status = "failed"
        audit.reason = f"Failed to load organ seg: {e}"
        audit.elapsed_sec = time.time() - t0
        return audit, existing_combined, existing_affine

    host_shape = seg_data.shape
    host_spacing = seg_spacing

    # ── 2a. Shape guard: reset combined mask if host geometry changed ─────
    # Multi-nodule cases (e.g. LUNA25) may span different CT series with
    # different array shapes.  When that happens we must re-initialise the
    # combined mask from the current organ seg so that later overlap
    # checks and voxel stamping don't crash on shape mismatches.
    if existing_combined is not None and existing_combined.shape != host_shape:
        logger.warning(
            "Row %s/%d: host geometry changed from %s to %s — "
            "re-initialising combined mask for this CT series.",
            case_id,
            nodule_idx,
            existing_combined.shape,
            host_shape,
        )
        existing_combined = None
        existing_affine = None

    # ── 2b. Sanity check: host seg should not already contain nodule label ─
    nodule_label_value = int(cfg.get("nodule_label_value", 23))
    if existing_combined is None and (seg_data == nodule_label_value).any():
        n_pre = int((seg_data == nodule_label_value).sum())
        logger.warning(
            "Row %s/%d: host organ seg already contains %d voxels with "
            "nodule label %d — these will be overwritten during insertion.",
            case_id,
            nodule_idx,
            n_pre,
            nodule_label_value,
        )

    # ── 3. Compute placement ─────────────────────────────────────────────
    case_seed = make_seed(trial_name, case_id, nodule_idx, seed)
    rng = np.random.default_rng(case_seed)

    lobe_labels = cfg.get("lobe_labels", None)
    lobe_aliases = cfg.get("lobe_aliases", None)
    lung_labels = cfg.get("lung_labels", None)

    placement = compute_placement(
        seg_data=seg_data,
        affine=seg_aff,
        voxel_spacing_mm=host_spacing,
        insertion_lobe=audit.insertion_lobe,
        cc_pct=audit.cc_pct,
        ml_pct=audit.ml_pct,
        ap_pct=audit.ap_pct,
        config=placement_cfg,
        existing_combined_mask=existing_combined,
        lobe_labels=lobe_labels,
        lobe_aliases=lobe_aliases,
        lung_labels=lung_labels,
        rng=rng,
    )

    audit.target_ijk = placement.center_ijk
    audit.target_world_mm = placement.center_world_mm
    audit.snapped = placement.snapped
    audit.snap_distance_vox = placement.snap_distance_vox
    audit.pleural_dist_mm = placement.pleural_dist_mm
    audit.placement_warnings = placement.warnings

    if not placement.is_feasible:
        audit.status = "failed"
        audit.reason = f"Placement infeasible: {placement.reason}"
        audit.elapsed_sec = time.time() - t0
        logger.warning("Row %s/%d: %s", case_id, nodule_idx, audit.reason)
        return audit, existing_combined, existing_affine

    if dry_run:
        audit.status = "dry_run"
        audit.elapsed_sec = time.time() - t0
        return audit, existing_combined, existing_affine

    # ── 4. Load donor nodule mask ────────────────────────────────────────
    try:
        donor_data, donor_aff, donor_spacing = load_nifti_canonical(paths.donor_nodule_mask)
    except Exception as e:
        audit.status = "failed"
        audit.reason = f"Failed to load donor mask: {e}"
        audit.elapsed_sec = time.time() - t0
        return audit, existing_combined, existing_affine

    audit.donor_mask_voxels = int((donor_data > 0).sum())

    # ── 5. Resample donor into host geometry ─────────────────────────────
    scale_factor = audit.scale_factor
    target_ijk = placement.center_ijk

    try:
        host_nodule_mask = resample_donor_to_host(
            donor_mask=donor_data,
            donor_affine=donor_aff,
            donor_spacing=donor_spacing,
            host_shape=host_shape,
            host_affine=seg_aff,
            host_spacing=host_spacing,
            target_center_ijk=target_ijk,
            scale_factor=scale_factor,
            config=resample_cfg,
        )
    except Exception as e:
        audit.status = "failed"
        audit.reason = f"Resampling failed: {e}"
        audit.elapsed_sec = time.time() - t0
        return audit, existing_combined, existing_affine

    audit.resampled_mask_voxels = int((host_nodule_mask > 0).sum())

    # ── 6. Overlap check & collision resolution ──────────────────────────
    if placement_cfg.check_overlap and existing_combined is not None:
        frac, ok = check_overlap(
            existing_combined,
            host_nodule_mask,
            placement_cfg.max_overlap_fraction,
        )
        audit.overlap_fraction = frac
        if not ok:
            new_center = resolve_collision(
                (seg_data > 0).astype(np.uint8),
                existing_combined,
                target_ijk,
                max_shift=placement_cfg.max_collision_shift_vox,
                rng=rng,
            )
            if new_center is not None:
                audit.placement_warnings.append(
                    f"Shifted from {target_ijk} to {new_center} to resolve collision."
                )
                # Re-resample at new centre
                host_nodule_mask = resample_donor_to_host(
                    donor_mask=donor_data,
                    donor_affine=donor_aff,
                    donor_spacing=donor_spacing,
                    host_shape=host_shape,
                    host_affine=seg_aff,
                    host_spacing=host_spacing,
                    target_center_ijk=new_center,
                    scale_factor=scale_factor,
                    config=resample_cfg,
                )
                audit.target_ijk = new_center
                audit.resampled_mask_voxels = int((host_nodule_mask > 0).sum())
            else:
                audit.placement_warnings.append("Could not resolve collision; mask may overlap.")

    # ── 7. Optionally save per-nodule mask ─────────────────────────────
    out_cfg = cfg.get("output", {})
    audit.nodule_label_value = nodule_label_value

    if out_cfg.get("save_per_nodule_mask", False):
        if out_cfg.get("use_case_subdirs", False):
            case_dir = os.path.join(output_dir, case_id)
        else:
            case_dir = output_dir

        mask_template = out_cfg.get(
            "nodule_mask_template",
            "case_{case_id}_nodule_{nodule_idx}_mask.nii.gz",
        )
        donor_id = str(row.get("donor_annotation_id", f"donor_{nodule_idx}"))
        mask_name = mask_template.format(
            case_id=str(case_id).zfill(4),
            nodule_idx=nodule_idx,
            donor_id=donor_id,
            host_patient_id=str(row.get("host_patient_id", "")),
            host_dataset=str(row.get("host_dataset", "")),
        )
        mask_path = os.path.join(case_dir, mask_name)

        labelled_nodule_mask = np.where(
            host_nodule_mask > 0,
            np.int16(nodule_label_value),
            np.int16(0),
        ).astype(np.int16)
        save_nifti(labelled_nodule_mask, seg_aff, mask_path, dtype=np.int16)
        audit.output_mask_path = mask_path

    # ── 8. Update combined mask ──────────────────────────────────────────
    if existing_combined is None:
        # Initialise from the host organ segmentation so the combined mask
        # carries full body context (lobe labels etc.) in addition to the
        # inserted nodule voxels.
        existing_combined = seg_data.astype(np.int16)
        existing_affine = seg_aff

    # Stamp nodule voxels with the label value.
    # Overlap check prevents overwriting an already-inserted nodule voxel;
    # it does NOT block placement over organ-seg labels (lobe labels are
    # overwritten by the nodule, which is the intended behaviour).
    insert_where = host_nodule_mask > 0
    if placement_cfg.check_overlap:
        # Only refuse to overwrite voxels that already carry the nodule label
        # (i.e. a previous nodule in this companion group).
        insert_where = insert_where & (existing_combined != nodule_label_value)
    existing_combined[insert_where] = nodule_label_value

    audit.status = "success"
    audit.elapsed_sec = time.time() - t0
    logger.info(
        "Inserted %s/nodule_%d: %d voxels at %s (%.1fs)",
        case_id,
        nodule_idx,
        audit.resampled_mask_voxels,
        audit.target_ijk,
        audit.elapsed_sec,
    )
    return audit, existing_combined, existing_affine


# ── Manifest-level orchestration ──────────────────────────────────────────


def insert_manifest(
    manifest_path: str,
    output_dir: str,
    *,
    config_path: str | None = None,
    resolver: ResolverBridge | None = None,
    path_resolver: object = None,
    base_dir: str | None = None,
    trial_name: str | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    n_jobs: int = 1,
) -> list[InsertionAuditRecord]:
    """Process an entire CohortManifest.

    Groups rows by case_id to handle multi-nodule companion groups,
    processing nodules within a case sequentially (so the combined mask
    accumulates correctly).

    Parameters
    ----------
    manifest_path : str
        Path to manifest CSV or JSON.
    output_dir : str
        Root output directory.
    config_path : str, optional
        YAML config overrides.
    resolver : ResolverBridge, optional
    path_resolver : object, optional
        iTrialSpace PathResolver to wrap.
    base_dir : str, optional
        Fallback base directory for relative paths.
    trial_name : str, optional
    seed : int, optional
    dry_run : bool
    n_jobs : int
        Parallelism (across cases, not within).

    Returns
    -------
    list[InsertionAuditRecord]
    """
    cfg = load_config(config_path)
    seed = seed or cfg.get("seed", 42)
    trial_name = trial_name or "unnamed"

    if resolver is None:
        resolver = ResolverBridge(path_resolver=path_resolver, base_dir=base_dir)

    # Load manifest
    df = _load_manifest_df(manifest_path)
    logger.info(
        "Loaded manifest with %d rows from %s",
        len(df),
        manifest_path,
    )

    # Group by case_id
    if "case_id" not in df.columns:
        raise ValueError("Manifest must have a 'case_id' column.")

    groups = list(df.groupby("case_id", sort=False))
    logger.info("Processing %d cases.", len(groups))

    all_records: list[InsertionAuditRecord] = []
    out_cfg = cfg.get("output", {})

    # Sequential per-case processing (multi-nodule must be sequential)
    for case_id, case_df in groups:
        if "nodule_idx" in case_df.columns:
            case_df = case_df.sort_values("nodule_idx")
        combined_mask = None
        combined_aff = None

        # Track per-geometry combined masks for cases that span multiple
        # CT series (e.g. LUNA25 patients with different-shaped CTs).
        # Each entry: (mask_array, affine, [audit_record_indices])
        geo_masks: list[tuple[np.ndarray, np.ndarray | None, list[int]]] = []
        current_geo_audits: list[int] = []

        for _, row in case_df.iterrows():
            prev_mask = combined_mask
            prev_aff = combined_aff

            audit, combined_mask, combined_aff = insert_case(
                row,
                output_dir=output_dir,
                config=cfg,
                resolver=resolver,
                existing_combined=combined_mask,
                existing_affine=combined_aff,
                trial_name=trial_name,
                seed=seed,
                dry_run=dry_run,
            )

            audit_idx = len(all_records)
            all_records.append(audit)

            # Detect geometry reset: insert_case created a new combined
            # mask array because the host shape changed.
            if prev_mask is not None and combined_mask is not prev_mask:
                geo_masks.append((prev_mask, prev_aff, current_geo_audits))
                current_geo_audits = []

            current_geo_audits.append(audit_idx)

        # Finalise last geometry
        if combined_mask is not None:
            geo_masks.append((combined_mask, combined_aff, current_geo_audits))

        # Save combined mask(s) per case
        if not dry_run and geo_masks:
            if out_cfg.get("use_case_subdirs", False):
                case_dir = os.path.join(output_dir, str(case_id))
            else:
                case_dir = output_dir

            first_row = case_df.iloc[0]
            combined_template = out_cfg.get(
                "combined_mask_template",
                "iTS--{trial_name}--C{case_id}--host-{host_id}--src-{host_dataset}--nod-{donor_id}_mask.nii.gz",
            )
            raw_host_id = str(first_row.get("host_patient_id", ""))
            compact_host_id = _compact_id(raw_host_id)

            if len(case_df) > 1:
                donor_id_str = f"Nnod{len(case_df)}"
            else:
                donor_id_str = str(first_row.get("donor_annotation_id", ""))

            base_combined_name = combined_template.format(
                trial_name=str(first_row.get("trial_name", trial_name)),
                case_id=str(case_id).zfill(4),
                host_id=compact_host_id,
                host_dataset=str(first_row.get("host_dataset", "")),
                donor_id=donor_id_str,
            )

            multi_geo = len(geo_masks) > 1
            if multi_geo:
                logger.info(
                    "Case %s spans %d CT geometries; saving separate "
                    "combined masks per geometry.",
                    case_id,
                    len(geo_masks),
                )

            for geo_idx, (mask, aff, audit_indices) in enumerate(geo_masks):
                if not mask.any():
                    continue
                if aff is None:
                    aff = np.eye(4)

                # Append _ct{N} suffix for multi-geometry cases so each
                # CT series gets its own combined mask file.
                if multi_geo:
                    combined_name = base_combined_name.replace(
                        "_mask.nii.gz",
                        f"_ct{geo_idx}_mask.nii.gz",
                    )
                else:
                    combined_name = base_combined_name

                combined_path = os.path.join(case_dir, combined_name)
                save_nifti(mask, aff, combined_path, dtype=np.int16)

                for ai in audit_indices:
                    all_records[ai].output_combined_path = combined_path
                    all_records[ai].host_id_compact = compact_host_id

    # Write audit JSON
    if not dry_run:
        audit_name = out_cfg.get("audit_name", "audit.json")
        audit_path = os.path.join(output_dir, audit_name)
        _write_audit(all_records, audit_path, trial_name, manifest_path)

    n_ok = sum(1 for r in all_records if r.status == "success")
    n_fail = sum(1 for r in all_records if r.status == "failed")
    logger.info(
        "Done: %d success, %d failed, %d total.",
        n_ok,
        n_fail,
        len(all_records),
    )

    return all_records


# ── Helpers ───────────────────────────────────────────────────────────────


def _load_manifest_df(path: str) -> pd.DataFrame:
    """Load manifest from CSV or JSON."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if "cases" in data:
            return pd.DataFrame(data["cases"])
        raise ValueError("JSON manifest must be a list or have a 'cases' key.")
    return pd.read_csv(path)


def _write_audit(
    records: list[InsertionAuditRecord],
    path: str,
    trial_name: str,
    manifest_path: str,
) -> None:
    """Write audit trail as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "trial_name": trial_name,
        "manifest_path": manifest_path,
        "n_total": len(records),
        "n_success": sum(1 for r in records if r.status == "success"),
        "n_failed": sum(1 for r in records if r.status == "failed"),
        "records": [_audit_to_dict(r) for r in records],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Audit written to %s", path)


def _audit_to_dict(r: InsertionAuditRecord) -> dict[str, Any]:
    """Convert audit record to a JSON-safe dict."""
    d = asdict(r)
    # Convert tuples to lists for JSON
    for key in ("target_ijk", "target_world_mm"):
        if d[key] is not None:
            d[key] = list(d[key])
    return d
