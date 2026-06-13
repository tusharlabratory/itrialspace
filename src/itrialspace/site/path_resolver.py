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
Path resolution — turns relative CSV paths and annotation IDs
into absolute server paths for CT scans, nodule masks, organ
segmentations, and refined segmentations.

Configuration is loaded from config/paths.yaml. The unified
iTrialSpace layout uses a single base_dir with standardised
subdirectories:

    base_dir/
    ├── raw_ct/{DATASET}/          CT scans
    ├── masks/{DATASET}/
    │   ├── nodule_seg/            Individual nodule masks
    │   ├── combined_seg/          Combined patient masks
    │   ├── organ_seg/             VISTA3D organ segmentations (raw, deprecated)
    │   └── refined_seg/           Refined organ + body segmentations (default)
    ├── profiles/                  Nodule profile CSVs
    └── meta/                      Dataset metadata CSVs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from itrialspace.config import settings

_DEFAULT_PATHS_YAML = Path(__file__).parent / "config" / "paths.yaml"


class PathResolver:
    """Config-driven path resolver for iTrialSpace datasets.

    Loads per-dataset path layouts from a YAML config file.
    All paths resolve relative to a single base_dir.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ):
        """
        Args:
            config_path: Path to paths.yaml. Defaults to bundled config.
            base_dir: Override the global base_dir from config.
        """
        # Resolve the config file: explicit arg → top-level configs/ → package default.
        if config_path:
            path = Path(config_path)
        else:
            path = settings.find_config("paths.yaml", package_default=_DEFAULT_PATHS_YAML)

        # load_yaml expands ${ITRIALSPACE_DATA_DIR} and other ${VAR} placeholders.
        cfg = settings.load_yaml(path)

        self._base_dir = base_dir or cfg.get("base_dir") or str(settings.data_dir())
        self._datasets: dict[str, dict] = cfg.get("datasets", {})

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def available_datasets(self) -> list[str]:
        return list(self._datasets.keys())

    def _get_ds(self, dataset: str) -> dict:
        if dataset not in self._datasets:
            raise KeyError(
                f"Unknown dataset '{dataset}'. "
                f"Available: {self.available_datasets}. "
                f"Add it to config/paths.yaml."
            )
        return self._datasets[dataset]

    # ── Individual resolvers ──────────────────────────────────────────────────

    def resolve_ct_path(self, dataset: str, ct_path_relative: str) -> str:
        """Resolve ct_path column value to an absolute path.

        ct_path_relative is e.g. "DLCS24/DLCS_0001.nii.gz",
        resolved as base_dir/raw_ct/{ct_path_relative}.
        """
        return os.path.join(self._base_dir, "raw_ct", ct_path_relative)

    def resolve_nodule_mask_path(
        self,
        dataset: str,
        annotation_id: str,
        ct_id: Optional[str] = None,
    ) -> str:
        """Individual nodule mask.

        Some datasets (e.g. IMDCT) have one nodule per CT and store masks
        keyed by ct_id rather than annotation_id. Set ``nodule_seg_use_ct_id: true``
        in the dataset config to activate this.
        """
        ds = self._get_ds(dataset)
        pattern = ds.get("nodule_seg_pattern", "{annotation_id}.nii.gz")
        if ds.get("nodule_seg_use_ct_id", False) and ct_id is not None:
            filename = pattern.format(ct_id=ct_id, annotation_id=annotation_id)
        else:
            filename = pattern.format(annotation_id=annotation_id)
        return os.path.join(self._base_dir, "masks", dataset, "nodule_seg", filename)

    def resolve_organ_seg_path(self, dataset: str, ct_id: str) -> str:
        """Organ + body segmentation (refined_seg preferred over organ_seg)."""
        ds = self._get_ds(dataset)
        pattern = ds.get("organ_seg_pattern", "{ct_id}_seg.nii.gz")
        filename = pattern.format(ct_id=ct_id)
        return os.path.join(self._base_dir, "masks", dataset, "refined_seg", filename)

    def resolve_combined_mask_path(self, dataset: str, ct_id: str) -> str:
        """Combined (all nodules in one file) segmentation mask."""
        ds = self._get_ds(dataset)
        pattern = ds.get("combined_mask_pattern", "{ct_id}_mask.nii.gz")
        filename = pattern.format(ct_id=ct_id)
        return os.path.join(self._base_dir, "masks", dataset, "combined_seg", filename)

    def resolve_refined_seg_path(self, dataset: str, ct_id: str) -> str:
        """Refined segmentation."""
        return os.path.join(
            self._base_dir,
            "masks",
            dataset,
            "refined_seg",
            f"{ct_id}_seg.nii.gz",
        )

    def resolve_profile_csv_path(self, dataset: str) -> str:
        """Nodule profile CSV for a dataset."""
        ds = self._get_ds(dataset)
        csv_name = ds.get("profile_csv", f"{dataset}_nodule_profiles.csv")
        return os.path.join(self._base_dir, "profiles", csv_name)

    def resolve_meta_csv_path(self, dataset: str) -> str:
        """Dataset metadata CSV."""
        ds = self._get_ds(dataset)
        csv_name = ds.get("meta_csv", "")
        if not csv_name:
            return ""
        return os.path.join(self._base_dir, "meta", csv_name)

    # ── Batch resolver ────────────────────────────────────────────────────────

    @staticmethod
    def extract_ct_id(ct_path: str) -> str:
        """Extract ct_id from a ct_path for organ seg / mask lookup.

        E.g. "DLCS24/DLCS_0001.nii.gz" → "DLCS_0001"
             "LUNA25/1.2.840...nii.gz" → "1.2.840..."
        """
        basename = os.path.basename(ct_path)
        if basename.endswith(".nii.gz"):
            return basename[:-7]
        if basename.endswith(".nii"):
            return basename[:-4]
        return basename

    def resolve_all_paths(self, row: pd.Series) -> dict[str, str]:
        """Resolve all paths from a NoduleIndex DataFrame row.

        Returns dict with keys:
            ct_path, nodule_mask_path, organ_seg_path,
            combined_mask_path, refined_seg_path
        """
        dataset = row["dataset"]
        ct_rel = row["ct_path"]
        annotation_id = row["annotation_id"]
        ct_id = self.extract_ct_id(ct_rel)

        return {
            "ct_path": self.resolve_ct_path(dataset, ct_rel),
            "nodule_mask_path": self.resolve_nodule_mask_path(dataset, annotation_id, ct_id=ct_id),
            "organ_seg_path": self.resolve_organ_seg_path(dataset, ct_id),
            "combined_mask_path": self.resolve_combined_mask_path(dataset, ct_id),
            "refined_seg_path": self.resolve_refined_seg_path(dataset, ct_id),
        }
