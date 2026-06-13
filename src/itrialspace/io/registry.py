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
DatasetRegistry — maps dataset names to CSV paths.
Reads from config/datasets.yaml or can be built programmatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Datasets permanently removed from the ecosystem.
_REMOVED_DATASETS = {"NLST3D"}


class DatasetRegistry:
    """
    Holds the mapping of dataset name → CSV path.

    Usage — from YAML config (recommended):
        registry = DatasetRegistry.from_yaml("/path/to/datasets.yaml")

    Usage — programmatic:
        registry = DatasetRegistry()
        registry.register("DLCS24", "/data/DLCS24_nodule_profiles.csv")
        registry.register("LUNA25", "/data/LUNA25_nodule_profiles.csv")

    Usage — auto-discover from a directory:
        registry = DatasetRegistry.from_directory("/data/profiles/")
        # Looks for files matching *_nodule_profiles.csv
    """

    # Known dataset names in canonical order
    KNOWN_DATASETS = ["DLCS24", "LUNA25", "LUNA16", "LUNGx", "LNDbv4", "NSCLCR", "IMDCT"]

    def __init__(self):
        self._entries: dict[str, str] = {}  # dataset → csv_path

    def register(self, dataset: str, csv_path: str) -> "DatasetRegistry":
        """Register a single dataset. Returns self for chaining."""
        if dataset in _REMOVED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset}' was removed due to quality issues and cannot be registered."
            )
        csv_path = str(Path(csv_path).expanduser())
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self._entries[dataset] = csv_path
        return self

    def register_if_exists(self, dataset: str, csv_path: str) -> "DatasetRegistry":
        """Register only if the file exists. Silent skip otherwise."""
        try:
            return self.register(dataset, csv_path)
        except FileNotFoundError:
            return self

    def get(self, dataset: str) -> Optional[str]:
        return self._entries.get(dataset)

    def datasets(self) -> list[str]:
        return list(self._entries.keys())

    def items(self):
        return self._entries.items()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        lines = [f"DatasetRegistry ({len(self)} datasets):"]
        for ds, path in self._entries.items():
            lines.append(f"  {ds:<10s}  {path}")
        return "\n".join(lines)

    # ── Class methods ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "DatasetRegistry":
        """Load from a YAML config file.

        When ``yaml_path`` is omitted, the file is discovered via
        ``itrialspace.config.settings`` (top-level ``configs/datasets.yaml`` →
        shipped ``datasets.example.yaml`` → bundled package default), and
        ``${ITRIALSPACE_DATA_DIR}`` style placeholders are expanded.
        """
        from itrialspace.config import settings

        _pkg_default = Path(__file__).resolve().parents[1] / "config" / "datasets.yaml"
        if yaml_path is None:
            yaml_path = settings.find_config("datasets.yaml", package_default=_pkg_default)

        config = settings.load_yaml(yaml_path)

        registry = cls()
        datasets = config.get("datasets", config)  # support flat or nested

        for name, entry in datasets.items():
            if isinstance(entry, dict):
                path = entry.get("csv_path") or entry.get("path")
            else:
                path = str(entry)
            if path:
                registry.register_if_exists(name, path)

        return registry

    @classmethod
    def from_directory(
        cls, directory: str, pattern: str = "*_nodule_profiles.csv"
    ) -> "DatasetRegistry":
        """
        Auto-discover CSVs in a directory.
        Filename must start with the dataset name, e.g. DLCS24_nodule_profiles.csv
        """
        import glob

        registry = cls()
        paths = sorted(glob.glob(os.path.join(directory, pattern)))
        for path in paths:
            fname = os.path.basename(path)
            # Extract dataset name as the part before '_nodule_profiles.csv'
            if "_nodule_profiles" in fname:
                dataset = fname.split("_nodule_profiles")[0]
                if dataset in cls.KNOWN_DATASETS:
                    registry.register_if_exists(dataset, path)
        return registry

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> "DatasetRegistry":
        """Build from a plain dict: {'DLCS24': '/path/...', ...}"""
        registry = cls()
        for name, path in mapping.items():
            registry.register_if_exists(name, path)
        return registry
