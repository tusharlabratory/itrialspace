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
NoduleIndex — the central object.
Loads all datasets, merges into a single DataFrame, exposes query/stats API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from itrialspace.core.schema import DATASET_FLAGS
from itrialspace.io.loader import DatasetLoader
from itrialspace.io.registry import DatasetRegistry

if TYPE_CHECKING:
    from itrialspace.query.query_api import NoduleQuery


class NoduleIndex:
    """
    A merged, queryable index over all iTrialSpace nodule profile datasets.

    The backing store is a single pandas DataFrame containing the 53 core
    pipeline columns plus `label`, `patient_id`, `annotation_id`, `dataset`,
    and all dataset-specific extra columns (NaN where not applicable).

    Usage:
        # From a registry (recommended)
        registry = DatasetRegistry.from_directory("/data/profiles/")
        idx = NoduleIndex.from_registry(registry)

        # From a single CSV
        idx = NoduleIndex.from_csv("DLCS24", "/data/DLCS24_nodule_profiles.csv")

        # From a pre-built DataFrame
        idx = NoduleIndex(df)

        # Query
        results = idx.query().label(1).lobe("right_lung_upper_lobe").fetch()

        # Stats
        print(idx.stats())
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy().reset_index(drop=True)

    @classmethod
    def from_csv(cls, dataset: str, csv_path: str) -> "NoduleIndex":
        """Load a single dataset CSV."""
        df = DatasetLoader.load_to_dataframe(dataset, csv_path)
        return cls(df)

    @classmethod
    def from_registry(cls, registry: DatasetRegistry, verbose: bool = True) -> "NoduleIndex":
        """Load all registered datasets and merge into one index."""
        frames = []
        for dataset, path in registry.items():
            if verbose:
                print(f"  Loading {dataset:<10s} ← {path}")
            try:
                df = DatasetLoader.load_to_dataframe(dataset, path)
                frames.append(df)
                if verbose:
                    mal = df["label"].sum() if df["label"].notna().any() else "?"
                    print(f"            {len(df):>5d} rows | label={mal}")
            except Exception as e:
                print(f"  ⚠ {dataset}: failed ({e})")

        if not frames:
            raise RuntimeError("No datasets loaded.")

        merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        if verbose:
            print(f"\n  Total: {len(merged):,} nodules across {len(frames)} datasets")
        return cls(merged)

    @classmethod
    def from_dict(cls, mapping: dict[str, str], verbose: bool = True) -> "NoduleIndex":
        """Shortcut: load from dict {'DLCS24': '/path/...', ...}"""
        registry = DatasetRegistry.from_dict(mapping)
        return cls.from_registry(registry, verbose=verbose)

    # ── Core properties ────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Access the underlying DataFrame directly."""
        return self._df

    @property
    def datasets(self) -> list[str]:
        return sorted(self._df["dataset"].unique().tolist())

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        n = len(self._df)
        ds = ", ".join(self.datasets)
        labelled = self._df["label"].notna().sum()
        return f"NoduleIndex({n:,} nodules | {ds} | {labelled:,} labelled)"

    # ── Query entry point ──────────────────────────────────────────────────────

    def query(self) -> "NoduleQuery":
        from itrialspace.query.query_api import NoduleQuery

        return NoduleQuery(self._df)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> pd.DataFrame:
        """
        Per-dataset summary table: n, malignancy rate, size stats, lobe coverage.
        """
        rows = []
        for ds in sorted(self._df["dataset"].unique()):
            sub = self._df[self._df["dataset"] == ds]
            flags = DATASET_FLAGS.get(ds, {})

            labelled = sub["label"].notna()
            mal_rate = (
                f"{sub.loc[labelled, 'label'].mean() * 100:.1f}%" if labelled.any() else "N/A"
            )
            mal_n = int(sub["label"].sum()) if labelled.any() else 0

            diam = sub["nodule_mean_diam_mm"]
            rows.append(
                {
                    "dataset": ds,
                    "n": len(sub),
                    "patients": sub["patient_id"].nunique(),
                    "labelled": int(labelled.sum()),
                    "malignant": mal_n,
                    "mal_rate": mal_rate,
                    "diam_median": round(diam.median(), 1),
                    "diam_mean": round(diam.mean(), 1),
                    "diam_max": round(diam.max(), 1),
                    "all_malignant": flags.get("all_malignant", False),
                    "all_unlabelled": flags.get("all_unlabelled", False),
                    "label_is_soft": flags.get("label_is_soft", False),
                    "notes": flags.get("notes", ""),
                }
            )
        return pd.DataFrame(rows).set_index("dataset")

    def lobe_stats(self) -> pd.DataFrame:
        """Lobe × dataset malignancy rates."""
        sub = self._df[self._df["label"].notna()]
        return (
            sub.groupby(["dataset", "lobe_name"])["label"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "malignant", "count": "total", "mean": "rate"})
            .assign(rate=lambda d: (d["rate"] * 100).round(1))
        )

    def size_stats(self) -> pd.DataFrame:
        """Size bucket × dataset malignancy rates."""
        bins = [0, 5, 10, 15, 20, 30, 200]
        lbls = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-30mm", ">30mm"]
        df = self._df.copy()
        df["size_bucket"] = pd.cut(df["nodule_mean_diam_mm"], bins=bins, labels=lbls)
        return (
            df.groupby(["dataset", "size_bucket"])["label"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "malignant", "count": "total", "mean": "rate"})
            .assign(rate=lambda d: (d["rate"] * 100).round(1))
        )

    # ── Flags ──────────────────────────────────────────────────────────────────

    def dataset_flags(self, dataset: str) -> dict:
        """Return known data quality flags for a dataset."""
        flags = DATASET_FLAGS.get(dataset, {})
        if not flags:
            return {"warning": f"No flags registered for '{dataset}'"}
        return flags

    # ── Convenience filters ───────────────────────────────────────────────────

    def labelled(self) -> "NoduleIndex":
        """Return a new NoduleIndex with only rows that have a label."""
        return NoduleIndex(self._df[self._df["label"].notna()])

    def malignant(self) -> "NoduleIndex":
        """Return only malignant rows."""
        return NoduleIndex(self._df[self._df["label"] == 1])

    def benign(self) -> "NoduleIndex":
        """Return only benign rows."""
        return NoduleIndex(self._df[self._df["label"] == 0])

    def subset(self, datasets: list[str]) -> "NoduleIndex":
        """Return only rows from specified datasets."""
        return NoduleIndex(self._df[self._df["dataset"].isin(datasets)])
