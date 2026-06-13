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

"""Shared skip-existing logic for VLM inference runners.

When ``--skip-existing`` is enabled, each ``run_task`` call checks whether a
results CSV from a previous run already exists.  Rows whose ``(case_id,
image_path)`` pair matches an entry in the current eval dataset are reused;
only genuinely new or changed rows are sent through the model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_cached_results(
    output_dir: str,
    task_name: str,
    current_df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """Load cached inference results and partition the dataset.

    Parameters
    ----------
    output_dir : str
        Directory containing ``{task_name}_results.csv`` from a prior run.
    task_name : str
        Task name (``presence``, ``lobe``, ``size``).
    current_df : pd.DataFrame
        Current eval dataset (already condition-remapped by run_conditions).

    Returns
    -------
    (cached_df, todo_df)
        *cached_df*: rows from the existing results that are still valid
        (``None`` if no cache file found).
        *todo_df*: rows from *current_df* that need fresh inference.
    """
    csv_path = os.path.join(output_dir, f"{task_name}_results.csv")

    if not os.path.isfile(csv_path):
        logger.info("skip-existing: no cached results for task '%s'", task_name)
        return None, current_df

    existing = pd.read_csv(csv_path)
    if existing.empty:
        return None, current_df

    # Build lookup of (case_id, image_path) from the cached results
    cached_keys = set(
        zip(
            existing["case_id"].astype(str),
            existing["image_path"].astype(str),
        )
    )

    # Determine which rows in the current dataset already have results
    current_keys = list(
        zip(
            current_df["case_id"].astype(str),
            current_df["png_path"].astype(str),
        )
    )
    is_cached = pd.Series([k in cached_keys for k in current_keys], index=current_df.index)
    todo_df = current_df[~is_cached]

    # Retain only cached rows whose key is still in the current dataset
    current_key_set = set(current_keys)
    existing_keys = list(
        zip(
            existing["case_id"].astype(str),
            existing["image_path"].astype(str),
        )
    )
    valid_mask = pd.Series([k in current_key_set for k in existing_keys])
    cached_df = existing[valid_mask.values].copy()

    n_reuse = len(cached_df)
    n_todo = len(todo_df)
    logger.info(
        "skip-existing [%s]: reusing %d cached, %d to infer (%d total)",
        task_name,
        n_reuse,
        n_todo,
        len(current_df),
    )

    return cached_df, todo_df


def merge_results(
    cached_df: Optional[pd.DataFrame],
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cached rows with freshly inferred rows."""
    if cached_df is not None and not cached_df.empty:
        return pd.concat([cached_df, new_df], ignore_index=True)
    return new_df
