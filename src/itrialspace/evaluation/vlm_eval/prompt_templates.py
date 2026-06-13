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
Prompt templates for VLM evaluation tasks.

Each task defines a set of text prompts and a mapping from prompt labels
to ground-truth column values in the iTrialSpace manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskDefinition:
    """A single evaluation task with its text prompts and label mapping."""

    name: str
    prompts: Dict[str, str]  # label -> text prompt
    manifest_column: str  # column name in the eval dataset CSV
    description: str = ""

    @property
    def labels(self) -> List[str]:
        return list(self.prompts.keys())

    @property
    def texts(self) -> List[str]:
        return list(self.prompts.values())


# ── Task 1: Nodule Presence ──────────────────────────────────────────────────

PRESENCE_TASK = TaskDefinition(
    name="presence",
    prompts={
        "present": "A CT scan containing a pulmonary nodule.",
        "absent": "A CT scan without a pulmonary nodule.",
    },
    manifest_column="ground_truth_presence",
    description="Binary classification: does the CT slice contain a nodule?",
)

# ── Task 2: Lobe Localisation ────────────────────────────────────────────────

LOBE_TASK = TaskDefinition(
    name="lobe",
    prompts={
        "right_lung_upper_lobe": "A pulmonary nodule in the right upper lobe.",
        "right_lung_middle_lobe": "A pulmonary nodule in the right middle lobe.",
        "right_lung_lower_lobe": "A pulmonary nodule in the right lower lobe.",
        "left_lung_upper_lobe": "A pulmonary nodule in the left upper lobe.",
        "left_lung_lower_lobe": "A pulmonary nodule in the left lower lobe.",
    },
    manifest_column="ground_truth_lobe",
    description="5-class lobe localisation from text-image similarity.",
)

# ── Task 3: Size Bucket ─────────────────────────────────────────────────────

SIZE_TASK = TaskDefinition(
    name="size",
    prompts={
        "<5mm": "A pulmonary nodule smaller than 5 millimeters.",
        "5-10mm": "A pulmonary nodule between 5 and 10 millimeters.",
        "10-20mm": "A pulmonary nodule between 10 and 20 millimeters.",
        ">20mm": "A pulmonary nodule larger than 20 millimeters.",
    },
    manifest_column="ground_truth_size_bucket",
    description="4-class size bucket classification from text-image similarity.",
)

# ── Registry ─────────────────────────────────────────────────────────────────

ALL_TASKS: Dict[str, TaskDefinition] = {
    "presence": PRESENCE_TASK,
    "lobe": LOBE_TASK,
    "size": SIZE_TASK,
}


def get_task(name: str) -> TaskDefinition:
    """Retrieve a task definition by name."""
    if name not in ALL_TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[name]


# ── Size-bucket helper ───────────────────────────────────────────────────────


def diameter_to_size_bucket(diam_mm: float) -> str:
    """Map effective diameter (mm) to a size bucket label."""
    if diam_mm < 5.0:
        return "<5mm"
    elif diam_mm < 10.0:
        return "5-10mm"
    elif diam_mm < 20.0:
        return "10-20mm"
    else:
        return ">20mm"


# ── Lobe normalisation ───────────────────────────────────────────────────────

_LOBE_ALIASES = {
    "right upper lobe": "right_lung_upper_lobe",
    "right middle lobe": "right_lung_middle_lobe",
    "right lower lobe": "right_lung_lower_lobe",
    "left upper lobe": "left_lung_upper_lobe",
    "left lower lobe": "left_lung_lower_lobe",
    "rul": "right_lung_upper_lobe",
    "rml": "right_lung_middle_lobe",
    "rll": "right_lung_lower_lobe",
    "lul": "left_lung_upper_lobe",
    "lll": "left_lung_lower_lobe",
}


def normalise_lobe(lobe: str) -> str:
    """Normalise a lobe name to the canonical form used in prompt labels."""
    lobe_lower = str(lobe).strip().lower().replace("_", " ")
    # Direct match after replacing underscores
    canonical = lobe_lower.replace(" ", "_")
    if canonical in LOBE_TASK.prompts:
        return canonical
    # Alias lookup
    if lobe_lower in _LOBE_ALIASES:
        return _LOBE_ALIASES[lobe_lower]
    # Partial match
    for alias, canon in _LOBE_ALIASES.items():
        if alias in lobe_lower:
            return canon
    return str(lobe)  # return as-is if no match


# ── Generative prompts (for instruction-following VLMs) ──────────────────────
# These are used by generative models like LLaVA-Med that produce free text
# rather than computing image-text similarity scores.

GENERATIVE_PROMPTS: Dict[str, str] = {
    "presence": (
        "Does this CT slice contain a pulmonary nodule? " "Answer using only one word: Yes or No."
    ),
    "lobe": (
        "Which lung lobe contains the nodule? "
        "Answer using only one option: "
        "Right upper lobe, Right middle lobe, Right lower lobe, "
        "Left upper lobe, Left lower lobe."
    ),
    "size": (
        "What is the approximate diameter category of the nodule? "
        "Answer using only one option: "
        "<5 mm, 5-10 mm, 10-20 mm, >20 mm."
    ),
}


def get_generative_prompt(task_name: str) -> str:
    """Get the generative (instruction-following) prompt for a task."""
    if task_name not in GENERATIVE_PROMPTS:
        raise ValueError(
            f"No generative prompt for task '{task_name}'. "
            f"Available: {list(GENERATIVE_PROMPTS.keys())}"
        )
    return GENERATIVE_PROMPTS[task_name]
