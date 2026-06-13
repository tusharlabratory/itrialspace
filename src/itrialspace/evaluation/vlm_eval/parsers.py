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
Answer normalisation for generative VLM outputs.

Maps free-text model responses to canonical task labels.
Handles common variants, abbreviations, and verbose answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a raw model output."""

    label: str  # canonical label or "unparsed"
    status: str  # "ok" | "unparsed"
    raw: str  # original text


# ── Presence parsing ─────────────────────────────────────────────────────────

_YES_PATTERNS = re.compile(
    r"\b(yes|yeah|yep|present|contains|detected|visible|seen|found|positive)\b",
    re.IGNORECASE,
)
_NO_PATTERNS = re.compile(
    r"\b(no|nope|absent|none|not\s+present|not\s+detected|negative"
    r"|no\s+nodule|does\s+not\s+contain|without)\b",
    re.IGNORECASE,
)


def parse_presence(raw: str) -> ParseResult:
    """Parse a generative answer into present/absent."""
    text = raw.strip()
    if not text:
        return ParseResult(label="unparsed", status="unparsed", raw=raw)

    # Check for explicit "no" first (must come before "yes" to handle
    # "No, there is no nodule" correctly)
    if _NO_PATTERNS.search(text):
        # Make sure it's not a double-negative situation
        # e.g. "not absent" → present
        if re.search(r"\bnot\s+(absent|negative)\b", text, re.IGNORECASE):
            return ParseResult(label="present", status="ok", raw=raw)
        return ParseResult(label="absent", status="ok", raw=raw)

    if _YES_PATTERNS.search(text):
        return ParseResult(label="present", status="ok", raw=raw)

    return ParseResult(label="unparsed", status="unparsed", raw=raw)


# ── Lobe parsing ─────────────────────────────────────────────────────────────

_LOBE_MAP = {
    # Canonical labels (underscore form)
    "right_lung_upper_lobe": "right_lung_upper_lobe",
    "right_lung_middle_lobe": "right_lung_middle_lobe",
    "right_lung_lower_lobe": "right_lung_lower_lobe",
    "left_lung_upper_lobe": "left_lung_upper_lobe",
    "left_lung_lower_lobe": "left_lung_lower_lobe",
    # Human-readable forms
    "right upper lobe": "right_lung_upper_lobe",
    "right middle lobe": "right_lung_middle_lobe",
    "right lower lobe": "right_lung_lower_lobe",
    "left upper lobe": "left_lung_upper_lobe",
    "left lower lobe": "left_lung_lower_lobe",
    # Abbreviations
    "rul": "right_lung_upper_lobe",
    "rml": "right_lung_middle_lobe",
    "rll": "right_lung_lower_lobe",
    "lul": "left_lung_upper_lobe",
    "lll": "left_lung_lower_lobe",
}

# Ordered by specificity (check "right middle" before "right")
_LOBE_SEARCH_ORDER = [
    ("right middle lobe", "right_lung_middle_lobe"),
    ("right upper lobe", "right_lung_upper_lobe"),
    ("right lower lobe", "right_lung_lower_lobe"),
    ("left upper lobe", "left_lung_upper_lobe"),
    ("left lower lobe", "left_lung_lower_lobe"),
    ("right middle", "right_lung_middle_lobe"),
    ("right upper", "right_lung_upper_lobe"),
    ("right lower", "right_lung_lower_lobe"),
    ("left upper", "left_lung_upper_lobe"),
    ("left lower", "left_lung_lower_lobe"),
    ("rml", "right_lung_middle_lobe"),
    ("rul", "right_lung_upper_lobe"),
    ("rll", "right_lung_lower_lobe"),
    ("lul", "left_lung_upper_lobe"),
    ("lll", "left_lung_lower_lobe"),
]


def parse_lobe(raw: str) -> ParseResult:
    """Parse a generative answer into a canonical lobe label."""
    text = raw.strip()
    if not text:
        return ParseResult(label="unparsed", status="unparsed", raw=raw)

    text_lower = text.lower()

    # Direct match of canonical underscore form (e.g. "right_lung_lower_lobe")
    if text_lower.replace(" ", "_") in _LOBE_MAP:
        return ParseResult(label=_LOBE_MAP[text_lower.replace(" ", "_")], status="ok", raw=raw)

    # Exact match after normalisation (underscores → spaces)
    normalised = text_lower.replace("_", " ").strip()
    if normalised in _LOBE_MAP:
        return ParseResult(label=_LOBE_MAP[normalised], status="ok", raw=raw)

    # Substring search in order of specificity
    for pattern, canonical in _LOBE_SEARCH_ORDER:
        if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
            return ParseResult(label=canonical, status="ok", raw=raw)

    # Abbreviation as standalone word (case-insensitive)
    for abbrev in ["rul", "rml", "rll", "lul", "lll"]:
        if re.search(r"\b" + abbrev + r"\b", text_lower):
            return ParseResult(label=_LOBE_MAP[abbrev], status="ok", raw=raw)

    return ParseResult(label="unparsed", status="unparsed", raw=raw)


# ── Size bucket parsing ──────────────────────────────────────────────────────

_SIZE_PATTERNS = [
    # Order: most specific first
    (re.compile(r"<\s*5\s*mm|less\s+than\s+5|smaller\s+than\s+5|under\s+5", re.I), "<5mm"),
    (
        re.compile(
            r">\s*20\s*mm|larger\s+than\s+20|greater\s+than\s+20|over\s+20|above\s+20", re.I
        ),
        ">20mm",
    ),
    (re.compile(r"10\s*[-–—to]+\s*20\s*mm|between\s+10\s+and\s+20", re.I), "10-20mm"),
    (re.compile(r"5\s*[-–—to]+\s*10\s*mm|between\s+5\s+and\s+10", re.I), "5-10mm"),
]


def parse_size_bucket(raw: str) -> ParseResult:
    """Parse a generative answer into a canonical size bucket."""
    text = raw.strip()
    if not text:
        return ParseResult(label="unparsed", status="unparsed", raw=raw)

    for pattern, label in _SIZE_PATTERNS:
        if pattern.search(text):
            return ParseResult(label=label, status="ok", raw=raw)

    return ParseResult(label="unparsed", status="unparsed", raw=raw)


# ── Dispatcher ───────────────────────────────────────────────────────────────

_TASK_PARSERS = {
    "presence": parse_presence,
    "lobe": parse_lobe,
    "size": parse_size_bucket,
}


def parse_answer(task_name: str, raw: str) -> ParseResult:
    """Route to the correct parser for a given task name."""
    parser_fn = _TASK_PARSERS.get(task_name)
    if parser_fn is None:
        raise ValueError(f"No parser for task '{task_name}'")
    return parser_fn(raw)
