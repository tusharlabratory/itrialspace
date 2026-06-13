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
iTrialSpace Retriever — interactive retrieval + visualization framework.

Three deployment modes:
    1. Full Web App:  Streamlit + FastAPI backend
    2. Headless:      FastAPI backend + Jupyter notebook client
    3. Library + CLI: Pure Python API + command-line tools

Quick start (library mode):
    from itrialspace.apps.retriever import RetrieverEngine
    engine = RetrieverEngine.from_defaults()
    results = engine.search(datasets=["DLCS24"], label=1, diameter_range=(5, 20))
    similar = engine.find_similar("DLCS24_n0001", k=10)
"""

__version__ = "0.1.0"

from itrialspace.apps.retriever.engine import RetrieverEngine
from itrialspace.apps.retriever.search import FacetedSearch, SearchFilters
from itrialspace.apps.retriever.similarity import SimilarityEngine
from itrialspace.apps.retriever.slicer import NIfTISlicer
