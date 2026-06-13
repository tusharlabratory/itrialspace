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
iTrialSpace Mask Insertion Engine
=================================

Downstream module that takes an iTrialSpace CohortManifest and produces
host-space nodule masks by inserting donor nodule masks into host CT
geometry according to the manifest's insertion plan.

Submodules:
  placement        – percentile→voxel conversion, snapping, pleural margin, collision
  resample         – scipy-based resampling, scaling, binary cleanup
  resolver_bridge  – PathResolver integration for host/donor path resolution
  inserter         – main per-row and per-manifest orchestration
  cli              – command-line interface for HPC batch execution
"""

__version__ = "0.1.0"

from itrialspace.mask_inserter.inserter import insert_case, insert_manifest

__all__ = ["insert_manifest", "insert_case", "__version__"]
