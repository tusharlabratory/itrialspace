#!/usr/bin/env bash
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

# Mode 3 — Location Sensitivity: cases per lobe at a fixed size band. CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

N_PER_LOBE="${N_PER_LOBE:-5}"      # cases per lobe (paper run: 100)
LABEL="${LABEL:-1}"                # 1 = malignant
DIAM_MIN="${DIAM_MIN:-6.0}"        # min diameter (mm)
DIAM_MAX="${DIAM_MAX:-15.0}"       # max diameter (mm)
TEMPLATE="${TEMPLATE:-NLST}"
SEED="${SEED:-42}"

its_trial location_sensitivity \
    --output-dir "${ITRIALSPACE_OUTPUT_DIR}/manifests/mode3_location_sensitivity" \
    --n-per-lobe "${N_PER_LOBE}" --label "${LABEL}" \
    --diam-min "${DIAM_MIN}" --diam-max "${DIAM_MAX}" \
    --template "${TEMPLATE}" --seed "${SEED}"
