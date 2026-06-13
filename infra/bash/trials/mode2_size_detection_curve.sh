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

# Mode 2 — Size Detection Curve: cases per size bucket (FROC). CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

N_PER_BUCKET="${N_PER_BUCKET:-5}"  # cases per size bucket (paper run: 100)
LABEL="${LABEL:-1}"                # 1 = malignant
TEMPLATE="${TEMPLATE:-NLST}"
SEED="${SEED:-42}"

its_trial size_detection_curve \
    --output-dir "${ITRIALSPACE_OUTPUT_DIR}/manifests/mode2_size_detection_curve" \
    --n-per-bucket "${N_PER_BUCKET}" --label "${LABEL}" \
    --template "${TEMPLATE}" --seed "${SEED}"
