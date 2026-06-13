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

# Mode 7 — Bootstrap Confidence: resampled replicate cohorts for CIs. CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

N_CASES="${N_CASES:-5}"            # cases per replicate (paper run: 200)
PREVALENCE="${PREVALENCE:-0.05}"
N_BOOTSTRAP="${N_BOOTSTRAP:-3}"    # number of replicates (paper run: 20)
TEMPLATE="${TEMPLATE:-NLST}"
SEED="${SEED:-42}"

its_trial bootstrap_confidence \
    --output-dir "${ITRIALSPACE_OUTPUT_DIR}/manifests/mode7_bootstrap_confidence" \
    --n-cases "${N_CASES}" --prevalence "${PREVALENCE}" \
    --n-bootstrap "${N_BOOTSTRAP}" --template "${TEMPLATE}" --seed "${SEED}"
