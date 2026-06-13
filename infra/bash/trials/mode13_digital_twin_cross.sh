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

# Mode 13 — Digital Twin Cross: donor nodules placed in a different patient's
# anatomy. One manifest per host×donor dataset pair (same-dataset pairs skipped —
# those are modes 11/12). CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

SEED="${SEED:-42}"
MAX_HOST_PATIENTS="${MAX_HOST_PATIENTS:-5}"
MAX_DONOR_NODULES="${MAX_DONOR_NODULES:-5}"                 # sample (paper run: ~250)
DONOR_TRANSFER_MODE="${DONOR_TRANSFER_MODE:-single}"
PAIRING_POLICY="${PAIRING_POLICY:-one_to_one}"
PLACEMENT_STRATEGY="${PLACEMENT_STRATEGY:-profile_faithful_transfer}"
read -ra HOST_DATASETS  <<< "${HOST_DATASETS:-DLCS24}"      # space-separated
read -ra DONOR_DATASETS <<< "${DONOR_DATASETS:-LUNA25}"     # space-separated
BASE="${ITRIALSPACE_OUTPUT_DIR}/manifests/mode13_digital_twin_cross"

for HOST_DS in "${HOST_DATASETS[@]}"; do
    for DONOR_DS in "${DONOR_DATASETS[@]}"; do
        [ "${HOST_DS}" = "${DONOR_DS}" ] && { echo "  skip same-dataset pair: ${HOST_DS} x ${DONOR_DS}"; continue; }
        its_trial digital_twin_cross \
            --output-dir "${BASE}/${HOST_DS}_x_${DONOR_DS}" \
            --host-dataset "${HOST_DS}" --donor-dataset "${DONOR_DS}" \
            --max-host-patients "${MAX_HOST_PATIENTS}" --max-donor-nodules "${MAX_DONOR_NODULES}" \
            --donor-transfer-mode "${DONOR_TRANSFER_MODE}" --pairing-policy "${PAIRING_POLICY}" \
            --placement-strategy "${PLACEMENT_STRATEGY}" --seed "${SEED}"
    done
done
