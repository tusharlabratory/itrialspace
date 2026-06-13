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

# Mode 11 — Digital Twin Isolation: one case per native nodule in its own CT. CPU.
# One manifest per dataset. Override DATASETS (space-separated) to widen coverage;
# the paper run uses all 7 with --all-patients.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

SEED="${SEED:-42}"
MAX_PATIENTS="${MAX_PATIENTS:-5}"          # paper run uses --all-patients (set MAX_PATIENTS=all)
read -ra DATASETS <<< "${DATASETS:-DLCS24}"   # space-separated; real run: DLCS24 LUNA25 LUNA16 LUNGx LNDbv4 NSCLCR IMDCT
BASE="${ITRIALSPACE_OUTPUT_DIR}/manifests/mode11_digital_twin_isolation"

for DS in "${DATASETS[@]}"; do
    if [ "${MAX_PATIENTS}" = "all" ]; then PAT=(--all-patients); else PAT=(--max-patients "${MAX_PATIENTS}"); fi
    its_trial digital_twin_isolation \
        --output-dir "${BASE}/${DS}" --dataset "${DS}" \
        "${PAT[@]}" --max-nodules-per-patient 1 --seed "${SEED}"
done
