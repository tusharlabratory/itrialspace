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

# Mode 12 — Digital Twin Complete: one case per CT with ALL its native nodules. CPU.
# One manifest per dataset. Override DATASETS (space-separated) to widen coverage.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

SEED="${SEED:-42}"
MAX_PATIENTS="${MAX_PATIENTS:-5}"          # set MAX_PATIENTS=all for the paper run
read -ra DATASETS <<< "${DATASETS:-DLCS24}"   # space-separated; real run: all 7
BASE="${ITRIALSPACE_OUTPUT_DIR}/manifests/mode12_digital_twin_complete"

for DS in "${DATASETS[@]}"; do
    if [ "${MAX_PATIENTS}" = "all" ]; then PAT=(--all-patients); else PAT=(--max-patients "${MAX_PATIENTS}"); fi
    its_trial digital_twin_complete \
        --output-dir "${BASE}/${DS}" --dataset "${DS}" \
        "${PAT[@]}" --seed "${SEED}"
done
