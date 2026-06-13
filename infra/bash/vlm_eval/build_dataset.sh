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

# ============================================================================
# infra/bash/vlm_eval/build_dataset.sh — VLM eval STEP 1/3: build the 2-D eval
# dataset (slices + overlays + ground truth). CPU. Bash counterpart of
# infra/slurm/vlm_eval/build_dataset.sub.
#
# Writes <EVAL_DIR>/{eval_dataset.csv, slices/…}.
#
# Env knobs (all optional — defaults = the small DEMO):
#   VLM_SET     synthetic | real                     (default: synthetic)
#   PROFILE     lung_axial | lung_axial_medgemma     (default: lung_axial)
#               BiomedCLIP/LLaVA-Med → lung_axial ; MedGemma → lung_axial_medgemma.
#   EVAL_DIR    output dir (default: $ITRIALSPACE_OUTPUT_DIR/vlm_eval_demo[_real]/$PROFILE)
#   WORKERS     parallel workers (default 8)
#   synthetic:  VLM_MODES  trial modes to include (default "1 2 3"; full "1 2 … 13")
#               OUT_BASE   holds generated_cts/manifests/inserted_masks (default $ITRIALSPACE_OUTPUT_DIR)
#   real:       VLM_DATASETS (default "DLCS24 LUNA25"; full = all 7)
#               VLM_MAX     TOTAL case cap across all datasets (default 40; VLM_MAX="" = ALL).
#                           NOTE: it is a GLOBAL total processed in --datasets order, so a small
#                           cap fills from the first dataset only (40 -> 40 DLCS24, no LUNA25).
#                           For a balanced small sample, build one dataset at a time, or use VLM_MAX="".
#               DATA_BASE   holds raw_ct/ profiles/ masks/ (default $ITRIALSPACE_DATA_DIR)
# ============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

VLM_SET="${VLM_SET:-synthetic}"
PROFILE="${PROFILE:-lung_axial}"
OUT_BASE="${OUT_BASE:-${ITRIALSPACE_OUTPUT_DIR}}"
DATA_BASE="${DATA_BASE:-${ITRIALSPACE_DATA_DIR}}"

# Log to terminal + file (ITS_NO_LOG=1 disables).
if [ -z "${ITS_NO_LOG:-}" ]; then
    LOG="${ITS_LOG_DIR}/vlm_build_${VLM_SET}_${PROFILE}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG}") 2>&1
    echo "log: ${LOG}"
fi

if [ "${VLM_SET}" = "synthetic" ]; then
    MODES="${VLM_MODES:-1 2 3}"
    EVAL_DIR="${EVAL_DIR:-${OUT_BASE}/vlm_eval_demo/${PROFILE}}"
    mkdir -p "${EVAL_DIR}"
    echo "=== VLM build (synthetic) — modes [${MODES}], profile ${PROFILE} -> ${EVAL_DIR} ==="
    MANIFESTS=()
    for m in ${MODES}; do
        while IFS= read -r f; do MANIFESTS+=("$f"); done \
            < <(find "${OUT_BASE}/manifests/mode${m}_"* -name '*.csv' 2>/dev/null)
    done
    [ "${#MANIFESTS[@]}" -gt 0 ] || { echo "ERROR: no manifests for modes [${MODES}] under ${OUT_BASE}/manifests."; exit 1; }
    echo "  manifests: ${#MANIFESTS[@]}"
    python3 -m itrialspace.evaluation.vlm_eval.build_dataset \
        --manifest "${MANIFESTS[@]}" \
        --output-dir "${EVAL_DIR}" \
        --ct-base "${OUT_BASE}/generated_cts" \
        --mask-base "${OUT_BASE}/inserted_masks" \
        --profile "${PROFILE}" \
        --overlays \
        --workers "${WORKERS:-8}"

elif [ "${VLM_SET}" = "real" ]; then
    DATASETS="${VLM_DATASETS:-DLCS24 LUNA25}"
    EVAL_DIR="${EVAL_DIR:-${OUT_BASE}/vlm_eval_demo_real/${PROFILE}}"
    mkdir -p "${EVAL_DIR}"
    MAX_ARG=()
    if   [ -z "${VLM_MAX+x}" ]; then MAX_ARG=(--max-cases 40)        # unset -> demo 40
    elif [ -n "${VLM_MAX}" ];   then MAX_ARG=(--max-cases "${VLM_MAX}"); fi   # ""  -> ALL
    echo "=== VLM build (real) — datasets [${DATASETS}], profile ${PROFILE} -> ${EVAL_DIR} ==="
    python3 -m itrialspace.evaluation.vlm_eval.build_real_dataset \
        --data-base "${DATA_BASE}" \
        --output-dir "${EVAL_DIR}" \
        --datasets ${DATASETS} \
        --profile "${PROFILE}" \
        "${MAX_ARG[@]}" \
        --overlays
else
    echo "ERROR: VLM_SET must be 'synthetic' or 'real' (got '${VLM_SET}')."; exit 1
fi

echo "Done. Eval dataset: ${EVAL_DIR}/eval_dataset.csv"
echo "Next: MODEL=biomedclip EVAL_DIR=${EVAL_DIR} bash infra/bash/vlm_eval/run_vlm.sh"
