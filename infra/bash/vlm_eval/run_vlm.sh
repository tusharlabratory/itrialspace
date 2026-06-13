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
# infra/bash/vlm_eval/run_vlm.sh — VLM eval STEP 2/3: run ONE model across all
# 4 conditions × 3 tasks on a built eval set. GPU. Bash counterpart of
# infra/slurm/vlm_eval/run_vlm.sub.
#
# Writes <EVAL_DIR>/<MODEL>/<condition>/<task>_results.csv.
#
# Env knobs:
#   MODEL       biomedclip | llava_med | medgemma   (default: biomedclip)
#   EVAL_DIR    dir holding eval_dataset.csv for this model's profile
#               (default $ITRIALSPACE_OUTPUT_DIR/vlm_eval_demo/<profile>).
#               NOTE: MedGemma needs the lung_axial_medgemma eval set; BiomedCLIP
#               and LLaVA-Med need lung_axial. (MedGemma is gated → needs HF_TOKEN.)
#   CONDITIONS  optional subset, e.g. "plain bbox"   (default: all 4)
#   CASE_IDS    optional frozen split file (--case-ids)
#   GPU         pin to one GPU, e.g. GPU=0           (default: all visible)
# ============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

MODEL="${MODEL:-biomedclip}"
case "${MODEL}" in
    biomedclip|llava_med) DEF_PROFILE="lung_axial" ;;
    medgemma)             DEF_PROFILE="lung_axial_medgemma" ;;
    *) echo "ERROR: MODEL must be biomedclip|llava_med|medgemma (got '${MODEL}')."; exit 1 ;;
esac
EVAL_DIR="${EVAL_DIR:-${ITRIALSPACE_OUTPUT_DIR}/vlm_eval_demo/${DEF_PROFILE}}"
DATASET_CSV="${EVAL_DIR}/eval_dataset.csv"
OUTPUT_DIR="${EVAL_DIR}/${MODEL}"
[ -f "${DATASET_CSV}" ] || { echo "ERROR: ${DATASET_CSV} not found. Build it first (build_dataset.sh)."; exit 1; }

# MedGemma (torch 2.6) needs the cuDNN 9 libs on LD_LIBRARY_PATH.
if [ "${MODEL}" = "medgemma" ]; then
    export LD_LIBRARY_PATH="$(python3 -c 'import os,nvidia.cudnn;print(os.path.dirname(nvidia.cudnn.__file__)+"/lib")' 2>/dev/null):${LD_LIBRARY_PATH:-}"
fi
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="${GPU}"

# Log to terminal + file (so failures are visible after the fact). ITS_NO_LOG=1 disables.
if [ -z "${ITS_NO_LOG:-}" ]; then
    LOG="${ITS_LOG_DIR}/vlm_run_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG}") 2>&1
    echo "log: ${LOG}"
fi

EXTRA=()
[ -n "${CASE_IDS:-}" ] && EXTRA+=(--case-ids "${CASE_IDS}")
if [ -n "${CONDITIONS:-}" ]; then EXTRA+=(--image-condition ${CONDITIONS}); else EXTRA+=(--run-all-conditions); fi

echo "=== VLM run ${MODEL} — all conditions × {presence,lobe,size}  ($(date '+%F %T')) ==="
echo "  dataset: ${DATASET_CSV}"
echo "  output : ${OUTPUT_DIR}"
echo "  GPU    : ${CUDA_VISIBLE_DEVICES:-all visible}"

python3 -m itrialspace.evaluation.vlm_eval.runners.run_conditions \
    --model "${MODEL}" \
    --dataset-csv "${DATASET_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --tasks presence lobe size \
    "${EXTRA[@]}"

echo "Done. Results: ${OUTPUT_DIR}"
