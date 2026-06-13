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
# infra/bash/vlm_eval/analyze.sh — VLM eval STEP 3/3: analyse results into
# accuracy / Δ-vs-plain / breakdowns / confusion / bootstrap CIs / McNemar /
# publication figures / report.md. CPU. Bash counterpart of
# infra/slurm/vlm_eval/analyze.sub. Auto-discovers whatever models/conditions
# /tasks are present under RESULTS.
#
# Env knobs:
#   RESULTS  result root to analyse (default $ITRIALSPACE_OUTPUT_DIR/vlm_eval_demo)
#   SPLIT    split name under <RESULTS>/splits or a file path; "" = all cases
#            (default "" for the demo; use release_v1_full on the full dataset)
#   OUT      output dir (default <RESULTS>/eval_analysis)
#   NBOOT    bootstrap resamples (default 1000; 0 to skip)
# ============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

RESULTS="${RESULTS:-${ITRIALSPACE_OUTPUT_DIR}/vlm_eval_demo}"
SPLIT="${SPLIT-}"                 # default empty = all cases (note: '-' not ':-')
NBOOT="${NBOOT:-1000}"
OUT="${OUT:-${RESULTS}/eval_analysis}"

# Log to terminal + file (ITS_NO_LOG=1 disables).
if [ -z "${ITS_NO_LOG:-}" ]; then
    LOG="${ITS_LOG_DIR}/vlm_analyze_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG}") 2>&1
    echo "log: ${LOG}"
fi

echo "=== VLM analyse — results=${RESULTS}  split=${SPLIT:-<all>}  out=${OUT}  ($(date '+%F %T')) ==="

SPLIT_ARG=()
[ -n "${SPLIT}" ] && SPLIT_ARG=(--split "${SPLIT}")

python3 -m itrialspace.evaluation.vlm_eval.eval_analysis \
    --results "${RESULTS}" \
    "${SPLIT_ARG[@]}" \
    --out "${OUT}" \
    --n-boot "${NBOOT}"

echo "Done. Report: ${OUT}/report.md"
