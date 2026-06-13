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
# infra/bash/run_sweep.sh — run ONE stage across many modes CONCURRENTLY, with a
# concurrency cap. The single-machine analogue of submitting many SLURM jobs at
# once: instead of waiting for mode 1 before mode 2, up to JOBS modes run in
# parallel. Each mode runs via run_pipeline.sh, so each still gets its own log.
#
#   infra/bash/run_sweep.sh <stage> [mode ...]
#     stage     trials | insert | synth | all
#     mode ...  modes to run (default: 1 2 … 13)
#
# Env knobs:
#   JOBS=N        max concurrent modes. Default: 4 for CPU stages (trials/insert),
#                 or the number of visible GPUs for synth/all.
#   GPUS="0 1 2"  GPUs to round-robin for synth/all (default: all visible). Each
#                 concurrent mode is pinned to one GPU via CUDA_VISIBLE_DEVICES.
#
# Notes:
#   * On a busy host keep JOBS modest — insertion already uses --n-jobs workers
#     per manifest, so total load ≈ JOBS × N_JOBS.
#   * Terminal output from concurrent modes interleaves; the per-mode log files
#     under $ITRIALSPACE_OUTPUT_DIR/logs/ are the clean per-mode record.
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/bash/env.sh
source "${HERE}/env.sh"

STAGE="${1:?usage: run_sweep.sh <trials|insert|synth|all> [mode ...]}"; shift || true
MODES=("$@"); [ "${#MODES[@]}" -eq 0 ] && MODES=($(seq 1 13))

# GPUs to spread synth/all across.
read -ra GPU_ARR <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}"
[ "${#GPU_ARR[@]}" -eq 0 ] && GPU_ARR=(0)

case "${STAGE}" in
    synth|all) DEF_JOBS="${#GPU_ARR[@]}" ;;
    trials|insert) DEF_JOBS=4 ;;
    *) echo "unknown stage: ${STAGE} (use: trials | insert | synth | all)"; exit 1 ;;
esac
JOBS="${JOBS:-${DEF_JOBS}}"

echo "=== sweep: stage=${STAGE}  modes=[${MODES[*]}]  concurrency=${JOBS}$([ "${STAGE}" = synth ] || [ "${STAGE}" = all ] && echo "  gpus=[${GPU_ARR[*]}]") ==="
i=0; running=0
for m in "${MODES[@]}"; do
    if [ "${STAGE}" = "synth" ] || [ "${STAGE}" = "all" ]; then
        gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
        CUDA_VISIBLE_DEVICES="${gpu}" bash "${HERE}/run_pipeline.sh" "${m}" "${STAGE}" &
        echo "  -> launched mode ${m} on GPU ${gpu} (pid $!)"
    else
        bash "${HERE}/run_pipeline.sh" "${m}" "${STAGE}" &
        echo "  -> launched mode ${m} (pid $!)"
    fi
    i=$((i + 1)); running=$((running + 1))
    if [ "${running}" -ge "${JOBS}" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
done
wait
echo "=== sweep complete: stage=${STAGE}  modes=[${MODES[*]}] ==="
