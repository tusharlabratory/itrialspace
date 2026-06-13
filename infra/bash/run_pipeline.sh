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
# infra/bash/run_pipeline.sh — run the iTrialSpace core pipeline on ONE machine
# (bash / Docker, no SLURM):  trials -> insert -> synth, sequentially.
#
# This is the single-machine counterpart to the SLURM orchestrator
# (infra/slurm/run_pipeline.sh). Both run the SAME per-mode logic and read the
# SAME .env; only the launcher differs (bash loop here vs. sbatch --array there).
#
# Usage:
#   infra/bash/run_pipeline.sh <mode 1-13> [all|trials|insert|synth]
#     all (default) = trials -> insert -> synth, in order.
#
# Examples:
#   infra/bash/run_pipeline.sh 1                  # full pipeline, mode 1
#   infra/bash/run_pipeline.sh 1 trials           # one stage only
#   for m in $(seq 1 13); do infra/bash/run_pipeline.sh "$m"; done
#
# trials / insert are CPU; synth needs a GPU + NodMAISI weights (NODMAISI_MODELS_DIR).
# Paths / run size come from your .env and the per-stage scripts. Run from anywhere.
# ============================================================================
set -uo pipefail

MODE="${1:?usage: run_pipeline.sh <mode 1-13> [all|trials|insert|synth]}"
STAGE="${2:-all}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/bash/env.sh
source "${HERE}/env.sh"
SLURM_DIR="$(cd "${HERE}/../slurm" && pwd)"   # insert/synth still drive these .sub files
cd "${PROJ_DIR}"

G="${ITRIALSPACE_OUTPUT_DIR}"
TRIALS_SH="$(ls "${HERE}"/trials/mode"${MODE}"_*.sh 2>/dev/null | head -1)"
INSERT_SH="${HERE}/mask_inserter/insert_masks.sh"
SYNTH_SH="${HERE}/synthesis/synthesize.sh"
mkdir -p "${G}/nodmaisi_case_lists" 2>/dev/null || true

# ── trials: clean bash stage script (infra/bash/trials/mode<MODE>_*.sh) ───────
run_trials() {
    [ -f "${TRIALS_SH}" ] || { echo "FATAL: no bash trials script for mode ${MODE}"; exit 1; }
    echo ">>> [mode ${MODE}] trials  (${TRIALS_SH##*/})"
    bash "${TRIALS_SH}"
}

# ── insert: clean generic bash script (one task per manifest, sequential) ─────
run_insert() {
    [ -f "${INSERT_SH}" ] || { echo "FATAL: insert script not found: ${INSERT_SH}"; exit 1; }
    echo ">>> [mode ${MODE}] insert"
    bash "${INSERT_SH}" "${MODE}"
}

# ── synth: clean bash script with case-level GPU parallelism (GPU) ────────────
run_synth() {
    [ -f "${SYNTH_SH}" ] || { echo "FATAL: synth script not found: ${SYNTH_SH}"; exit 1; }
    echo ">>> [mode ${MODE}] synth"
    bash "${SYNTH_SH}" "${MODE}"
}

run_stage() {
    case "${STAGE}" in
        trials) run_trials ;;
        insert) run_insert ;;
        synth)  run_synth ;;
        all)    run_trials; run_insert; run_synth ;;
        *) echo "unknown stage: ${STAGE} (use: all | trials | insert | synth)"; exit 1 ;;
    esac
    echo "[mode ${MODE}] stage '${STAGE}' complete."
}

# Tee the whole run to a timestamped log (terminal + file) — the bash counterpart
# of SLURM's logs/%j.out. Set ITS_NO_LOG=1 to print to the terminal only.
if [ -n "${ITS_NO_LOG:-}" ]; then
    run_stage
else
    LOG="${ITS_LOG_DIR}/mode${MODE}_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
    run_stage 2>&1 | tee "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "log: ${LOG}"
    exit "${rc}"
fi
