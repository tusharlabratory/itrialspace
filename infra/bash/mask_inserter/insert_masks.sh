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
# infra/bash/mask_inserter/insert_masks.sh — insert donor nodule masks (label 23)
# into host anatomy for ONE trial mode. CPU.
#
#   infra/bash/mask_inserter/insert_masks.sh <mode 1-13>
#
# Generic across all 13 modes: the SLURM tree's per-mode *_insert_masks_array.sub
# files differ only by the mode-name string, which we resolve here from the
# manifests directory. Reads the same config + writes the same layout as SLURM:
#   $ITRIALSPACE_OUTPUT_DIR/inserted_masks/<mode>_<name>/<manifest_stem>/{...,audit.json}
#
# Parallelism (two levels, the bash equivalent of SLURM's insert array):
#   JOBS=N    process N manifests CONCURRENTLY (default 1 = sequential). This is
#             the SLURM "--array over manifests" equivalent.
#   N_JOBS=M  workers PER manifest, across cases (default 8). SLURM "--n-jobs".
#   => total in-flight workers ≈ JOBS × N_JOBS. On a shared/busy host keep the
#      product at or below the free core count; on the demo (seconds/manifest)
#      leave JOBS=1.
#
# Other knobs: SEED (42), CONFIG_PATH (config/defaults.yaml under the package),
#              DRY_RUN (false).
# ============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

MODE="${1:?usage: insert_masks.sh <mode 1-13>}"
JOBS="${JOBS:-1}"             # concurrent manifests
N_JOBS="${N_JOBS:-8}"         # workers per manifest
SEED="${SEED:-42}"
CONFIG_PATH="${CONFIG_PATH:-config/defaults.yaml}"
DRY_RUN="${DRY_RUN:-false}"

# All manifest CSVs for this mode (recurses into the modes 11-13 dataset subdirs).
mapfile -t MANIFESTS < <(find "${ITRIALSPACE_OUTPUT_DIR}/manifests/mode${MODE}_"*/ -name '*.csv' 2>/dev/null | sort)
[ "${#MANIFESTS[@]}" -gt 0 ] || { echo "FATAL: no manifests for mode ${MODE} (run trials first)"; exit 1; }

# TRIAL_MODE = the "mode<N>_<name>" path segment (resolved from the first manifest).
TRIAL_MODE="$(grep -oE "mode${MODE}_[^/]+" <<< "${MANIFESTS[0]}" | head -1)"
OUT_ROOT="${ITRIALSPACE_OUTPUT_DIR}/inserted_masks/${TRIAL_MODE}"

# Config: prefer the packaged default, else a literal path if given.
CFG_ARG=()
if [ -f "${PROJ_DIR}/src/itrialspace/mask_inserter/${CONFIG_PATH}" ]; then
    CFG_ARG=(--config "${PROJ_DIR}/src/itrialspace/mask_inserter/${CONFIG_PATH}")
elif [ -f "${CONFIG_PATH}" ]; then
    CFG_ARG=(--config "${CONFIG_PATH}")
fi
EXTRA=(); [ "${DRY_RUN}" = "true" ] && EXTRA=(--dry-run)

FAILDIR="$(mktemp -d)"
trap 'rm -rf "${FAILDIR}"' EXIT

# Insert one manifest. ${2} is a unique job index (avoids fail-marker collisions
# when modes 11-13 nest same-named CSVs under different dataset subdirs).
process_one() {
    local m="$1" jid="$2" base out
    base="$(basename "${m}" .csv)"
    out="${OUT_ROOT}/${base}"
    echo "--- [${jid}] ${base}  ->  ${out}"
    mkdir -p "${out}"
    if ! python3 -m itrialspace.mask_inserter run \
            --manifest "${m}" --output-dir "${out}" \
            "${CFG_ARG[@]}" --trial-name "${TRIAL_MODE}_${base}" \
            --seed "${SEED}" --n-jobs "${N_JOBS}" "${EXTRA[@]}"; then
        echo "    WARN: insert failed for ${base}"
        : > "${FAILDIR}/${jid}.fail"
    fi
}

echo "=== iTrialSpace insert: ${TRIAL_MODE} — ${#MANIFESTS[@]} manifest(s), JOBS=${JOBS} x N_JOBS=${N_JOBS}  ($(date '+%F %T')) ==="
idx=0; running=0
for m in "${MANIFESTS[@]}"; do
    process_one "${m}" "${idx}" &
    idx=$((idx + 1)); running=$((running + 1))
    if [ "${running}" -ge "${JOBS}" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
done
wait

fail=$(find "${FAILDIR}" -name '*.fail' 2>/dev/null | wc -l)
echo "=== insert done: ${TRIAL_MODE} ($((${#MANIFESTS[@]} - fail))/${#MANIFESTS[@]} manifests ok, ${fail} failed) ==="
[ "${fail}" -eq 0 ]
