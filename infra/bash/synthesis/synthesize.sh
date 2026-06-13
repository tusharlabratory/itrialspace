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
# infra/bash/synthesis/synthesize.sh — NodMAISI CT synthesis for ONE trial mode,
# with CASE-LEVEL GPU parallelism. GPU.
#
#   infra/bash/synthesis/synthesize.sh <mode 1-13>
#
# The faithful bash equivalent of the SLURM synth array (`--array` over cases):
# it gathers every successful case from the mode's insertion audit(s) and
# dispatches them across the available GPUs. Each GPU runs a "lane" that pulls
# cases sequentially; with N GPUs you get N cases synthesizing at once.
#
# Reads/writes the same paths as the SLURM path:
#   in :  $ITRIALSPACE_OUTPUT_DIR/inserted_masks/<mode>_<name>/<manifest>/audit.json
#   out:  $ITRIALSPACE_OUTPUT_DIR/generated_cts/mode<MODE>/<manifest>/<case>/...
#
# Parallelism / memory knobs (env):
#   GPUS="0 1 2 3"   GPUs to use (default: all visible)
#   PER_GPU=1        concurrent cases PER gpu (default 1). NodMAISI needs ~20 GB
#                    peak per case, so 1/GPU is safe; raise only on big idle cards.
#   => lanes = (#GPUS) × PER_GPU cases in flight.
#   CONFIG_PATH      integration config (default: tools/integration_config.yaml)
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/bash/env.sh
source "${HERE}/../env.sh"

MODE="${1:?usage: synthesize.sh <mode 1-13>}"
NODMAISI_DIR="${PROJ_DIR}/src/itrialspace/synthesis"
CONFIG_PATH="${CONFIG_PATH:-${NODMAISI_DIR}/tools/integration_config.yaml}"
PER_GPU="${PER_GPU:-1}"

# NodMAISI runtime env (matches the SLURM synth .sub) + fragmentation guard so the
# big full-res VAE-decode allocation doesn't trip on a partially-used card.
export MONAI_DATA_DIRECTORY="${NODMAISI_DIR}/"
export PYTHONPATH="${NODMAISI_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -ra GPU_ARR <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}"
[ "${#GPU_ARR[@]}" -eq 0 ] && GPU_ARR=(0)
NLANE=$(( ${#GPU_ARR[@]} * PER_GPU ))

G="${ITRIALSPACE_OUTPUT_DIR}"
TASKS="$(mktemp)"; FAILDIR="$(mktemp -d)"
trap 'rm -f "${TASKS}" "${TASKS}".lane* ; rm -rf "${FAILDIR}"' EXIT

# Build the flat task list: one line per case  =  audit <TAB> case_id <TAB> outdir
naudit=0
while IFS= read -r audit; do
    [ -z "${audit}" ] && continue
    naudit=$((naudit + 1))
    sub="$(basename "$(dirname "${audit}")")"                       # manifest stem
    out="${G}/generated_cts/mode${MODE}/${sub}"
    cl="${G}/nodmaisi_case_lists/mode${MODE}_${sub}.txt"
    mkdir -p "$(dirname "${cl}")"
    python3 "${PROJ_DIR}/infra/slurm/_gen_caselist.py" "${audit}" "${cl}" >/dev/null
    while IFS= read -r case; do
        [ -z "${case}" ] && continue
        printf '%s\t%s\t%s\n' "${audit}" "${case}" "${out}" >> "${TASKS}"
    done < "${cl}"
done < <(find "${G}/inserted_masks/mode${MODE}_"*/ -name audit.json 2>/dev/null | sort)

[ "${naudit}" -gt 0 ] || { echo "FATAL: no audits for mode ${MODE} (run insert first)"; exit 1; }
ntask=$(wc -l < "${TASKS}")
[ "${ntask}" -gt 0 ] || { echo "FATAL: no successful cases for mode ${MODE}"; exit 1; }

echo "=== iTrialSpace synth: mode ${MODE} — ${ntask} case(s) across ${#GPU_ARR[@]} GPU(s) × ${PER_GPU} = ${NLANE} lane(s)  ($(date '+%F %T')) ==="

run_case() {
    local gpu="$1" audit="$2" case="$3" out="$4"
    mkdir -p "${out}"
    echo "[gpu ${gpu}] >> ${case}"
    if CUDA_VISIBLE_DEVICES="${gpu}" python3 "${NODMAISI_DIR}/tools/run_itrialspace_to_ct.py" \
            --audit "${audit}" --config "${CONFIG_PATH}" --outdir "${out}" --case-ids "${case}" -v \
            >/dev/null 2>&1; then
        echo "[gpu ${gpu}] OK ${case}"
    else
        echo "[gpu ${gpu}] FAIL ${case}"; : > "${FAILDIR}/$(printf '%s' "${case}" | md5sum | cut -c1-16).fail"
    fi
}

# Round-robin tasks into per-lane files; each lane is pinned to one GPU.
for ((l = 0; l < NLANE; l++)); do : > "${TASKS}.lane${l}"; done
k=0
while IFS=$'\t' read -r audit case out; do
    printf '%s\t%s\t%s\n' "${audit}" "${case}" "${out}" >> "${TASKS}.lane$(( k % NLANE ))"
    k=$((k + 1))
done < "${TASKS}"

for ((l = 0; l < NLANE; l++)); do
    gpu="${GPU_ARR[$(( l % ${#GPU_ARR[@]} ))]}"
    ( while IFS=$'\t' read -r audit case out; do run_case "${gpu}" "${audit}" "${case}" "${out}"; done < "${TASKS}.lane${l}" ) &
done
wait

fail=$(find "${FAILDIR}" -name '*.fail' 2>/dev/null | wc -l)
echo "=== synth done: mode ${MODE} ($((ntask - fail))/${ntask} cases ok, ${fail} failed) ==="
[ "${fail}" -eq 0 ]
