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
# infra/bash/env.sh — shared environment for the NON-SLURM (bash / Docker) pipeline.
#
# Scheduler-agnostic counterpart to infra/slurm/env.sh: it reads the SAME repo
# `.env`, sets the SAME PYTHONPATH, but makes no SLURM assumptions, writes logs to
# a WRITABLE directory (not the read-only image code dir), and treats conda as
# OPTIONAL — silent no-op when the package already imports (e.g. inside the Docker
# image, or any ready venv). Source it from any infra/bash/* stage script:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # from infra/bash/<stage>/x.sh
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"      # from infra/bash/x.sh
#
# Override any value by exporting it before calling (shell exports win over .env).
# ============================================================================

# ── Repo root (this file lives at <repo>/infra/bash/env.sh) ──────────────────
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ_DIR="${PROJ_DIR:-$(cd "${_ENV_DIR}/../.." && pwd)}"

# ── Load repo-local .env (only keys not already set; safe line-by-line parse) ─
if [[ -f "${PROJ_DIR}/.env" ]]; then
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        case "${_line}" in ''|\#*) continue ;; esac
        _key="${_line%%=*}"
        [[ "${_key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        if [[ -z "${!_key+x}" ]]; then
            export "${_key}=$(eval "printf '%s' \"${_line#*=}\"")"
        fi
    done < "${PROJ_DIR}/.env"
    unset _line _key
fi

# ── Data / output roots ──────────────────────────────────────────────────────
export ITRIALSPACE_DATA_DIR="${ITRIALSPACE_DATA_DIR:-${HOME}/.itrialspace/data}"
export ITRIALSPACE_OUTPUT_DIR="${ITRIALSPACE_OUTPUT_DIR:-${ITRIALSPACE_DATA_DIR}}"

# ── Make the package importable from source; isolate from ~/.local ───────────
export PYTHONPATH="${PROJ_DIR}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# ── Writable log dir (NEVER under the read-only image code dir) ───────────────
export ITS_LOG_DIR="${ITS_LOG_DIR:-${ITRIALSPACE_OUTPUT_DIR}/logs}"
mkdir -p "${ITS_LOG_DIR}" 2>/dev/null || true

# ── Conda is OPTIONAL ────────────────────────────────────────────────────────
# Activate only if python can't already import the package (i.e. a host conda
# install). Inside the Docker image / any ready env this is a silent no-op — no
# more "conda profile not found" warnings on the happy path.
its_maybe_activate_conda() {
    python3 -c 'import itrialspace' >/dev/null 2>&1 && return 0
    command -v conda >/dev/null 2>&1 || return 0
    local prof="${CONDA_PROFILE:-}"
    if [[ -z "${prof}" ]]; then
        for c in "${HOME}/miniconda3/etc/profile.d/conda.sh" \
                 "${HOME}/anaconda3/etc/profile.d/conda.sh" \
                 "/opt/conda/etc/profile.d/conda.sh" \
                 "/cm/shared/apps/miniconda/etc/profile.d/conda.sh"; do
            [[ -f "${c}" ]] && prof="${c}" && break
        done
    fi
    [[ -n "${prof}" && -f "${prof}" ]] || return 0
    # shellcheck disable=SC1090
    source "${prof}" && conda activate "${ITRIALSPACE_CONDA_ENV:-itrialspace}" 2>/dev/null || true
    # torch's pip-installed CUDA libs (cuDNN etc.) aren't auto-found in a clean env.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        local d
        for d in "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/*/lib; do
            [[ -d "${d}" ]] && export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
        done
    fi
}
its_maybe_activate_conda

# ── Helper: run a trial mode (used by infra/bash/trials/*.sh) ─────────────────
its_trial() {
    local mode="$1"; shift
    echo "=== iTrialSpace trials: ${mode}  ($(date '+%F %T')) ==="
    python3 "${PROJ_DIR}/src/itrialspace/trials/run_trial.py" --mode "${mode}" "$@"
}

cd "${PROJ_DIR}" 2>/dev/null || true
