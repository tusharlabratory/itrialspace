#!/usr/bin/env bash
# Shared environment for all iTrialSpace SLURM scripts.
#
# Every .sub script sources this file instead of hardcoding paths or the conda env:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
#
# Override any value by exporting it before submitting, or by editing this file /
# a local copy. Nothing here is host-specific by default — set the variables for
# your cluster in your shell profile or a `.env` next to the repo.

# ── Repo root (auto-detected: this file lives at <repo>/infra/slurm/env.sh) ──
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ_DIR="${PROJ_DIR:-$(cd "${_ENV_SH_DIR}/../.." && pwd)}"

# Load a repo-local .env for defaults. Precedence: variables already set in the
# environment (shell export, or `VAR=x sbatch ...`) WIN over .env, which in turn
# wins over the built-in defaults below. So a per-job override works:
#     ITRIALSPACE_OUTPUT_DIR=/somewhere sbatch infra/slurm/trials/mode1_*.sub
if [[ -f "${PROJ_DIR}/.env" ]]; then
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        case "${_line}" in ''|\#*) continue ;; esac        # skip blanks/comments
        _key="${_line%%=*}"
        [[ "${_key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        if [[ -z "${!_key+x}" ]]; then                     # only if not already set
            export "${_key}=$(eval "printf '%s' \"${_line#*=}\"")"
        fi
    done < "${PROJ_DIR}/.env"
    unset _line _key
fi

# ── Data root (unified layout: raw_ct/ masks/ profiles/ ...) ────────────────
export ITRIALSPACE_DATA_DIR="${ITRIALSPACE_DATA_DIR:-${HOME}/.itrialspace/data}"
export DATA_BASE="${DATA_BASE:-${ITRIALSPACE_DATA_DIR}}"
# Output root defaults to the data root — computed AFTER .env so it follows it.
export ITRIALSPACE_OUTPUT_DIR="${ITRIALSPACE_OUTPUT_DIR:-${ITRIALSPACE_DATA_DIR}}"

# ── Conda environment ───────────────────────────────────────────────────────
export ITRIALSPACE_CONDA_ENV="${ITRIALSPACE_CONDA_ENV:-itrialspace}"
# Isolate from ~/.local user-site packages so the conda env is authoritative
# (avoids shadowing the env's torch/pytest/etc. with stale user-site copies).
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# Path to your conda profile script (cluster-specific). Common locations tried below.
export CONDA_PROFILE="${CONDA_PROFILE:-}"

# ── SLURM partitions / account (override per cluster) ───────────────────────
export SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-gpu}"
export SLURM_PARTITION_CPU="${SLURM_PARTITION_CPU:-cpu}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"

# ── Make the package importable ─────────────────────────────────────────────
export PYTHONPATH="${PROJ_DIR}/src:${PYTHONPATH:-}"

# ── Helper: activate the conda env portably ─────────────────────────────────
itrialspace_activate_conda() {
    local profile="${CONDA_PROFILE}"
    if [[ -z "${profile}" ]]; then
        for cand in \
            "/cm/shared/apps/miniconda/etc/profile.d/conda.sh" \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            [[ -f "${cand}" ]] && profile="${cand}" && break
        done
    fi
    if [[ -n "${profile}" && -f "${profile}" ]]; then
        # shellcheck disable=SC1090
        source "${profile}"
        conda activate "${ITRIALSPACE_CONDA_ENV}"
    else
        echo "WARNING: conda profile not found; set CONDA_PROFILE. Continuing without activation." >&2
    fi
    # torch's pip-installed CUDA libs (cuDNN etc.) aren't auto-found in a clean env →
    # put every nvidia/*/lib on LD_LIBRARY_PATH (otherwise: "libcudnn.so.9 not found" /
    # CUDNN_STATUS_NOT_INITIALIZED).
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        local _d
        for _d in "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/*/lib; do
            [[ -d "${_d}" ]] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
        done
    fi
}

# Run from the repo root and ensure the log directory exists. (The `#SBATCH
# --output=logs/...` path is opened by SLURM *before* this runs, so also keep a
# committed logs/ dir and submit from the repo root — see infra/slurm/README.md.)
cd "${PROJ_DIR}" 2>/dev/null || true
mkdir -p "${PROJ_DIR}/logs"
