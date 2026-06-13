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
# infra/bash/docker_run.sh — convenience wrapper to run the GPU image with the
# right --gpus / --user / mounts / env, then exec a command inside it.
#
# Build once:   docker build -t itrialspace:gpu -f docker/Dockerfile.gpu .
#
# Paths and the HF token are read from the repo-local **.env** (gitignored), so
# nothing host-specific or secret is ever hardcoded in this committed script.
# Set them once in .env (ITRIALSPACE_DATA_DIR, ITRIALSPACE_OUTPUT_DIR,
# NODMAISI_MODELS_DIR, HF_TOKEN) and just run:
#
#   # `its` CLI (default entrypoint — pass the subcommand):
#   infra/bash/docker_run.sh config
#
#   # a shell / pipeline (ENTRY=bash):
#   ENTRY=bash infra/bash/docker_run.sh -lc 'infra/bash/run_pipeline.sh 1'
#
#   # a module run (ENTRY=python):
#   ENTRY=python infra/bash/docker_run.sh -m itrialspace.cli --help
#
# Any value may still be overridden from the shell (ITS_DATA / ITS_OUT /
# ITS_MODELS / HF_TOKEN / ITS_PORTS); shell exports win over .env.
#
# Runs as the *host* uid:gid so mounted data/weights owned by a non-1000 user
# (a group-restricted share, weights under a home dir, etc.) are readable and
# /out is writable. The image is built world-readable for exactly this reason.
# HOME=/tmp gives the (passwd-less) uid a writable home for HF / matplotlib caches.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Load paths + token from the gitignored .env (only these keys; shell wins) ─
# Same safe line-by-line parse as infra/slurm/env.sh — never `source`s the file.
if [ -f "${REPO_ROOT}/.env" ]; then
    while IFS= read -r _line || [ -n "${_line}" ]; do
        case "${_line}" in ''|\#*) continue ;; esac
        _key="${_line%%=*}"
        case "${_key}" in
            HF_TOKEN|ITRIALSPACE_DATA_DIR|ITRIALSPACE_OUTPUT_DIR|NODMAISI_MODELS_DIR|NODULEMAP_ARTIFACTS) ;;
            *) continue ;;
        esac
        [ -z "${!_key+x}" ] && export "${_key}=$(printf '%s' "${_line#*=}")"
    done < "${REPO_ROOT}/.env"
    unset _line _key
fi

IMAGE="${ITS_IMAGE:-itrialspace:gpu}"
ITS_DATA="${ITS_DATA:-${ITRIALSPACE_DATA_DIR:?set ITRIALSPACE_DATA_DIR in .env (or export ITS_DATA)}}"
ITS_OUT="${ITS_OUT:-${ITRIALSPACE_OUTPUT_DIR:-${ITS_DATA}/outputs}}"
ITS_MODELS="${ITS_MODELS:-${NODMAISI_MODELS_DIR:-}}"   # NodMAISI weights (only for synthesis)
ENTRY="${ENTRY:-its}"                                  # override: ENTRY=bash / python / pytest

mkdir -p "${ITS_OUT}"

# Allocate a TTY only when attached to one (so it also works in scripts / CI).
TTY_FLAGS=(-i); [ -t 0 ] && [ -t 1 ] && TTY_FLAGS=(-it)

ARGS=(--rm "${TTY_FLAGS[@]}" --gpus all
      --user "$(id -u):$(id -g)"
      --shm-size "${ITS_SHM:-16g}"
      -e HOME=/tmp
      -e HF_HOME=/out/.hf_cache               # persist HF model downloads across runs (avoids re-download + rate limits)
      -e ITRIALSPACE_DATA_DIR=/data
      -e ITRIALSPACE_OUTPUT_DIR=/out
      -v "${ITS_DATA}:/data"
      -v "${ITS_OUT}:/out"
      --entrypoint "${ENTRY}")

[ -n "${HF_TOKEN:-}" ]   && ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")          # gated VLMs (MedGemma)
[ -n "${ITS_MODELS}" ]   && ARGS+=(-e NODMAISI_MODELS_DIR=/models -v "${ITS_MODELS}:/models")

# Remap NODULEMAP_ARTIFACTS (a HOST path in .env) onto its container mount, so the
# NoduleMap app writes/reads a path that exists inside the container (not /mnt/…).
if [ -n "${NODULEMAP_ARTIFACTS:-}" ]; then
    case "${NODULEMAP_ARTIFACTS}" in
        "${ITS_OUT}"/*|"${ITS_OUT}")   ARGS+=(-e "NODULEMAP_ARTIFACTS=/out${NODULEMAP_ARTIFACTS#"${ITS_OUT}"}") ;;
        "${ITS_DATA}"/*|"${ITS_DATA}") ARGS+=(-e "NODULEMAP_ARTIFACTS=/data${NODULEMAP_ARTIFACTS#"${ITS_DATA}"}") ;;
        *)                             ARGS+=(-e "NODULEMAP_ARTIFACTS=/out/nodulemap_artifacts") ;;
    esac
fi

# Dev mode: mount the host repo over /app so code/script edits take effect WITHOUT
# rebuilding the image (the editable install resolves /app/src either way). For a
# self-contained / shareable run, omit ITS_DEV and rebuild instead.
[ -n "${ITS_DEV:-}" ]    && ARGS+=(-v "${REPO_ROOT}:/app")

# Apps (NoduleMap 8422 / Retriever 8421+8501): publish ports when requested.
[ -n "${ITS_PORTS:-}" ]  && for p in ${ITS_PORTS}; do ARGS+=(-p "${p}:${p}"); done

exec docker run "${ARGS[@]}" "${IMAGE}" "$@"
