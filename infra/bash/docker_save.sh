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
# infra/bash/docker_save.sh — package the built GPU image into a single
# portable tarball so it can be shared (USB, scp, shared filesystem) and loaded
# on another host that has no internet / no access to your registry.
#
#   # on the source host (after building itrialspace:gpu):
#   infra/bash/docker_save.sh                       # -> itrialspace-gpu.tar.gz
#   infra/bash/docker_save.sh /shared/itrialspace-gpu.tar.gz
#
#   # on the target host (needs Docker + nvidia-container-toolkit):
#   docker load -i itrialspace-gpu.tar.gz
#   ITS_DATA=/host/iTrialSpace infra/bash/docker_run.sh config
#
# To share via a registry instead:
#   docker tag itrialspace:gpu <registry>/itrialspace:gpu
#   docker push <registry>/itrialspace:gpu
# ============================================================================
set -euo pipefail

IMAGE="${ITS_IMAGE:-itrialspace:gpu}"
OUT="${1:-itrialspace-gpu.tar.gz}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: image '${IMAGE}' not found. Build it first:" >&2
    echo "  docker build -t ${IMAGE} -f docker/Dockerfile.gpu ." >&2
    exit 1
fi

echo "Saving ${IMAGE} -> ${OUT} (this is ~10 GB compressed; may take a few minutes)…"
docker save "${IMAGE}" | gzip > "${OUT}"
echo "Done: $(du -h "${OUT}" | cut -f1)  ${OUT}"
echo "Load on the target host with:  docker load -i ${OUT}"
