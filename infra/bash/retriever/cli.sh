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
# infra/bash/retriever/cli.sh — Retriever command-line passthrough. Bash counterpart
# of infra/slurm/retriever/cli.sub. Subcommands: info / search / similar / match /
# detail / export / serve.
#
#   bash infra/bash/retriever/cli.sh info
#   bash infra/bash/retriever/cli.sh search --label 1 --lobe right_lung_upper_lobe --limit 50
#   bash infra/bash/retriever/cli.sh similar --id DLCS_0001_01 --k 10
#   bash infra/bash/retriever/cli.sh match --lobe left_lung_upper_lobe --diameter 12 --cc-pct 50 --k 10
#   # in Docker:
#   ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/retriever/cli.sh info'
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

exec python3 -m itrialspace.apps.retriever.cli "$@"
