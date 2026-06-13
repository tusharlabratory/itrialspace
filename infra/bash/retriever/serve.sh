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
# infra/bash/retriever/serve.sh — serve the Retriever app: FastAPI API + Streamlit
# UI on one machine. Bash counterpart of infra/slurm/retriever/app.sub (without
# the HPC reverse-tunnel).
#
#   bash infra/bash/retriever/serve.sh
#   # in Docker (publish both ports):
#   ITS_PORTS="8421 8501" ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/retriever/serve.sh'
#
# Then open the UI at http://localhost:<RETRIEVER_UI_PORT>. Knobs:
#   RETRIEVER_API_PORT (8421), RETRIEVER_UI_PORT (8501). Ctrl-C stops both.
# ============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${PROJ_DIR}"

API_PORT="${RETRIEVER_API_PORT:-8421}"
UI_PORT="${RETRIEVER_UI_PORT:-8501}"

echo "=== Retriever: FastAPI on :${API_PORT}  +  Streamlit UI on :${UI_PORT}  ($(date '+%F %T')) ==="

uvicorn itrialspace.apps.retriever.api.app:create_app \
    --factory --host 0.0.0.0 --port "${API_PORT}" --workers 1 &
API_PID=$!
trap 'kill "${API_PID}" "${UI_PID:-}" 2>/dev/null || true' EXIT INT TERM

# Wait (up to ~2 min) for the API /health before starting the UI. Uses Python's
# urllib (curl is not installed in the image), so this works inside the container.
python3 - "${API_PORT}" <<'PY'
import sys, time, urllib.request
port = sys.argv[1]
for _ in range(120):
    try:
        if urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2).status == 200:
            print(f"  API ready on :{port}")
            break
    except Exception:
        time.sleep(1)
else:
    print(f"  WARNING: API /health not ready after 120s — starting UI anyway")
PY

streamlit run src/itrialspace/apps/retriever/ui/app.py \
    --server.port "${UI_PORT}" --server.headless true --server.address 0.0.0.0 \
    -- --api-url "http://localhost:${API_PORT}" &
UI_PID=$!

echo "  API: http://localhost:${API_PORT}    UI: http://localhost:${UI_PORT}"
wait
