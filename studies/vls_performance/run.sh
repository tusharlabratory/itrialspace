#!/usr/bin/env bash
# VLS performance study — thin orchestration over the public analysis engine.
#
# This layer is intentionally tiny: it only chooses *what* to analyse (which split,
# where to write) and calls the reusable engine
# (itrialspace.evaluation.vlm_eval.eval_analysis). All logic lives in the package,
# so numbers can never drift from the code. This folder is gitignored (not part of
# the public release).
#
# Usage:
#   ITRIALSPACE_DATA_DIR=/path/to/iTrialSpace  bash studies/vls_performance/run.sh
set -euo pipefail

DATA="${ITRIALSPACE_DATA_DIR:?set ITRIALSPACE_DATA_DIR to the dataset root}"
OUT="${1:-$(dirname "$0")/output}"
SPLIT="${SPLIT:-release_v1_full}"     # all-4-conditions set (41,502 syn / 13,047 real)
NBOOT="${NBOOT:-1000}"

echo "=== VLS performance study ==="
echo "  data  : $DATA/vlm_dataset"
echo "  split : $SPLIT"
echo "  out   : $OUT"

python -m itrialspace.evaluation.vlm_eval.eval_analysis \
    --results "$DATA/vlm_dataset" \
    --split   "$SPLIT" \
    --out     "$OUT" \
    --n-boot  "$NBOOT"

echo "Done. See $OUT/report.md"
