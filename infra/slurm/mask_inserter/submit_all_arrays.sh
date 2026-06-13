#!/bin/bash

# ── Load shared environment (PROJ_DIR, ITRIALSPACE_*, conda) — before any var use ──
ITS_ENV="${SLURM_SUBMIT_DIR:-$PWD}"; while [ "$ITS_ENV" != "/" ] && [ ! -f "$ITS_ENV/infra/slurm/env.sh" ]; do ITS_ENV="$(dirname "$ITS_ENV")"; done; source "$ITS_ENV/infra/slurm/env.sh"

# ──────────────────────────────────────────────────────────────────────────────
# Submit mask insertion ARRAY jobs for all 13 iTrialSpace trial modes.
#
# For each mode this script:
#   1. Generates the manifest list file  (manifests_modeN.txt)
#   2. Counts how many manifests exist
#   3. Submits sbatch --array=0-(N-1)  modeN_insert_masks_array.sub
#
# Modes with exactly 1 manifest fall back to the single (non-array) .sub.
# Modes with 0 manifests are skipped.
#
# Usage:
#   ./submit_all_arrays.sh              # submit all 13 modes
#   ./submit_all_arrays.sh 2 3 7        # submit only modes 2, 3, 7
#   ./submit_all_arrays.sh --dry-run    # submit all with dry-run enabled
#   ./submit_all_arrays.sh --list-only  # generate manifest lists, don't submit
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Ensure logs directory exists
mkdir -p logs

BASE_DIR="${ITRIALSPACE_DATA_DIR}"
MANIFEST_ROOT="${ITRIALSPACE_OUTPUT_DIR}/manifests"
LIST_MANIFESTS="${PROJECT_DIR}/itrialspace_mask_inserter/tools/list_manifests.py"

MODES=(
    "mode1_insert_masks"     # Controlled Prevalence
    "mode2_insert_masks"     # Size Detection Curve
    "mode3_insert_masks"     # Location Sensitivity
    "mode4_insert_masks"     # Demographic Stratification
    "mode5_insert_masks"     # Counterfactual
    "mode6_insert_masks"     # Cross-Dataset
    "mode7_insert_masks"     # Bootstrap Confidence
    "mode8_insert_masks"     # Algorithm Comparison
    "mode9_insert_masks"     # Screening Simulation
    "mode10_insert_masks"    # Multi-Nodule Realism
    "mode11_insert_masks"    # Digital Twin Isolation
    "mode12_insert_masks"    # Digital Twin Complete
    "mode13_insert_masks"    # Digital Twin Cross
)

MODE_DIRS=(
    "mode1_controlled_prevalence"
    "mode2_size_detection_curve"
    "mode3_location_sensitivity"
    "mode4_demographic_stratification"
    "mode5_counterfactual"
    "mode6_cross_dataset"
    "mode7_bootstrap_confidence"
    "mode8_algorithm_comparison"
    "mode9_screening_simulation"
    "mode10_multi_nodule_realism"
    "mode11_digital_twin_isolation"
    "mode12_digital_twin_complete"
    "mode13_digital_twin_cross"
)

MODE_NAMES=(
    "Controlled Prevalence"
    "Size Detection Curve"
    "Location Sensitivity"
    "Demographic Stratification"
    "Counterfactual"
    "Cross-Dataset"
    "Bootstrap Confidence"
    "Algorithm Comparison"
    "Screening Simulation"
    "Multi-Nodule Realism"
    "Digital Twin Isolation"
    "Digital Twin Complete"
    "Digital Twin Cross"
)

# ── Parse flags ───────────────────────────────────────────────────────────────
DRY_RUN_FLAG=""
LIST_ONLY=false
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN_FLAG="--export=DRY_RUN=true" ;;
        --list-only) LIST_ONLY=true ;;
        *)           POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

echo "=================================================================="
echo " iTrialSpace Mask Insertion — Array Batch Submission"
echo " Date: $(date)"
if [ -n "$DRY_RUN_FLAG" ]; then
    echo " *** DRY RUN MODE ***"
fi
if $LIST_ONLY; then
    echo " *** LIST-ONLY MODE (no sbatch) ***"
fi
echo "=================================================================="
echo ""

# ── Determine which modes to process ─────────────────────────────────────────
if [ $# -eq 0 ]; then
    INDICES=($(seq 0 $((${#MODES[@]} - 1))))
else
    INDICES=()
    for num in "$@"; do
        idx=$((num - 1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#MODES[@]} ]; then
            INDICES+=($idx)
        else
            echo "  Invalid mode number: $num (valid: 1-${#MODES[@]})"
        fi
    done
fi

SUBMITTED=0
SKIPPED=0

for idx in "${INDICES[@]}"; do
    num=$((idx + 1))
    mode="${MODES[$idx]}"
    dir="${MODE_DIRS[$idx]}"
    name="${MODE_NAMES[$idx]}"
    manifest_dir="${MANIFEST_ROOT}/${dir}"
    list_file="${SCRIPT_DIR}/manifests_mode${num}.txt"

    echo -n "  Mode ${num}: ${name} ... "

    # ── Check if manifest directory exists ────────────────────────────────
    if [ ! -d "${manifest_dir}" ]; then
        echo "SKIPPED (no directory: ${dir}/)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # ── Generate manifest list file ───────────────────────────────────────
    python3 "${LIST_MANIFESTS}" \
        --dir "${manifest_dir}" \
        --output "${list_file}" \
        --quiet 2>/dev/null || true

    if [ ! -f "${list_file}" ]; then
        echo "SKIPPED (list_manifests.py failed)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    N=$(wc -l < "${list_file}" | tr -d ' ')

    if [ "$N" -eq 0 ]; then
        echo "SKIPPED (0 manifests)"
        rm -f "${list_file}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $LIST_ONLY; then
        echo "${N} manifest(s) → ${list_file}"
        continue
    fi

    # ── Decide: single job  vs  array job ─────────────────────────────────
    if [ "$N" -eq 1 ]; then
        # Single manifest → prefer the non-array .sub (simpler logs)
        sub_file="${mode}.sub"
        if [ -f "$sub_file" ]; then
            sbatch $DRY_RUN_FLAG "${sub_file}"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "SKIPPED (${sub_file} not found)"
            SKIPPED=$((SKIPPED + 1))
        fi
    else
        # Multiple manifests → array job
        sub_file="${mode}_array.sub"
        if [ -f "$sub_file" ]; then
            ARRAY_RANGE="0-$((N - 1))"
            sbatch $DRY_RUN_FLAG --array="${ARRAY_RANGE}" "${sub_file}"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "SKIPPED (${sub_file} not found)"
            SKIPPED=$((SKIPPED + 1))
        fi
    fi
done

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo " Submitted: ${SUBMITTED}   Skipped: ${SKIPPED}"
echo ""
echo " Check status:  squeue -u $USER"
echo " Cancel all:    scancel -u $USER --name='ins_M*'"
echo " Logs:          ${SCRIPT_DIR}/logs/"
echo "──────────────────────────────────────────────────────────────────"
