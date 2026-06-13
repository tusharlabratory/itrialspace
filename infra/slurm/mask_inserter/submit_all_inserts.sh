#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Submit mask insertion jobs for all 13 iTrialSpace trial modes.
#
# Usage:
#   ./submit_all_inserts.sh              # submit all 13 modes (single-manifest)
#   ./submit_all_inserts.sh 1 3 8        # submit only modes 1, 3, 8
#   ./submit_all_inserts.sh --dry-run    # submit all with dry-run enabled
#
# For array jobs (modes with multiple manifests), use the array sub files
# directly — see README.md for details.
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure logs directory exists
mkdir -p logs

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

# Parse --dry-run flag
DRY_RUN_FLAG=""
POSITIONAL=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN_FLAG="--export=DRY_RUN=true"
    else
        POSITIONAL+=("$arg")
    fi
done
set -- "${POSITIONAL[@]}"

echo "=================================================================="
echo " iTrialSpace Mask Insertion — Batch Submission"
echo " Date: $(date)"
if [ -n "$DRY_RUN_FLAG" ]; then
    echo " *** DRY RUN MODE ***"
fi
echo "=================================================================="
echo ""

if [ $# -eq 0 ]; then
    # Submit all
    echo "Submitting all ${#MODES[@]} trial modes..."
    echo ""
    for i in "${!MODES[@]}"; do
        mode="${MODES[$i]}"
        name="${MODE_NAMES[$i]}"
        sub_file="${mode}.sub"
        if [ -f "$sub_file" ]; then
            echo -n "  Mode $((i+1)): ${name} ... "
            sbatch $DRY_RUN_FLAG "${sub_file}"
        else
            echo "  Mode $((i+1)): ${name} — SKIPPED (${sub_file} not found)"
        fi
    done
else
    # Submit selected modes
    echo "Submitting selected trial modes..."
    echo ""
    for num in "$@"; do
        idx=$((num - 1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#MODES[@]} ]; then
            mode="${MODES[$idx]}"
            name="${MODE_NAMES[$idx]}"
            sub_file="${mode}.sub"
            if [ -f "$sub_file" ]; then
                echo -n "  Mode ${num}: ${name} ... "
                sbatch $DRY_RUN_FLAG "${sub_file}"
            else
                echo "  Mode ${num}: ${name} — SKIPPED (${sub_file} not found)"
            fi
        else
            echo "  Invalid mode number: $num (valid: 1-${#MODES[@]})"
        fi
    done
fi

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo "Check status:  squeue -u $USER"
echo "Cancel all:    scancel -u $USER --name='ins_M*'"
echo "Logs:          ${SCRIPT_DIR}/logs/"
echo "──────────────────────────────────────────────────────────────────"
