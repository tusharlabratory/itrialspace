#!/bin/bash
# Submit all iTrialSpace trial mode jobs to SLURM
#
# Usage:
#   ./submit_all.sh              # submit all modes
#   ./submit_all.sh 1 3 8       # submit only modes 1, 3, 8
#
# Each mode runs independently — no dependencies between jobs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure logs directory exists
mkdir -p logs

MODES=(
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

if [ $# -eq 0 ]; then
    # Submit all
    echo "Submitting all ${#MODES[@]} trial modes..."
    for mode in "${MODES[@]}"; do
        echo -n "  $mode: "
        sbatch "${mode}.sub"
    done
else
    # Submit selected modes
    echo "Submitting selected trial modes..."
    for num in "$@"; do
        idx=$((num - 1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#MODES[@]} ]; then
            mode="${MODES[$idx]}"
            echo -n "  $mode: "
            sbatch "${mode}.sub"
        else
            echo "  Invalid mode number: $num (valid: 1-${#MODES[@]})"
        fi
    done
fi

echo ""
echo "Check status with: squeue -u $USER"
echo "Check logs in: $SCRIPT_DIR/logs/"
