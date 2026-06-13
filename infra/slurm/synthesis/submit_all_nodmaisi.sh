#!/bin/bash

# ── Load shared environment (PROJ_DIR, ITRIALSPACE_*, conda) — before any var use ──
ITS_ENV="${SLURM_SUBMIT_DIR:-$PWD}"; while [ "$ITS_ENV" != "/" ] && [ ! -f "$ITS_ENV/infra/slurm/env.sh" ]; do ITS_ENV="$(dirname "$ITS_ENV")"; done; source "$ITS_ENV/infra/slurm/env.sh"

# ============================================================================
# Submit NodMAISI CT synthesis jobs for all 9 iTrialSpace trial modes.
#
# Reads audit.json from each mode's inserted_masks directory.
#
# Usage:
#   ./submit_all_nodmaisi.sh              # submit all 9 modes
#   ./submit_all_nodmaisi.sh 1 3 8        # submit only modes 1, 3, 8
#   ./submit_all_nodmaisi.sh --dry-run    # all modes, dry-run
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NOMAISI_DIR="${PROJ_DIR}/src/itrialspace/synthesis"
ITRIALSPACE_BASE="${ITRIALSPACE_OUTPUT_DIR}"
MASK_BASE="${ITRIALSPACE_BASE}/inserted_masks"
CT_BASE="${ITRIALSPACE_BASE}/generated_cts"

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
)

# Parse --dry-run flag
DRY_RUN_EXPORT=""
POSITIONAL=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN_EXPORT="--export=ALL,DRY_RUN=true"
    else
        POSITIONAL+=("$arg")
    fi
done
set -- "${POSITIONAL[@]}"

echo "=================================================================="
echo " iTrialSpace -> NodMAISI CT Synthesis — Batch Submission"
echo " Date: $(date)"
if [ -n "$DRY_RUN_EXPORT" ]; then
    echo " *** DRY RUN MODE ***"
fi
echo "=================================================================="
echo ""

submit_mode() {
    local idx=$1
    local mode="${MODES[$idx]}"
    local name="${MODE_NAMES[$idx]}"
    local mask_dir="${MASK_BASE}/${mode}"
    local audit_path="${mask_dir}/audit.json"
    local out_dir="${CT_BASE}/${mode}"

    echo -n "  Mode $((idx+1)): ${name} ... "

    # Check for audit.json at root or in any subdirectory
    local audit_path="${mask_dir}/audit.json"
    local has_sub_audits
    has_sub_audits=$(find "${mask_dir}" -mindepth 2 -name "audit.json" 2>/dev/null | head -1)

    if [ -f "${audit_path}" ]; then
        sbatch $DRY_RUN_EXPORT \
            --export=ALL,AUDIT_PATH="${audit_path}",OUT_DIR="${out_dir}" \
            --job-name="NM_M$((idx+1))" \
            template_nodmaisi_from_masks.sub
    elif [ -n "${has_sub_audits}" ]; then
        echo -n "(subdirectory audits found, using --mask-root) "
        sbatch $DRY_RUN_EXPORT \
            --export=ALL,MASK_ROOT="${mask_dir}",OUT_DIR="${out_dir}" \
            --job-name="NM_M$((idx+1))" \
            template_nodmaisi_from_masks.sub
    elif [ -d "${mask_dir}" ] && find "${mask_dir}" -name "*.nii.gz" -print -quit | grep -q .; then
        echo -n "(no audit.json, using --mask-root) "
        sbatch $DRY_RUN_EXPORT \
            --export=ALL,MASK_ROOT="${mask_dir}",OUT_DIR="${out_dir}" \
            --job-name="NM_M$((idx+1))" \
            template_nodmaisi_from_masks.sub
    else
        echo "SKIPPED (no audit.json and no masks in ${mask_dir})"
        return
    fi
}

if [ $# -eq 0 ]; then
    echo "Submitting all 9 trial modes..."
    echo ""
    for i in "${!MODES[@]}"; do
        submit_mode "$i"
    done
else
    echo "Submitting selected trial modes..."
    echo ""
    for num in "$@"; do
        idx=$((num - 1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#MODES[@]} ]; then
            submit_mode "$idx"
        else
            echo "  Invalid mode number: $num (valid: 1-9)"
        fi
    done
fi

echo ""
echo "--------------------------------------------------------------"
echo "Check status:  squeue -u $USER"
echo "Cancel all:    scancel -u $USER --name='NM_M*'"
echo "Logs:          ${SCRIPT_DIR}/logs/"
echo "--------------------------------------------------------------"
