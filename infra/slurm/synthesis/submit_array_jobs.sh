#!/bin/bash

# ── Load shared environment (PROJ_DIR, ITRIALSPACE_*, conda) — before any var use ──
ITS_ENV="${SLURM_SUBMIT_DIR:-$PWD}"; while [ "$ITS_ENV" != "/" ] && [ ! -f "$ITS_ENV/infra/slurm/env.sh" ]; do ITS_ENV="$(dirname "$ITS_ENV")"; done; source "$ITS_ENV/infra/slurm/env.sh"

# ============================================================================
# Submit NodMAISI CT synthesis ARRAY jobs for iTrialSpace trial modes.
#
# For each mode: generates a case list, then submits a SLURM array job
# (one case per GPU task) for maximum parallelism.
#
# Usage:
#   ./submit_array_jobs.sh                  # all 13 modes
#   ./submit_array_jobs.sh 1 3 8            # selected modes
#   ./submit_array_jobs.sh --dry-run        # all modes, dry-run
#   ./submit_array_jobs.sh --max-concurrent 20  # limit concurrent GPU tasks
#   ./submit_array_jobs.sh --time 04:00:00  # override walltime per task
#   ./submit_array_jobs.sh --skip-existing   # skip cases with synthetic_ct
#   ./submit_array_jobs.sh --nodelist node001  # pin jobs to specific node
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ITRIALSPACE_BASE="${ITRIALSPACE_OUTPUT_DIR}"
MASK_BASE="${ITRIALSPACE_BASE}/inserted_masks"
CT_BASE="${ITRIALSPACE_BASE}/generated_cts"
CASE_LIST_DIR="${ITRIALSPACE_BASE}/nodmaisi_case_lists"

MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
TIME_OVERRIDE=""
DRY_RUN=""
SKIP_EXISTING=""
NODELIST=""

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

# ── Parse arguments ──────────────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN="true"; shift ;;
        --skip-existing)  SKIP_EXISTING="true"; shift ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --time)          TIME_OVERRIDE="$2"; shift 2 ;;
        --nodelist)      NODELIST="$2"; shift 2 ;;
        *)               POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

mkdir -p logs "${CASE_LIST_DIR}"

echo "=================================================================="
echo " iTrialSpace -> NodMAISI — Array Job Submission"
echo " Date: $(date)"
echo " Max concurrent tasks per mode: ${MAX_CONCURRENT}"
[ -n "$DRY_RUN" ] && echo " *** DRY RUN MODE ***"
[ -n "$SKIP_EXISTING" ] && echo " Skip existing: YES (requires synthetic_ct + audit + QC slices)"
[ -n "$NODELIST" ] && echo " Node constraint: ${NODELIST}"
echo "=================================================================="
echo ""

# ── Generate case list from audit.json or mask directory ─────────────────────
generate_case_list() {
    local mode="$1"
    local mask_dir="${MASK_BASE}/${mode}"
    local case_list="${CASE_LIST_DIR}/${mode}_cases.txt"

    if [ -d "${mask_dir}" ]; then
        # Hybrid discovery: use audit.json where available, fall back to
        # file-scan for subdirectories that lack one (e.g. timed-out jobs).
        python3 -c "
import json, os, glob

mask_dir = '${mask_dir}'
case_ids = set()

# --- Pass 1: extract from audit.json files ---
audited_dirs = set()
all_audits = sorted(glob.glob(os.path.join(mask_dir, '**', 'audit.json'), recursive=True))
for af in all_audits:
    audited_dirs.add(os.path.dirname(af))
    try:
        audit = json.load(open(af))
    except Exception:
        continue
    for r in audit.get('records', []):
        if r.get('status') != 'success':
            continue
        cp = r.get('output_combined_path', '')
        if not cp or not os.path.isfile(cp):
            continue
        name = os.path.basename(cp)
        for sfx in ('_mask.nii.gz', '.nii.gz'):
            if name.endswith(sfx):
                name = name[:-len(sfx)]
                break
        case_ids.add(name)

# --- Pass 2: file-scan for dirs WITHOUT audit.json ---
for root, dirs, files in os.walk(mask_dir):
    if root in audited_dirs:
        continue  # already covered by audit
    for f in files:
        if not f.endswith('.nii.gz'):
            continue
        name = f
        for sfx in ('_mask.nii.gz', '.nii.gz'):
            if name.endswith(sfx):
                name = name[:-len(sfx)]
                break
        case_ids.add(name)

cases = sorted(case_ids)
if cases:
    open('${case_list}', 'w').write('\n'.join(cases) + '\n')
print(len(cases))
"
    else
        echo "0"
        return 1
    fi
}

# ── Submit a single mode ────────────────────────────────────────────────────
submit_mode() {
    local idx=$1
    local mode="${MODES[$idx]}"
    local name="${MODE_NAMES[$idx]}"
    local mask_dir="${MASK_BASE}/${mode}"
    local out_dir="${CT_BASE}/${mode}"
    local case_list="${CASE_LIST_DIR}/${mode}_cases.txt"

    printf "  Mode %d: %-30s " "$((idx+1))" "${name}"

    # Generate case list
    local n_cases
    n_cases=$(generate_case_list "$mode" 2>/dev/null)
    if [ -z "$n_cases" ] || [ "$n_cases" -eq 0 ]; then
        echo "SKIPPED (no cases found in ${mask_dir})"
        return
    fi

    # Optionally filter out already-completed cases
    if [ -n "$SKIP_EXISTING" ] && [ -d "${out_dir}" ]; then
        local orig_count=$n_cases
        python3 -c "
import os, glob
case_list = '${case_list}'
out_dir = '${out_dir}'
cases = open(case_list).read().strip().split('\n')

def is_complete(case_dir):
    \"\"\"A case is complete only if it has synthetic_ct, audit, and QC slices.\"\"\"
    if not os.path.isdir(case_dir):
        return False
    has_ct    = os.path.isfile(os.path.join(case_dir, 'synthetic_ct.nii.gz'))
    has_audit = os.path.isfile(os.path.join(case_dir, 'nodmaisi_audit.json'))
    has_qc    = len(glob.glob(os.path.join(case_dir, 'qc', 'qc_slice_*.png'))) > 0
    return has_ct and has_audit and has_qc

pending = []
for c in cases:
    # Check both flat and sub-directory layouts
    case_dir = os.path.join(out_dir, c)
    found = is_complete(case_dir)
    if not found:
        for entry in (os.listdir(out_dir) if os.path.isdir(out_dir) else []):
            sub_dir = os.path.join(out_dir, entry, c)
            if is_complete(sub_dir):
                found = True
                break
    if not found:
        pending.append(c)
open(case_list, 'w').write('\n'.join(pending) + '\n' if pending else '')
print(len(pending))
"
        n_cases=$(wc -l < "${case_list}" | tr -d ' ')
        local skipped=$((orig_count - n_cases))
        if [ "$skipped" -gt 0 ]; then
            printf "(%d cases, skipped %d existing) " "$n_cases" "$skipped"
        else
            printf "(%4d cases) " "$n_cases"
        fi
        if [ "$n_cases" -eq 0 ]; then
            echo "ALL DONE"
            return
        fi
    else
        printf "(%4d cases) " "$n_cases"
    fi

    # SLURM MaxArraySize is typically 1001; chunk large case lists
    local MAX_ARRAY_SIZE=1000
    local offset=0
    local chunk_idx=0
    local job_ids=()

    while [ $offset -lt $n_cases ]; do
        local remaining=$((n_cases - offset))
        local chunk_size=$remaining
        if [ $chunk_size -gt $MAX_ARRAY_SIZE ]; then
            chunk_size=$MAX_ARRAY_SIZE
        fi

        local sbatch_args="--array=0-$((chunk_size - 1))%${MAX_CONCURRENT}"
        sbatch_args+=" --job-name=NM_M$((idx+1))_arr"

        if [ -n "$TIME_OVERRIDE" ]; then
            sbatch_args+=" --time=${TIME_OVERRIDE}"
        fi

        if [ -n "$NODELIST" ]; then
            sbatch_args+=" --nodelist=${NODELIST}"
        fi

        local export_vars="ALL,TRIAL_MODE=${mode},CASE_LIST=${case_list},OUT_DIR=${out_dir},CASE_OFFSET=${offset}"
        sbatch_args+=" --export=${export_vars}"

        if [ -n "$DRY_RUN" ]; then
            echo "Submitted batch job DRY-RUN (chunk $chunk_idx, ${chunk_size} tasks)"
        else
            # Wait for SLURM queue capacity (MaxJobCount=10000)
            while true; do
                local qcount
                qcount=$(squeue -u "$USER" -r --noheader 2>/dev/null | wc -l)
                local avail=$((10000 - qcount))
                if [ "$avail" -ge "$((chunk_size + 200))" ]; then
                    break
                fi
                printf "\n    [waiting] Queue %d/10000, need %d free slots..." "$qcount" "$((chunk_size + 200))"
                sleep 120
            done

            local sbatch_out
            sbatch_out=$(sbatch ${sbatch_args} nodmaisi_array.sub)
            echo "$sbatch_out"
            local jid
            jid=$(echo "$sbatch_out" | grep -oP '\d+')
            if [ -n "$jid" ]; then
                job_ids+=("$jid")
            fi
        fi

        offset=$((offset + chunk_size))
        chunk_idx=$((chunk_idx + 1))
    done

    # Submit post-run audit job (runs after all array chunks finish)
    if [ ${#job_ids[@]} -gt 0 ]; then
        local dep_str
        dep_str=$(IFS=:; echo "${job_ids[*]}")
        local audit_export="ALL,TRIAL_MODE=${mode},OUT_DIR=${out_dir},MASK_DIR=${mask_dir}"
        local audit_out
        audit_out=$(sbatch --dependency=afterany:${dep_str} \
            --job-name="audit_M$((idx+1))" \
            --export="${audit_export}" \
            audit_mode.sub 2>&1)
        local audit_jid
        audit_jid=$(echo "$audit_out" | grep -oP '\d+')
        printf "\n    -> Audit job %s (after %s)\n" "${audit_jid:-??}" "${dep_str}"
    elif [ -n "$DRY_RUN" ]; then
        printf "\n    -> Audit job DRY-RUN (after all chunks)\n"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
if [ ${#POSITIONAL[@]} -eq 0 ] 2>/dev/null || [ $# -eq 0 ]; then
    echo "Submitting all ${#MODES[@]} trial modes (1-${#MODES[@]})..."
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
            echo "  Invalid mode number: $num (valid: 1-${#MODES[@]})"
        fi
    done
fi

echo ""
echo "--------------------------------------------------------------"
echo "Case lists:  ${CASE_LIST_DIR}/"
echo "Check status:  squeue -u $USER"
echo "Cancel all:    scancel -u $USER --name='NM_M*'"
echo "Logs:          ${SCRIPT_DIR}/logs/"
echo "--------------------------------------------------------------"
