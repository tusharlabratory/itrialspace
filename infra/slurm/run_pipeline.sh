#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — drive the iTrialSpace pipeline end-to-end for one trial mode
#   trials (manifest) -> insertion (label-23 masks + audit) -> synthesis (CT)
#
# Stages are chained with SLURM dependencies that wait for the *array* jobs to
# finish (not just their submitters), so synthesis only runs after every mask +
# audit exists. Manifests are discovered recursively (modes 11-13 nest them under
# a dataset subdirectory). Run from the repo root.
#
# Usage:
#   infra/slurm/run_pipeline.sh <mode_number> [stage]
#     <mode_number>  1..13
#     [stage]        all (default) | trials | insert | synth
#                    (_insert_chain / _synth_chain are internal dependency steps)
#
# Examples:
#   infra/slurm/run_pipeline.sh 1
#   for m in $(seq 1 13); do infra/slurm/run_pipeline.sh $m; done
#
# Run size comes from the mode .sub files (edit those / see docs §5).
# Paths/conda-env come from your .env via infra/slurm/env.sh.
# ============================================================================
set -euo pipefail

MODE="${1:?usage: run_pipeline.sh <mode 1-13> [all|trials|insert|synth]}"
STAGE="${2:-all}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
cd "${PROJ_DIR}"

TRIALS_SUB=$(ls "${HERE}"/trials/mode${MODE}_*.sub 2>/dev/null | head -1)
INSERT_SUB="${HERE}/mask_inserter/mode${MODE}_insert_masks_array.sub"
SYNTH_SUB="${HERE}/synthesis/mode1_nodmaisi_array.sub"   # generic; driven by env overrides
SELF="${HERE}/run_pipeline.sh"
G="${ITRIALSPACE_OUTPUT_DIR}"
ML="${G}/manifest_lists/mode${MODE}.txt"
mkdir -p "${G}/manifest_lists" "${G}/nodmaisi_case_lists" logs

# Recursively find a mode's manifest CSVs (handles the modes 11-13 dataset subdirs).
list_manifests() { find "${G}/manifests/mode${MODE}_"*/ -name '*.csv' 2>/dev/null | sort; }

case "${STAGE}" in

  trials|all)
    [ -f "${TRIALS_SUB}" ] || { echo "No trials .sub for mode ${MODE}"; exit 1; }
    TJOB=$(sbatch --parsable "${TRIALS_SUB}")
    echo "[mode ${MODE}] trials -> job ${TJOB}"
    [ "${STAGE}" = "trials" ] && exit 0
    # After trials finish, build the manifest list and submit insertion (+ chain synth).
    sbatch --parsable --dependency=afterok:${TJOB} \
        --job-name=drv_ins_M${MODE} --partition="${SLURM_PARTITION_CPU}" --time=00:15:00 \
        --output=logs/drv_ins_M${MODE}_%j.out --error=logs/drv_ins_M${MODE}_%j.err \
        --wrap="bash '${SELF}' ${MODE} _insert_chain"
    echo "[mode ${MODE}] insertion+synthesis chained after trials"
    ;;

  insert)
    # trials already done — submit insertion array now (no synth chain)
    list_manifests > "${ML}"; N=$(wc -l < "${ML}")
    [ "${N}" -gt 0 ] || { echo "No manifests for mode ${MODE} (run trials first)"; exit 1; }
    sbatch --array=0-$((N-1)) --export=ALL,MANIFEST_LIST="${ML}" "${INSERT_SUB}"
    echo "[mode ${MODE}] insertion -> ${N} manifest task(s)"
    ;;

  _insert_chain)
    # runs as a dependency job AFTER trials: submit the insert array, then chain
    # the synth driver on that ARRAY (so it waits for all masks/audits).
    list_manifests > "${ML}"; N=$(wc -l < "${ML}")
    [ "${N}" -gt 0 ] || { echo "No manifests for mode ${MODE}"; exit 1; }
    IARR=$(sbatch --parsable --array=0-$((N-1)) --export=ALL,MANIFEST_LIST="${ML}" "${INSERT_SUB}")
    echo "[mode ${MODE}] insertion array -> ${IARR} (${N} task(s))"
    sbatch --parsable --dependency=afterok:${IARR} \
        --job-name=drv_syn_M${MODE} --partition="${SLURM_PARTITION_CPU}" --time=00:15:00 \
        --output=logs/drv_syn_M${MODE}_%j.out --error=logs/drv_syn_M${MODE}_%j.err \
        --wrap="bash '${SELF}' ${MODE} synth"
    ;;

  synth)
    # one synthesis array per audit.json (one task per case). Audits exist now.
    nsub=0
    while IFS= read -r audit; do
        [ -z "${audit}" ] && continue
        sub=$(basename "$(dirname "${audit}")")
        cl="${G}/nodmaisi_case_lists/mode${MODE}_${sub}.txt"
        n=$(python "${HERE}/_gen_caselist.py" "${audit}" "${cl}")
        [ "${n}" -gt 0 ] || continue
        sbatch --array=0-$((n-1)) \
            --export=ALL,AUDIT_PATH="${audit}",CASE_LIST="${cl}",OUT_DIR="${G}/generated_cts/mode${MODE}/${sub}" \
            "${SYNTH_SUB}"
        nsub=$((nsub+1))
    done < <(find "${G}/inserted_masks/mode${MODE}_"*/ -name audit.json 2>/dev/null | sort)
    echo "[mode ${MODE}] synthesis -> ${nsub} array job(s)"
    [ "${nsub}" -gt 0 ] || { echo "[mode ${MODE}] no audits found — insertion incomplete?"; exit 1; }
    ;;

  *) echo "unknown stage: ${STAGE}"; exit 1 ;;
esac
