#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VAL_SIZE=${VAL_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
T_MAX=${T_MAX:-3000}
RUN_DIR=${RUN_DIR:-"result_n2s/Fixed_Assignment/run_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${RUN_DIR}/fixed_assignment" "${RUN_DIR}/full_mvn2s"

if ! full_result=$(find_latest_valid_result result_n2s/Fleet_Scalability \
    '*/K_2/*_results_*.json' 20 2 proposed "${VAL_SIZE}" "${T_MAX}"); then
    echo "[ERROR] Missing matching Fleet Scalability K=2 result. Run fleet_scalability_eval.sh first." >&2
    exit 1
fi

result_dir="${RUN_DIR}/fixed_assignment"
if fixed_result=$(result_is_complete "${result_dir}" 20 2 proposed "${VAL_SIZE}" "${T_MAX}"); then
    echo "[SKIP] Complete fixed-assignment result: ${fixed_result}"
else
    if ! model_dir=$(find_latest_training_run outputs/mvpdtsp_fixed_20 \
        mvpdtsp20_mv2_fixed_proposed_log_ mvpdtsp_fixed 20 2 proposed complete); then
        echo "[ERROR] Missing completed fixed-assignment model. Run fixed_assignment_train.sh first." >&2
        exit 1
    fi
    checkpoint=$(latest_checkpoint "${model_dir}")
    cmd=("${PYTHON}" -u run.py --eval_only --problem mvpdtsp_fixed
        --graph_size 20 --num_vehicles 2 --objective proposed --seed 1234
        --val_dataset ./datasets/pdp_20.pkl --val_size "${VAL_SIZE}"
        --val_batch_size "${VAL_BATCH_SIZE}" --T_max "${T_MAX}"
        --load_path "${checkpoint}" --results_dir "${result_dir}"
        --run_name fixed_assignment_eval --no_tb)
    write_command "${result_dir}/command.txt" "${cmd[@]}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        run_or_print "${cmd[@]}"
        fixed_result="${result_dir}/DRY_RUN.json"
    else
        "${cmd[@]}" 2>&1 | tee "${result_dir}/evaluation.log"
        fixed_result=$(latest_result "${result_dir}")
    fi
fi

run_or_print "${PYTHON}" "${TOOLS}" write-manifest --source "${full_result}" \
    --output "${RUN_DIR}/full_mvn2s/source_manifest.json" --role full_mvn2s
run_or_print "${PYTHON}" "${TOOLS}" summarize --mode fixed \
    --output "${RUN_DIR}/summary.csv" \
    --entry "fixed_assignment=${fixed_result}" --entry "full_mvn2s=${full_result}"
echo "[INFO] Fixed Assignment artifacts: ${RUN_DIR}"
