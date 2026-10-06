#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
VAL_SIZE=${VAL_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
T_MAX=${T_MAX:-3000}
RUN_DIR=${RUN_DIR:-"result_n2s/Objective_Ablation/run_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${RUN_DIR}"

if ! fleet_result=$(find_latest_valid_result result_n2s/Fleet_Scalability \
    '*/K_2/*_results_*.json' 20 2 proposed "${VAL_SIZE}" "${T_MAX}"); then
    echo "[ERROR] Missing matching Fleet Scalability K=2 result. Run fleet_scalability_eval.sh first." >&2
    exit 1
fi

summary_entries=()
for objective in distance only_makespan; do
    result_dir="${RUN_DIR}/${objective}"
    mkdir -p "${result_dir}"
    if existing=$(result_is_complete "${result_dir}" 20 2 "${objective}" "${VAL_SIZE}" "${T_MAX}"); then
        result="${existing}"
    else
        prefix="mvpdtsp20_mv2_${objective}_log_"
        if ! model_dir=$(find_latest_training_run outputs/mvpdtsp_20 "${prefix}" mvpdtsp 20 2 "${objective}" complete); then
            echo "[ERROR] Missing completed ${objective} model. Run objective_ablation_train.sh first." >&2
            exit 1
        fi
        checkpoint=$(latest_checkpoint "${model_dir}")
        cmd=("${PYTHON}" -u run.py --eval_only --problem mvpdtsp
            --graph_size 20 --num_vehicles 2 --objective "${objective}" --seed 1234
            --val_dataset ./datasets/pdp_20.pkl --val_size "${VAL_SIZE}"
            --val_batch_size "${VAL_BATCH_SIZE}" --T_max "${T_MAX}"
            --load_path "${checkpoint}" --results_dir "${result_dir}"
            --run_name "objective_${objective}_eval" --no_tb)
        write_command "${result_dir}/command.txt" "${cmd[@]}"
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            run_or_print "${cmd[@]}"
            result="${result_dir}/DRY_RUN.json"
        else
            "${cmd[@]}" 2>&1 | tee "${result_dir}/evaluation.log"
            result=$(latest_result "${result_dir}")
        fi
    fi
    summary_entries+=(--entry "${objective}=${result}")
done

mkdir -p "${RUN_DIR}/proposed"
run_or_print "${PYTHON}" "${TOOLS}" write-manifest --source "${fleet_result}" \
    --output "${RUN_DIR}/proposed/source_manifest.json" --role proposed
summary_entries+=(--entry "proposed=${fleet_result}")
run_or_print "${PYTHON}" "${TOOLS}" summarize --mode objective \
    --output "${RUN_DIR}/summary.csv" "${summary_entries[@]}"
echo "[INFO] Objective Ablation artifacts: ${RUN_DIR}"
