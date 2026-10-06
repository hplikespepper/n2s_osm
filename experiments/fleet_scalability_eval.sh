#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VAL_SIZE=${VAL_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
T_MAX=${T_MAX:-3000}
RUN_DIR=${RUN_DIR:-"result_n2s/Fleet_Scalability/run_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${RUN_DIR}"

declare -a summary_entries=()
for k in 1 2 3 4 5; do
    result_dir="${RUN_DIR}/K_${k}"
    mkdir -p "${result_dir}"
    if existing=$(result_is_complete "${result_dir}" 20 "${k}" proposed "${VAL_SIZE}" "${T_MAX}"); then
        echo "[SKIP] Complete Fleet Scalability result: ${existing}"
        summary_entries+=(--entry "K_${k}=${existing}")
        continue
    fi

    case "${k}" in
        1) checkpoint="pre-trained/pdtsp/20/epoch-156.pt" ;;
        2) checkpoint="outputs/mvpdtsp_20/mvpdtsp20_makespan_log_20260130T200924/epoch-198.pt" ;;
        *)
            prefix="mvpdtsp20_mv${k}_proposed_log_"
            if ! model_dir=$(find_latest_training_run outputs/mvpdtsp_20 "${prefix}" mvpdtsp 20 "${k}" proposed complete); then
                echo "[ERROR] No completed proposed model for N=20 K=${k}. Run fleet_scalability_train.sh first." >&2
                exit 1
            fi
            checkpoint=$(latest_checkpoint "${model_dir}")
            ;;
    esac
    [[ -f "${checkpoint}" ]] || { echo "[ERROR] Missing checkpoint: ${checkpoint}" >&2; exit 1; }

    cmd=("${PYTHON}" -u run.py --eval_only --problem mvpdtsp
        --graph_size 20 --num_vehicles "${k}" --objective proposed --seed 1234
        --val_dataset ./datasets/pdp_20.pkl --val_size "${VAL_SIZE}"
        --val_batch_size "${VAL_BATCH_SIZE}" --T_max "${T_MAX}"
        --load_path "${checkpoint}" --results_dir "${result_dir}"
        --run_name "fleet_n20_k${k}_eval" --no_tb)
    write_command "${result_dir}/command.txt" "${cmd[@]}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        run_or_print "${cmd[@]}"
        summary_entries+=(--entry "K_${k}=${result_dir}/DRY_RUN.json")
    else
        "${cmd[@]}" 2>&1 | tee "${result_dir}/evaluation.log"
        result=$(latest_result "${result_dir}")
        [[ -n "${result}" ]] || { echo "[ERROR] Evaluation produced no JSON for K=${k}" >&2; exit 1; }
        summary_entries+=(--entry "K_${k}=${result}")
    fi
done

summary_cmd=("${PYTHON}" "${TOOLS}" summarize --mode fleet --output "${RUN_DIR}/summary.csv" "${summary_entries[@]}")
run_or_print "${summary_cmd[@]}"
echo "[INFO] Fleet Scalability artifacts: ${RUN_DIR}"
