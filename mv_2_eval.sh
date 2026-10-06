#!/usr/bin/env bash
# Activate the rlor environment before running. Both configurations run sequentially.
# Overrides: MODEL_PATH_50, MODEL_PATH_100, T_MAX (or T_MAX_50/T_MAX_100),
# VAL_SIZE, VAL_BATCH_SIZE, PRINT_SOLUTION, CUDA_VISIBLE_DEVICES, PYTHON.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trap 'echo "[ERROR] mv_2_eval.sh failed at line ${LINENO}." >&2' ERR
cd "${SCRIPT_DIR}"

PYTHON=${PYTHON:-python}
VAL_SIZE=${VAL_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
PRINT_SOLUTION=${PRINT_SOLUTION:-0}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1,3}"

# Use the highest numbered available epoch in the two requested training runs.
# This selects the latest saved checkpoint, not a claim of best validation score.
latest_checkpoint() {
    local model_dir=$1
    find "${model_dir}" -maxdepth 1 -type f -name 'epoch-*.pt' | sort -V | tail -n 1
}

MODEL_DIR_20='./outputs/mvpdtsp_20/mvpdtsp20_makespan_log_20260130T200924'
MODEL_DIR_50='./outputs/mvpdtsp_50/mvpdtsp50_makespan_log_20260311T220152'
MODEL_DIR_100='./outputs/mvpdtsp_100/mvpdtsp100_makespan_log_20260316T105307'
MODEL_PATH_20=${MODEL_PATH_20:-$(latest_checkpoint "${MODEL_DIR_20}")}
MODEL_PATH_50=${MODEL_PATH_50:-$(latest_checkpoint "${MODEL_DIR_50}")}
MODEL_PATH_100=${MODEL_PATH_100:-$(latest_checkpoint "${MODEL_DIR_100}")}

# Validate both configurations before starting either evaluation.
for graph_size in 20 50 100; do
    model_var="MODEL_PATH_${graph_size}"
    if [[ ! -f "${!model_var}" ]]; then
        echo "[ERROR] Checkpoint not found for ${graph_size} nodes: ${!model_var}" >&2
        exit 1
    fi
    if [[ ! -f "datasets/pdp_${graph_size}.pkl" ]]; then
        echo "[ERROR] Missing datasets/pdp_${graph_size}.pkl" >&2
        exit 1
    fi
done

mkdir -p "${SCRIPT_DIR}/result_n2s"
RUN_DIR=$(mktemp -d "${SCRIPT_DIR}/result_n2s/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "[INFO] Results: ${RUN_DIR}"

# for graph_size in 20 50 100; do
for graph_size in 20; do
    model_var="MODEL_PATH_${graph_size}"
    steps_var="T_MAX_${graph_size}"
    steps=${!steps_var:-${T_MAX:-3000}}
    result_dir="${RUN_DIR}/n2s_${graph_size}_mv2"
    mkdir -p "${result_dir}"
    cmd=(
        "${PYTHON}" -u run.py
        --eval_only --problem mvpdtsp
        --graph_size "${graph_size}" --num_vehicles 2 --makespan
        --val_size "${VAL_SIZE}" --val_batch_size "${VAL_BATCH_SIZE}"
        --T_max "${steps}"
        --run_name "mvpdtsp${graph_size}_mv2_makespanw1_eval"
        --load_path "${!model_var}"
        --val_dataset "./datasets/pdp_${graph_size}.pkl"
        --results_dir "${result_dir}"
        --no_tb
    )
    if [[ "${PRINT_SOLUTION}" == 1 ]]; then
        cmd+=(--print_solution)
    fi
    # Save an executable command record including the chosen GPU visibility.
    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "${result_dir}/command.txt"
    echo "[INFO] Evaluating ${graph_size} nodes, 2 vehicles, makespan weight 1"
    echo "[INFO] Checkpoint: ${!model_var}; T_max=${steps}"
    "${cmd[@]}" 2>&1 | tee "${result_dir}/evaluation.log"
done

echo "[INFO] Both evaluations completed. JSON, statistics (*_summary.txt), and logs: ${RUN_DIR}"
