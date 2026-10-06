#!/usr/bin/env bash
# Run the requested MVN2S comparisons with starts parallelized per instance.
# Example: VAL_SIZE=256 NUM_STARTS=4 START_WORKERS=4 NUM_WORKERS=16 bash mv_lkh3.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trap 'echo "[ERROR] mv_lkh3.sh failed at line ${LINENO}." >&2' ERR
cd "${SCRIPT_DIR}"

PYTHON=${PYTHON:-python}
LKH_PATH=${LKH_PATH:-${SCRIPT_DIR}/../LKH3/LKH-3.0.14/LKH}
VAL_SIZE=${VAL_SIZE:-256}
MAX_TRIALS_LIST=${MAX_TRIALS_LIST:-"5000 10000"}
NUM_STARTS=${NUM_STARTS:-4}
START_WORKERS=${START_WORKERS:-4}
NUM_WORKERS=${NUM_WORKERS:-16}
SEED=${SEED:-42}
CONFIGS=${CONFIGS:-"50:5 100:2 100:5 20:2 50:2"}

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -c 'import numpy, torch; print("[INFO] Python dependencies available; CPU execution")'
if [[ ! -x "${LKH_PATH}" ]]; then
    echo "[ERROR] LKH executable not found or not executable: ${LKH_PATH}" >&2
    exit 1
fi
for graph_size in 20 50 100; do
    if [[ ! -f "datasets/pdp_${graph_size}.pkl" ]]; then
        echo "[ERROR] Missing datasets/pdp_${graph_size}.pkl" >&2
        exit 1
    fi
done

mkdir -p "${SCRIPT_DIR}/result_lkh3"
if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_DIR=$(mktemp -d "${SCRIPT_DIR}/result_lkh3/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
else
    mkdir -p "${RUN_DIR}"
fi
echo "[INFO] Results: ${RUN_DIR}"
echo "[INFO] val_size=${VAL_SIZE}, max_trials_list=${MAX_TRIALS_LIST}, starts=${NUM_STARTS}, start_workers=${START_WORKERS}, instance_workers=${NUM_WORKERS}, no time limit"
echo "[INFO] configurations=${CONFIGS}"

for config in ${CONFIGS}; do
    IFS=: read -r graph_size num_vehicles <<< "${config}"
    for max_trials in ${MAX_TRIALS_LIST}; do
        name="lkh3_${graph_size}_mv${num_vehicles}_t${max_trials}"
        output="${RUN_DIR}/${name}.json"
        echo "[INFO] Running ${name}: D + $((num_vehicles - 1)) * M; MAX_TRIALS=${max_trials}/start"
        "${PYTHON}" lkh3_baseline.py \
            --lkh_path "${LKH_PATH}" \
            --val_dataset "./datasets/pdp_${graph_size}.pkl" \
            --graph_size "${graph_size}" \
            --num_vehicles "${num_vehicles}" \
            --val_size "${VAL_SIZE}" \
            --max_trials "${max_trials}" \
            --num_starts "${NUM_STARTS}" \
            --start_workers "${START_WORKERS}" \
            --num_workers "${NUM_WORKERS}" \
            --seed "${SEED}" \
            --output "${output}" 2>&1 | tee "${RUN_DIR}/${name}.log"
        "${PYTHON}" baseline_summary.py "${RUN_DIR}"
    done
done

echo "[INFO] All requested configurations completed: ${RUN_DIR}"
echo "[INFO] Summary: ${RUN_DIR}/experiment_summary.csv"
