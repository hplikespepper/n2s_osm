#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=6,7
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

PYTHON_BIN=${PYTHON_BIN:-/mnt/beegfs/home/phuang2023/miniconda3/envs/rlor/bin/python}
VAL_DATASET="${PROJECT_DIR}/../pdp_n2s/datasets/pdp_20.pkl"
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${VAL_DATASET}" ]] || { echo "Dataset not found: ${VAL_DATASET}" >&2; exit 1; }

if [[ -n "${VEHICLE_CAPACITIES:-}" ]]; then
    read -r -a capacities <<< "${VEHICLE_CAPACITIES}"
    [[ ${#capacities[@]} -eq 2 ]] || { echo "VEHICLE_CAPACITIES must contain two values" >&2; exit 1; }
    capacity_args=(--vehicle_capacities "${capacities[@]}")
else
    capacity_args=(--capacity "${CAPACITY:-3}")
fi

OUTPUT_ROOT="${SCRIPT_DIR}/output"
RESULT_ROOT="${SCRIPT_DIR}/result"
mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
RESULTS_DIR=$(mktemp -d "${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_XXXX")
RUN_NAME=${RESULTS_DIR##*/}
OUTPUT_RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
MODEL_ROOT="${OUTPUT_RUN_DIR}/models"
mkdir -p "${OUTPUT_RUN_DIR}"

common_args=(--problem mvpdtsp --graph_size 20 --num_vehicles 2
    "${capacity_args[@]}" --objective proposed --seed 1234)
train_cmd=("${PYTHON_BIN}" -u run.py "${common_args[@]}"
    --warm_up 2 --max_grad_norm 0.05
    --batch_size 600 --epoch_size 12000 --epoch_end 1 --T_train 250
    --val_dataset "${VAL_DATASET}" --val_size 1000 --val_batch_size 1000
    --T_max 1500 --checkpoint_epochs 1 --run_name n20_k2_capacity
    --output_dir "${MODEL_ROOT}" --results_dir "${RESULTS_DIR}"
    --log_dir "${OUTPUT_RUN_DIR}/tensorboard")
printf '%q ' "${train_cmd[@]}" > "${RESULTS_DIR}/train_command.txt"
printf '\n' >> "${RESULTS_DIR}/train_command.txt"
echo "Training output: ${OUTPUT_RUN_DIR}"
echo "Training results: ${RESULTS_DIR}"
"${train_cmd[@]}" 2>&1 | tee "${RESULTS_DIR}/train.log"

shopt -s nullglob
checkpoints=("${MODEL_ROOT}"/mvpdtsp_20/*/epoch-0.pt)
shopt -u nullglob
[[ ${#checkpoints[@]} -eq 1 ]] || { echo "Expected one epoch-0.pt under ${MODEL_ROOT}" >&2; exit 1; }
checkpoint=${checkpoints[0]}

eval_cmd=("${PYTHON_BIN}" -u run.py --eval_only "${common_args[@]}"
    --val_dataset "${VAL_DATASET}" --val_size 1000 --val_batch_size 1000
    --T_max 1500 --load_path "${checkpoint}"
    --output_dir "${MODEL_ROOT}" --results_dir "${RESULTS_DIR}"
    --run_name n20_k2_capacity_eval --no_saving --no_tb)
printf '%q ' "${eval_cmd[@]}" > "${RESULTS_DIR}/eval_command.txt"
printf '\n' >> "${RESULTS_DIR}/eval_command.txt"
"${eval_cmd[@]}" 2>&1 | tee "${RESULTS_DIR}/eval.log"

shopt -s nullglob
result_files=("${RESULTS_DIR}"/mvpdtsp_results_*.json)
shopt -u nullglob
[[ ${#result_files[@]} -eq 1 ]] || { echo "Expected one evaluation JSON under ${RESULTS_DIR}" >&2; exit 1; }
echo "Checkpoint: ${checkpoint}"
echo "Evaluation result: ${result_files[0]}"
