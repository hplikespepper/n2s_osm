#!/usr/bin/env bash
# 评估 OSM 20 客户节点、2 辆车的 epoch-198；由用户手动运行。
# 可通过环境变量覆盖 PYTHON、CUDA_VISIBLE_DEVICES、MODEL_PATH、VAL_DATASET、
# VAL_SIZE、VAL_BATCH_SIZE、VAL_M、T_MAX、SEED、PRINT_SOLUTION、RESULTS_DIR。
# 额外 run.py 参数可直接附在脚本命令末尾。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'echo "[ERROR] Evaluation failed at line ${LINENO}." >&2' ERR

PYTHON=${PYTHON:-/home/data/phuang/miniconda3/envs/rlor/bin/python}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
MODEL_PATH=${MODEL_PATH:-./outputs/mvpdtsp_osm_20/mvpdtsp_osm20_mv2_proposed_20261006T190051/epoch-198.pt}
VAL_DATASET=${VAL_DATASET:-./datasets/osm_val_20.pkl}
VAL_SIZE=${VAL_SIZE:-100}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-100}
VAL_M=${VAL_M:-1}
T_MAX=${T_MAX:-5000}
SEED=${SEED:-1234}
PRINT_SOLUTION=${PRINT_SOLUTION:-0}

for input_file in "${MODEL_PATH}" "${VAL_DATASET}"; do
    if [[ ! -f "${input_file}" ]]; then
        echo "[ERROR] File not found: ${input_file}" >&2
        exit 1
    fi
done
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON}" >&2
    exit 1
fi

if [[ -n "${RESULTS_DIR:-}" ]]; then
    mkdir -p "${RESULTS_DIR}"
else
    mkdir -p ./result_n2s/mvpdtsp_osm20_mv2
    RESULTS_DIR=$(mktemp -d "./result_n2s/mvpdtsp_osm20_mv2/epoch198_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi

cmd=(
    "${PYTHON}" -u run.py
    --eval_only --problem mvpdtsp_osm
    --graph_size 20 --num_vehicles 2 --objective proposed
    --load_path "${MODEL_PATH}"
    --val_dataset "${VAL_DATASET}"
    --val_size "${VAL_SIZE}" --val_batch_size "${VAL_BATCH_SIZE}"
    --val_m "${VAL_M}" --T_max "${T_MAX}"
    --init_val_met random --seed "${SEED}"
    --run_name mvpdtsp_osm20_mv2_proposed_epoch198_eval
    --results_dir "${RESULTS_DIR}" --no_tb --no_saving
)
if [[ "${PRINT_SOLUTION}" == 1 ]]; then
    cmd+=(--print_solution)
fi
cmd+=("$@")

# 记录工作目录及完整命令，便于复现本次评估。
{
    printf 'cd -- %q\n' "${PWD}"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
} > "${RESULTS_DIR}/command.txt"
echo "[INFO] Checkpoint: ${MODEL_PATH}"
echo "[INFO] Dataset: ${VAL_DATASET}; samples=${VAL_SIZE}; T_max=${T_MAX}; val_m=${VAL_M}"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES}; results: ${RESULTS_DIR}"
"${cmd[@]}" 2>&1 | tee "${RESULTS_DIR}/evaluation.log"
echo "[INFO] Evaluation completed. Results and logs: ${RESULTS_DIR}"
