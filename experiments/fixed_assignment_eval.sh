#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VAL_SIZE=${VAL_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
T_MAX=${T_MAX:-3000}
RUN_DIR=${RUN_DIR:-"result_n2s/Fixed_Assignment/run_$(date +%Y%m%d_%H%M%S)"}
FULL_MVN2S_RESULT=${FULL_MVN2S_RESULT:-"result_n2s/run_20261005_202445_6h2JWG/n2s_20_mv2/mvpdtsp_results_20261005_202624.json"}
mkdir -p "${RUN_DIR}/fixed_assignment" "${RUN_DIR}/full_mvn2s"

full_result="${FULL_MVN2S_RESULT}"
if [[ ! -f "${full_result}" ]]; then
    echo "[ERROR] Missing full MVN2S result: ${full_result}" >&2
    exit 1
fi
if ! "${PYTHON}" - "${full_result}" "${VAL_SIZE}" "${T_MAX}" <<'PY'
import json
import sys

path, val_size, t_max = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path) as f:
    data = json.load(f)
instances = data.get('instances', [])
objective = data.get('objective')
objective_is_proposed = objective == 'proposed' or (
    objective is None
    and all(instance.get('makespan_weight') == 1 for instance in instances)
)
valid = (
    data.get('graph_size') == 20
    and data.get('num_vehicles') == 2
    and objective_is_proposed
    and data.get('val_size') == val_size
    and data.get('T_max') == t_max
    and len(instances) == val_size
)
raise SystemExit(0 if valid else 1)
PY
then
    echo "[ERROR] Full MVN2S result does not match N=20, K=2, objective=proposed, VAL_SIZE=${VAL_SIZE}, T_MAX=${T_MAX}: ${full_result}" >&2
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
