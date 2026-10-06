#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VAL_SIZE=${VAL_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
T_MAX=${T_MAX:-3000}
HISTORY_INTERVAL=${HISTORY_INTERVAL:-100}
RUN_DIR=${RUN_DIR:-"result_n2s/Search_Convergence/run_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${RUN_DIR}"

declare -a entries=()
for k in 2 5; do
    result_dir="${RUN_DIR}/K_${k}"
    mkdir -p "${result_dir}"
    if existing=$(result_is_complete "${result_dir}" 50 "${k}" proposed "${VAL_SIZE}" "${T_MAX}"); then
        if "${PYTHON}" - "${existing}" "${HISTORY_INTERVAL}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
interval = int(sys.argv[2])
expected = list(range(0, d['T_max'] + 1, interval))
if expected[-1] != d['T_max']:
    expected.append(d['T_max'])
raise SystemExit(0 if d.get('component_history_steps') == expected else 1)
PY
        then
            echo "[SKIP] Complete convergence result: ${existing}"
            entries+=(--entry "K_${k}=${existing}")
            continue
        fi
    fi

    case "${k}" in
        2) checkpoint="outputs/mvpdtsp_50/mvpdtsp50_makespan_log_20260311T220152/epoch-199.pt" ;;
        5) checkpoint="outputs/mvpdtsp_50/mvpdtsp50_mv5_makespanw4_log_20260506T152110/epoch-198.pt" ;;
    esac
    [[ -f "${checkpoint}" ]] || { echo "[ERROR] Missing checkpoint: ${checkpoint}" >&2; exit 1; }
    cmd=("${PYTHON}" -u run.py --eval_only --problem mvpdtsp
        --graph_size 50 --num_vehicles "${k}" --objective proposed --seed 1234
        --val_dataset ./datasets/pdp_50.pkl --val_size "${VAL_SIZE}"
        --val_batch_size "${VAL_BATCH_SIZE}" --T_max "${T_MAX}"
        --record_component_history --history_interval "${HISTORY_INTERVAL}"
        --load_path "${checkpoint}" --results_dir "${result_dir}"
        --run_name "convergence_n50_k${k}_eval" --no_tb)
    write_command "${result_dir}/command.txt" "${cmd[@]}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        run_or_print "${cmd[@]}"
        entries+=(--entry "K_${k}=${result_dir}/DRY_RUN.json")
    else
        "${cmd[@]}" 2>&1 | tee "${result_dir}/evaluation.log"
        result=$(latest_result "${result_dir}")
        entries+=(--entry "K_${k}=${result}")
    fi
done

run_or_print "${PYTHON}" "${TOOLS}" summarize --mode convergence \
    --output "${RUN_DIR}/history_summary.csv" "${entries[@]}"
echo "[INFO] Search Convergence artifacts: ${RUN_DIR}"
