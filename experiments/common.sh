#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
N2S_DIR=$(cd "${EXPERIMENT_DIR}/.." && pwd)
PYTHON=${PYTHON:-python}
TOOLS="${EXPERIMENT_DIR}/experiment_tools.py"

latest_checkpoint() {
    local run_dir=$1
    find "${run_dir}" -maxdepth 1 -type f -name 'epoch-*.pt' | sort -V | tail -n 1
}

latest_result() {
    local result_dir=$1
    find "${result_dir}" -maxdepth 1 -type f -name '*_results_*.json' | sort | tail -n 1
}

find_latest_training_run() {
    local root=$1
    local prefix=$2
    local problem=$3
    local graph_size=$4
    local num_vehicles=$5
    local objective=$6
    local wanted_state=$7
    local run_dir state

    while IFS= read -r run_dir; do
        [[ -z "${run_dir}" ]] && continue
        set +e
        state=$("${PYTHON}" "${TOOLS}" training-state \
            --run-dir "${run_dir}" --problem "${problem}" \
            --graph-size "${graph_size}" --num-vehicles "${num_vehicles}" \
            --objective "${objective}" 2>/dev/null)
        local status=$?
        set -e
        if [[ "${wanted_state}" == complete && ${status} -eq 0 ]]; then
            echo "${run_dir}"
            return 0
        fi
        if [[ "${wanted_state}" == incomplete && ${status} -eq 1 && "${state}" == incomplete ]]; then
            echo "${run_dir}"
            return 0
        fi
    done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}*" 2>/dev/null | sort -r)
    return 1
}

result_is_complete() {
    local result_dir=$1
    local graph_size=$2
    local num_vehicles=$3
    local objective=$4
    local val_size=$5
    local t_max=$6
    "${PYTHON}" "${TOOLS}" result-complete \
        --result-dir "${result_dir}" --graph-size "${graph_size}" \
        --num-vehicles "${num_vehicles}" --objective "${objective}" \
        --val-size "${val_size}" --t-max "${t_max}"
}

result_file_is_valid() {
    local result_file=$1
    local graph_size=$2
    local num_vehicles=$3
    local objective=$4
    local val_size=$5
    local t_max=$6
    "${PYTHON}" "${TOOLS}" result-file-valid \
        --result-file "${result_file}" --graph-size "${graph_size}" \
        --num-vehicles "${num_vehicles}" --objective "${objective}" \
        --val-size "${val_size}" --t-max "${t_max}" >/dev/null
}

find_latest_valid_result() {
    local root=$1
    local path_pattern=$2
    local graph_size=$3
    local num_vehicles=$4
    local objective=$5
    local val_size=$6
    local t_max=$7
    local candidate
    while IFS= read -r candidate; do
        [[ -z "${candidate}" ]] && continue
        if result_file_is_valid "${candidate}" "${graph_size}" "${num_vehicles}" \
            "${objective}" "${val_size}" "${t_max}"; then
            echo "${candidate}"
            return 0
        fi
    done < <(find "${root}" -path "${path_pattern}" -type f 2>/dev/null | sort -r)
    return 1
}

run_or_print() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '[DRY RUN] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

write_command() {
    local output=$1
    shift
    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES-}"
        printf '%q ' "$@"
        printf '\n'
    } > "${output}"
}

run_training_logged() {
    local log_root=$1
    local label=$2
    shift 2
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        run_or_print "$@"
        return
    fi
    local launch_dir="${log_root}/launch_$(date +%Y%m%d_%H%M%S)_${label}"
    mkdir -p "${launch_dir}"
    write_command "${launch_dir}/command.txt" "$@"
    "$@" 2>&1 | tee "${launch_dir}/training.log"
}
