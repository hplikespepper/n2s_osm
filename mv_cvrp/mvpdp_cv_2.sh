#!/usr/bin/env bash
set -euo pipefail

vehicles=2
# 50-node experiment: edit these values independently.
capacity_50=10
warm_up_50=1.5
max_grad_norm_50=0.15
batch_size_50=600
epoch_size_50=12000
epoch_end_50=200
t_train_50=250
t_max_50=2000
eval_t_max_50=3000
lr_model_50=8e-5
lr_critic_50=2e-5
train_val_size_50=1000
train_val_batch_size_50=1000
eval_size_50=256
eval_batch_size_50=256

# 100-node experiment: edit these values independently.
capacity_100=15
warm_up_100=1
max_grad_norm_100=0.3
batch_size_100=600
epoch_size_100=12000
epoch_end_100=200
t_train_100=250
t_max_100=3000
eval_t_max_100=3000
lr_model_100=8e-5
lr_critic_100=2e-5
train_val_size_100=1000
train_val_batch_size_100=1000
eval_size_100=256
eval_batch_size_100=256

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "${script_dir}/.." && pwd)
cd "${project_dir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

python_bin=${PYTHON_BIN:-/mnt/beegfs/home/phuang2023/miniconda3/envs/rlor/bin/python}
[[ -x "${python_bin}" ]] || { echo "Python not found: ${python_bin}" >&2; exit 1; }

read_checkpoint_epoch() {
    "${python_bin}" - "$1" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu')
epoch = checkpoint.get('epoch')
if not isinstance(epoch, int):
    raise SystemExit('resume checkpoint has no epoch')
print(epoch)
PY
}

result_is_complete() {
    "${python_bin}" - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json
import sys
try:
    with open(sys.argv[1]) as stream:
        result = json.load(stream)
    graph_size, vehicles, capacity, val_size, t_max = map(int, sys.argv[2:])
    valid = (result.get('graph_size') == graph_size
             and result.get('num_vehicles') == vehicles
             and result.get('vehicle_capacities') == [capacity] * vehicles
             and result.get('val_size') == val_size
             and result.get('T_max') == t_max
             and len(result.get('instances', [])) == val_size)
except (OSError, ValueError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

for graph_size in 50 100; do
    [[ -f "${project_dir}/../pdp_n2s/datasets/pdp_${graph_size}.pkl" ]] || {
        echo "Dataset not found: pdp_${graph_size}.pkl" >&2
        exit 1
    }
done

result_root="${script_dir}/result"
output_root="${script_dir}/output"
mkdir -p "${result_root}" "${output_root}"
exec 8>"${result_root}/.mv${vehicles}.lock"
flock -n 8 || { echo "Another ${vehicles}-vehicle run is active" >&2; exit 1; }
active_file="${result_root}/.active_mv${vehicles}"
if [[ "${NEW_RUN:-0}" == 1 || ! -f "${active_file}" ]]; then
    result_run_dir=$(mktemp -d "${result_root}/run_$(date +%Y%m%d_%H%M%S)_mv${vehicles}_XXXX")
    run_id=${result_run_dir##*/}
    printf '%s\n' "${run_id}" > "${active_file}.tmp"
    mv "${active_file}.tmp" "${active_file}"
else
    read -r run_id < "${active_file}"
    [[ "${run_id}" == run_*_mv${vehicles}_* ]] || { echo "Invalid active run ID" >&2; exit 1; }
    result_run_dir="${result_root}/${run_id}"
    [[ -d "${result_run_dir}" ]] || { echo "Active result directory missing" >&2; exit 1; }
fi
output_run_dir="${output_root}/${run_id}"
mkdir -p "${output_run_dir}"
echo "Results: ${result_run_dir}"
echo "Weights: ${output_run_dir}"

for graph_size in 50 100; do
    if [[ ${graph_size} -eq 50 ]]; then
        capacity=${capacity_50}
        warm_up=${warm_up_50}; max_grad_norm=${max_grad_norm_50}
        batch_size=${batch_size_50}; epoch_size=${epoch_size_50}; epoch_end=${epoch_end_50}
        t_train=${t_train_50}; t_max=${t_max_50}; eval_t_max=${eval_t_max_50}
        lr_model=${lr_model_50}; lr_critic=${lr_critic_50}
        train_val_size=${train_val_size_50}; train_val_batch_size=${train_val_batch_size_50}
        eval_size=${eval_size_50}; eval_batch_size=${eval_batch_size_50}
    else
        capacity=${capacity_100}
        warm_up=${warm_up_100}; max_grad_norm=${max_grad_norm_100}
        batch_size=${batch_size_100}; epoch_size=${epoch_size_100}; epoch_end=${epoch_end_100}
        t_train=${t_train_100}; t_max=${t_max_100}; eval_t_max=${eval_t_max_100}
        lr_model=${lr_model_100}; lr_critic=${lr_critic_100}
        train_val_size=${train_val_size_100}; train_val_batch_size=${train_val_batch_size_100}
        eval_size=${eval_size_100}; eval_batch_size=${eval_batch_size_100}
    fi

    val_dataset="${project_dir}/../pdp_n2s/datasets/pdp_${graph_size}.pkl"
    label="n${graph_size}_mv${vehicles}_cap${capacity}"
    result_dir="${result_run_dir}/${label}"
    model_root="${output_run_dir}/${label}/models"
    mkdir -p "${result_dir}" "${model_root}"

    common_args=(--problem mvpdtsp --graph_size "${graph_size}"
        --num_vehicles "${vehicles}" --capacity "${capacity}"
        --objective proposed --seed 1234)
    train_cmd=("${python_bin}" -u run.py "${common_args[@]}"
        --warm_up "${warm_up}" --max_grad_norm "${max_grad_norm}"
        --batch_size "${batch_size}" --epoch_size "${epoch_size}"
        --checkpoint_epochs 1
        --epoch_end "${epoch_end}" --T_train "${t_train}"
        --lr_model "${lr_model}" --lr_critic "${lr_critic}"
        --val_dataset "${val_dataset}" --val_size "${train_val_size}"
        --val_batch_size "${train_val_batch_size}"
        --T_max "${t_max}"
        --run_name "${label}" --output_dir "${model_root}"
        --results_dir "${result_dir}"
        --log_dir "${output_run_dir}/${label}/tensorboard")
    signature=$(
        printf '%q ' "${train_cmd[@]}"
        printf 'eval_size=%q eval_batch_size=%q eval_t_max=%q' \
            "${eval_size}" "${eval_batch_size}" "${eval_t_max}"
    )
    signature_file="${result_dir}/training_signature.txt"
    if [[ -f "${signature_file}" ]]; then
        [[ "$(cat "${signature_file}")" == "${signature}" ]] || {
            echo "Settings changed for ${label}; use NEW_RUN=1 to start a separate run" >&2
            exit 1
        }
    else
        printf '%s\n' "${signature}" > "${signature_file}"
    fi

    completed_result=''
    shopt -s nullglob
    for candidate in "${result_dir}"/mvpdtsp_results_*.json; do
        if result_is_complete "${candidate}" "${graph_size}" "${vehicles}" \
            "${capacity}" "${eval_size}" "${eval_t_max}"; then
            completed_result=${candidate}
            break
        fi
    done
    shopt -u nullglob
    if [[ -n "${completed_result}" ]]; then
        echo "Already completed ${label}: ${completed_result}"
        continue
    fi

    resume_checkpoint=''
    save_root="${model_root}/mvpdtsp_${graph_size}"
    if [[ -d "${save_root}" ]]; then
        resume_checkpoint=$(find "${save_root}" -mindepth 2 -maxdepth 2 \
            -type f -name 'resume-latest.pt' | sort -r | head -n 1)
    fi
    resume_epoch=-1
    if [[ -n "${resume_checkpoint}" ]]; then
        resume_epoch=$(read_checkpoint_epoch "${resume_checkpoint}")
        (( resume_epoch < epoch_end )) || {
            echo "Resume checkpoint exceeds epoch_end for ${label}" >&2
            exit 1
        }
    fi
    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
        printf '%q ' "${train_cmd[@]}"
        printf '\n'
    } > "${result_dir}/train_command.txt"
    if (( resume_epoch < epoch_end - 1 )); then
        if [[ -n "${resume_checkpoint}" ]]; then
            train_cmd+=(--resume "${resume_checkpoint}")
            echo "Resuming ${label} after epoch ${resume_epoch}"
        else
            echo "Training ${label}: ${epoch_end} epochs on GPU ${CUDA_VISIBLE_DEVICES}"
        fi
        {
            printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
            printf '%q ' "${train_cmd[@]}"
            printf '\n'
        } >> "${result_dir}/train_commands.txt"
        "${train_cmd[@]}" 2>&1 | tee -a "${result_dir}/train.log"
    else
        echo "Training finished for ${label}; continuing with evaluation"
    fi

    # Evaluate the separate copy of the best validated checkpoint.
    checkpoint=$(find "${model_root}/mvpdtsp_${graph_size}" -type f \
        -name 'optimal-*.pt' | sort -V | tail -n 1)
    [[ -f "${checkpoint}" ]] || { echo "Optimal checkpoint not found for ${label}" >&2; exit 1; }
    eval_cmd=("${python_bin}" -u run.py --eval_only "${common_args[@]}"
        --val_dataset "${val_dataset}" --val_size "${eval_size}"
        --val_batch_size "${eval_batch_size}"
        --T_max "${eval_t_max}" --load_path "${checkpoint}"
        --results_dir "${result_dir}" --output_dir "${model_root}"
        --run_name "${label}_eval" --no_saving --no_tb)
    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
        printf '%q ' "${eval_cmd[@]}"
        printf '\n'
    } > "${result_dir}/eval_command.txt"
    echo "Evaluating ${label} from ${checkpoint}"
    "${eval_cmd[@]}" 2>&1 | tee -a "${result_dir}/eval.log"

    result_files=''
    shopt -s nullglob
    for candidate in "${result_dir}"/mvpdtsp_results_*.json; do
        if result_is_complete "${candidate}" "${graph_size}" "${vehicles}" \
            "${capacity}" "${eval_size}" "${eval_t_max}"; then
            result_files=${candidate}
            break
        fi
    done
    shopt -u nullglob
    [[ -n "${result_files}" ]] || {
        echo "Evaluation JSON not found for ${label}" >&2
        exit 1
    }
    echo "Completed ${label}; evaluation result: ${result_files}"
done
