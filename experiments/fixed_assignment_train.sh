#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_ROOT="outputs/mvpdtsp_fixed_20"
PREFIX="mvpdtsp20_mv2_fixed_proposed_log_"
mkdir -p "${MODEL_ROOT}"

if run_dir=$(find_latest_training_run "${MODEL_ROOT}" "${PREFIX}" mvpdtsp_fixed 20 2 proposed complete); then
    echo "[SKIP] Completed fixed-assignment training: ${run_dir}"
    exit 0
fi

resume_args=()
if run_dir=$(find_latest_training_run "${MODEL_ROOT}" "${PREFIX}" mvpdtsp_fixed 20 2 proposed incomplete); then
    checkpoint=$(latest_checkpoint "${run_dir}")
    [[ -z "${checkpoint}" ]] || resume_args=(--resume "${checkpoint}")
fi

cmd=("${PYTHON}" -u run.py
    --problem mvpdtsp_fixed --graph_size 20 --num_vehicles 2
    --objective proposed --seed 1234
    --warm_up 2 --max_grad_norm 0.05
    --batch_size 600 --epoch_size 12000 --epoch_end 200 --T_train 250
    --val_dataset ./datasets/pdp_20.pkl --val_size 1000 --val_batch_size 1000
    --T_max 1500 --run_name mvpdtsp20_mv2_fixed_proposed_log
    "${resume_args[@]}")
echo "[INFO] Training Fixed Assignment N=20 K=2"
run_training_logged result_n2s/Fixed_Assignment/training_logs fixed_assignment "${cmd[@]}"
