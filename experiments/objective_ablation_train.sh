#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_ROOT="outputs/mvpdtsp_20"

for objective in distance only_makespan; do
    prefix="mvpdtsp20_mv2_${objective}_log_"
    if run_dir=$(find_latest_training_run "${MODEL_ROOT}" "${prefix}" mvpdtsp 20 2 "${objective}" complete); then
        echo "[SKIP] Completed N=20 K=2 ${objective} training: ${run_dir}"
        continue
    fi
    resume_args=()
    if run_dir=$(find_latest_training_run "${MODEL_ROOT}" "${prefix}" mvpdtsp 20 2 "${objective}" incomplete); then
        checkpoint=$(latest_checkpoint "${run_dir}")
        [[ -z "${checkpoint}" ]] || resume_args=(--resume "${checkpoint}")
    fi
    cmd=("${PYTHON}" -u run.py
        --problem mvpdtsp --graph_size 20 --num_vehicles 2
        --objective "${objective}" --seed 1234
        --warm_up 2 --max_grad_norm 0.05
        --batch_size 600 --epoch_size 12000 --epoch_end 200 --T_train 250
        --val_dataset ./datasets/pdp_20.pkl --val_size 1000 --val_batch_size 1000
        --T_max 1500 --run_name "mvpdtsp20_mv2_${objective}_log"
        "${resume_args[@]}")
    echo "[INFO] Training Objective Ablation variant=${objective}"
    run_training_logged result_n2s/Objective_Ablation/training_logs "${objective}" "${cmd[@]}"
done
