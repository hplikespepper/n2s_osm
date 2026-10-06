#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_ROOT="outputs/mvpdtsp_20"
mkdir -p "${MODEL_ROOT}"

for k in 3 4 5; do
    prefix="mvpdtsp20_mv${k}_proposed_log_"
    if run_dir=$(find_latest_training_run "${MODEL_ROOT}" "${prefix}" mvpdtsp 20 "${k}" proposed complete); then
        echo "[SKIP] Completed N=20 K=${k} proposed training: ${run_dir}"
        continue
    fi

    resume_args=()
    if run_dir=$(find_latest_training_run "${MODEL_ROOT}" "${prefix}" mvpdtsp 20 "${k}" proposed incomplete); then
        checkpoint=$(latest_checkpoint "${run_dir}")
        if [[ -n "${checkpoint}" ]]; then
            resume_args=(--resume "${checkpoint}")
            echo "[INFO] Resuming N=20 K=${k} from ${checkpoint}"
        fi
    fi

    cmd=("${PYTHON}" -u run.py
        --problem mvpdtsp --graph_size 20 --num_vehicles "${k}"
        --objective proposed --seed 1234
        --warm_up 2 --max_grad_norm 0.05
        --batch_size 600 --epoch_size 12000 --epoch_end 200 --T_train 250
        --val_dataset ./datasets/pdp_20.pkl --val_size 1000 --val_batch_size 1000
        --T_max 1500 --run_name "mvpdtsp20_mv${k}_proposed_log"
        "${resume_args[@]}")
    echo "[INFO] Training Fleet Scalability N=20 K=${k}"
    run_training_logged result_n2s/Fleet_Scalability/training_logs "K_${k}" "${cmd[@]}"
done
